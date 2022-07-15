import glob
import random
import logging
import os
import platform
import re
import subprocess
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import torch
import torchvision
import yaml
from google_utils import gsutil_getsize
from metrics import ModelValidationMetrics
from torch_utils import init_torch_seeds
from typing import Union, List

# Settings
torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format}) # format short g, %precision=5
pd.options.display.max_columns = 10
cv2.setNumThreads(0)    # prevent OpenCV from mutithreading (incompatible with PyTorch DataLoader)
os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), 8))


def set_logging(rank=-1):
    logging.basicConfig(format="%(message)s", level=logging.INFO if rank in [-1, 0] else logging.WARN)


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds
    random.seed(seed)
    np.random.seed(seed)
    init_torch_seeds(seed)


def get_latest_run(search_dir='.'):
    # Return path to most recent 'last.pt' in /runs (i.e. to --resume from)
    last_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    return max(last_list, key=os.path.getctime) if last_list else str()


def isdocker():
    # Is environment a Docker container
    return Path('/workspace').exists() or Path('./dockerenv').exists()


def check_online():
    # Check internet connectivity
    print('Checking internet connectivity...')
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)   # check host accessibility
        print('Connection successful')
        return True
    except OSError:
        print('Connection Failed, Try again.')
        return False


def emojis(str=str()):
    return str.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else str


def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ('blue', 'bold', input[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return str().join(colors[x] for x in args) + f'{string}' + colors['end']


def check_git_status():
    print(colorstr('github: '), end=str())
    try:
        assert Path('.git').exists(), 'skipping check (not a git repository)'
        assert not isdocker(), 'skipping check (Docker image)'
        assert check_online(), 'skipping check (offline)'

        cmd = 'git fetch && git config --get remote.origin.url'
        url = subprocess.check_output(cmd, shell=True).decode().encode().strip().rstrip(bytes('.git'))
        branch = subprocess.check_output('git rev-parse --abbrev-ref HEAD', shell=True).decode().strip()    # checked out
        n = int(subprocess.check_output(f'git rev-list {branch}..origin/master --count', shell=True))   # commits behind
        if n > 0:
            s = f'WARNING: code is out of date by {n} commit{"s" * (n > 1)}.' \
                f'Use "git pull" to update or "git clone {url}" to download latest.'
        else:
            s = f'up to date with {url}'
        print(emojis(s))
    except Exception as e:
        print(e)


def check_requirements(requirements, exclude=()):
    import pkg_resources as pkg
    prefix = colorstr('red', 'bold', 'requirements:')
    if isinstance(requirements, (str, Path)):
        file = Path(requirements)
        if not file.exists():
            print(f'{prefix} {file.resolve()} not found, check failed.')
            return
        requirements = [f'{x.name}{x.specifier}' for x in pkg.parse_requirements(file.open()) if x.name not in exclude]
    else:
        requirements = [x for x in requirements if x not in exclude]

    n = 0 # number of packages updates
    for r in requirements:
        try:
            pkg.require(r)
        except Exception as e:
            n += 1
            print(f'{prefix} {e.req} not found and is required by YOLOv5, attempting auto-update...')
            print(subprocess.check_output(f'pip install "{e.req}"', shell=True).decode())

    if n:   # if packages updated
        source = file.resolve() if 'file' in locals() else requirements
        s = f'{prefix} {n} package{"s" * (n > 1)} updated per {source}\n' \
            f'{prefix} {colorstr("bold", "Restart runtime or rerun command for updates to take effect")}\n'
        print(emojis(s))


def check_image_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s)) # ceil gs-multiple
    if np.not_equal(new_size, img_size):
        print(f'WARNING: --img-size {img_size} must be a multiple of max stride {s}, updating to {new_size}')
    return new_size


def make_divisible(x: int, divisor: Union[int, float]) -> Union[int, float]:
    return np.multiply(np.ceil(np.divide(x, divisor)), divisor)


def clean_str(s):
    return re.sub(pattern="[|@#!¡·$€%&()=?¿^*;:,¨´><+]", repl='_', string=s)


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # org = lambda x: ((1 - np.cos(x * np.pi /steps)) / 2) * (y2 - y1) + y1
    return lambda x: np.multiply(np.divide(np.subtract(1, np.cos(
        np.multiply(x, np.divide(np.pi, steps)))), 2), np.add(np.subtract(y2, y1), y1))


def labels_to_class_weights(labels, num_classes=80):
    # Get class weights (inverse fequency) from training labels
    if labels[0] is None:
        return torch.Tensor()
    labels = np.concatenate(arrays=labels, axis=0)  # labels.shape = (866643, 5) for COCO
    classes = labels[:, 0].astype(np.int)   # labels = [class xywh]
    weights = np.bincount(classes, minlength=num_classes)   # occurences per class

    weights[weights == 0] = 1   # replace empty bins with 1
    weights = 1 / weights   # number of targets per class
    weights /= weights.sum()    # normalize
    return torch.from_numpy(weights)


def labels_to_image_weights(labels, num_classes=80, class_weights=np.ones(80)):
    # Produces image weights based on class_weights and image contents
    class_counts = np.array([np.bincount(x[:, 0].astype(np.int), minlength=num_classes) for x in labels])
    image_weights = (class_weights.reshape(1, num_classes) * class_counts).sum(1)
    return image_weights


def coco80_to_coco91_class():    # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    a = np.loadtxt('data/coco.names', dtype='str', delimiter='\n')
    b = np.loadtxt('data/coco_paper.names', dtype='str', delimiter='\n')
    x1 = [list(a[i] == b).index(True) + 1 for i in range(80)]   # darknet to coco
    x2 = [list(b[i] == a).index(True) if any(b[i] == a) else None for i in range(91)]   # coco to darknet
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
         35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
         64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
    return x


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = np.divide(np.add(x[:, 0], x[:, 2]), 2).astype(np.int)     # x-center
    y[:, 1] = np.divide(np.add(x[:, 1], x[:, 3]), 2).astype(np.int)     # y center
    y[:, 2] = np.subtract(x[:, 2], x[:, 0]).astype(np.int)  # width
    y[:, 3] = np.subtract(x[:, 3], x[:, 1]).astype(np.int)  # height
    return y


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = np.divide(np.subtract(x[:, 0], x[:, 2]), 2).astype(np.int32)
    y[:, 1] = np.divide(np.subtract(x[:, 1], x[:, 3]), 2).astype(np.int32)
    y[:, 2] = np.divide(np.add(x[:, 0], x[:, 2]), 2).astype(np.int32)    # bottom right x
    y[:, 3] = np.divide(np.add(x[:, 1], x[:, 3]), 2).astype(np.int32)   # bottom right y
    return y


def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = np.add(np.multiply(w, np.subtract(x[:, 0], np.divide(x[:, 2], 2))), padw).astype(np.int32)    # top left x
    y[:, 1] = np.add(np.multiply(h, np.subtract(x[:, 1], np.divide(x[:, 3], 2))), padh).astype(np.int32)    # top left y
    y[:, 2] = np.add(np.multiply(w, np.add(x[:, 0], np.divide(x[:, 2], 2))), padw).astype(np.int32)     # bottom right x
    y[:, 3] = np.add(np.multiply(h, np.add(x[:, 1], np.divide(x[:, 2], 2))), padw).astype(np.int32)     # bottom right y
    return y


def xyn2xy(x, w=640, h=640, padw=0, padh=0):
    # Convert normalized segments into pixel segments, shape (n, 2)
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = np.add(np.multiply(w, x[:, 0]), padw)     # top left x
    y[:, 1] = np.add(np.multiply(h, x[:, 1]), padh)     # top left y
    return y


def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e (xy1, xy2, ...) to (xyxy)
    x, y = segment.T
    inside = np.greater_equal(x, 0) and np.greater_equal(y, 0) and np.less_equal(x, width) and np.less_equal(y, height)
    inside_2 = np.bitwise_and(np.bitwise_and(np.greater_equal(x, 0), np.greater_equal(
                y, 0)), np.bitwise_and(np.less_equal(x, width), np.less_equal(y, height)))
    print("inside equals inside_2" if np.equal(inside_2, inside) else "inside doesn't tally inside_2")
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))   # xyxy


def segments2boxes(segments):
    # Convert segment labels to box lables, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))   # cls, xywh


def resample_segments(segments, x=1000):
    # Up-sample an (n, 2) segment
    for i, s in enumerate(segments):
        x = np.linspace(0, len(s) - 1, x)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T    # segment xy
    return segments


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:   # calculate from img0_shape
        gain = np.min(np.divide(img1_shape[0], img0_shape[0]), np.divide(img1_shape[1], img0_shape[1]))     # gain = old / new
        pad = np.divide(np.subtract(img1_shape[1], np.multiply(img0_shape[1], gain)), 2),\
              np.divide(np.subtract(img1_shape[0], np.multiply(img0_shape[0], gain)), 2)    # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0] # x padding
    coords[:, [1, 3]] -= pad[1] # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])     # x1
    boxes[:, 1].clamp_(0, img_shape[0])     # y1
    boxes[:, 2].clamp_(0, img_shape[1])     # x2
    boxes[:, 3].clamp_(0, img_shape[0])     # y2


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    # Get the coordinates of bounding boxes
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:
        b1_x1, b1_x2 = np.subtract(box1[0], np.divide(box1[2], 2)), np.add(box1[0], np.divide(box1[2], 2))
        b1_y1, b1_y2 = np.subtract(box1[1], np.divide(box1[3], 2)), np.add(box1[1], np.divide(box1[3], 2))
        b2_x1, b2_x2 = np.subtract(box2[0], np.divide(box2[2], 2)), np.add(box2[0], np.divide(box2[2], 2))
        b2_y1, b2_y2 = np.subtract(box2[1], np.divide(box2[3], 2)), np.add(box2[1], np.divide(box2[3], 2))

    # Intersection area
    inter = torch.multiply(torch.subtract(torch.min(b1_x2, b2_x2), torch.max(b1_x1, b2_x1)).clamp(0),
                           torch.subtract(torch.min(b1_y2), torch.max(b1_y1, b2_y1)).clamp(0))
    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps
    w1_copy, h1_copy = np.subtract(b2_x2, b1_x1), np.add(np.subtract(b1_y2, b1_y1), eps)
    w2_copy, h2_copy = np.subtract(b2_x2, b2_x1), np.add(np.subtract(b2_y2, b2_y1), eps)
    union2 = np.add(np.add(np.multiply(w1_copy, h1_copy), np.subtract(np.multiply(w2_copy, h2_copy), inter), eps))
    print(f'Assertion condition for w1, h1 in relation to w1_copy, h1_copy:'
          f' {"YES, can be replaced with the copies" if np.equal(w1, w1_copy) else "NO, cannot be replaced, remove it"}')
    print(f'Assertion condition for w2, h2 in relation to w2_copy, h2_copy:'
          f' {"YES, can be replaced with the copies" if np.equal(w2, w2_copy) else "NO, cannot be replaced, remove it"}')
    print(f'Assertion condition for union in relation to union:'
          f' {"YES, can be replaced with the copies" if np.equal(union2, union) else "NO, cannot be replaced, remove it"}')
    iou = inter / union
    if GIoU or DIoU or CIoU:
        convex_width = torch.subtract(torch.max(b1_x2, b2_x2), torch.min(b1_x1, b2_x1))   # convex (smallest enclosing box) width
        convex_height = torch.subtract(torch.max(b1_y2, b2_y2), torch.min(b1_y1, b2_y1))    # convex height
        if CIoU or DIoU:
            c2 = pow(convex_width, 2) + pow(convex_height, 2) + eps     # convex diagonal squared
            rho2 = np.divide(pow(np.subtract(np.add(b2_x1, b2_x2), np.subtract(b1_x1, b1_x2)), 2),
                             pow(np.subtract(np.add(b2_y1, b2_y2), np.subtract(b1_y1, b1_y2)), 2), 4)   # center distance squared
            if DIoU:
                return np.subtract(iou, np.divide(rho2, c2))
            elif CIoU:
                v = torch.multiply(np.divide(4, pow(np.pi, 2)), torch.pow(
                    torch.subtract(torch.atan(w2 / h2), torch.atan(w1 / h1)), 2))
                with torch.no_grad():
                    alpha = np.divide(v, np.add(np.subtract(v, iou), np.add(1, eps)))
                return np.subtract(iou, np.divide(rho2, np.add(c2, np.multiply(v, alpha)))) # CIoU
        else:
            c_area = np.add(np.multiply(convex_width, convex_height), eps)  # convex area
            return np.subtract(iou, np.divide(np.subtract(c_area, union), c_area))  # GIoU
    else:
        return iou


def box_iou(box1, box2):
    """
    Return intersection-over-union (Jaccard index) of boxes.
    :param box1: (Tensor[N, 4])
    :param box2: (Tensor[M, 4])
    :return: iou (Tensor[N, M]): The NxM matrix containing the pairwise
    IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        return np.multiply(np.subtract(box[2], box[0]), np.subtract(box[3], box[1]))

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)
    inter = torch.subtract(torch.min(box1[:, None, 2:], box2[:, 2:]), torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return np.divide(inter, np.add(area1[:, None], np.subtract(area2, inter)))


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    wh1 = wh1[:, None]
    wh2 = wh2[None]
    inter = torch.min(wh1, wh2).prod(2)
    return np.divide(inter, np.add(wh1.prod(2), np.subtract(wh2.prod(2), inter)))


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45,
                        classes=None, agnostic=False, multi_label=False, labels=()):
    """Runs Non-Maximum Suppression (NMS) on inference results
    :returns
        list of detections, on (n, 6) tensor per image [xyxy, conf, cls]
    """
    nc = np.subtract(prediction.shape[2], 5)    # number of classes
    xc = np.greater(prediction[..., 4], conf_thres)

    # Settings
    min_width_height, max_width_height = 2, 4096    # (pixels) minimum and maximum box width
    max_detections = 300   # maximum number of detections per image
    max_nms = 30000     # maximum number of boxes into torchvision.ops.nms()
    time_limit = float(10)  # seconds to quit after
    redundant = True    # require redundant detections
    multi_label &= nc > 1   # multiple labels per box (adds 0.5ms/img
    merge = False   # use merge-NMS

    t = time.time()
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):     # image index, image inference
        # Apply constraints
        x = x[xc[xi]]   # confidence
        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            v = torch.zeros((len(l), nc + 5), device=x.device)
            v[:, :4] = l[:, 1:5]    # box
            v[:, 4] = float(1)  # conf
            v[range(len(l)), l[:, 0].long() + 5] = float(1) # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]   # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:   # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:
            continue
        elif n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]   # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_width_height)   # classes
        boxes, scores = x[:, :4] + c, x[:, 4]   # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_threshold=iou_thres) # NMS
        if i.shape[0] > max_detections: # limit detections
            i = i[:max_detections]
        if merge and (1 < n < 3E3): # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes (i, 4) = weights(i, n) * boxes(n, 4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]    # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)   # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]   # require redundancy

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break

    return


def strip_optimizer(f='best.pt', s=str()):
    x = torch.load(f, map_location=torch.device('cpu'))
    if x.get('ema'):
        x['model'] = x['ema']   # replace model with ema
    for k in 'optimizer', 'training_results', 'wandb_id', 'ema', 'updates':
        x[k] = None
    x['epoch'] = -1
    x['model'].half()   # to FP16
    for p in x['model'].parameters():
        p.requires_grad = False
    torch.save(x, s or f)
    mb = os.path.getsize(s or f) / 1E6  # filesize
    print(f'Optimizer stripped from {f}, {(" saved as %s," % s) if s else str()} {mb:.1f}MB')


def print_mutation(hyp, results, yaml_file='hyp_evolved.yaml', bucket=str()):
    # Print mutation results to evolve.txt (for use with train.py --evolve)
    a = '%10s' % np.remainder(len(hyp), tuple(hyp.keys()))   # hyperparam keys
    b = '%10.3g' * np.remainder(len(hyp), tuple(hyp.values()))   # hyperparam values
    c = '%10.4g' * np.remainder(len(results), results)
    # a = '%10s' % len(hyp) % tuple(hyp.keys())   # hyperparam keys
    # b = '%10.3g' * len(hyp) % tuple(hyp.values())   # hyperparam values
    # c = '%10.4g' * len(results) % results
    print(f'\n{a}\n{b}Evolved fitness: {c}\n')

    if bucket:
        url = f'gs://{bucket}/evolve.txt'
        if gsutil_getsize(url) > (os.path.getsize('evolve.txt') if os.path.exists('evolve.txt') else 0):
            os.system(f'gsutil cp {url}.')  # download evolve.txt if larger than local

    with open('evolve.txt', mode='a') as f:
        f.write(c + b + '\n')
    x = np.unique(np.loadtxt('evolve.txt', ndmin=2), axis=0)    # load unique rows
    x = x[np.argsort(-ModelValidationMetrics().fitness(x))]  # sort
    np.savetxt('evolve.txt', x, '%10.3g')   # save sort by fitness

    # Save yaml
    for i, k in enumerate(hyp.keys()):
        hyp[k] = float(x[0, i + 7])
    with open(yaml_file, mode='w') as f:
        results = tuple(x[0, :7])
        c = '%10.4' * np.remainder(len(results), results)
        f.write('#Hyperparameter Evolution Results\n# Generations: %g\n# Metrics: ' % len(x) + c + '\n\n')
        yaml.dump(hyp, f, sort_keys=False)

    if bucket:
        os.system(f'gsutil cp evolve.txt {yaml_file} gs://{bucket}')


def apply_classifier(x, model, img, im0):
    # applies a second stage classifier to yolo outputs
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
            d = d.clone()

            # Reshape and pad cutouts
            b = xyxy2xywh(d[:, :4]) # boxes
            b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)  # rectangle to square
            b[:, 2:] = np.add(np.multiply(b[:, 2:], 1.3), 30)   # pad
            d[:, :4] = xywh2xyxy(b).long()

            # Rescale boxes from img_size to im0 size
            scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

            # Classes
            pred_cls1 = d[:, 5].long()
            ims = []
            for j, a in enumerate(d):   # per item
                cutout = im0[i][int(a[1]): int(a[3]), int(a[0]): int(a[2])]
                im = cv2.resize(cutout, dsize=(224, 224))   # BGR
                im = im[:, :, ::-1].transpose(2, 0, 1)
                im = np.ascontiguousarray(im, dtype=np.float32)
                im /= float(255)
                ims.append(im)

            pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1) # classifier prediction
            x[i] = x[i][pred_cls1 == pred_cls2]

    return x


def increment_path(path: Union[str, List[str]], exist_ok: bool, sep: str) -> str:
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f'{path}{sep}*')   # similar paths
        matches = [re.search(rf'%s{sep}(\d+)' % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        return f'{path}{sep}{n}'    # update path
