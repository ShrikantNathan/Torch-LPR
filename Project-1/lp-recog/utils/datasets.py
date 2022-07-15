import glob
import logging
import os
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread
import cv2
import numpy as np
import torch
import random
import torch.nn.functional as F
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm
from general import check_requirements, xyxy2xywh, xywh2xyxy, xywhn2xyxy, xyn2xy,\
    segment2box, segments2boxes, resample_segments, clean_str
from torch_utils import torch_distributed_zero_first

# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']    # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv'] # acceptable video suffixes
logger = logging.getLogger(__name__)

# Get orientation exif tag
for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def get_hash(files):
    # Returns a single hash value of a list of files
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:
            s = (s[1], s[0])
        elif rotation == 8:
            s = (s[1], s[0])
    except:
        pass

    return s


def create_dataloader(path, imgsz, batch_size, stride, opt, hyp=None, augment=False, cache=False, pad=0.0, rect=False,
                      rank=-1, world_size=1, workers=8, image_weights=False, quad=False, prefix=str()):
    # Make sure only the first process in DDP process the dataset first, and the following others can use the cache
    with torch_distributed_zero_first(rank):
        dataset = LoadImagesAndLabels(path=path, img_size=imgsz, batch_size=batch_size, augment=augment,
                                      hyp=hyp, rect=rect, cache_images=cache, single_cls=opt.single_cls,
                                      stride=int(stride), pad=pad, image_weights=image_weights, prefix=prefix)
    batch_size = min(batch_size, len(dataset))
    worker_nos = min([np.floor_divide(os.cpu_count(), world_size), batch_size if np.greater(batch_size, 1) else 0, workers])
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if np.not_equal(rank, -1) else None
    loader = torch.utils.data.DataLoader if image_weights else InfiniteDataLoader
    # Use torch.utils.data.DataLoader() if dataset.properties will update during training else InfiniteDataLoader()
    dataloader = loader(dataset=dataset, batch_size=batch_size, num_workers=worker_nos, sampler=sampler,
                        pin_memory=True, collate_fn=LoadImagesAndLabels.collate_fn4 if quad else LoadImagesAndLabels.collate_fn)
    return dataloader, dataset


class InfiniteDataLoader(torch.utils.data.DataLoader):
    """Dataloader that reuses workers
    Uses same syntax as vanilla DataLoader"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        object.__setattr__(self, 'batch_sampler', _RepeatSampler(self.batch_sampler))
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever

    Args:
        sampler (Sampler)
    """
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class LoadImages: # For inference
    def __init__(self, path, img_size=640, stride=32):
        p = str(Path(path).absolute())  # os-agnostic absolute path
        if '*' in p:
            files = sorted(glob.glob(p, recursive=True))
        elif os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.*')))
        elif os.path.isfile(p):
            files = [p]
        else:
            raise Exception(f'ERROR: {p} does not exist')

        images = [x for x in files if str(x).split('.')[-1].lower() in img_formats]
        videos = [x for x in files if str(x).split('.')[-1].lower() in vid_formats]
        ni, nv = len(images), len(videos)

        self.img_size = img_size
        self.stride = stride
        self.files = images + videos
        self.nf = ni + nv   # number of files
        self.video_flag = [False] * ni + [True] * nv
        self.mode = 'image'
        if any(videos):
            self.new_video(videos[0])   # new video
        else:
            self.cap = None
        assert np.greater(self.nf, 0), f'No images or videos found in {p}.' \
                                       f' Supported formats are:\nimages: {img_formats}\nvideos: {vid_formats}'

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if np.equal(self.count, self.nf):
            raise StopIteration
        path = self.files[self.count]

        if self.video_flag[self.count]:
            # Read video
            self.mode = 'video'
            ret, img0 = self.cap.read()
            if not ret:
                self.count += 1
                self.cap.release()
                if np.equal(self.count, self.nf):    # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret, img0 = self.cap.read()

            self.frame += 1
            print(f'video {self.count + 1}/{self.nf} ({self.frame}/{self.nframes}) {path}: ', end=str())

        else:
            # Read image
            self.count += 1
            img0 = cv2.imread(path) # BGR
            assert img0 is not None, f'Image Not Found {path}'
            print(f'image {self.count}/{self.nf} {path}: ', end=str())

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)    # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return path, img, img0, self.cap

    def new_video(self, path: str):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.nframes = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def __len__(self):
        return self.nf


class LoadWebcam:   # for inference
    def __init__(self, pipe='0', img_size=640, stride=32):
        self.img_size = img_size
        self.stride = stride

        if pipe.isnumeric():
            pipe = eval(pipe)   # local camera

        self.pipe = pipe
        self.cap = cv2.VideoCapture(pipe)   # video capture object
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)    # set buffer size
        # self.cap.setExceptionMode(enable=True)

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        if cv2.waitKey(1) == ord('q'):
            self.cap.release()
            cv2.destroyAllWindows()
            raise StopIteration

        # Read frame
        if np.equal(self.pipe, 0):
            ret, img0 = self.cap.read()
            img0 = cv2.flip(img0, 1)    # flip left-right
        else:   # IP camera
            n = 0
            while True:
                n += 1
                self.cap.grab()
                if np.equal(np.remainder(n, 30), 0):    # skip frames
                    ret, img0 = self.cap.retrieve()
                    if ret:
                        break

        assert ret, f'Camera Error {self.pipe}'
        img_path = 'webcam.jpg'
        print(f'webcam {self.count}: ', end=str())

        # Padded resize
        img = letterbox(img0, self.img_size, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)    # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img_path, img, img0, None

    def __len__(self):
        return 0


class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, mode='r') as f:
                sources = [str(x).strip() for x in f.read().strip().splitlines() if len(str(x).strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end=str())
            url = eval(s) if s.isnumeric() else s
            if 'youtube.com/' in url or 'youtu.be/' in url:
                check_requirements(('pafy', 'youtube_dl'))
                import pafy
                url = pafy.new(url).getbest(preftype='mp4').url
            cap = cv2.VideoCapture(url)
            assert cap.isOpened(), f'Failed to open {s}'
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = np.remainder(cap.get(cv2.CAP_PROP_FPS), 100)
            ret, self.imgs[i] = cap.read()  # guarantee first frame
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print(f'SUCCESS ({width} x {height} at {self.fps:.2f}) FPS.')
            thread.start()
        print('')   # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)    # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print("WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.")

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            if n == 4:  # read every 4th frame
                success, im = cap.retrieve()
                self.imgs[index] = im if success else self.imgs[index] * 0
                n = 0
            time.sleep(1 / self.fps)    # wait time

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0    # 1E12 frames = 32 streams at 30 FPS for 30 years


def img2label_paths(image_paths):
    # Define label paths as a function of image paths
    sa, sb = f'{os.sep}/"images"/{os.sep}', f'{os.sep}/"labels"/{os.sep}'
    return ['txt'.join(str(x).replace(sa, sb, 1).rsplit(str(x).split('.')[-1], 1)) for x in image_paths]


class LoadImagesAndLabels(Dataset):
    def __init__(self, path, img_size=640, batch_size=16, augment=False, hyp=None, rect=False, image_weights=False,
                 cache_images=False, single_cls=False, stride=32, pad=0.0, prefix=str()):
        self.img_size = img_size
        self.augment = augment
        self.hyp = hyp
        self.image_weights = image_weights
        self.rect = False if image_weights else rect
        self.mosaic = self.augment and not self.rect
        self.mosaic_border = [-img_size // 2, -img_size // 2]
        self.stride = stride
        self.path = path

        try:
            f = []  # image files
            for p in path if isinstance(path, list) else [path]:
                p = Path(p) # os-agnostic
                if p.is_dir():
                    f += glob.glob(str(p / '**' / '*.*'), recursive=True)
                elif p.is_file():
                    with open(p, mode='r') as t:
                        t = t.read().strip().splitlines()
                        parent = str(p.parent) + os.sep
                        f += [x.replace('./', parent) if str(x).startswith('./') else x for x in t]     # local to global path
                else:
                    raise Exception(f'{prefix}{p} does not exist')
            self.img_files = sorted([str(x).replace('/', os.sep) for x in f if str(x).split('.')[-1].lower() if img_formats])
            assert self.img_files, f'{prefix}No images found'
        except Exception as e:
            raise Exception(f'{prefix}Error loading data from {path}: {e}\nSee {help_url}')

        # Check cache
        self.label_files = img2label_paths(self.img_files)    # labels
        cache_path = (p if p.is_file() else Path(self.label_files[0].parent).with_suffix('.cache')) # cached labels
        if cache_path.is_file():
            cache, exists = torch.load(cache_path), True
            if np.not_equal(cache['hash'], get_hash(self.label_files + self.img_files)) or 'version' not in cache:
                cache, exists = self.cache_labels(cache_path, prefix), False
        else:
            cache, exists = self.cache_labels(cache_path, prefix), False    # cache

        # Display cache
        nf, nm, ne, nc, n = cache.pop('results')    # found, missing, empty, corrupted, total
        if exists:
            d = f'Scanning "{cache_path}" images and labels... {nf} found, {nm} missing, {ne} empty, {nc} corrupted'
            tqdm(iterable=None, desc=prefix + d, total=n, initial=n)    # display cache results
        assert np.greater(nf, 0) or not augment, f'{prefix}No labels in {cache_path}. Cannot train without labels. See {help_url}'

        # Read cache
        cache.pop('hash')   # remove hash
        cache.pop('version')    # remove version
        labels, shapes, self.segments = zip(*cache.values())
        self.labels = list(labels)
        self.shapes = np.array(shapes, dtype=np.float64)
        self.img_files = list(cache.keys()) # update
        self.label_files = img2label_paths(cache.keys())  # update
        if single_cls:
            for x in self.labels:
                x[:, 0] = 0

        n = len(shapes) # number of images
        bi = np.divide(np.floor(np.arange(n), batch_size).astype(np.int))   # batch index
        nb = bi[-1] + 1 # number of batches
        self.batch = bi
        self.n = n
        self.indices = range(n)

        # Rectangular Training
        if self.rect:
            # Sort by aspect ratio
            s = self.shapes # wh
            ar = s[:, 1] / s[:, 0]  # aspect ratio
            irect = ar.argsort()
            self.img_files = [self.img_files[i] for i in irect]
            self.label_files = [self.label_files[i] for i in irect]
            self.labels = [self.labels[i] for i in irect]
            self.shapes = s[irect]
            ar = ar[irect]

            # Set training image shapes
            shapes = [[1, 1]] * nb
            for i in range(nb):
                ari = ar[bi == i]
                mini, maxi = ari.min(), ari.max()
                if maxi < 1:
                    shapes[i] = [maxi, 1]
                elif mini > 1:
                    shapes[i] = [1, 1 / mini]

            self.batch_shapes = np.ceil(np.multiply(np.array(shapes),
                                        np.divide(img_size, np.add(stride, pad)))).astype(np.int) * stride

        # Cache images into memory for faster training (WARNING: large datasets may exceed system RAM)
        self.imgs = [None] * n
        if cache_images:
            gb = 0  # Gigabytes of cached images
            self.img_hw0, self.img_hw = [None] * n, [None] * n
            results = ThreadPool(8).imap(lambda x: self.load_image(*x), zip(repeat(self), range(n))) # 8 threads
            pbar = tqdm(enumerate(results), total=n)
            for i, x in pbar:
                self.imgs[i], self.img_hw0[i], self.img_hw[i] = x
                gb += self.imgs[i].nbytes
                pbar.desc = f'{prefix}Caching images ({gb / 1E9:.1f}GB)'
            pbar.close()

    def cache_labels(self, path=Path('./labels.cache'), prefix=str()):
        # Cache dataset labels, check images and read shapes
        x = {}  # dict
        nm, nf, ne, nc = 0, 0, 0, 0
        pbar = tqdm(zip(self.img_files, self.label_files), desc='Scanning images', total=len(self.img_files))
        for i, (im_file, lb_file) in enumerate(pbar):
            try:
                # verify images
                im = Image.open(im_file)
                im.verify() # PIL verify
                shape = exif_size(im)   # image size
                segments = []   # instance segments
                assert np.bitwise_and(np.greater(shape[0], 9), np.greater(shape[1], 9)), f'image size {shape} < 10 pixels'
                assert im.format.lower() in img_formats, f'invalid image format {im.format}'
                # assert (shape[0] > 9) & (shape[1] > 9), f'image size {shape} < 10 pixels'

                # verify labels
                if os.path.isfile(lb_file):
                    nf += 1 # label found
                    with open(lb_file, mode='r') as f:
                        l = [str(x).split() for x in f.read().splitlines()]
                        if any([len(x) > 8 for x in l]):
                            classes = np.array([x[0] for x in l], dtype=np.float32)
                            segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in l]    # (cls, xy1...)
                            l = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)   # (cls, xywh)
                        l = np.array(l, dtype=np.float32)
                    if len(l):
                        assert l.shape[1] == 5, 'labels require 5 columns each'
                        assert (l >= 0).all(), 'negative labels'
                        assert (l[:, 1:] <= 1).all(), 'non-normalized or out of bounds coordinate labels'
                        assert np.unique(l, axis=0).shape[0] == l.shape[0], 'duplicate labels'
                    else:
                        ne += 1 # label empty
                        l = np.zeros((0, 5), dtype=np.float32)
                    x[im_file] = [l, shape, segments]
            except Exception as e:
                nc += 1
                print(f'{prefix}WARNING: Ignoring corrupted image and/or label {im_file}: {e}')

            pbar.desc = f'{prefix}Scanning, "{path.parent / path.stem}" images and labels...' \
                        f'{nf} found, {nm} missing, {ne} empty, {nc} corrupted'
        pbar.close()

        if np.equal(nf, 0):
            print(f'{prefix}WARNING: No labels found in {path}. See {help_url}')

        x['hash'] = get_hash(self.label_files + self.img_files)
        x['results'] = nf, nm, ne, nc, i + 1
        x['version'] = 0.1  # cache version
        torch.save(x, path) # save for next time
        logging.info(f'{prefix}New cache created: {path}')
        return x

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        index = self.indices[index]     # linear, shuffled, or image_weights
        hyp = self.hyp
        mosaic = self.mosaic and np.less(random.random(), hyp['mosaic'])
        if mosaic:
            # Load mosaic
            img, labels = self.load_mosaic(index)
            shapes = None
            # MixUP https://arxiv.org/pdf/1710.09412.pdf
            if np.less(random.random(), hyp['mixup']):
                img2, labels2 = self.load_mosaic(random.randint(0, self.n - 1))
                r = np.random.beta(8.0, 8.0)    # mixup ratio, alpha=beta=8.0
                img = np.add(np.multiply(img, r), np.multiply(img2, np.subtract(1, r))).astype(np.uint8)
                labels = np.concatenate((labels, labels2), 0)
        else:
            # Load image
            img, (h0, w0), (h, w) = self.load_image(index)

            # Letterbox
            shape = self.batch_shapes[self.batch[index]] if self.rect else self.img_size    # final letterboxed shape
            img, ratio, pad = letterbox(image=img, new_shape=shape, auto=False, scaleup=self.augment)
            shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP recaling

            labels = self.labels[index].copy()
            if labels.size: # normalized xywh to pixel xyxy format
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], ratio[0] * w, ratio[1] * h, padw=pad[0], padh=pad[1])

        if self.augment:
            if not mosaic:
                img, labels = random_perspective(img, labels, degrees=hyp['degrees'], translate=hyp['translate'],
                                                 scale=hyp['scale'], shear=hyp['shear'], perspective=hyp['perspective'])
                # Augment colorspace
                self.augment_hsv(image=img, hgain=hyp['hsv_h'], sgain=hyp['hsv_s'], vgain=hyp['hsv_v'])

                # Apply cutouts
                if np.less(random.random(), 0.9):
                    labels = cutout(image=img, labels=labels)

        nL = len(labels)    # number of labels
        if nL:
            labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])  # convert xyxy to xywh
            labels[:, [2, 4]] /= img.shape[0]   # normalized height 0-1
            labels[:, [1, 3]] /= img.shape[1]   # normalized width 0-1

        if self.augment:
            # flip up-down
            if np.less(random.random(), hyp['flipud']):
                img = np.flipud(img)
                if nL:
                    labels[:, 2] = 1 - labels[:, 2]

            # flip left-right
            if np.less(random.random(), hyp['fliplr']):
                img = np.fliplr(img)
                if nL:
                    labels[:, 1] = 1 - labels[:, 1]

            labels_out = torch.zeros((nL, 6))
            if nL:
                labels_out[:, 1:] = torch.from_numpy(labels)

            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)    # BGR to RGB, to 3 x 416 x 416
            img = np.ascontiguousarray(img)

            return torch.from_numpy(img), labels_out, self.img_files[index], shapes

    @staticmethod
    def collate_fn(batch):
        img, label, path, shapes = zip(*batch)
        for i, l in enumerate(label):
            l[:, 0] = i # add target image index for build_targets()
        return torch.stack(img, 0), torch.cat(label, 0), path, shapes

    @staticmethod
    def collate_fn4(batch):
        img, label, path, shapes = zip(*batch)  # transposed
        n = np.floor_divide(len(shapes), 4)
        img4, label4, path4, shapes4 = [], [], path[:n], shapes[:n]

        ho = torch.tensor([[0., 0, 0, 1, 0, 0]])
        wo = torch.tensor([[0., 0, 1, 0, 0, 0]])
        s = torch.tensor([[1, 1, .5, .5, .5, .5]])  # scale
        for i in range(n):
            i *= 4
            if np.less(random.random(), 0.5):
                im = F.interpolate(img[i].unsqueeze(0).float(), scale_factor=2,
                                   mode='bilinear', align_corners=False)[0].type(img[i].type())
                l = label[i]
            else:
                im = torch.cat((torch.cat((img[i], img[i + 1]), 1), torch.cat((img[i + 2], img[i + 3]), 1)), 2)
                l = torch.cat((label[i], label[i + 1] + ho, label[i + 2] + wo, label[i + 3] + ho + wo), 0) * s
            img4.append(im)
            label4.append(l)

        for i, l in enumerate(label4):
            l[:, 0] = i

        return torch.stack(img4, 0), torch.cat(label4, 0), path4, shapes4

    # Ancillary functions
    def load_image(self, index):
        # loads 1 image from dataset, returns img, original hw, resized hw
        img = self.imgs[index]
        if img is None:
            path = self.img_files[index]
            img = cv2.imread(path)  # BGR
            assert img is not None, f'Image not found in path {path}'
            height0, width0 = img.shape[:2]     # orig hw
            r = np.divide(self.img_size, max(height0, width0))  # resize file to img_size
            if np.not_equal(r, 1):
                interp = cv2.INTER_AREA if np.less(r, 1) and not self.augment else cv2.INTER_LINEAR
                img = cv2.resize(img, (int(width0 * r), int(height0 * r)), interpolation=interp)
            return img, (height0, width0), img.shape[:2]    # img, hw_original, hw_resized
        else:
            return self.imgs[index], self.img_hw0[index], self.img_hw[index]    # img, hw_original, hw_resized

    def augment_hsv(self, image, hgain=0.5, sgain=0.5, vgain=0.5):
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1     # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
        dtype = image.dtype
        x = np.arange(0, 256, dtype=np.int16)

        lut_hue = np.remainder(np.multiply(x, r[0]), 180).astype(dtype)
        lut_sat = np.clip(np.multiply(x, r[1]), 0, 255).astype(dtype)
        lut_val = np.clip(np.multiply(x, r[2]), 0, 255).astype(dtype)

        img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
        cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=image)     # no return needed

    def hist_equalize(image, clahe=True, bgr=False):
        # Equalize histogram on BGR image 'img' with img.shape(n, m, 3) and wrange (0-255)
        yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV if bgr else cv2.COLOR_RGB2YUV)
        if clahe:
            c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            yuv[:, :, 0] = c.apply(yuv[:, :, 0])
        else:
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])   # equalize Y channel histogram
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR if bgr else cv2.COLOR_YUV2RGB)   # convert YUV image to RGB

    def load_mosaic(self, index):
        # loads images in a 4-mosaic
        labels4, segments4 = [], []
        s = self.img_size
        yc, xc = [int(np.random.uniform(-x, 2 * s + x)) for x in self.mosaic_border]    # mosaic center x, y
        indices = [index] + random.choices(self.indices, k=3)   # 3 additional image indices
        x1a, y1a, x2a, y2a = 0, 0, 0, 0
        x1b, y1b, x2b, y2b = 0, 0, 0, 0
        for i, index in enumerate(indices):
            # Load image
            img, _, (height, width) = self.load_image(index)

            # place img in img4
            if np.equal(i, 0):  # top left
                img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)    # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - width, 0), max(yc - height, 0), xc, yc # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = np.subtract(width, np.subtract(x2a, x1a)),\
                                     np.subtract(height, np.subtract(y2a, y1a)), width, height  # xmin, ymin, xmax, ymax (small image)
            elif np.equal(i, 1):    # top right
                x1a, y1a, x2a, y2a = xc, max(yc - height, 0), min(xc + width, s * 2), yc
                x1b, y1b, x2b, y2b = np.subtract(width, np.subtract(x2a, x1a)),\
                                     np.subtract(height, np.subtract(y2a, y1a)), width, height
            elif np.equal(i, 2):    # bottom left
                x1a, y1a, x2a, y2a = max(xc - width, 0), yc, xc, min(s * 2, yc + height)
                x1b, y1b, x2b, y2b = np.subtract(width, np.subtract(x2a, x1a), 0, width, min(np.subtract(y2a, y1a), height))
            elif np.equal(i, 3):    # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + width, s * 2), min(s * 2, yc + height)
                x1b, y1b, x2b, y2b = 0, 0, min(width, np.subtract(x2a, x1a)), min(np.subtract(y2a, y1a), height)

            img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            padwidth = np.subtract(x1a, x1b)
            padheight = np.subtract(y1a, y1b)

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], width, height, padwidth, padheight)
                segments = [xyn2xy(x, width, height, padwidth, padheight) for x in segments]
            labels4.append(labels)
            segments4.append(segments)

        # Concat/clip labels
        labels = np.concatenate(labels4, 0)
        for x in (labels[:, 1:], *segments4):
            np.clip(x, 0, 2 * s, out=x) # clip when using random_perspective()

        # Augment
        img4, labels4 = random_perspective(
            img4, labels4, segments4, degrees=self.hyp['degrees'],
            translate=self.hyp['translate'], scale=self.hyp['scale'],
            shear=self.hyp['shear'], perspective=self.hyp['perspective'], border=self.mosaic_border)
        return img4, labels4

    def load_mosaic9(self, index):
        # loads images in a 9-mosaic
        labels9, segments9 = list(), list()
        s = self.img_size
        indices = [index] + random.choices(self.indices, k=8)   # 8 additional image indices
        for i, index in enumerate(indices):
            # Load image
            img, ret, (height, width) = self.load_image(index)
            # place img in img9
            if i == 0:
                img9 = np.full((s * 3, s * 3, img.shape[2]), 114, dtype=np.uint8)   # base image with 4 tiles
                height0, width0 = height, width
                c = s, s, s + width, s + height
            elif i == 1:
                c = s, s - height, s + width, s
            elif i == 2:
                c = s + wp, s - height, s + wp + width, s
            elif i == 3:
                c = s + width0, s, s + width0 + width, s + height
            elif i == 4:
                c = s + width0, s + hp, s + width0 + width, s + hp + height
            elif i == 5:
                c = s + width0 - width, s + height0, s + width0 - wp, s + height0 + height
            elif i == 6:
                c = s + width0 - wp - width, s + height0, s + width0 - wp, s + height0 + height
            elif i == 7:
                c = s - width, s + height0 - height, s, s + height0
            elif i == 8:
                c = s - width, s + height0 - hp - height, s, s + height0 - hp

            padx, pady = c[:2]
            x1, y1, x2, y2 = [max(x, 0) for x in c] # allocate coords

            # Labels
            labels, segments = self.labels[index].copy(), self.segments[index].copy()
            if labels.size:
                labels[:, 1:] = xywhn2xyxy(labels[:, 1:], width, height, padx, pady)    # normalized xywh to pixel xyxy format
                segments = [xyn2xy(x, width, height, padx, pady) for x in segments]
            labels9.append(labels)
            segments9.extend(segments)

            # Image
            img9[y1:y2, x1: x2] = img[y1 - pady:, x1 - padx:]
            height_prev, width_prev = height, width     # height, width previous

        # Offset
        ycenter, xcenter = [int(random.uniform(0, s)) for i in self.mosaic_border]  # mosaic center x, y
        img9 = img9[ycenter: np.add(ycenter, np.multiply(2, s)), xcenter: np.add(xcenter, np.multiply(2, s))]

        # Concat/clip labels
        labels9 = np.concatenate(labels9, 0)
        labels9[:, [1, 3]] -= xcenter
        labels9[:, [2, 4]] -= ycenter
        c = np.array([xcenter, ycenter])    # centers
        segments9 = [x - c for x in segments9]

        for x in (labels9[:, 1:], *segments9):
            np.clip(x, 0, 2 * s, out=x)     # clip when using random_perspective()

        # Augment
        img9, labels9 = random_perspective(img9, labels9, segments9, degrees=self.hyp['degrees'],
            translate=self.hyp['translate'], scale=self.hyp['scale'],
            shear=self.hyp['shear'], perspective=self.hyp['perspective'],
            border=self.hyp['mosaic_border'])
        return img9, labels9


def replicate(img, labels):
    # Replicate labels
    height, width = img.shape[:2]
    boxes = labels[:, 1:].astype(int)
    x1, y1, x2, y2 = boxes.T
    s = np.divide(np.add(np.subtract(x2, x1), np.subtract(y2, y1)), 2)  # side length (pixels)
    for i in s.argsort()[:round(s.size * 0.5)]: # smallest indices
        x1b, y1b, x2b, y2b = boxes[i]
        bh, bw = y2b - y1b, x2b - x1b
        yc, xc = int(np.random.uniform(0, height - bh)), int(np.random.uniform(0, width - bw))  # offset x, y
        x1a, y1a, x2a, y2a = [xc, yc, xc + bw, yc + bh]
        img[y1a:y2a, x1a:x2a] = img[y1b: y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        labels = np.append(labels, [[labels[i, 0], x1a, y1a, x2a, y2a]], axis=0)

    return img, labels


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = image.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = np.subtract(new_shape[1], new_unpad[0]), np.subtract(new_shape[0], new_unpad[1])   # wh padding
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)     # wh padding
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = np.divide(new_shape[1], shape[1]), np.divide(new_shape[0], shape[0])    # width, height ratios

    dw /= 2     # divide padding into 2 sides
    dh /= 2
    if np.not_equal(shape[::-1], new_unpad):
        img = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, ratio, (dw, dh)


def random_perspective(img, targets=(), segments=(), degrees=10,
                       translate=.1, scale=.1, shear=10, perspective=0.0, border=(0, 0)):
    height = np.add(img.shape[0], np.multiply(border[0], 2))
    width = np.add(img.shape[1], np.multiply(border[1], 2))

    # Center
    C = np.eye(3)
    C[0, 2] = np.divide(-img.shape[1], 2)   # x translation (pixels)
    C[1, 2] = np.divide(-img.shape[0], 2)   # y translation (pixels)

    # Perspective
    P = np.eye(3)
    P[2, 0] = random.uniform(-perspective, perspective) # x perspective (about y)
    P[2, 1] = random.uniform(-perspective, perspective) # y perspective (about x)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    s = random.uniform(1 - scale, 1 + scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = np.tan(np.multiply(np.random.uniform(-shear, shear), np.divide(np.pi, 180)))  # x shear (deg)
    S[1, 0] = np.tan(np.multiply(np.random.uniform(-shear, shear)), np.divide(np.pi, 180))  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = np.multiply(np.random.uniform(np.subtract(0.5, translate), np.add(0.5, translate)), width) # x translation (pixels)
    T[1, 2] = np.multiply(np.random.uniform(np.subtract(0.5, translate), np.add(0.5, translate)), height)   # y translation (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ P @ C   # order of operations (right to left) is IMPORTANT
    if np.not_equal(border[0], 0) or np.not_equal(border[1], 0) or np.not_equal(M, np.eye(3)).any():    # image changed
        if perspective:
            img = cv2.warpPerspective(img, M, dsize=(width, height), borderValue=(114, 114, 114))
        else:
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    n = len(targets)
    if n:
        use_segments = any(x.any() for x in segments)
        new = np.zeros((n, 4))
        if use_segments:
            segments = resample_segments(segments)  # upsample
            for i, segment in enumerate(segments):
                xy = np.ones((len(segment), 3))
                xy[:, :2] = segment
                xy = xy @ M.T   # transform
                xy = np.divide(xy[:, :2], xy[:, 2:3]) if perspective else xy[:, :2] # perspective rescale or affine
                # clip
                new[i] = segment2box(xy, width, height)
        else:
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)
            xy = xy @ M.T   # transform
            xy = (np.divide(xy[:, :2], xy[:, 2:3]) if perspective else xy[:, :2]).reshape(n, 8) # perspective rescale or affine

            # Create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # clip
            new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
            new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=targets[:, 1:5].T * s, box2=new.T, area_thr=0.01 if use_segments else 0.10)
        targets = targets[i]
        targets[:, 1:5] = new[i]
    return img, targets


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr = 0.1, eps=1e-16):
    # Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio
    w1, h1 = np.subtract(box1[2], box1[0]), np.subtract(box1[3], box1[1])
    w2, h2 = np.subtract(box2[2], box2[0]), np.subtract(box2[3], box2[1])
    ar = np.max(np.divide(w2, np.add(h2, eps)), np.divide(h2, np.add(w2, eps))) # aspect ratio
    return (w2 < wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)   # candidates


def cutout(image, labels):
    # Applies image cutout augmentation https://arxiv.org/abs/1708.04552
    height, width = image.shape[:2]

    def bbox_ioa(box1, box2):
        # Returns the intersection over box2 area given box1, box2. box1 is 4, box2 is nx4. boxes are x1y1x2y2
        box2 = box2.transpose()

        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

        # Intersection area
        inter_area = np.multiply(np.subtract(np.minimum(b1_x2, b2_x2), np.maximum(b1_x1, b2_x1).clip(0)),
                                 np.subtract(np.minimum(b1_y2, b2_y2), np.maximum(b1_y1, b2_y1)).clip(0))
        # box2 area
        box2_area = np.add(np.multiply(np.subtract(b2_x2, b2_x1), np.subtract(b2_y2, b2_y1)), 1e-16)
        # Intersection over box2 area
        return np.divide(inter_area, box2_area)

    # create random masks
    scales = [0.5] * 1 + [0.25] * 2 + [0.125] * 4 + [0.0625] * 8 + [0.03125] * 16   # image size fraction
    for s in scales:
        mask_height = np.random.randint(1, int(height * s))
        mask_width = np.random.randint(1, int(width * s))
        # box
        xmin = max(0, np.subtract(random.randint(0, width), np.floor_divide(mask_width, 2)))
        ymin = max(0, np.subtract(random.randint(0, height), np.floor_divide(mask_height, 2)))
        xmax = min(width, np.add(xmin, mask_width))
        ymax = min(height, np.add(ymin, mask_height))
        # apply random color mask
        image[ymin:ymax, xmin:xmax] = [random.randint(64, 191) for i in range(3)]

        # return unobscured masks
        if len(labels) and np.greater(s, 0.03):
            box = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
            ioa = bbox_ioa(box, labels[:, 1:5]) # intersection over area
            labels = labels[ioa < 0.60] # remove >60% unobscured labels

    return labels


def create_folder(path='./new'):
    # Create folder
    if os.path.exists(path):
        shutil.rmtree(path)     # delete output folder
    os.makedirs(path)   # make new output folder


def flatten_recursive(path='../coco128'):
    # Flatten a recursive directory by bringing all files to top level
    new_path = Path(path + '_flat')
    create_folder(str(new_path))
    for file in tqdm(glob.glob(str(Path(path)) + '/**/*.*', recursive=True)):
        shutil.copyfile(file, new_path / Path(file).name)


def extract_boxes(path='../coco128/'):
    # Convert detection dataset into classification dataset, with one directory per class
    path = Path(path)   # images dir
    shutil.rmtree(path / 'classifier') if (path / 'classifier').is_dir() else None  # remove existing
    files = list(path.rglob('*.*'))
    n = len(files)
    for im_file in tqdm(files, total=n):
        if im_file.suffix[1:] in img_formats:
            im = cv2.imread(str(im_file))[..., ::-1]    # BGR to RGB
            height, width = im.shape[:2]
            # labels
            lb_file = Path(img2label_paths([str(im_file)])[0])
            if Path(lb_file).exists():
                with open(lb_file, mode='r') as f:
                    lb = np.array([x.split() for x in f.read().strip().splitlines()], dtype=np.float32) # labels
                for j, x in enumerate(lb):
                    c = int(x[0])   # class
                    f = (path / 'classifier') / f'{c}' / f'{path.stem}_{im_file.stem}_{j}.jpg'  # new filename
                    if not f.parent.is_dir():
                        f.parent.mkdir(parents=True)
                    b = x[1:] * [width, height, width, height]  # box
                    b[2:] = np.add(np.multiply(b[2:], 1.2), 3)
                    b = xywh2xyxy(b.reshape(-1, 4)).ravel().astype(np.int)
                    b[[0, 2]] = np.clip(b[[0, 2]], 0, width)    # clip boxes outside of image
                    b[[1, 3]] = np.clip(b[[1, 3]], 0, height)
                    assert cv2.imwrite(str(f), im[b[1]: b[3], b[0]: b[2]]), f'box failure in {f}'


def autosplit(path='../coco128', weights=(0.9, 0.1, 0.0), annotated_only=False):
    """ Autosplit a dataset into train/val/test splits and save path/autosplit_*.txt files
    Usage: from utils.datasets import *; autosplit('../coco128')
    Arguments
        path:   Path to images directory
        weights:    Train, val, test weights (list)
        annotated_only: Only use images with an annotated txt file"""
    path = Path(path)   # images dir
    files = sum([list(path.rglob(f"*.{img_ext}")) for img_ext in img_formats], [])  # image files only
    n = len(files)
    indices = random.choices([0, 1, 2], weights=weights, k=n)   # assign each image to a aplit

    txt = ['autosplit_train.txt', 'autosplit_val.txt', 'autosplit_test.txt']    # 3 txt files
    [(path / x).unlink() for x in txt if (path / x).exists()]   # remove existing
    print(f'Autosplitting images from {path}' + ', using *.txt labeled images only' * annotated_only)

    for i, img in tqdm(zip(indices, files), total=n):
        if not annotated_only or Path(img2label_paths([str(img)])[0]).exists(): # check label
            with open(path / txt[i], mode='a') as f:
                f.write(str(img) + '\n')    # add image to txt file
