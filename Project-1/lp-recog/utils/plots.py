import datetime
import glob
import os
import random
from copy import copy
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from PIL import Image, ImageFont, ImageDraw
from scipy.signal import butter, filtfilt
from general import xywh2xyxy, xyxy2xywh
from metrics import ModelValidationMetrics

# Settings
# matplotlib.rc(group='rc', **{'size': 11})
# matplotlib.use('Agg')   # for writing to files only


def color_list():
    # Return first 10 plt colors as (r, g, b)
    def hex2rgb(h):
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

    return [hex2rgb(h) for h in matplotlib.colors.TABLEAU_COLORS.values()]


def hist2d(x, y, n=100):
    # 2d histogram used in labels.png and evolve.png
    xedges, yedges = np.linspace(x.min(), x.max(), n), np.linspace(y.min(), y.max(), n)
    hist, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    xidx = np.clip(np.digitize(x, bins=xedges) - 1, 0, hist.shape[0] - 1)
    yidx = np.clip(np.digitize(y, bins=yedges) - 1, 0, hist.shape[1] - 1)
    return np.log(hist[xidx, yidx])


def butter_lowpass_filtfilt(data, cutoff=1500, fs=50000, order=5):
    def butter_lowpass(cutoff, fs, order):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    b, a = butter_lowpass(cutoff, fs, order=order)
    return filtfilt(b, a, data)


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # Plots one bounding box on image img
    tl = line_thickness or round(np.multiply(0.002, np.add(img.shape[0], np.divide(img.shape[1], 2)))) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for i in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)     # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = np.add(c1[0], t_size[0]), np.subtract(c1[1], np.subtract(t_size[1], 3))
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


def plot_one_box_PIL(box, img, color=None, label=None, line_thickness=None):
    img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    line_thickness = line_thickness or max(int(min(img.size) / 200), 2)
    draw.rectangle(box, width=line_thickness, outline=tuple(color)) # plot
    if label:
        fontsize = max(round(max(img.size) / 40), 12)
        font = ImageFont.truetype('Arial.ttf', fontsize)
        text_width, text_height = font.getsize(label)
        draw.rectangle([box[0], box[1] - text_height + 4, box[0] + text_width, box[1]], fill=tuple(color))
        draw.text((box[0], box[1] - text_height + 1), label, fill=(255, 255, 255), font=font)
    return np.asarray(img)


def plot_width_height_methods():
    x = np.arange(-4.0, 4.0, .1)
    ya = np.exp(x)
    yb = torch.sigmoid(torch.from_numpy(x)).numpy() * 2

    fig = plt.figure(figsize=(6, 3), tight_layout=True)
    plt.plot(x, ya, '.-', label='YOLOv3')
    plt.plot(x, pow(yb, 2), '.-', label='YOLOv5 ^2')
    plt.plot(x, pow(yb, 1.6), '.-', label='YOLOv5 ^1.6')
    plt.xlim(left=-4, right=4)
    plt.ylim(bottom=0, top=6)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.grid()
    plt.legend()
    plt.savefig('comparison.png', dpi=200)


def output_to_target(output):
    # Convert model output to target format [batch_id, class_id, x, y, w, h, conf]
    targets = []
    for i, o in enumerate(output):
        for *box, conf, cls in o.cpu().numpy():
            targets.append([i, cls, *list(*xyxy2xywh(np.array(box)[None])), conf])
    return np.array(targets)


def plot_images(images, targets, paths=None, fname='images.jpg', names=None, max_size=640, max_subplots=16):
    if isinstance(images, torch.Tensor):
        images = images.cpu().float().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()

    # un-normalize
    if np.less_equal(np.max(images[0]), 1):
        images *= 255

    tl = 3  # line thickness
    tf = max(tl - 1, 1)     # font thickness
    batch_size, ret, height, width = images.shape
    batch_size = min(batch_size, max_subplots)  # limit plot images
    num_subplots = np.ceil(np.power(batch_size, 0.5))   # number of subplots (square)

    # check if we should resize
    scale_factor = np.divide(max_size, max(height, width))
    if np.less(scale_factor, 1):
        height = np.ceil(scale_factor * height)
        width = np.ceil(scale_factor * width)

    colors = color_list()   # list of colors
    mosaic = np.full((int(num_subplots * height), int(num_subplots * width), 3), 255, dtype=np.uint8)   # init
    for i, img in enumerate(images):
        if np.equal(i, max_subplots):
            break
        block_x = np.multiply(width, np.floor_divide(i, num_subplots)).astype('int')
        block_y = np.multiply(height, np.remainder(i, num_subplots)).astype('int')
        img = img.transpose(1, 2, 0)
        if np.less(scale_factor, 1):
            img = cv2.resize(img, (width, height))

        mosaic[block_y: block_y + height, block_x: block_x + width, :] = img
        if np.greater(len(targets), 0):
            image_targets = targets[targets[:, 0] == i]
            boxes = xywh2xyxy(image_targets[:, 2:6]).T
            classes = image_targets[:, 1].astype('int')
            labels = np.equal(image_targets.shape[1], 6)    # labels if no conf column
            conf = None if labels else image_targets[:, 6]  # check for confidence presence (label vs pred)

            if boxes.shape[1]:
                if np.less_equal(boxes.max(), 1.01):    # if normalized with tolerance 0.01
                    boxes[[0, 2]] *= width  # scale to pixels
                    boxes[[1, 3]] *= height
                elif np.less(scale_factor, 1):  # absolute coords need scale if image scales
                    boxes *= scale_factor
            boxes[[0, 2]] += block_x
            boxes[[1, 3]] += block_y
            for j, box in enumerate(boxes.T):
                cls = int(classes[j])
                color = colors[cls % len(colors)]
                cls = names[cls] if names else cls
                if labels or np.less(conf[j], 0.25):    # 0.25 conf thresh
                    label = f'{cls}' if labels else f'{cls} {conf[j]:.1f}'
                    plot_one_box(box, mosaic, label=label, color=color, line_thickness=tl)

        # Draw image filename labels
        if paths:
            label = Path(paths[i].name[:40])    # trim to 40 char
            t_size = cv2.getTextSize(text=label, fontFace=0, fontScale=tl / 3, thickness=tf)[0]
            cv2.putText(mosaic, label, (block_x + 5, block_y + t_size[1] + 5), 0,
                        tl / 3, [220, 220, 220], thickness=tf, lineType=cv2.LINE_AA)
            # Image Border
            cv2.rectangle(mosaic, (block_x, block_y), (block_x + width, block_y + height), (255, 255, 255), thickness=3)

    if fname:
        r = min(1280. / max(height, width) / num_subplots, float(1))    # ratio to limit limage size
        mosaic = cv2.resize(mosaic, (int(num_subplots * width * r), int(num_subplots * height * r)), interpolation=cv2.INTER_AREA)
        Image.fromarray(mosaic).save(fname) # PIL save
    return mosaic


def plot_lr_scheduler(optimizer, scheduler, epochs=300, save_dir=str()):
    # Plot LR simulating training for full epochs
    y = list()
    for i in range(epochs):
        scheduler.step()
        y.append(optimizer.param_groups[0]['lr'])
    plt.plot(y, '.-', label='LR')
    plt.xlabel('epoch')
    plt.ylabel('LR')
    plt.grid()
    plt.xlim(0, epochs)
    plt.ylim(0)
    plt.savefig(Path(save_dir) / 'LR.png', dpi=200)
    plt.close()


def plot_test_txt():
    # Plot test.txt histograms
    x = np.loadtxt('test.txt', dtype=np.float32)
    box = xyxy2xywh(x[:, :4])
    cx, cy = box[:, 0], box[:, 1]

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
    ax.hist2d(cx, cy, bins=600, cmax=10, cmin=0)
    ax.set_aspect('equal')
    plt.savefig('hist2d.png', dpi=300)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    ax[0].hist(cx, bins=600)
    ax[1].hist(cy, bins=600)
    plt.savefig('hist1d.png', dpi=200)


def plot_targets_txt():
    # Plot targets.txt histograms
    x = np.loadtxt('targets.txt', dtype=np.float32).T
    s = ['x targets', 'y targets', 'width targets', 'height targets']
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(8, 8), tight_layout=True)
    ax = ax.ravel()
    for i in range(4):
        ax[i].hist(x[i], bins=100, label='%.3g +/- %.3g' % (x[i].mean(), x[i].std()))
        ax[i].legend()
        ax[i].set_title(s[i])
    plt.savefig('targets.jpg', dpi=200)


def plot_study_txt(path=str(), x=None):
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(10, 6), tight_layout=True)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(8, 4), tight_layout=True)
    for f in sorted(Path(path).glob('study*.txt')):
        y = np.loadtxt(fname=f, dtype=np.float32, usecols=[0, 1, 2, 3, 7, 8, 9], ndmin=2).T
        x = np.arange(y.shape[1]) if x is None else np.array(x)
        s = ['P', 'R', 'mAP@0.5', 'mAP@.5:.95', 't_inference (ms/img)', 't_NMS (ms/img)', 't_total (ms/img)']
        j = y[3].argmax() + 1
        ax2.plot(y[6, 1:j], y[3, 1:j] * 1E2, '.-', linewidth=2, markersize=8,
                 label=f.stem.replace('study_coco_', str()).replace('yolo', 'YOLO'))

    ax2.plot(1E3 / np.array([209, 140, 97, 58, 35, 18]), [34.6, 40.5, 43.0, 47.5, 49.7, 51.5],
             'k.-', linewidth=2, markersize=8, alpha=.25, label='EfficientDet')
    ax2.grid(alpha=0.2)
    ax2.set_yticks(np.arange(20, 60, 5))
    ax2.set_xlim(0, 57)
    ax2.set_ylim(30, 55)
    ax2.set_xlabel('GPU Speed (ms/img)')
    ax2.set_ylabel('COCO AP val')
    ax2.legend(loc='lower right')
    plt.savefig(f'{str(Path(path).name)}.png', dpi=300)


def plot_labels(labels, names=(), save_dir=Path(''), loggers=None):
    print('Plotting labels... ')
    classes, boxes = labels[:, 0], labels[:, 1:].transpose()
    num_classes = int(classes.max() + 1)
    colors = color_list()
    x = pd.DataFrame(boxes.transpose(), columns=['x', 'y', 'width', 'height'])

    # seaborn correlogram
    sns.pairplot(x, corner=True, diag_kind='auto', kind='hist', diag_kws=dict(bins=50), plot_kws=dict(pmax=0.9))
    plt.savefig(save_dir / 'labels_correlogram.jpg', dpi=200)
    plt.close()

    # matplotlib labels
    matplotlib.use('svg')   # faster
    ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 8), tight_layout=True)[1].ravel()
    ax[0].hist(classes, bins=np.linspace(0, num_classes, num_classes + 1) - 0.5, rwidth=0.8)
    ax[0].set_ylabel('instances')
    if 0 < len(names) < 30:
        ax[0].set_xticks(range(len(names)))
        ax[0].set_xticklabels(names, rotation=90, fontsize=10)
    else:
        ax[0].set_xlabel('classes')
    sns.histplot(x, x='x', y='y', ax=ax[2], bins=50, pmax=0.9)
    sns.histplot(x, x='width', y='height', ax=ax[3], bins=50, pmax=0.9)

    # rectangles
    labels[:, 1:3] = 0.5    # center
    labels[:, 1:] = xywh2xyxy(labels[:, 1:]) * 2000
    img = Image.fromarray(np.ones((2000, 2000, 3), dtype=np.uint8) * 255)
    for cls, *box in labels[:1000]:
        ImageDraw.Draw(img).rectangle(box, width=1, outline=colors[int(cls) % 10])  # plot
    ax[1].imshow(img)
    ax[1].axis('off')

    for a in [0, 1, 2, 3]:
        for s in ['top', 'right', 'left', 'bottom']:
            ax[a].spines[s].set_visible(False)

    plt.savefig(save_dir / 'labels.jpg', dpi=200)
    matplotlib.use('Agg')
    plt.close()

    # loggers
    for key, val in loggers.items() or {}:
        if key == 'wandb' and val:
            val.log({"Labels": [val.Image(str(x), caption=x.name) for x in save_dir.glob('*labels*.jpg')]}, commit=False)


def plot_evolution(yaml_file='data/hyp.finetune.yaml'):
    with open(yaml_file) as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)
    x = np.loadtxt('evolve.txt', ndmin=2)
    f = ModelValidationMetrics().fitness(x)
    plt.figure(figsize=(10, 12), tight_layout=True)
    matplotlib.rc('font', **{'size': 8})
    for i, (key, val) in enumerate(hyp.items()):
        y = x[:, i + 7]
        mu = y[np.argmax(f)]    # best single result
        plt.subplot(6, 5, i + 1)
        plt.scatter(y, f, c=hist2d(y, f, 20), cmap='viridis', alpha=0.8, edgecolors='none')
        plt.plot(mu, max(f), 'k+', markersize=15)
        plt.title('%s = %.3g' % (key, mu), fontdict={'size': 9})    # limit to 40 characters
        if np.not_equal(np.remainder(i, 5), 0):
            plt.yticks([])
        print('%15s: %.3g' % (key, mu))
    plt.savefig('evolve.png', dpi=200)
    print('\nPlot saved as evolve.png')


def profile_idetection(start=0, stop=0, labels=(), save_dir=str()):
    ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 6), tight_layout=True)[1].ravel()
    s = ['Images', 'Free Storage (GB)', 'RAM Usage (GB)', 'Battery', 'dt_raw (ms)', 'dt_smooth (ms)', 'real-world FPS']
    files = list(Path(save_dir).glob('frames*.txt'))
    for file_idx, file in enumerate(files):
        try:
            results = np.loadtxt(file, ndmin=2).T[:, 90:-30]    # clip first and last rows
            num_rows = results.shape[1] # number of rows
            x = np.arange(start, min(stop, num_rows) if stop else num_rows)
            results = results[:, x]
            t = (results[0] - results[0].min())
            results[0] = x
            for i, a in enumerate(ax):
                if np.less(i, len(results)):
                    label = labels[file_idx] if len(labels) else file.stem.replace('frames_', str())
                    a.plot(t, results[i], marker='.', labels=label, linewidth=1, markersize=5)
                    a.set_title(s[i])
                    a.set_xlabel('time (s)')
                    for side in ['top', 'right']:
                        a.spines[side].set_visible(False)
                else:
                    a.remove()
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (file, e))

    ax[1].legend()
    plt.savefig(Path(save_dir) / 'idetection_profile.png', dpi=200)


def plot_results_overlay(start=0, stop=0):
    # Plot training 'results*.txt', overlaying train and val losses
    s = ['train', 'train', 'train', 'Precision', 'mAP@0.5', 'val', 'val', 'val', 'Recall', 'mAP@0.5:0.95']  # legends
    t = ['Box', 'Objectness', 'Classification', 'P-R', 'mAP-F1']    # titles
    for f in sorted(glob.glob('results*.txt') + glob.glob('../../Downloads/results*.txt')):
        results = np.loadtxt(f, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
        n = results.shape[1]    # number of rows
        x = range(start, min(stop, n) if stop else n)
        fig, ax = plt.subplots(nrows=1, ncols=5, figsize=(14, 3.5), tight_layout=True)
        ax = ax.ravel()
        # ax = np.ravel(ax)
        for i in range(5):
            for j in [i, i + 5]:
                y = results[j, x]
                ax[i].plot(x, y, marker='.', label=s[j])

            ax[i].set_title(t[i])
            ax[i].legend()
            ax[i].set_ylabel(f) if i == 0 else None     # add filename
        fig.savefig(f.replace('.txt', '.png'), dpi=200)


def plot_results(start=0, stop=0, bucket=str(), id=(), labels=(), save_dir=str()):
    # Plot training 'results*.txt.
    fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(12, 6), tight_layout=True)
    # ax = np.ravel(ax)
    ax = ax.ravel()
    s = ['Box', 'Objectness', 'Classification', 'Precision', 'Recall', 'val Box',
         'val Objectness', 'val Classification', 'mAP@0.5', 'mAP@0.5:0.95']
    if bucket:
        files = ['results%g.txt' % x for x in id]
        c = ('gsutil cp' + '%s' * len(files) + '.') % tuple('gs://%s/results%g.txt' % (bucket, x) for x in id)
        os.system(c)
    else:
        files = list(Path(save_dir).glob('results*.txt'))
    assert len(files), 'No results.txt files found in %s, nothing to plot.' % os.path.abspath(save_dir)
    for file_idx, file in enumerate(files):
        try:
            results = np.loadtxt(file, usecols=[2, 3, 4, 8, 9, 12, 13, 14, 10, 11], ndmin=2).T
            num_rows = results.shape[1]
            x = range(start, min(stop, num_rows) if stop else num_rows)
            for i in range(10):
                y = results[i, x]
                if i in [0, 1, 2, 5, 6, 7]:
                    y[y == 0] = np.nan  # don't show zero loss values
                label = labels[file_idx] if len(labels) else file.stem
                ax[i].plot(x, y, market='.', label=label, linewidth=2, markersize=8)
                ax.set_title(s[i])
        except Exception as e:
            print('Warning: Plotting error for %s; %s' % (file, e))

    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)