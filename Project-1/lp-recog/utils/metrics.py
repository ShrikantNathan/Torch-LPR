from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import general


class ModelValidationMetrics:
    def fitness(self, x):
        # Model fitness as a weighted combination of metrics
        w = [0.0, 0.0, 0.1, 0.9]
        return (x[:, :4] * w).sum(1)

    def ap_per_class(self, tp, conf, pred_cls, target_cls, plot=False, save_dir='.', names=()):
        """ Compute the average precision, given the recall and precision curves.
        Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
        # Arguments
            tp: True positives (nparray, nx1 or nx10).
            conf: Objectness value from 0-1 (nparray).
            pred_cls: Predicted object classes (nparray).
            target_cls: True object classes (nparray).
            plot: Plot precision-recall curve at mAP@0.5
            save_dir: Plot save directory
        # Returns
            The average precision as computed in py-faster-rcnn"""
        # Sort by objectness
        i = np.argsort(-conf)
        tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

        # Find unique class
        unique_classes = np.unique(target_cls)
        nc = unique_classes.shape[0] # number of classes, number of detections

        # Create Precision-Recall curve and compute AP for each class
        px, py = np.linspace(0, 1, 1000), [] # for plotting
        ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
        for ci, c in enumerate(unique_classes):
            i = pred_cls == c
            n_l = (target_cls == c).sum()
            n_p = i.sum()
            if n_p == 0 or n_l == 0:
                continue
            else:
                fpc = (1 - tp[i]).cumsum(0)
                tpc = tp[i].cumsum(0)
                # Recall
                recall = np.divide(tpc, np.add(n_l, 1e-16)) # recall curve
                r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)
                # Precision
                precision = np.divide(tpc, np.add(tpc, fpc))    # precision curve
                p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)
                # AP from recall-precision curve
                for j in range(tp.shape[1]):
                    ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                    if plot and j == 0:
                        py.append(np.interp(px, mrec, mpre))

        # Compute F1 (harmonic mean of precision and recall)
        f1 = 2 * p * r / (p + r + 1e-16)
        print('Computed F1 (harmonic mean):', f1)
        if plot:
            plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
            plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
            plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
            plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

        i = f1.mean(0).argmax()
        return p[:, i], r[:, i], ap, f1[:, i], unique_classes.astype('int32')

    def compute_ap(self, recall, precision):
        """ Compute the average precision, given the recall and precision curves
        # Arguments
            recall: The recall curve (list)
            precision: The precision curve (list)
        # Returns
            Average precision, precision curve, recall curve"""
        # Append sentinel values to beginning and end
        mrec = np.concatenate(([0.], recall, [recall[-1] + 0.01]))
        mpre = np.concatenate(([1.], precision, [0.]))
        # Compute the precision envelope
        mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

        # Integrate area under curve
        method = 'interp'   # methods: 'continuous', 'interp'
        if method == 'interp':
            x = np.linspace(0, 1, 102)  # 101-point interp (COCO)
            ap = np.trapz(np.interp(x, mrec, mpre), x)
        else:   # Continuous
            i = np.where(np.not_equal(mrec[1:], mrec[:-1]))[0]
            ap = np.sum(np.multiply(np.subtract(mrec[i + 1], mrec[i]), mpre[i + 1])) # area under curve

        return ap, mpre, mrec


class ConfusionMatrix:
    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc    # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """Return intersection over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly"""
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = general