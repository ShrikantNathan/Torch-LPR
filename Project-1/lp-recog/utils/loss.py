import torch
import torch.nn as nn
from general import bbox_iou
from torch_utils import is_parallel
import numpy as np


def smooth_BCE(eps=0.1):
    # return positive, negative label smoothing BCE targets
    return np.subtract(float(1), np.multiply(0.5, eps)), np.multiply(0.5, eps)


class BCEBlurWithLogitsLoss(nn.Module):
    # BCEwithLogitLoss() with reduced missing label effects.
    def __init__(self, alpha=0.05):
        super(BCEBlurWithLogitsLoss, self).__init__()
        self.loss_fcn = nn.BCEWithLogitsLoss(reduction='none')  # must be nn.BCEWithLogitsLoss()
        self.alpha = alpha

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred = torch.sigmoid(pred)      # prob from logits
        dx = pred - true    # reduce only missing label effects
        alpha_factor = 1 - torch.divide(torch.exp(torch.subtract(dx, 1)), torch.add(self.alpha, 1e-4))
        # alpha_factor_2 = 1 - torch.exp((dx - 1) / (self.alpha + 1e-4))
        loss *= alpha_factor
        return loss.mean()


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'    # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred)
        print('loss:', loss, 'pred_prob:', pred_prob)
        prob_true = torch.add(torch.multiply(true, pred_prob), torch.multiply(np.subtract(1, true), np.subtract(1, self.alpha)))
        alpha_factor = torch.add(torch.multiply(true, self.alpha), torch.multiply(np.subtract(1, true), np.subtract(1, self.alpha)))

        modulating_factor = torch.pow(np.subtract(float(1), prob_true), self.gamma)
        loss *= torch.multiply(alpha_factor, modulating_factor)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class QFocalLoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super(QFocalLoss, self).__init__()
        self.loss_fcn = loss_fcn
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        pred_prob = torch.sigmoid(pred) # prob from logits
        alpha_factor = torch.add(torch.multiply(true, self.alpha), torch.multiply(np.subtract(1, true), np.subtract(1, self.alpha)))
        modulating_factor = torch.pow(torch.abs(torch.subtract(true, pred_prob)), self.gamma)
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class ComputeLoss:
    def __init__(self, model, autobalance=False):
        super(ComputeLoss, self).__init__()
        device = next(model.parameters()).device    # get model device
        h = model.hyp   # hyperparameters

        # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['cls_pw']], device=device))
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h['obj_pw']], device=device))

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        self.cp, self.cn = smooth_BCE(eps=h.get('label_smoothing', 0.0))    # positive, negative BCE targets

        # Focal loss
        g = h['f1_gamma']   # focal loss gamma
        if g > 0:
            BCEcls, BCEobj = FocalLoss(loss_fcn=BCEcls, gamma=g), FocalLoss(loss_fcn=BCEobj, gamma=g)

        det = model.module.model[-1] if is_parallel(model) else model.model[-1]     # Detect() module
        self.balance = {3: [float(4), float(1), 0.4]}.get(det.nl, [float(4), float(1), 0.25, 0.06, .02])
        self.ssi = list(det.stride).index(16) if autobalance else 0     # stride 16 index
        self.BCEcls, self.BCEobj, self.gr, self.hyp, self.autobalance = BCEcls, BCEobj, model.gr, h, autobalance
        for k in 'na', 'nc', 'nl', 'anchors':
            setattr(self, k, getattr(det, k))

    def __call__(self, p, targets):     # predictions, targets, model
        device = targets.device
        lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)
        tcls, tbox, indices, anchors = self.build_targets(p, targets)   # targets

        # Losses
        for layer_index, predicted_layer in enumerate(p):   # layer index, layer predictions
            image, anchor, grid_y, grid_x = indices[layer_index]
            target_obj = torch.zeros_like(predicted_layer[..., 0], device=device)   # target obj
            target_nos = image.shape[0]     # number of targets
            if target_nos:   # prediction subset corresponding to targets (shown below)
                pred_subsets = predicted_layer[image, anchor, grid_y, grid_x]
                print('predicted_subsets:', pred_subsets)
                # Regression
                predicted_xy = torch.subtract(torch.multiply(pred_subsets[:, :2].sigmoid(), 2.), 0.5)
                predicted_wh = torch.multiply(torch.pow(torch.multiply(pred_subsets[:, 2:4].sigmoid(), 2), 2), anchors[layer_index])
                predicted_box = torch.cat(tensors=(predicted_xy, predicted_wh), dim=1)  # predicted box
                iou = bbox_iou(box1=predicted_box.T, box2=tbox[layer_index], x1y1x2y2=False, CIoU=True) # iou(prediction, target)
                lbox += (float(1) - iou).mean()
                # Objectness
                target_obj[predicted_box, anchor, grid_y, grid_x] = torch.add(np.subtract(
                    float(1), self.gr), torch.multiply(self.gr, iou.detach().clamp(0).type(target_obj.dtype)))
                # Classification
                if np.greater(self.nc, 1):
                    t = torch.full_like(pred_subsets[:, 5:], self.cn, device=device)    # targets
                    t[range(target_nos), tcls[layer_index]] = self.cp
                    lcls += self.BCEcls(pred_subsets[:, 5:], t) # BCE

                obji = self.BCEobj(predicted_layer[..., 4], target_obj)
                lobj += obji * self.balance[layer_index]  # obj loss
                if self.autobalance:
                    self.balance[layer_index] = torch.add(np.multiply(
                        self.balance[layer_index], 0.9999), np.divide(0.0001, obji.detach().item()))

            if self.autobalance:
                self.balance = [x / self.balance[self.ssi] for x in self.balance]
            lbox *= self.hyp['box']
            lobj *= self.hyp['obj']
            lcls *= self.hyp['cls']
            batch_size = target_obj.shape[0]    # batch size
            loss = lbox + lobj + lcls
            return loss * batch_size, torch.cat((lbox, lobj, lcls, loss)).detach()

    def build_targets(self, p, targets):
        num_anchors, num_targets = self.na, targets.shape[0]    # number of anchors, targets
        tcls, tbox, indices, anch = list(), list(), list(), list()
        gain = torch.ones(7, device=targets.device)     # normalized to gridspace gain
        ai = torch.arange(num_anchors, device=targets.device).float().view(num_anchors, 1).repeat(1, num_targets)
        targets = torch.cat((targets.repeat(num_anchors, 1, 1), ai[:, :, None]), 2)     # append anchor indices
        g = 0.5 # bias
        off = torch.tensor([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], ], device=targets.device).float() * g

        for i in range(self.nl):
            anchors = self.anchors[i]
            gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # xyxy gain

            # Match targets to anchors
            t = targets * gain
            if num_targets:
                # Matches
                r = t[:, :, 4:6] / anchors[:, None] # wh ratio
                j = torch.less(torch.max(r, 1. / r).max(2)[0], self.hyp['anchor_t'])    # compare
                t = t[j]    # filter

                # Offsets
                gxy = t[:, 2:4]     # grid xy
                gxi = gain[[2, 3]] - gxy    # inverse
                j, k = torch.bitwise_and(torch.less(torch.remainder(gxy, 1), g), torch.greater(gxy, 1.)).T
                l, m = torch.bitwise_and(torch.less(torch.remainder(gxi, 1), g), torch.greater(gxi, 1.)).T
                j = torch.stack((torch.ones_like(j), j, k, l, m))
                t = t.repeat((5, 1, 1))[j]
                offsets = (torch.zeros_like(gxy)[None] + off[:, None])[j]
            else:
                t = targets[0]
                offsets = 0

            # Define
            b, c = t[:, :2].long().T    # image, class
            gxy = t[:, 2:4]     # grid xy
            gwh = t[:, 4:6]     # grid wh
            gij = (gxy - offsets).long()
            gi, gj = gij.T  # grid xy indices

            # Append
            a = t[:, 6].long()  # anchor indices
            indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1)))   # image, anchor, grid indices
            tbox.append(torch.cat((gxy - gij, gwh), 1))     # box
            anch.append(anchors[a])     # anchors
            tcls.append(c)  # class

        return tcls, tbox, indices, anch



