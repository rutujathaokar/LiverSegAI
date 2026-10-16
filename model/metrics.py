"""
metrics.py
==========
Loss functions and evaluation metrics for Medical Image Segmentation.
Meets requirements:
- Combined Loss: Binary Cross Entropy (BCE) + Dice Loss
- Metrics: Dice Score, IoU (Jaccard), Precision, Recall / Sensitivity, Specificity
"""

import torch
import torch.nn as nn
import numpy as np


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Loss = 1 - (2 * |P ∩ G| + smooth) / (|P| + |G| + smooth)
    """
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        probs_flat = probs.contiguous().view(-1)
        targets_flat = targets.contiguous().view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice = (2.0 * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        return 1.0 - dice


class CombinedLoss(nn.Module):
    """
    Combined BCE + Dice Loss as required:
    Loss = bce_weight * BCE + dice_weight * DiceLoss
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-5):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def dice_coef(preds, targets, smooth=1e-5):
    """
    Computes Dice coefficient for binary masks (torch or numpy).
    """
    if isinstance(preds, torch.Tensor):
        preds_flat = (preds > 0.5).float().contiguous().view(-1)
        targets_flat = (targets > 0.5).float().contiguous().view(-1)
        intersection = (preds_flat * targets_flat).sum()
        total = preds_flat.sum() + targets_flat.sum()
        if total == 0:
            return 1.0
        return float((2.0 * intersection + smooth) / (total + smooth))
    else:
        p = (preds > 0.5).astype(np.float32).flatten()
        t = (targets > 0.5).astype(np.float32).flatten()
        intersection = np.sum(p * t)
        total = np.sum(p) + np.sum(t)
        if total == 0:
            return 1.0
        return float((2.0 * intersection + smooth) / (total + smooth))


def iou_score(preds, targets, smooth=1e-5):
    """
    Computes Jaccard index / IoU (Intersection over Union).
    """
    if isinstance(preds, torch.Tensor):
        p = (preds > 0.5).float().contiguous().view(-1)
        t = (targets > 0.5).float().contiguous().view(-1)
        intersection = (p * t).sum()
        union = p.sum() + t.sum() - intersection
        if union == 0:
            return 1.0
        return float((intersection + smooth) / (union + smooth))
    else:
        p = (preds > 0.5).astype(np.float32).flatten()
        t = (targets > 0.5).astype(np.float32).flatten()
        intersection = np.sum(p * t)
        union = np.sum(p) + np.sum(t) - intersection
        if union == 0:
            return 1.0
        return float((intersection + smooth) / (union + smooth))


def compute_all_metrics(preds, targets, smooth=1e-5):
    """
    Computes comprehensive segmentation metrics:
    - Dice Score
    - IoU (Jaccard)
    - Precision (PPV)
    - Recall (Sensitivity)
    - Specificity (TNR)
    - Pixel Accuracy
    """
    if isinstance(preds, torch.Tensor):
        p = (preds > 0.5).cpu().numpy().astype(bool).flatten()
    else:
        p = (preds > 0.5).astype(bool).flatten()

    if isinstance(targets, torch.Tensor):
        t = (targets > 0.5).cpu().numpy().astype(bool).flatten()
    else:
        t = (targets > 0.5).astype(bool).flatten()

    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, ~t).sum()
    fn = np.logical_and(~p, t).sum()
    tn = np.logical_and(~p, ~t).sum()

    # Dice
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    if tp + fp + fn == 0:
        dice = 1.0

    # IoU
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    if tp + fp + fn == 0:
        iou = 1.0

    # Precision
    precision = (tp + smooth) / (tp + fp + smooth)
    if tp + fp == 0:
        precision = 1.0 if fn == 0 else 0.0

    # Recall / Sensitivity
    recall = (tp + smooth) / (tp + fn + smooth)
    if tp + fn == 0:
        recall = 1.0

    # Specificity
    specificity = (tn + smooth) / (tn + fp + smooth)

    # Overall pixel accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    return {
        "dice": float(round(dice, 4)),
        "iou": float(round(iou, 4)),
        "precision": float(round(precision, 4)),
        "recall": float(round(recall, 4)),
        "specificity": float(round(specificity, 4)),
        "accuracy": float(round(accuracy, 4)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn)
    }
