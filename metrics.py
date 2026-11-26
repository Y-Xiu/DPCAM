import torch
import torch.nn.functional as F

NUM_CLASSES = 5

EPS = 1e-6

def one_hot_encode(mask, num_classes=NUM_CLASSES):
    """
    mask: [B,H,W] long
    return: [B,C,H,W] float
    """
    return F.one_hot(mask, num_classes).permute(0,3,1,2).float()


def _prepare_onehot_predictions(pred, target):
    """Convert logits and labels to hard one-hot maps for metric computation."""
    num_classes = pred.shape[1]
    pred_classes = pred.argmax(dim=1)
    pred_onehot = one_hot_encode(pred_classes, num_classes=num_classes)
    target_onehot = one_hot_encode(target, num_classes=num_classes)
    return pred_onehot, target_onehot


def _class_stats(pred, target):
    """
    Compute per-class TP/FP/FN over the whole batch.
    pred/target: [B, C, H, W] one-hot
    """
    tp = (pred * target).sum(dim=(0, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(0, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(0, 2, 3))
    return tp, fp, fn


def dice_coef(pred, target, eps=EPS):
    """
    pred: [B,C,H,W] logits
    target: [B,H,W] long
    """
    pred_onehot, target_onehot = _prepare_onehot_predictions(pred, target)
    tp = (pred_onehot * target_onehot).sum(dim=(2, 3))
    fp = (pred_onehot * (1 - target_onehot)).sum(dim=(2, 3))
    fn = ((1 - pred_onehot) * target_onehot).sum(dim=(2, 3))
    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    return dice.mean().item()

def iou_score(pred, target, eps=EPS):
    pred_onehot, target_onehot = _prepare_onehot_predictions(pred, target)
    intersection = (pred_onehot * target_onehot).sum(dim=(2, 3))
    union = (pred_onehot + target_onehot - pred_onehot * target_onehot).sum(dim=(2, 3))
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()

def precision_score(pred, target, eps=EPS):
    pred_onehot, target_onehot = _prepare_onehot_predictions(pred, target)
    tp = (pred_onehot * target_onehot).sum(dim=(2, 3))
    fp = (pred_onehot * (1 - target_onehot)).sum(dim=(2, 3))
    precision = (tp + eps) / (tp + fp + eps)
    return precision.mean().item()

def recall_score(pred, target, eps=EPS):
    pred_onehot, target_onehot = _prepare_onehot_predictions(pred, target)
    tp = (pred_onehot * target_onehot).sum(dim=(2, 3))
    fn = ((1 - pred_onehot) * target_onehot).sum(dim=(2, 3))
    recall = (tp + eps) / (tp + fn + eps)
    return recall.mean().item()


def mpa_score(pred, target, eps=EPS):
    """
    mPA (mean Pixel Accuracy): mean of per-class pixel accuracy.
    mPA = mean_c( correctly predicted pixels of class c / total GT pixels of class c )
    """
    pred_onehot, target_onehot = _prepare_onehot_predictions(pred, target)
    correct_pixels_per_class = (pred_onehot * target_onehot).sum(dim=(0, 2, 3))
    total_gt_pixels_per_class = target_onehot.sum(dim=(0, 2, 3))
    valid = total_gt_pixels_per_class > 0
    if not valid.any():
        return 0.0
    class_pa = (correct_pixels_per_class[valid] + eps) / (total_gt_pixels_per_class[valid] + eps)
    return class_pa.mean().item()
