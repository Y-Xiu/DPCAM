"""One global confusion matrix; same class set and reduction for all metrics."""
import numpy as np
import torch


class ConfusionMatrix:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, prediction, target):
        if prediction.ndim == target.ndim+1:
            prediction = prediction.argmax(1)
        values = self.num_classes*target.reshape(-1).long() + prediction.reshape(-1).long()
        counts = torch.bincount(values, minlength=self.num_classes**2)
        self.matrix += counts.reshape(self.num_classes, self.num_classes).cpu().numpy()

    def compute(self, classes):
        cm = self.matrix.astype(np.float64)
        tp, gt, pred = np.diag(cm), cm.sum(1), cm.sum(0)
        union = gt+pred-tp
        iou = np.divide(tp, union, out=np.zeros_like(tp), where=union>0)
        dice = np.divide(2*tp, gt+pred, out=np.zeros_like(tp), where=(gt+pred)>0)
        recall = np.divide(tp, gt, out=np.zeros_like(tp), where=gt>0)
        precision = np.divide(tp, pred, out=np.zeros_like(tp), where=pred>0)
        per_class = {name: {'IoU': float(iou[i]), 'Dice': float(dice[i]),
                            'Recall': float(recall[i]), 'Precision': float(precision[i]),
                            'gt_pixels': int(gt[i])} for i, name in enumerate(classes)}
        # Fixed C, including background. Empty GT classes get recall=0 and are flagged.
        return {'mIoU': float(iou.mean()), 'mDice': float(dice.mean()),
                'mPA': float(recall.mean()), 'foreground_mIoU': float(iou[1:].mean()),
                'pixel_accuracy': float(tp.sum()/max(cm.sum(), 1)), 'per_class': per_class,
                'absent_gt_classes': [classes[i] for i in np.flatnonzero(gt == 0)],
                'metric_protocol': 'global_confusion_macro_all_classes_empty_zero_v1'}
