"""Multiclass extension of the manuscript's binary Dice/Focal equations.

Softmax + one-hot labels; binary focal averaged over all class/pixel pairs.
This retains both alpha*y and (1-alpha)*(1-y), rather than silently treating
alpha as an unused or uniform scalar multiplier on categorical focal loss.
"""
import torch
from torch import nn
from torch.nn import functional as F


class PaperLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2., dice_weight=.5, focal_weight=.5, smooth=1e-6):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
        self.dice_weight, self.focal_weight, self.smooth = dice_weight, focal_weight, smooth

    def forward(self, logits, target):
        probs = logits.float().softmax(1)
        labels = F.one_hot(target, logits.shape[1]).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        dice = 1 - ((2*(probs*labels).sum(dims)+self.smooth) /
                    (probs.sum(dims)+labels.sum(dims)+self.smooth)).mean()
        p = probs.clamp(1e-7, 1-1e-7)
        focal = -(self.alpha*(1-p).pow(self.gamma)*labels*p.log() +
                  (1-self.alpha)*p.pow(self.gamma)*(1-labels)*torch.log1p(-p)).mean()
        return self.dice_weight*dice + self.focal_weight*focal
