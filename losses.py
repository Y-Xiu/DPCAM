
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth=1.0, ignore_index=-100, include_background=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.include_background = include_background
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        """
        # Convert logits to probabilities
        predictions = F.softmax(predictions, dim=1)
        
        # Convert targets to one-hot encoding
        num_classes = predictions.shape[1]
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index)
            targets_for_one_hot = targets.masked_fill(~valid_mask, 0)
        else:
            valid_mask = None
            targets_for_one_hot = targets

        targets_one_hot = F.one_hot(targets_for_one_hot, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = valid_mask.float().unsqueeze(1)
            predictions = predictions * mask
            targets_one_hot = targets_one_hot * mask
        
        # Calculate Dice coefficient
        intersection = (predictions * targets_one_hot).sum(dim=(2, 3))
        union = predictions.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        if not self.include_background and dice.shape[1] > 1:
            dice = dice[:, 1:]
        
        # Average over batch and classes
        dice_loss = 1 - dice.mean()
        
        return dice_loss


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance"""
    
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        """
        Args:
            alpha: (C,) tensor of class weights, or None for no weighting
            gamma: focusing parameter (default: 2.0)
            ignore_index: class index to ignore in loss calculation
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        """
        # Get log probabilities
        log_probs = F.log_softmax(predictions, dim=1)
        
        # Gather the log probabilities for the target classes
        if self.ignore_index >= 0:
            valid_mask = (targets != self.ignore_index)
            targets_for_gather = targets.masked_fill(~valid_mask, 0)
        else:
            valid_mask = None
            targets_for_gather = targets

        targets_expanded = targets_for_gather.unsqueeze(1)  # (B, 1, H, W)
        log_probs_target = log_probs.gather(1, targets_expanded).squeeze(1)  # (B, H, W)
        
        # Calculate probabilities
        probs_target = log_probs_target.exp()
        
        # Calculate focal weight
        focal_weight = (1 - probs_target) ** self.gamma
        
        # Calculate loss
        loss = -focal_weight * log_probs_target
        
        # Apply class weights if provided
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha = torch.tensor(self.alpha, device=predictions.device, dtype=predictions.dtype)
            else:
                alpha = self.alpha.to(device=predictions.device, dtype=predictions.dtype)
            alpha_t = alpha.gather(0, targets_for_gather.view(-1)).view_as(targets_for_gather)
            loss = alpha_t * loss
        
        # Handle ignore_index
        if self.ignore_index >= 0:
            mask = valid_mask.float()
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1.0)
        
        return loss.mean()


class CombinedLoss(nn.Module):
    """Combination of multiple loss functions"""
    
    def __init__(self, losses, weights=None):
        """
        Args:
            losses: dict of {'name': loss_fn}
            weights: dict of {'name': weight}, or None for equal weights
        """
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleDict(losses)
        
        if weights is None:
            # Equal weights
            self.weights = {name: 1.0 / len(losses) for name in losses.keys()}
        else:
            self.weights = weights
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: (B, C, H, W) - logits
            targets: (B, H, W) - class indices
        """
        total_loss = 0
        loss_dict = {}
        
        for name, loss_fn in self.losses.items():
            loss_value = loss_fn(predictions, targets)
            weight = self.weights.get(name, 1.0)
            total_loss += weight * loss_value
            loss_dict[name] = loss_value.item()
        
        return total_loss, loss_dict


def get_loss_function(loss_config):
    """
    根据配置创建损失函数
    
    Args:
        loss_config: 损失函数配置字典
        
    Returns:
        loss function
    """
    loss_type = loss_config.get('type', 'cross_entropy')
    
    if loss_type == 'cross_entropy':
        weight = loss_config.get('weight', None)
        if weight is not None:
            weight = torch.tensor(weight)
        return nn.CrossEntropyLoss(
            weight=weight,
            ignore_index=loss_config.get('ignore_index', -100)
        )
    
    elif loss_type == 'dice':
        return DiceLoss(
            smooth=loss_config.get('smooth', 1.0),
            ignore_index=loss_config.get('ignore_index', -100),
            include_background=loss_config.get('include_background', True)
        )
    
    elif loss_type == 'focal':
        alpha = loss_config.get('alpha', None)
        if alpha is not None:
            alpha = torch.tensor(alpha)
        return FocalLoss(
            alpha=alpha,
            gamma=loss_config.get('gamma', 2.0),
            ignore_index=loss_config.get('ignore_index', -100)
        )
    
    elif loss_type == 'combined':
        # 组合损失函数
        losses_config = loss_config.get('losses', {})
        weights = loss_config.get('weights', None)
        
        losses = {}
        for name, cfg in losses_config.items():
            losses[name] = get_loss_function(cfg)
        
        return CombinedLoss(losses, weights)
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage configurations:
"""
# 1. Cross Entropy Loss
LOSS_CONFIG = {
    'type': 'cross_entropy',
    'weight': None,  # or [0.1, 1.0, 1.5, 2.0, 2.5] for class weights
    'ignore_index': -100,
}

# 2. Dice Loss
LOSS_CONFIG = {
    'type': 'dice',
    'smooth': 1.0,
    'ignore_index': -100,
}

# 3. Focal Loss
LOSS_CONFIG = {
    'type': 'focal',
    'alpha': None,  # or [0.25, 0.75, 1.0, 1.0, 1.5] for class weights
    'gamma': 2.0,
    'ignore_index': -100,
}

# 4. Combined Loss (Dice + Focal)
LOSS_CONFIG = {
    'type': 'combined',
    'losses': {
        'dice': {
            'type': 'dice',
            'smooth': 1.0,
            'ignore_index': -100,
        },
        'focal': {
            'type': 'focal',
            'alpha': None,
            'gamma': 2.0,
            'ignore_index': -100,
        }
    },
    'weights': {
        'dice': 0.5,
        'focal': 0.5,
    }
}
"""
