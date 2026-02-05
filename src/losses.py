"""
Loss functions for earthquake precursor detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    Reference: Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
    
    Args:
        alpha: Weighting factor for each class
        gamma: Focusing parameter (higher = more focus on hard examples)
        reduction: Reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(
        self, 
        alpha: Optional[torch.Tensor] = None, 
        gamma: float = 2.0, 
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)
            alpha_t = alpha[targets]
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross entropy with label smoothing.
    
    Args:
        smoothing: Label smoothing factor (0.0 = no smoothing)
    """
    
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        n_classes = inputs.size(-1)
        log_probs = F.log_softmax(inputs, dim=-1)
        
        # Create smoothed targets
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (n_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        loss = (-smooth_targets * log_probs).sum(dim=-1)
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Combined loss for multi-task learning.
    
    Args:
        mag_weight: Weight for magnitude loss
        azi_weight: Weight for azimuth loss
        mag_class_weights: Class weights for magnitude
        azi_class_weights: Class weights for azimuth
        use_focal: Whether to use focal loss
        focal_gamma: Gamma for focal loss
        label_smoothing: Label smoothing factor
    """
    
    def __init__(
        self,
        mag_weight: float = 1.0,
        azi_weight: float = 0.5,
        mag_class_weights: Optional[torch.Tensor] = None,
        azi_class_weights: Optional[torch.Tensor] = None,
        use_focal: bool = False,
        focal_gamma: float = 2.0,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        
        self.mag_weight = mag_weight
        self.azi_weight = azi_weight
        
        if use_focal:
            self.mag_criterion = FocalLoss(alpha=mag_class_weights, gamma=focal_gamma)
            self.azi_criterion = FocalLoss(alpha=azi_class_weights, gamma=focal_gamma)
        elif label_smoothing > 0:
            self.mag_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
            self.azi_criterion = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        else:
            self.mag_criterion = nn.CrossEntropyLoss(weight=mag_class_weights)
            self.azi_criterion = nn.CrossEntropyLoss(weight=azi_class_weights)
    
    def forward(
        self, 
        mag_logits: torch.Tensor, 
        azi_logits: torch.Tensor,
        mag_targets: torch.Tensor, 
        azi_targets: torch.Tensor
    ) -> tuple:
        """
        Compute combined loss.
        
        Returns:
            Tuple of (total_loss, mag_loss, azi_loss)
        """
        mag_loss = self.mag_criterion(mag_logits, mag_targets)
        azi_loss = self.azi_criterion(azi_logits, azi_targets)
        
        total_loss = self.mag_weight * mag_loss + self.azi_weight * azi_loss
        
        return total_loss, mag_loss, azi_loss
