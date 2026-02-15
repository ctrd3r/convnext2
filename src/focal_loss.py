#!/usr/bin/env python3
"""
Earthquake CNN V3.0 - State-of-the-Art Implementation
Multi-task CNN for earthquake prediction from geomagnetic spectrograms

Features:
- ConvNeXt-Tiny backbone (modern CNN architecture)
- Multi-task learning (Magnitude + Azimuth)
- Focal Loss for extreme class imbalance
- Progressive resizing training
- Advanced augmentation (RandAugment + MixUp/CutMix)
- Automatic Mixed Precision (AMP)
- EMA model for better performance

Author: Earthquake Prediction Research Team
Date: 30 January 2026
Version: 3.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.models import convnext_tiny
import numpy as np
import math
from typing import Tuple, Dict, Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing extreme class imbalance
    Paper: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, 
                 weight: Optional[torch.Tensor] = None):
        """
        Initialize Focal Loss
        
        Args:
            alpha: Weighting factor for rare class (default: 0.25)
            gamma: Focusing parameter (default: 2.0)
            weight: Manual class weights
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss
        
        Args:
            inputs: Predictions (batch_size, num_classes)
            targets: Ground truth labels (batch_size,)
            
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Label Smoothing Cross Entropy Loss
    Prevents overconfident predictions
    """
    
    def __init__(self, smoothing: float = 0.1):
        """
        Initialize label smoothing loss
        
        Args:
            smoothing: Smoothing factor (default: 0.1)
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing cross entropy
        
        Args:
            pred: Predictions (batch_size, num_classes)
            target: Ground truth labels (batch_size,)
            
        Returns:
            Label smoothing loss
        """
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=1)
        nll_loss = F.nll_loss(log_probs, target, reduction='none')
        smooth_loss = -log_probs.mean(dim=1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class EarthquakeCNNV3(nn.Module):
    """
    State-of-the-Art Multi-Task CNN for Earthquake Prediction
    
    Architecture:
    - ConvNeXt-Tiny backbone (modern CNN)
    - Multi-task heads (Magnitude + Azimuth)
    - Advanced regularization
    """
    
    def __init__(self, 
                 num_magnitude_classes: int = 4,
                 num_azimuth_classes: int = 9,
                 dropout_rate: float = 0.3,
                 drop_path_rate: float = 0.1):
        """
        Initialize EarthquakeCNN V3.0
        
        Args:
            num_magnitude_classes: Number of magnitude classes (default: 4)
            num_azimuth_classes: Number of azimuth classes (default: 9)
            dropout_rate: Dropout rate (default: 0.3)
            drop_path_rate: DropPath rate for ConvNeXt (default: 0.1)
        """
        super(EarthquakeCNNV3, self).__init__()
        
        self.num_magnitude_classes = num_magnitude_classes
        self.num_azimuth_classes = num_azimuth_classes
        
        # ConvNeXt-Tiny backbone (modern CNN architecture)
        self.backbone = convnext_tiny(pretrained=True)
        
        # Get feature dimension from ConvNeXt
        feature_dim = self.backbone.classifier[2].in_features  # 768 for ConvNeXt-Tiny
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Shared feature processing with modern components
        self.shared_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 512),
            nn.GELU(),  # Modern activation function
            nn.Dropout(dropout_rate),
            nn.LayerNorm(512)
        )
        
        # Task-specific attention mechanisms
        self.magnitude_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        self.azimuth_attention = nn.MultiheadAttention(512, num_heads=8, batch_first=True)
        
        # Magnitude classification head
        self.magnitude_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, num_magnitude_classes)
        )
        
        # Azimuth classification head  
        self.azimuth_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.LayerNorm(256),
            nn.Linear(256, num_azimuth_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor (batch_size, 3, H, W)
            
        Returns:
            magnitude_logits: Magnitude predictions (batch_size, num_magnitude_classes)
            azimuth_logits: Azimuth predictions (batch_size, num_azimuth_classes)
        """
        # Extract features with ConvNeXt backbone
        features = self.backbone(x)
        
        # Shared feature processing
        shared_features = self.shared_head(features)  # (batch_size, 512)
        
        # Add sequence dimension for attention
        shared_features_seq = shared_features.unsqueeze(1)  # (batch_size, 1, 512)
        
        # Task-specific attention
        magnitude_features, _ = self.magnitude_attention(
            shared_features_seq, shared_features_seq, shared_features_seq
        )
        azimuth_features, _ = self.azimuth_attention(
            shared_features_seq, shared_features_seq, shared_features_seq
        )
        
        # Remove sequence dimension
        magnitude_features = magnitude_features.squeeze(1)  # (batch_size, 512)
        azimuth_features = azimuth_features.squeeze(1)  # (batch_size, 512)
        
        # Task-specific predictions
        magnitude_logits = self.magnitude_head(magnitude_features)
        azimuth_logits = self.azimuth_head(azimuth_features)
        
        return magnitude_logits, azimuth_logits
    
    def freeze_backbone(self):
        """Freeze backbone for transfer learning"""
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        """Unfreeze backbone for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True


class MultiTaskLossV3(nn.Module):
    """
    Advanced Multi-Task Loss with multiple loss functions
    
    Features:
    - Focal Loss for extreme imbalance
    - Label Smoothing for overconfidence
    - Uncertainty weighting for task balancing
    """
    
    def __init__(self, 
                 magnitude_weights: Optional[torch.Tensor] = None,
                 azimuth_weights: Optional[torch.Tensor] = None,
                 focal_alpha: float = 0.25,
                 focal_gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 learn_task_weights: bool = True):
        """
        Initialize multi-task loss
        
        Args:
            magnitude_weights: Class weights for magnitude
            azimuth_weights: Class weights for azimuth
            focal_alpha: Focal loss alpha parameter
            focal_gamma: Focal loss gamma parameter
            label_smoothing: Label smoothing factor
            learn_task_weights: Learn task weights automatically
        """
        super(MultiTaskLossV3, self).__init__()
        
        # Loss functions
        self.magnitude_focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, 
                                       weight=magnitude_weights)
        self.azimuth_focal = FocalLoss(alpha=focal_alpha, gamma=focal_gamma,
                                     weight=azimuth_weights)
        
        self.magnitude_smooth = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        self.azimuth_smooth = LabelSmoothingCrossEntropy(smoothing=label_smoothing)
        
        # Task weight learning
        self.learn_task_weights = learn_task_weights
        if learn_task_weights:
            self.log_var_magnitude = nn.Parameter(torch.zeros(1))
            self.log_var_azimuth = nn.Parameter(torch.zeros(1))
        else:
            self.weight_magnitude = 1.0
            self.weight_azimuth = 1.0
            
    def forward(self, 
                magnitude_logits: torch.Tensor, 
                azimuth_logits: torch.Tensor,
                magnitude_targets: torch.Tensor, 
                azimuth_targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            magnitude_logits: Magnitude predictions
            azimuth_logits: Azimuth predictions  
            magnitude_targets: Magnitude ground truth
            azimuth_targets: Azimuth ground truth
            
        Returns:
            Dictionary with loss components
        """
        # Individual losses (combine focal + label smoothing)
        magnitude_focal_loss = self.magnitude_focal(magnitude_logits, magnitude_targets)
        magnitude_smooth_loss = self.magnitude_smooth(magnitude_logits, magnitude_targets)
        magnitude_loss = 0.7 * magnitude_focal_loss + 0.3 * magnitude_smooth_loss
        
        azimuth_focal_loss = self.azimuth_focal(azimuth_logits, azimuth_targets)
        azimuth_smooth_loss = self.azimuth_smooth(azimuth_logits, azimuth_targets)
        azimuth_loss = 0.7 * azimuth_focal_loss + 0.3 * azimuth_smooth_loss
        
        # Task weighting
        if self.learn_task_weights:
            # Uncertainty weighting (Kendall et al., 2018)
            precision_magnitude = torch.exp(-self.log_var_magnitude)
            precision_azimuth = torch.exp(-self.log_var_azimuth)
            
            total_loss = (
                precision_magnitude * magnitude_loss + self.log_var_magnitude +
                precision_azimuth * azimuth_loss + self.log_var_azimuth
            )
        else:
            total_loss = self.weight_magnitude * magnitude_loss + self.weight_azimuth * azimuth_loss
        
        return {
            'total_loss': total_loss,
            'magnitude_loss': magnitude_loss,
            'azimuth_loss': azimuth_loss,
            'magnitude_focal': magnitude_focal_loss,
            'magnitude_smooth': magnitude_smooth_loss,
            'azimuth_focal': azimuth_focal_loss,
            'azimuth_smooth': azimuth_smooth_loss
        }


class EMAModel:
    """
    Exponential Moving Average Model
    Maintains shadow copy of model weights for better performance
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        """
        Initialize EMA model
        
        Args:
            model: Model to track
            decay: EMA decay rate (default: 0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
                
    def update(self):
        """Update EMA weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
                
    def apply_shadow(self):
        """Apply shadow weights to model"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]
                
    def restore(self):
        """Restore original weights"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def create_model_v3(config: Dict) -> Tuple[EarthquakeCNNV3, MultiTaskLossV3]:
    """
    Create EarthquakeCNN V3.0 model and loss function
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        model: EarthquakeCNNV3 instance
        criterion: MultiTaskLossV3 instance
    """
    # Create model
    model = EarthquakeCNNV3(
        num_magnitude_classes=config.get('num_magnitude_classes', 3),
        num_azimuth_classes=config.get('num_azimuth_classes', 8),
        dropout_rate=config.get('dropout_rate', 0.3),
        drop_path_rate=config.get('drop_path_rate', 0.1)
    )
    
    # Create loss function
    criterion = MultiTaskLossV3(
        magnitude_weights=config.get('magnitude_weights'),
        azimuth_weights=config.get('azimuth_weights'),
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
        label_smoothing=config.get('label_smoothing', 0.1),
        learn_task_weights=config.get('learn_task_weights', True)
    )
    
    return model, criterion


def get_model_config() -> Dict:
    """
    Get default model configuration following golden standards
    
    Returns:
        Configuration dictionary
    """
    return {
        # Architecture
        'num_magnitude_classes': 4,  # Medium, Large, Moderate, Normal
        'num_azimuth_classes': 9,    # N, NE, E, SE, S, SW, W, NW, Normal
        'dropout_rate': 0.3,
        'drop_path_rate': 0.1,
        
        # Loss function
        'focal_alpha': 0.1,   # More aggressive for extreme imbalance
        'focal_gamma': 3.0,   # Higher gamma for better focus on hard examples
        'label_smoothing': 0.1,
        'learn_task_weights': True,
        
        # Training
        'optimizer': 'adamw',
        'base_lr': 1e-3,
        'weight_decay': 0.05,
        'batch_size': 32,
        'epochs': 60,
        
        # Augmentation
        'randaugment_magnitude': 9,
        'randaugment_num_ops': 2,
        'mixup_alpha': 0.2,
        'cutmix_alpha': 1.0,
        
        # Progressive resizing
        'progressive_resizing': True,
        'resize_schedule': {
            0: 112,   # epochs 0-7
            7: 168,   # epochs 7-14  
            14: 224   # epochs 14-20
        },
        
        # EMA
        'ema_decay': 0.9999,
        
        # Reproducibility
        'seed': 42
    }


if __name__ == '__main__':
    # Test model
    print("🚀 Testing EarthquakeCNN V3.0...")
    
    # Get configuration
    config = get_model_config()
    
    # Create model
    model, criterion = create_model_v3(config)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 3, 224, 224)
    
    print(f"\n📊 Model Architecture:")
    print(f"Input shape: {x.shape}")
    
    # Forward pass
    magnitude_logits, azimuth_logits = model(x)
    
    print(f"Magnitude output: {magnitude_logits.shape}")
    print(f"Azimuth output: {azimuth_logits.shape}")
    
    # Test loss
    magnitude_targets = torch.randint(0, 4, (batch_size,))
    azimuth_targets = torch.randint(0, 9, (batch_size,))
    
    loss_dict = criterion(magnitude_logits, azimuth_logits, 
                         magnitude_targets, azimuth_targets)
    
    print(f"\n💰 Loss Components:")
    for key, value in loss_dict.items():
        print(f"{key}: {value.item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n🔢 Model Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.1f} MB")
    
    # Test EMA
    ema_model = EMAModel(model, decay=0.9999)
    print(f"\n✅ EMA model initialized")
    
    print(f"\n🎯 EarthquakeCNN V3.0 - Ready for training!")
    print(f"Architecture: ConvNeXt-Tiny backbone")
    print(f"Features: Focal Loss + Label Smoothing + EMA + Attention")
    print(f"Target: Multi-task earthquake prediction")