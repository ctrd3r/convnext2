"""
ConvNeXt Model for Earthquake Precursor Detection

ConvNeXt is a pure convolutional model that incorporates design choices from
Vision Transformers (ViT) while maintaining the efficiency of CNNs.

Reference: Liu et al., "A ConvNet for the 2020s" (CVPR 2022)
"""

import torch
import torch.nn as nn
from torchvision.models import (
    convnext_tiny, convnext_small, convnext_base,
    ConvNeXt_Tiny_Weights, ConvNeXt_Small_Weights
)
from typing import Dict, Optional, Tuple
import json
from pathlib import Path


class ConvNeXtMultiTask(nn.Module):
    """
    ConvNeXt model with multi-task heads for magnitude and azimuth classification.
    
    Args:
        variant: Model variant ("tiny", "small", "base")
        pretrained: Whether to use ImageNet pretrained weights
        num_mag_classes: Number of magnitude classes (default: 4)
        num_azi_classes: Number of azimuth classes (default: 9)
        dropout: Dropout rate for classification heads
    """
    
    def __init__(
        self, 
        variant: str = "tiny", 
        pretrained: bool = True, 
        num_mag_classes: int = 4, 
        num_azi_classes: int = 9, 
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.variant = variant
        self.num_mag_classes = num_mag_classes
        self.num_azi_classes = num_azi_classes
        
        # Load pretrained ConvNeXt backbone
        if variant == "tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
            num_features = 768
        elif variant == "small":
            weights = ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = convnext_small(weights=weights)
            num_features = 768
        elif variant == "base":
            self.backbone = convnext_base(weights=None)
            num_features = 1024
        else:
            raise ValueError(f"Unknown variant: {variant}. Choose from: tiny, small, base")
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Magnitude classification head
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_mag_classes)
        )
        
        # Azimuth classification head
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_azi_classes)
        )
        
        self.num_features = num_features
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 3, H, W)
            
        Returns:
            Tuple of (magnitude_logits, azimuth_logits)
        """
        # Extract features
        features = self.backbone(x)
        
        # Flatten if needed
        if features.dim() == 4:
            features = features.flatten(1)
        
        # Classification heads
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        
        return mag_out, azi_out
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract feature embeddings (for visualization)."""
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.flatten(1)
        return features
    
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Get predictions with probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary with predictions and probabilities
        """
        self.eval()
        with torch.no_grad():
            mag_logits, azi_logits = self.forward(x)
            
            mag_probs = torch.softmax(mag_logits, dim=1)
            azi_probs = torch.softmax(azi_logits, dim=1)
            
            mag_pred = torch.argmax(mag_probs, dim=1)
            azi_pred = torch.argmax(azi_probs, dim=1)
            
            mag_conf = mag_probs.max(dim=1).values
            azi_conf = azi_probs.max(dim=1).values
            
        return {
            'magnitude_pred': mag_pred,
            'magnitude_prob': mag_probs,
            'magnitude_conf': mag_conf,
            'azimuth_pred': azi_pred,
            'azimuth_prob': azi_probs,
            'azimuth_conf': azi_conf,
        }


class ConvNeXtPrecursorModel:
    """
    High-level wrapper for ConvNeXt earthquake precursor model.
    
    Provides easy-to-use interface for loading, inference, and evaluation.
    """
    
    # Default class mappings
    DEFAULT_MAG_CLASSES = ['Large', 'Medium', 'Moderate', 'Normal']
    DEFAULT_AZI_CLASSES = ['E', 'N', 'NE', 'NW', 'Normal', 'S', 'SE', 'SW', 'W']
    
    def __init__(
        self,
        model: ConvNeXtMultiTask,
        device: torch.device,
        mag_classes: list = None,
        azi_classes: list = None
    ):
        self.model = model
        self.device = device
        self.mag_classes = mag_classes or self.DEFAULT_MAG_CLASSES
        self.azi_classes = azi_classes or self.DEFAULT_AZI_CLASSES
        
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def load_pretrained(
        cls, 
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> 'ConvNeXtPrecursorModel':
        """
        Load pretrained model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint (.pth file)
            device: Device to load model on (default: auto-detect)
            
        Returns:
            ConvNeXtPrecursorModel instance
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Get config from checkpoint
        config = checkpoint.get('config', {})
        variant = config.get('model_variant', 'tiny')
        num_mag_classes = config.get('num_mag_classes', 4)
        num_azi_classes = config.get('num_azi_classes', 9)
        
        # Create model
        model = ConvNeXtMultiTask(
            variant=variant,
            pretrained=False,
            num_mag_classes=num_mag_classes,
            num_azi_classes=num_azi_classes
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load class mappings if available
        mag_classes = None
        azi_classes = None
        
        mappings_path = Path(checkpoint_path).parent / 'class_mappings.json'
        if mappings_path.exists():
            with open(mappings_path) as f:
                mappings = json.load(f)
                mag_classes = [mappings['magnitude'][str(i)] for i in range(num_mag_classes)]
                azi_classes = [mappings['azimuth'][str(i)] for i in range(num_azi_classes)]
        
        return cls(model, device, mag_classes, azi_classes)
    
    def predict(self, image: torch.Tensor) -> Dict:
        """
        Predict magnitude and azimuth for input image.
        
        Args:
            image: Preprocessed image tensor (B, 3, 224, 224)
            
        Returns:
            Dictionary with predictions
        """
        image = image.to(self.device)
        
        results = self.model.predict(image)
        
        # Convert to class names
        batch_size = image.size(0)
        predictions = []
        
        for i in range(batch_size):
            mag_idx = results['magnitude_pred'][i].item()
            azi_idx = results['azimuth_pred'][i].item()
            
            predictions.append({
                'magnitude_class': self.mag_classes[mag_idx],
                'magnitude_prob': results['magnitude_conf'][i].item(),
                'magnitude_probs': {
                    cls: results['magnitude_prob'][i, j].item()
                    for j, cls in enumerate(self.mag_classes)
                },
                'azimuth_class': self.azi_classes[azi_idx],
                'azimuth_prob': results['azimuth_conf'][i].item(),
                'azimuth_probs': {
                    cls: results['azimuth_prob'][i, j].item()
                    for j, cls in enumerate(self.azi_classes)
                },
            })
        
        return predictions[0] if batch_size == 1 else predictions
    
    def save(self, path: str, config: dict = None):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'config': config or {},
            'mag_classes': self.mag_classes,
            'azi_classes': self.azi_classes,
        }
        torch.save(checkpoint, path)
        
        # Save class mappings
        mappings = {
            'magnitude': {str(i): c for i, c in enumerate(self.mag_classes)},
            'azimuth': {str(i): c for i, c in enumerate(self.azi_classes)}
        }
        mappings_path = Path(path).parent / 'class_mappings.json'
        with open(mappings_path, 'w') as f:
            json.dump(mappings, f, indent=2)


def create_model(
    variant: str = "tiny",
    pretrained: bool = True,
    num_mag_classes: int = 4,
    num_azi_classes: int = 9,
    dropout: float = 0.5
) -> ConvNeXtMultiTask:
    """
    Factory function to create ConvNeXt model.
    
    Args:
        variant: Model variant ("tiny", "small", "base")
        pretrained: Use ImageNet pretrained weights
        num_mag_classes: Number of magnitude classes
        num_azi_classes: Number of azimuth classes
        dropout: Dropout rate
        
    Returns:
        ConvNeXtMultiTask model
    """
    return ConvNeXtMultiTask(
        variant=variant,
        pretrained=pretrained,
        num_mag_classes=num_mag_classes,
        num_azi_classes=num_azi_classes,
        dropout=dropout
    )


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }
