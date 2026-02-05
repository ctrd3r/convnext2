"""
Inference utilities for earthquake precursor detection.
"""

import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
import json

from .model import ConvNeXtPrecursorModel, ConvNeXtMultiTask
from .dataset import get_transforms


class PrecursorPredictor:
    """
    High-level predictor for earthquake precursor detection.
    
    Example:
        predictor = PrecursorPredictor.from_checkpoint('model.pth')
        result = predictor.predict('spectrogram.png')
        print(f"Magnitude: {result['magnitude_class']}")
    """
    
    def __init__(
        self,
        model: ConvNeXtMultiTask,
        device: torch.device,
        transform: transforms.Compose,
        mag_classes: List[str],
        azi_classes: List[str]
    ):
        self.model = model
        self.device = device
        self.transform = transform
        self.mag_classes = mag_classes
        self.azi_classes = azi_classes
        
        self.model.to(device)
        self.model.eval()
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        device: Optional[torch.device] = None
    ) -> 'PrecursorPredictor':
        """Load predictor from checkpoint."""
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        config = checkpoint.get('config', {})
        
        # Create model
        model = ConvNeXtMultiTask(
            variant=config.get('model_variant', 'tiny'),
            pretrained=False,
            num_mag_classes=config.get('num_mag_classes', 4),
            num_azi_classes=config.get('num_azi_classes', 9)
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get transforms
        transform = get_transforms(
            image_size=config.get('image_size', 224),
            is_training=False
        )
        
        # Get class mappings
        mag_classes = checkpoint.get('mag_classes', ['Large', 'Medium', 'Moderate', 'Normal'])
        azi_classes = checkpoint.get('azi_classes', ['E', 'N', 'NE', 'NW', 'Normal', 'S', 'SE', 'SW', 'W'])
        
        # Try loading from separate file
        mappings_path = Path(checkpoint_path).parent / 'class_mappings.json'
        if mappings_path.exists():
            with open(mappings_path) as f:
                mappings = json.load(f)
                mag_classes = [mappings['magnitude'][str(i)] for i in range(len(mappings['magnitude']))]
                azi_classes = [mappings['azimuth'][str(i)] for i in range(len(mappings['azimuth']))]
        
        return cls(model, device, transform, mag_classes, azi_classes)
    
    def predict(
        self, 
        image: Union[str, Path, Image.Image, torch.Tensor]
    ) -> Dict:
        """
        Predict earthquake precursor from spectrogram.
        
        Args:
            image: Image path, PIL Image, or tensor
            
        Returns:
            Dictionary with predictions
        """
        # Load and preprocess image
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        
        if isinstance(image, Image.Image):
            image = self.transform(image)
        
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Predict
        with torch.no_grad():
            mag_logits, azi_logits = self.model(image)
            
            mag_probs = torch.softmax(mag_logits, dim=1)
            azi_probs = torch.softmax(azi_logits, dim=1)
            
            mag_pred = torch.argmax(mag_probs, dim=1).item()
            azi_pred = torch.argmax(azi_probs, dim=1).item()
        
        return {
            'magnitude_class': self.mag_classes[mag_pred],
            'magnitude_prob': mag_probs[0, mag_pred].item(),
            'magnitude_probs': {
                cls: mag_probs[0, i].item() 
                for i, cls in enumerate(self.mag_classes)
            },
            'azimuth_class': self.azi_classes[azi_pred],
            'azimuth_prob': azi_probs[0, azi_pred].item(),
            'azimuth_probs': {
                cls: azi_probs[0, i].item() 
                for i, cls in enumerate(self.azi_classes)
            },
            'is_precursor': self.mag_classes[mag_pred] != 'Normal',
        }
    
    def predict_batch(
        self, 
        images: List[Union[str, Path, Image.Image]]
    ) -> List[Dict]:
        """Predict on batch of images."""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results


def predict_spectrogram(
    model: Union[ConvNeXtPrecursorModel, ConvNeXtMultiTask, str],
    image_path: Union[str, Path],
    device: Optional[torch.device] = None
) -> Dict:
    """
    Convenience function to predict from spectrogram image.
    
    Args:
        model: Model instance or path to checkpoint
        image_path: Path to spectrogram image
        device: Device to use
        
    Returns:
        Prediction dictionary
    """
    if isinstance(model, str):
        predictor = PrecursorPredictor.from_checkpoint(model, device)
    elif isinstance(model, ConvNeXtPrecursorModel):
        predictor = PrecursorPredictor(
            model.model, model.device,
            get_transforms(is_training=False),
            model.mag_classes, model.azi_classes
        )
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
    
    return predictor.predict(image_path)


def generate_gradcam(
    model: ConvNeXtMultiTask,
    image: torch.Tensor,
    target_layer: str = 'backbone.features.7',
    target_class: Optional[int] = None
) -> np.ndarray:
    """
    Generate GradCAM visualization.
    
    Args:
        model: ConvNeXt model
        image: Input image tensor (1, 3, H, W)
        target_layer: Layer to visualize
        target_class: Target class (None for predicted class)
        
    Returns:
        GradCAM heatmap as numpy array
    """
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        raise ImportError("Please install grad-cam: pip install grad-cam")
    
    # Get target layer
    layers = target_layer.split('.')
    target = model
    for layer in layers:
        target = getattr(target, layer)
    
    # Create GradCAM
    cam = GradCAM(model=model, target_layers=[target])
    
    # Generate heatmap
    grayscale_cam = cam(input_tensor=image, targets=None)
    
    return grayscale_cam[0]
