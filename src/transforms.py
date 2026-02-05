"""
Data transforms and augmentation for earthquake precursor detection.
"""

import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from typing import Tuple, Optional


class SpectrogramTransform:
    """
    Custom transform for spectrogram images.
    
    Includes normalization specific to spectrogram data.
    """
    
    def __init__(
        self,
        image_size: int = 224,
        is_training: bool = True,
        use_augmentation: bool = True
    ):
        self.image_size = image_size
        self.is_training = is_training
        self.use_augmentation = use_augmentation
        
        # ImageNet normalization (for pretrained models)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    
    def __call__(self, image: Image.Image) -> torch.Tensor:
        # Resize
        image = transforms.functional.resize(
            image, (self.image_size, self.image_size)
        )
        
        # Augmentation (training only)
        if self.is_training and self.use_augmentation:
            # Random horizontal flip
            if np.random.random() < 0.5:
                image = transforms.functional.hflip(image)
            
            # Random rotation
            angle = np.random.uniform(-15, 15)
            image = transforms.functional.rotate(image, angle)
            
            # Color jitter
            image = transforms.functional.adjust_brightness(
                image, 1 + np.random.uniform(-0.2, 0.2)
            )
            image = transforms.functional.adjust_contrast(
                image, 1 + np.random.uniform(-0.2, 0.2)
            )
        
        # Convert to tensor
        image = transforms.functional.to_tensor(image)
        
        # Normalize
        image = self.normalize(image)
        
        # Random erasing (training only)
        if self.is_training and self.use_augmentation:
            if np.random.random() < 0.1:
                image = transforms.functional.erase(
                    image,
                    i=np.random.randint(0, self.image_size // 2),
                    j=np.random.randint(0, self.image_size // 2),
                    h=np.random.randint(10, self.image_size // 4),
                    w=np.random.randint(10, self.image_size // 4),
                    v=0
                )
        
        return image


class MixUp:
    """
    MixUp augmentation.
    
    Reference: Zhang et al., "mixup: Beyond Empirical Risk Minimization" (ICLR 2018)
    """
    
    def __init__(self, alpha: float = 0.8):
        self.alpha = alpha
    
    def __call__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply MixUp to batch.
        
        Returns:
            mixed_images, labels_a, labels_b, lambda
        """
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        return mixed_images, labels, labels[index], lam


class CutMix:
    """
    CutMix augmentation.
    
    Reference: Yun et al., "CutMix: Regularization Strategy to Train Strong Classifiers" (ICCV 2019)
    """
    
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
    
    def __call__(
        self, 
        images: torch.Tensor, 
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """
        Apply CutMix to batch.
        
        Returns:
            mixed_images, labels_a, labels_b, lambda
        """
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.alpha, self.alpha)
        
        # Random permutation
        index = torch.randperm(batch_size)
        
        # Get bounding box
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = images[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return mixed_images, labels, labels[index], lam


def get_train_transforms(
    image_size: int = 224,
    use_augmentation: bool = True
) -> transforms.Compose:
    """Get training transforms."""
    if use_augmentation:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        return get_val_transforms(image_size)


def get_val_transforms(image_size: int = 224) -> transforms.Compose:
    """Get validation/test transforms."""
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize tensor for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean
