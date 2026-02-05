"""
Dataset classes for earthquake precursor detection.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np


class EarthquakeDataset(Dataset):
    """
    Dataset for earthquake precursor spectrograms.
    
    Args:
        metadata_df: DataFrame with image paths and labels
        img_dir: Base directory for images
        transform: Image transforms
        mag_mapping: Magnitude class to index mapping
        azi_mapping: Azimuth class to index mapping
    """
    
    def __init__(
        self, 
        metadata_df: pd.DataFrame, 
        img_dir: str,
        transform: Optional[Callable] = None, 
        mag_mapping: Optional[Dict[str, int]] = None, 
        azi_mapping: Optional[Dict[str, int]] = None
    ):
        self.metadata = metadata_df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        
        # Create class mappings
        if mag_mapping is None:
            mag_classes = sorted(self.metadata['magnitude_class'].unique())
            self.mag_mapping = {c: i for i, c in enumerate(mag_classes)}
        else:
            self.mag_mapping = mag_mapping
            
        if azi_mapping is None:
            azi_classes = sorted(self.metadata['azimuth_class'].unique())
            self.azi_mapping = {c: i for i, c in enumerate(azi_classes)}
        else:
            self.azi_mapping = azi_mapping
        
        # Reverse mappings
        self.mag_classes = {v: k for k, v in self.mag_mapping.items()}
        self.azi_classes = {v: k for k, v in self.azi_mapping.items()}
    
    def __len__(self) -> int:
        return len(self.metadata)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int]:
        row = self.metadata.iloc[idx]
        
        # Get image path
        img_path = self._get_image_path(row)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        mag_label = self.mag_mapping.get(row['magnitude_class'], 0)
        azi_label = self.azi_mapping.get(row['azimuth_class'], 0)
        
        return image, mag_label, azi_label
    
    def _get_image_path(self, row: pd.Series) -> Path:
        """Get image path from metadata row."""
        # Support multiple column names
        if 'unified_path' in row.index and pd.notna(row['unified_path']):
            return Path("dataset_unified") / row['unified_path']
        elif 'spectrogram_file' in row.index:
            return self.img_dir / row['spectrogram_file']
        elif 'filename' in row.index:
            return self.img_dir / row['filename']
        elif 'image_path' in row.index:
            return Path(row['image_path'])
        else:
            raise KeyError(f"No valid image path column found. Available: {row.index.tolist()}")
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling."""
        mag_counts = self.metadata['magnitude_class'].value_counts()
        weights = self.metadata['magnitude_class'].map(
            lambda x: 1.0 / mag_counts[x]
        ).values
        return torch.tensor(weights, dtype=torch.float)


class LOEODataset(EarthquakeDataset):
    """
    Dataset for Leave-One-Event-Out cross-validation.
    
    Ensures all samples from the same earthquake event are in the same split.
    """
    
    def __init__(
        self,
        metadata_df: pd.DataFrame,
        img_dir: str,
        event_column: str = 'event_id',
        transform: Optional[Callable] = None,
        mag_mapping: Optional[Dict[str, int]] = None,
        azi_mapping: Optional[Dict[str, int]] = None
    ):
        super().__init__(metadata_df, img_dir, transform, mag_mapping, azi_mapping)
        self.event_column = event_column
        
        # Get unique events
        self.events = self.metadata[event_column].unique()
        self.n_events = len(self.events)
    
    def get_fold_indices(self, fold: int, n_folds: int = 10) -> Tuple[List[int], List[int]]:
        """
        Get train/test indices for a specific fold.
        
        Args:
            fold: Fold number (0-indexed)
            n_folds: Total number of folds
            
        Returns:
            Tuple of (train_indices, test_indices)
        """
        # Split events into folds
        events_per_fold = len(self.events) // n_folds
        
        start_idx = fold * events_per_fold
        if fold == n_folds - 1:
            end_idx = len(self.events)
        else:
            end_idx = start_idx + events_per_fold
        
        test_events = set(self.events[start_idx:end_idx])
        
        # Get sample indices
        test_mask = self.metadata[self.event_column].isin(test_events)
        test_indices = self.metadata[test_mask].index.tolist()
        train_indices = self.metadata[~test_mask].index.tolist()
        
        return train_indices, test_indices


def get_transforms(
    image_size: int = 224,
    is_training: bool = True,
    use_augmentation: bool = True
) -> transforms.Compose:
    """
    Get image transforms.
    
    Args:
        image_size: Target image size
        is_training: Whether this is for training
        use_augmentation: Whether to use data augmentation
        
    Returns:
        Composed transforms
    """
    if is_training and use_augmentation:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    
    return transform


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create class mappings from all data
    all_data = pd.concat([train_df, val_df, test_df])
    mag_classes = sorted(all_data['magnitude_class'].unique())
    azi_classes = sorted(all_data['azimuth_class'].unique())
    
    mag_mapping = {c: i for i, c in enumerate(mag_classes)}
    azi_mapping = {c: i for i, c in enumerate(azi_classes)}
    
    # Create datasets
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    train_dataset = EarthquakeDataset(
        train_df, img_dir, train_transform, mag_mapping, azi_mapping
    )
    val_dataset = EarthquakeDataset(
        val_df, img_dir, val_transform, mag_mapping, azi_mapping
    )
    test_dataset = EarthquakeDataset(
        test_df, img_dir, val_transform, mag_mapping, azi_mapping
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader
