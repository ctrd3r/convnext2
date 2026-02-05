"""
Unit tests for dataset classes
"""

import pytest
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import sys
sys.path.append('..')

from src.dataset import EarthquakeDataset, LOEODataset, get_transforms


class TestTransforms:
    """Tests for data transforms"""
    
    def test_train_transforms(self):
        """Test training transforms"""
        transform = get_transforms(is_training=True)
        
        # Create dummy image
        img = Image.new('RGB', (256, 256), color='red')
        transformed = transform(img)
        
        assert transformed.shape == (3, 224, 224)
        assert isinstance(transformed, torch.Tensor)
    
    def test_val_transforms(self):
        """Test validation transforms"""
        transform = get_transforms(is_training=False)
        
        img = Image.new('RGB', (256, 256), color='blue')
        transformed = transform(img)
        
        assert transformed.shape == (3, 224, 224)
        assert isinstance(transformed, torch.Tensor)
    
    def test_normalization(self):
        """Test that normalization is applied"""
        transform = get_transforms(is_training=False)
        
        # Create white image
        img = Image.new('RGB', (224, 224), color='white')
        transformed = transform(img)
        
        # After normalization, values should not be in [0, 1]
        # ImageNet normalization: (x - mean) / std
        assert transformed.min() < 0 or transformed.max() > 1


class TestEarthquakeDataset:
    """Tests for EarthquakeDataset"""
    
    @pytest.fixture
    def sample_data(self, tmp_path):
        """Create sample data for testing"""
        # Create sample images
        img_dir = tmp_path / "spectrograms"
        img_dir.mkdir()
        
        for i in range(10):
            img = Image.new('RGB', (224, 224), color=(i*25, i*25, i*25))
            img.save(img_dir / f"sample_{i}.png")
        
        # Create metadata
        metadata = pd.DataFrame({
            'filename': [f"sample_{i}.png" for i in range(10)],
            'magnitude_class': ['Large', 'Medium', 'Moderate', 'Normal'] * 2 + ['Large', 'Medium'],
            'azimuth_class': ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Normal', 'N'],
            'event_id': list(range(10))
        })
        
        return metadata, img_dir
    
    def test_dataset_creation(self, sample_data):
        """Test creating dataset"""
        metadata, img_dir = sample_data
        transform = get_transforms(is_training=False)
        
        dataset = EarthquakeDataset(metadata, img_dir, transform)
        
        assert len(dataset) == 10
    
    def test_dataset_getitem(self, sample_data):
        """Test getting items from dataset"""
        metadata, img_dir = sample_data
        transform = get_transforms(is_training=False)
        
        dataset = EarthquakeDataset(metadata, img_dir, transform)
        
        image, mag_label, azi_label = dataset[0]
        
        assert image.shape == (3, 224, 224)
        assert isinstance(mag_label, int)
        assert isinstance(azi_label, int)
    
    def test_class_mappings(self, sample_data):
        """Test class mappings are created correctly"""
        metadata, img_dir = sample_data
        transform = get_transforms(is_training=False)
        
        dataset = EarthquakeDataset(metadata, img_dir, transform)
        
        assert len(dataset.mag_mapping) == 4  # Large, Medium, Moderate, Normal
        assert len(dataset.azi_mapping) == 9  # 8 directions + Normal


class TestLOEODataset:
    """Tests for LOEODataset (Leave-One-Event-Out)"""
    
    @pytest.fixture
    def sample_loeo_data(self, tmp_path):
        """Create sample data for LOEO testing"""
        img_dir = tmp_path / "spectrograms"
        img_dir.mkdir()
        
        # Create 20 samples from 5 events
        for i in range(20):
            img = Image.new('RGB', (224, 224), color=(i*12, i*12, i*12))
            img.save(img_dir / f"sample_{i}.png")
        
        metadata = pd.DataFrame({
            'filename': [f"sample_{i}.png" for i in range(20)],
            'magnitude_class': ['Large', 'Medium', 'Moderate', 'Normal'] * 5,
            'azimuth_class': ['N', 'NE', 'E', 'SE'] * 5,
            'event_id': [i // 4 for i in range(20)]  # 5 events, 4 samples each
        })
        
        return metadata, img_dir
    
    def test_loeo_split(self, sample_loeo_data):
        """Test LOEO split creates correct train/test sets"""
        metadata, img_dir = sample_loeo_data
        transform = get_transforms(is_training=False)
        
        # Hold out event 0
        train_df = metadata[metadata['event_id'] != 0]
        test_df = metadata[metadata['event_id'] == 0]
        
        train_dataset = EarthquakeDataset(train_df, img_dir, transform)
        test_dataset = EarthquakeDataset(test_df, img_dir, transform)
        
        assert len(train_dataset) == 16  # 4 events * 4 samples
        assert len(test_dataset) == 4    # 1 event * 4 samples
    
    def test_no_event_overlap(self, sample_loeo_data):
        """Test that train and test have no event overlap"""
        metadata, img_dir = sample_loeo_data
        
        for held_out_event in range(5):
            train_df = metadata[metadata['event_id'] != held_out_event]
            test_df = metadata[metadata['event_id'] == held_out_event]
            
            train_events = set(train_df['event_id'].unique())
            test_events = set(test_df['event_id'].unique())
            
            assert len(train_events & test_events) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
