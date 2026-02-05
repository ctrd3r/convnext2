"""
Unit tests for inference module
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
from pathlib import Path
import sys
sys.path.append('..')

from src.model import ConvNeXtMultiTask, create_model
from src.inference import PrecursorPredictor, predict_spectrogram, preprocess_image


class TestPreprocessing:
    """Tests for image preprocessing"""
    
    def test_preprocess_pil_image(self):
        """Test preprocessing PIL image"""
        img = Image.new('RGB', (256, 256), color='red')
        tensor = preprocess_image(img)
        
        assert tensor.shape == (1, 3, 224, 224)
        assert isinstance(tensor, torch.Tensor)
    
    def test_preprocess_from_path(self, tmp_path):
        """Test preprocessing from file path"""
        img_path = tmp_path / "test.png"
        img = Image.new('RGB', (256, 256), color='blue')
        img.save(img_path)
        
        tensor = preprocess_image(str(img_path))
        
        assert tensor.shape == (1, 3, 224, 224)
    
    def test_preprocess_numpy_array(self):
        """Test preprocessing numpy array"""
        arr = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        tensor = preprocess_image(arr)
        
        assert tensor.shape == (1, 3, 224, 224)


class TestPrecursorPredictor:
    """Tests for PrecursorPredictor class"""
    
    @pytest.fixture
    def predictor(self):
        """Create predictor with untrained model"""
        model = create_model(variant="tiny", pretrained=False)
        device = torch.device('cpu')
        return PrecursorPredictor(model, device)
    
    def test_predictor_creation(self, predictor):
        """Test predictor creation"""
        assert predictor is not None
        assert predictor.model is not None
    
    def test_predict_single_image(self, predictor):
        """Test prediction on single image"""
        img = Image.new('RGB', (224, 224), color='green')
        result = predictor.predict(img)
        
        assert 'magnitude_class' in result
        assert 'magnitude_prob' in result
        assert 'azimuth_class' in result
        assert 'azimuth_prob' in result
        
        assert result['magnitude_class'] in ['Large', 'Medium', 'Moderate', 'Normal']
        assert 0 <= result['magnitude_prob'] <= 1
    
    def test_predict_batch(self, predictor):
        """Test batch prediction"""
        images = [Image.new('RGB', (224, 224), color=c) 
                  for c in ['red', 'green', 'blue']]
        
        results = predictor.predict_batch(images)
        
        assert len(results) == 3
        for result in results:
            assert 'magnitude_class' in result
            assert 'azimuth_class' in result
    
    def test_class_names(self, predictor):
        """Test that class names are correct"""
        assert len(predictor.mag_classes) == 4
        assert len(predictor.azi_classes) == 9
        
        assert 'Large' in predictor.mag_classes
        assert 'Normal' in predictor.mag_classes
        assert 'N' in predictor.azi_classes


class TestPredictSpectrogram:
    """Tests for predict_spectrogram function"""
    
    def test_predict_spectrogram_basic(self):
        """Test basic spectrogram prediction"""
        model = create_model(variant="tiny", pretrained=False)
        img = Image.new('RGB', (224, 224), color='purple')
        
        result = predict_spectrogram(model, img)
        
        assert 'magnitude_class' in result
        assert 'azimuth_class' in result
    
    def test_predict_spectrogram_from_file(self, tmp_path):
        """Test prediction from file"""
        model = create_model(variant="tiny", pretrained=False)
        
        img_path = tmp_path / "spectrogram.png"
        img = Image.new('RGB', (224, 224), color='orange')
        img.save(img_path)
        
        result = predict_spectrogram(model, str(img_path))
        
        assert result is not None


class TestConfidenceScores:
    """Tests for confidence score handling"""
    
    def test_confidence_range(self):
        """Test that confidence scores are in valid range"""
        model = create_model(variant="tiny", pretrained=False)
        device = torch.device('cpu')
        predictor = PrecursorPredictor(model, device)
        
        img = Image.new('RGB', (224, 224), color='cyan')
        result = predictor.predict(img)
        
        assert 0 <= result['magnitude_prob'] <= 1
        assert 0 <= result['azimuth_prob'] <= 1
    
    def test_probability_distribution(self):
        """Test that probabilities sum to 1"""
        model = create_model(variant="tiny", pretrained=False)
        device = torch.device('cpu')
        predictor = PrecursorPredictor(model, device)
        
        img = Image.new('RGB', (224, 224), color='magenta')
        result = predictor.predict(img)
        
        if 'magnitude_probs' in result:
            mag_sum = sum(result['magnitude_probs'].values())
            assert abs(mag_sum - 1.0) < 0.01
        
        if 'azimuth_probs' in result:
            azi_sum = sum(result['azimuth_probs'].values())
            assert abs(azi_sum - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
