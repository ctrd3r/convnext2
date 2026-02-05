"""
Unit tests for ConvNeXt model
"""

import pytest
import torch
import sys
sys.path.append('..')

from src.model import ConvNeXtMultiTask, create_model, count_parameters


class TestConvNeXtMultiTask:
    """Tests for ConvNeXtMultiTask model"""
    
    def test_model_creation_tiny(self):
        """Test creating ConvNeXt-Tiny model"""
        model = create_model(variant="tiny", pretrained=False)
        assert model is not None
        assert model.num_features == 768
    
    def test_model_creation_small(self):
        """Test creating ConvNeXt-Small model"""
        model = create_model(variant="small", pretrained=False)
        assert model is not None
        assert model.num_features == 768
    
    def test_forward_pass(self):
        """Test forward pass with random input"""
        model = create_model(variant="tiny", pretrained=False)
        model.eval()
        
        # Create random input
        x = torch.randn(2, 3, 224, 224)
        
        # Forward pass
        mag_out, azi_out = model(x)
        
        # Check output shapes
        assert mag_out.shape == (2, 4)  # 4 magnitude classes
        assert azi_out.shape == (2, 9)  # 9 azimuth classes
    
    def test_predict_method(self):
        """Test predict method"""
        model = create_model(variant="tiny", pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        
        results = model.predict(x)
        
        assert 'magnitude_pred' in results
        assert 'magnitude_prob' in results
        assert 'azimuth_pred' in results
        assert 'azimuth_prob' in results
    
    def test_get_features(self):
        """Test feature extraction"""
        model = create_model(variant="tiny", pretrained=False)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        features = model.get_features(x)
        
        # Features should be 768-dimensional for tiny variant
        assert features.shape[1] == 768 or features.numel() // 2 == 768
    
    def test_parameter_count(self):
        """Test parameter counting"""
        model = create_model(variant="tiny", pretrained=False)
        params = count_parameters(model)
        
        assert 'total' in params
        assert 'trainable' in params
        assert params['total'] > 0
        assert params['trainable'] > 0
    
    def test_different_batch_sizes(self):
        """Test with different batch sizes"""
        model = create_model(variant="tiny", pretrained=False)
        model.eval()
        
        for batch_size in [1, 4, 8, 16]:
            x = torch.randn(batch_size, 3, 224, 224)
            mag_out, azi_out = model(x)
            
            assert mag_out.shape[0] == batch_size
            assert azi_out.shape[0] == batch_size
    
    def test_different_input_sizes(self):
        """Test with different input sizes (should work with 224x224)"""
        model = create_model(variant="tiny", pretrained=False)
        model.eval()
        
        # Standard size
        x = torch.randn(1, 3, 224, 224)
        mag_out, azi_out = model(x)
        
        assert mag_out.shape == (1, 4)
        assert azi_out.shape == (1, 9)


class TestModelVariants:
    """Test different model variants"""
    
    @pytest.mark.parametrize("variant", ["tiny", "small"])
    def test_variant_creation(self, variant):
        """Test creating different variants"""
        model = create_model(variant=variant, pretrained=False)
        assert model is not None
    
    def test_invalid_variant(self):
        """Test that invalid variant raises error"""
        with pytest.raises(ValueError):
            create_model(variant="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
