#!/usr/bin/env python3
"""
Enhanced ConvNeXt Model with Attention Mechanisms
Includes CBAM attention, hierarchical azimuth classification, and directional features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class SpatialAttention(nn.Module):
    """Spatial attention to focus on important regions in spectrograms"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise max and average pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention


class ChannelAttention(nn.Module):
    """Channel attention to focus on important feature channels"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class DirectionalFeatureExtractor(nn.Module):
    """Extract direction-specific features for azimuth classification"""
    def __init__(self, in_channels):
        super().__init__()
        
        # Vertical emphasis (for N-S directions)
        self.vertical_conv = nn.Conv2d(
            in_channels, in_channels // 4,
            kernel_size=(7, 3), padding=(3, 1)
        )
        
        # Horizontal emphasis (for E-W directions)
        self.horizontal_conv = nn.Conv2d(
            in_channels, in_channels // 4,
            kernel_size=(3, 7), padding=(1, 3)
        )
        
        # Diagonal emphasis (for NE-SW directions)
        self.diagonal1_conv = nn.Conv2d(
            in_channels, in_channels // 4,
            kernel_size=5, padding=2
        )
        
        # Diagonal emphasis (for NW-SE directions)
        self.diagonal2_conv = nn.Conv2d(
            in_channels, in_channels // 4,
            kernel_size=5, padding=2
        )
        
        # Fusion layer
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        v = self.vertical_conv(x)
        h = self.horizontal_conv(x)
        d1 = self.diagonal1_conv(x)
        d2 = self.diagonal2_conv(x)
        
        combined = torch.cat([v, h, d1, d2], dim=1)
        fused = self.fusion(combined)
        fused = self.bn(fused)
        fused = self.relu(fused)
        
        return fused


class HierarchicalAzimuthHead(nn.Module):
    """Two-stage hierarchical azimuth classification"""
    def __init__(self, in_features, dropout=0.4):
        super().__init__()
        
        # Stage 1: Cardinal directions (N/S/E/W + Center)
        self.cardinal_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 5)  # N, S, E, W, Center
        )
        
        # Stage 2: Intercardinal directions (NE/SE/SW/NW)
        self.intercardinal_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 4)  # NE, SE, SW, NW
        )
        
        # Fusion layer to combine predictions
        self.fusion = nn.Linear(9, 9)
    
    def forward(self, x):
        cardinal = self.cardinal_head(x)
        intercardinal = self.intercardinal_head(x)
        
        # Combine predictions
        # Map: N, NE, E, SE, S, SW, W, NW, Center
        combined = torch.cat([
            cardinal[:, 0:1],      # N
            intercardinal[:, 0:1], # NE
            cardinal[:, 2:3],      # E
            intercardinal[:, 1:2], # SE
            cardinal[:, 1:2],      # S
            intercardinal[:, 2:3], # SW
            cardinal[:, 3:4],      # W
            intercardinal[:, 3:4], # NW
            cardinal[:, 4:5],      # Center
        ], dim=1)
        
        return self.fusion(combined)


class EnhancedClassificationHead(nn.Module):
    """Enhanced classification head with batch norm and progressive dropout"""
    def __init__(self, in_features, num_classes, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)


class EnhancedConvNeXt(nn.Module):
    """
    Enhanced ConvNeXt with:
    - CBAM attention mechanisms
    - Directional feature extraction for azimuth
    - Hierarchical azimuth classification
    - Enhanced classification heads
    """
    def __init__(self, 
                 num_mag_classes=4, 
                 num_azi_classes=9, 
                 dropout=0.4,
                 use_attention=True,
                 use_hierarchical_azimuth=True,
                 use_directional_features=True):
        super().__init__()
        
        # Load pretrained ConvNeXt-Tiny
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = models.convnext_tiny(weights=weights)
        
        # Get feature dimension (768 for ConvNeXt-Tiny)
        num_features = 768
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Add attention after backbone features
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(num_features, reduction=16)
        
        # Add directional feature extraction for azimuth
        self.use_directional_features = use_directional_features
        if use_directional_features:
            self.directional_features = DirectionalFeatureExtractor(num_features)
        
        # Classification heads
        self.magnitude_head = EnhancedClassificationHead(
            num_features, num_mag_classes, dropout
        )
        
        # Hierarchical or standard azimuth head
        self.use_hierarchical_azimuth = use_hierarchical_azimuth
        if use_hierarchical_azimuth:
            self.azimuth_head = HierarchicalAzimuthHead(num_features, dropout)
        else:
            self.azimuth_head = EnhancedClassificationHead(
                num_features, num_azi_classes, dropout
            )
    
    def forward(self, x):
        # Extract features from backbone
        features = self.backbone.features(x)
        
        # Apply attention if enabled
        if self.use_attention:
            features = self.attention(features)
        
        # Apply directional feature extraction for azimuth
        if self.use_directional_features:
            directional_feat = self.directional_features(features)
            # Residual connection
            features = features + directional_feat
        
        # Global average pooling
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Classification
        mag_out = self.magnitude_head(features)
        azi_out = self.azimuth_head(features)
        
        return mag_out, azi_out


def create_enhanced_model(config):
    """Factory function to create enhanced model from config"""
    model = EnhancedConvNeXt(
        num_mag_classes=config.get('num_magnitude_classes', 4),
        num_azi_classes=config.get('num_azimuth_classes', 9),
        dropout=config.get('dropout_rate', 0.4),
        use_attention=config.get('use_attention', True),
        use_hierarchical_azimuth=config.get('use_hierarchical_azimuth', True),
        use_directional_features=config.get('use_directional_features', True)
    )
    return model


if __name__ == '__main__':
    # Test the model
    print("Testing Enhanced ConvNeXt Model...")
    
    # Create model
    model = EnhancedConvNeXt(
        num_mag_classes=4,
        num_azi_classes=9,
        dropout=0.4,
        use_attention=True,
        use_hierarchical_azimuth=True,
        use_directional_features=True
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    mag_out, azi_out = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Magnitude output shape: {mag_out.shape}")
    print(f"Azimuth output shape: {azi_out.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Model size: {total_params * 4 / 1024 / 1024:.2f} MB")
    
    print("\n✅ Model test passed!")
