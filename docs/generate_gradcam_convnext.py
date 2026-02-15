#!/usr/bin/env python3
"""
Generate Grad-CAM Visualizations for ConvNeXt Model

This script generates Grad-CAM (Gradient-weighted Class Activation Mapping)
visualizations for the ConvNeXt earthquake precursor model.

Author: Earthquake Prediction Research Team
Date: 5 February 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from glob import glob

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
import cv2

# Configuration
OUTPUT_DIR = Path("publication_convnext/figures/gradcam")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find latest model
EXPERIMENT_DIRS = sorted(glob("experiments_convnext/convnext_tiny_*"))
if EXPERIMENT_DIRS:
    MODEL_PATH = Path(EXPERIMENT_DIRS[-1]) / "best_model.pth"
else:
    MODEL_PATH = None

DATASET_DIR = Path("dataset_unified/spectrograms")
METADATA_PATH = Path("dataset_unified/metadata/unified_metadata.csv")

print("=" * 60)
print("GRAD-CAM VISUALIZATION - ConvNeXt Model")
print("=" * 60)


class ConvNeXtMultiTask(nn.Module):
    """ConvNeXt model with multi-task heads"""
    
    def __init__(self, num_mag_classes=4, num_azi_classes=9):
        super().__init__()
        
        self.backbone = convnext_tiny(weights=None)
        num_features = 768
        self.backbone.classifier = nn.Identity()
        
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_mag_classes)
        )
        
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(0.25),
            nn.Linear(512, num_azi_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.flatten(1)
        
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        
        return mag_out, azi_out


class GradCAM:
    """Grad-CAM implementation for ConvNeXt"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class, task='magnitude'):
        self.model.zero_grad()
        
        mag_out, azi_out = self.model(input_tensor)
        
        if task == 'magnitude':
            output = mag_out
        else:
            output = azi_out
        
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()


def load_model():
    """Load trained ConvNeXt model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = ConvNeXtMultiTask(num_mag_classes=4, num_azi_classes=9)
    
    if MODEL_PATH and MODEL_PATH.exists():
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print("WARNING: Model not found!")
        return None, device
    
    model.to(device)
    model.eval()
    return model, device


def get_transform():
    """Get image transforms"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def generate_gradcam_visualization(model, device, img_path, class_mappings, output_prefix):
    """Generate Grad-CAM visualization for a sample"""
    
    # Load image
    image = Image.open(img_path).convert('RGB')
    transform = get_transform()
    img_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        mag_out, azi_out = model(img_tensor)
        mag_pred = torch.argmax(mag_out, dim=1).item()
        azi_pred = torch.argmax(azi_out, dim=1).item()
        mag_conf = torch.softmax(mag_out, dim=1)[0].cpu().numpy()
        azi_conf = torch.softmax(azi_out, dim=1)[0].cpu().numpy()
    
    # Get target layer (last conv layer of ConvNeXt)
    target_layer = model.backbone.features[-1]
    
    # Create Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmaps
    mag_heatmap = gradcam.generate(img_tensor, mag_pred, 'magnitude')
    azi_heatmap = gradcam.generate(img_tensor, azi_pred, 'azimuth')
    
    # Resize heatmaps
    mag_heatmap_resized = cv2.resize(mag_heatmap, (224, 224))
    azi_heatmap_resized = cv2.resize(azi_heatmap, (224, 224))
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    img_array = np.array(image.resize((224, 224)))
    
    # Magnitude row
    axes[0, 0].imshow(img_array)
    axes[0, 0].set_title('Original Spectrogram', fontsize=12)
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(mag_heatmap_resized, cmap='jet')
    axes[0, 1].set_title(f'Magnitude Grad-CAM\nPred: {class_mappings["magnitude"][str(mag_pred)]}', fontsize=12)
    axes[0, 1].axis('off')
    
    mag_colored = cv2.applyColorMap(np.uint8(255 * mag_heatmap_resized), cv2.COLORMAP_JET)
    mag_colored = cv2.cvtColor(mag_colored, cv2.COLOR_BGR2RGB)
    mag_overlay = cv2.addWeighted(img_array, 0.6, mag_colored, 0.4, 0)
    axes[0, 2].imshow(mag_overlay)
    axes[0, 2].set_title(f'Overlay (Conf: {mag_conf.max():.1%})', fontsize=12)
    axes[0, 2].axis('off')
    
    # Azimuth row
    axes[1, 0].imshow(img_array)
    axes[1, 0].set_title('Original Spectrogram', fontsize=12)
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(azi_heatmap_resized, cmap='jet')
    axes[1, 1].set_title(f'Azimuth Grad-CAM\nPred: {class_mappings["azimuth"][str(azi_pred)]}', fontsize=12)
    axes[1, 1].axis('off')
    
    azi_colored = cv2.applyColorMap(np.uint8(255 * azi_heatmap_resized), cv2.COLORMAP_JET)
    azi_colored = cv2.cvtColor(azi_colored, cv2.COLOR_BGR2RGB)
    azi_overlay = cv2.addWeighted(img_array, 0.6, azi_colored, 0.4, 0)
    axes[1, 2].imshow(azi_overlay)
    axes[1, 2].set_title(f'Overlay (Conf: {azi_conf.max():.1%})', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle(f'ConvNeXt Grad-CAM Analysis: {Path(img_path).stem}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = OUTPUT_DIR / f"{output_prefix}_gradcam.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return output_path


def main():
    """Main function"""
    
    # Load model
    model, device = load_model()
    if model is None:
        print("Cannot proceed without model. Exiting.")
        return
    
    # Load class mappings
    class_mappings = {
        "magnitude": {"0": "Large", "1": "Medium", "2": "Moderate", "3": "Normal"},
        "azimuth": {"0": "E", "1": "N", "2": "NE", "3": "NW", "4": "Normal",
                   "5": "S", "6": "SE", "7": "SW", "8": "W"}
    }
    
    # Try to load from experiment
    if EXPERIMENT_DIRS:
        mappings_path = Path(EXPERIMENT_DIRS[-1]) / "class_mappings.json"
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                class_mappings = json.load(f)
    
    # Load metadata
    if METADATA_PATH.exists():
        metadata_df = pd.read_csv(METADATA_PATH)
        print(f"Loaded {len(metadata_df)} samples")
    else:
        print("Metadata not found!")
        return
    
    # Select samples for visualization
    print("\nGenerating Grad-CAM visualizations...")
    
    # Sample from each magnitude class
    for mag_class in ['Large', 'Medium', 'Moderate', 'Normal']:
        samples = metadata_df[metadata_df['magnitude_class'] == mag_class]
        if len(samples) > 0:
            sample = samples.iloc[0]
            
            if 'unified_path' in sample.index:
                img_path = Path("dataset_unified") / sample['unified_path']
            else:
                img_path = DATASET_DIR / sample.get('spectrogram_file', sample.get('filename', ''))
            
            if img_path.exists():
                generate_gradcam_visualization(
                    model, device, str(img_path), class_mappings,
                    f"mag_{mag_class.lower()}"
                )
    
    print("\n" + "=" * 60)
    print("GRAD-CAM GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
