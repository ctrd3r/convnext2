#!/usr/bin/env python3
"""
Evaluate ConvNeXt Model Performance

This script evaluates the trained ConvNeXt model and generates
comprehensive performance metrics for publication.

Author: Earthquake Prediction Research Team
Date: 5 February 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from glob import glob
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny
from PIL import Image
from sklearn.metrics import (classification_report, confusion_matrix, 
                            f1_score, precision_recall_fscore_support)

# Configuration
OUTPUT_DIR = Path("publication_convnext/evaluation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find latest model
EXPERIMENT_DIRS = sorted(glob("experiments_convnext/convnext_tiny_*"))
if EXPERIMENT_DIRS:
    MODEL_PATH = Path(EXPERIMENT_DIRS[-1]) / "best_model.pth"
    EXPERIMENT_DIR = Path(EXPERIMENT_DIRS[-1])
else:
    MODEL_PATH = None
    EXPERIMENT_DIR = None

DATASET_DIR = Path("dataset_unified/spectrograms")
TEST_SPLIT_PATH = Path("dataset_unified/metadata/test_split.csv")

print("=" * 60)
print("CONVNEXT MODEL EVALUATION")
print("=" * 60)


class EarthquakeDataset(Dataset):
    """Dataset for earthquake spectrograms"""
    
    def __init__(self, metadata_df, transform=None, mag_mapping=None, azi_mapping=None):
        self.metadata = metadata_df.reset_index(drop=True)
        self.transform = transform
        self.mag_mapping = mag_mapping or {}
        self.azi_mapping = azi_mapping or {}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        if 'unified_path' in row.index:
            img_path = Path("dataset_unified") / row['unified_path']
        else:
            img_path = DATASET_DIR / row.get('spectrogram_file', row.get('filename', ''))
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        mag_label = self.mag_mapping.get(row['magnitude_class'], 0)
        azi_label = self.azi_mapping.get(row['azimuth_class'], 0)
        
        return image, mag_label, azi_label


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


def load_model():
    """Load trained model"""
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


def evaluate_model(model, dataloader, device, class_mappings):
    """Evaluate model on test set"""
    
    all_mag_preds = []
    all_mag_labels = []
    all_azi_preds = []
    all_azi_labels = []
    all_mag_probs = []
    all_azi_probs = []
    
    with torch.no_grad():
        for images, mag_labels, azi_labels in dataloader:
            images = images.to(device)
            
            mag_out, azi_out = model(images)
            
            mag_probs = torch.softmax(mag_out, dim=1)
            azi_probs = torch.softmax(azi_out, dim=1)
            
            mag_pred = torch.argmax(mag_out, dim=1)
            azi_pred = torch.argmax(azi_out, dim=1)
            
            all_mag_preds.extend(mag_pred.cpu().numpy())
            all_mag_labels.extend(mag_labels.numpy())
            all_azi_preds.extend(azi_pred.cpu().numpy())
            all_azi_labels.extend(azi_labels.numpy())
            all_mag_probs.extend(mag_probs.cpu().numpy())
            all_azi_probs.extend(azi_probs.cpu().numpy())
    
    # Calculate metrics
    mag_acc = np.mean(np.array(all_mag_preds) == np.array(all_mag_labels))
    azi_acc = np.mean(np.array(all_azi_preds) == np.array(all_azi_labels))
    
    mag_f1 = f1_score(all_mag_labels, all_mag_preds, average='weighted')
    azi_f1 = f1_score(all_azi_labels, all_azi_preds, average='weighted')
    
    return {
        'mag_acc': mag_acc,
        'azi_acc': azi_acc,
        'mag_f1': mag_f1,
        'azi_f1': azi_f1,
        'mag_preds': all_mag_preds,
        'mag_labels': all_mag_labels,
        'azi_preds': all_azi_preds,
        'azi_labels': all_azi_labels,
        'mag_probs': all_mag_probs,
        'azi_probs': all_azi_probs
    }


def plot_confusion_matrices(results, mag_classes, azi_classes, output_path):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Magnitude
    mag_cm = confusion_matrix(results['mag_labels'], results['mag_preds'])
    sns.heatmap(mag_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=mag_classes, yticklabels=mag_classes, ax=axes[0])
    axes[0].set_xlabel('Predicted', fontsize=12)
    axes[0].set_ylabel('True', fontsize=12)
    axes[0].set_title('Magnitude Classification', fontsize=14)
    
    # Azimuth
    azi_cm = confusion_matrix(results['azi_labels'], results['azi_preds'])
    sns.heatmap(azi_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=azi_classes, yticklabels=azi_classes, ax=axes[1])
    axes[1].set_xlabel('Predicted', fontsize=12)
    axes[1].set_ylabel('True', fontsize=12)
    axes[1].set_title('Azimuth Classification', fontsize=14)
    
    plt.suptitle('ConvNeXt-Tiny Confusion Matrices', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def generate_classification_report(results, mag_classes, azi_classes, output_path):
    """Generate detailed classification report"""
    
    with open(output_path, 'w') as f:
        f.write("# ConvNeXt Model Evaluation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Summary
        f.write("## Summary\n\n")
        f.write(f"- **Magnitude Accuracy**: {results['mag_acc']*100:.2f}%\n")
        f.write(f"- **Azimuth Accuracy**: {results['azi_acc']*100:.2f}%\n")
        f.write(f"- **Magnitude F1 (weighted)**: {results['mag_f1']:.4f}\n")
        f.write(f"- **Azimuth F1 (weighted)**: {results['azi_f1']:.4f}\n\n")
        
        # Magnitude report
        f.write("## Magnitude Classification Report\n\n")
        f.write("```\n")
        f.write(classification_report(results['mag_labels'], results['mag_preds'],
                                     target_names=mag_classes))
        f.write("```\n\n")
        
        # Azimuth report
        f.write("## Azimuth Classification Report\n\n")
        f.write("```\n")
        f.write(classification_report(results['azi_labels'], results['azi_preds'],
                                     target_names=azi_classes))
        f.write("```\n\n")
    
    print(f"Saved: {output_path}")


def main():
    """Main evaluation function"""
    
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
    
    if EXPERIMENT_DIR:
        mappings_path = EXPERIMENT_DIR / "class_mappings.json"
        if mappings_path.exists():
            with open(mappings_path, 'r') as f:
                class_mappings = json.load(f)
    
    mag_classes = [class_mappings['magnitude'][str(i)] for i in range(len(class_mappings['magnitude']))]
    azi_classes = [class_mappings['azimuth'][str(i)] for i in range(len(class_mappings['azimuth']))]
    
    mag_mapping = {v: int(k) for k, v in class_mappings['magnitude'].items()}
    azi_mapping = {v: int(k) for k, v in class_mappings['azimuth'].items()}
    
    # Load test data
    if TEST_SPLIT_PATH.exists():
        test_df = pd.read_csv(TEST_SPLIT_PATH)
        print(f"Loaded {len(test_df)} test samples")
    else:
        print("Test split not found!")
        return
    
    # Create dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    test_dataset = EarthquakeDataset(test_df, transform, mag_mapping, azi_mapping)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device, class_mappings)
    
    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Magnitude Accuracy: {results['mag_acc']*100:.2f}%")
    print(f"Azimuth Accuracy: {results['azi_acc']*100:.2f}%")
    print(f"Magnitude F1: {results['mag_f1']:.4f}")
    print(f"Azimuth F1: {results['azi_f1']:.4f}")
    
    # Generate outputs
    print("\nGenerating outputs...")
    
    plot_confusion_matrices(results, mag_classes, azi_classes,
                           OUTPUT_DIR / "confusion_matrices.png")
    
    generate_classification_report(results, mag_classes, azi_classes,
                                  OUTPUT_DIR / "EVALUATION_REPORT.md")
    
    # Save results JSON
    summary = {
        'model': 'ConvNeXt-Tiny',
        'test_samples': len(test_df),
        'magnitude_accuracy': float(results['mag_acc']),
        'azimuth_accuracy': float(results['azi_acc']),
        'magnitude_f1': float(results['mag_f1']),
        'azimuth_f1': float(results['azi_f1']),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / "evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
