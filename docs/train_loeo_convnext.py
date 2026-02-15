#!/usr/bin/env python3
"""
LOEO (Leave-One-Event-Out) Cross-Validation for ConvNeXt Model

This script performs rigorous cross-validation to validate model generalization
to unseen earthquake events.

Author: Earthquake Prediction Research Team
Date: 5 February 2026
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from glob import glob
import logging
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from PIL import Image
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GroupKFold

# Configuration
CONFIG = {
    "n_folds": 10,
    "epochs_per_fold": 30,
    "batch_size": 32,
    "learning_rate": 1e-4,
    "weight_decay": 0.05,
    "early_stopping_patience": 5,
}

OUTPUT_DIR = Path("loeo_convnext_results")
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_DIR = Path("dataset_unified/spectrograms")
METADATA_PATH = Path("dataset_unified/metadata/unified_metadata.csv")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

print("=" * 60)
print("LOEO CROSS-VALIDATION - ConvNeXt Model")
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
        
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = convnext_tiny(weights=weights)
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


def get_transforms(is_training=True):
    """Get data transforms"""
    if is_training:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


def train_fold(model, train_loader, val_loader, device, fold_num):
    """Train model for one fold"""
    
    criterion_mag = nn.CrossEntropyLoss()
    criterion_azi = nn.CrossEntropyLoss()
    
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG["learning_rate"], 
                           weight_decay=CONFIG["weight_decay"])
    
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(CONFIG["epochs_per_fold"]):
        # Training
        model.train()
        train_loss = 0
        
        for images, mag_labels, azi_labels in train_loader:
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            azi_labels = azi_labels.to(device)
            
            optimizer.zero_grad()
            mag_out, azi_out = model(images)
            
            loss = criterion_mag(mag_out, mag_labels) + 0.5 * criterion_azi(azi_out, azi_labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        mag_correct = 0
        azi_correct = 0
        total = 0
        
        with torch.no_grad():
            for images, mag_labels, azi_labels in val_loader:
                images = images.to(device)
                mag_labels = mag_labels.to(device)
                azi_labels = azi_labels.to(device)
                
                mag_out, azi_out = model(images)
                
                mag_pred = torch.argmax(mag_out, dim=1)
                azi_pred = torch.argmax(azi_out, dim=1)
                
                mag_correct += (mag_pred == mag_labels).sum().item()
                azi_correct += (azi_pred == azi_labels).sum().item()
                total += images.size(0)
        
        val_mag_acc = mag_correct / total
        val_azi_acc = azi_correct / total
        
        if val_mag_acc > best_val_acc:
            best_val_acc = val_mag_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG["early_stopping_patience"]:
            logger.info(f"Fold {fold_num}: Early stopping at epoch {epoch+1}")
            break
    
    return best_val_acc, val_azi_acc


def evaluate_fold(model, test_loader, device):
    """Evaluate model on test fold"""
    model.eval()
    
    all_mag_preds = []
    all_mag_labels = []
    all_azi_preds = []
    all_azi_labels = []
    
    with torch.no_grad():
        for images, mag_labels, azi_labels in test_loader:
            images = images.to(device)
            
            mag_out, azi_out = model(images)
            
            mag_pred = torch.argmax(mag_out, dim=1)
            azi_pred = torch.argmax(azi_out, dim=1)
            
            all_mag_preds.extend(mag_pred.cpu().numpy())
            all_mag_labels.extend(mag_labels.numpy())
            all_azi_preds.extend(azi_pred.cpu().numpy())
            all_azi_labels.extend(azi_labels.numpy())
    
    mag_acc = np.mean(np.array(all_mag_preds) == np.array(all_mag_labels))
    azi_acc = np.mean(np.array(all_azi_preds) == np.array(all_azi_labels))
    
    return mag_acc, azi_acc


def main():
    """Main LOEO cross-validation"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load metadata
    metadata_df = pd.read_csv(METADATA_PATH)
    logger.info(f"Loaded {len(metadata_df)} samples")
    
    # Create event groups
    metadata_df['event_group'] = metadata_df['station'] + '_' + metadata_df['date'].astype(str)
    groups = metadata_df['event_group'].values
    
    # Class mappings
    mag_classes = sorted(metadata_df['magnitude_class'].unique())
    azi_classes = sorted(metadata_df['azimuth_class'].unique())
    mag_mapping = {c: i for i, c in enumerate(mag_classes)}
    azi_mapping = {c: i for i, c in enumerate(azi_classes)}
    
    # Cross-validation
    gkf = GroupKFold(n_splits=CONFIG["n_folds"])
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(metadata_df, groups=groups)):
        logger.info(f"\n{'='*50}")
        logger.info(f"FOLD {fold+1}/{CONFIG['n_folds']}")
        logger.info(f"{'='*50}")
        
        train_df = metadata_df.iloc[train_idx]
        test_df = metadata_df.iloc[test_idx]
        
        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")
        
        # Create datasets
        train_dataset = EarthquakeDataset(train_df, get_transforms(True), mag_mapping, azi_mapping)
        test_dataset = EarthquakeDataset(test_df, get_transforms(False), mag_mapping, azi_mapping)
        
        train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2)
        
        # Create model
        model = ConvNeXtMultiTask(num_mag_classes=len(mag_classes), num_azi_classes=len(azi_classes))
        model = model.to(device)
        
        # Train
        train_fold(model, train_loader, test_loader, device, fold+1)
        
        # Evaluate
        mag_acc, azi_acc = evaluate_fold(model, test_loader, device)
        
        results.append({
            'fold': fold + 1,
            'mag_acc': mag_acc,
            'azi_acc': azi_acc,
            'test_samples': len(test_df)
        })
        
        logger.info(f"Fold {fold+1} Results: Mag={mag_acc*100:.2f}%, Azi={azi_acc*100:.2f}%")
    
    # Summary
    results_df = pd.DataFrame(results)
    
    print("\n" + "=" * 60)
    print("LOEO CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nMagnitude Accuracy: {results_df['mag_acc'].mean()*100:.2f}% ± {results_df['mag_acc'].std()*100:.2f}%")
    print(f"Azimuth Accuracy: {results_df['azi_acc'].mean()*100:.2f}% ± {results_df['azi_acc'].std()*100:.2f}%")
    
    # Save results
    results_df.to_csv(OUTPUT_DIR / "loeo_results.csv", index=False)
    
    summary = {
        'model': 'ConvNeXt-Tiny',
        'n_folds': CONFIG['n_folds'],
        'mag_acc_mean': float(results_df['mag_acc'].mean()),
        'mag_acc_std': float(results_df['mag_acc'].std()),
        'azi_acc_mean': float(results_df['azi_acc'].mean()),
        'azi_acc_std': float(results_df['azi_acc'].std()),
        'timestamp': datetime.now().isoformat()
    }
    
    with open(OUTPUT_DIR / "loeo_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
