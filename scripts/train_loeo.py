#!/usr/bin/env python3
"""
Leave-One-Event-Out (LOEO) Cross-Validation Training Script.

This script performs rigorous cross-validation by ensuring all samples
from the same earthquake event are in the same fold.

Usage:
    python scripts/train_loeo.py --config configs/loeo_validation.yaml
    python scripts/train_loeo.py --n-folds 10 --epochs 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

from src.model import ConvNeXtMultiTask
from src.dataset import EarthquakeDataset, get_transforms
from src.losses import MultiTaskLoss
from src.utils import setup_logging, set_seed, get_device, EarlyStopping


def parse_args():
    parser = argparse.ArgumentParser(description='LOEO Cross-Validation')
    
    parser.add_argument('--config', type=str, help='Config file')
    parser.add_argument('--variant', type=str, default='tiny')
    parser.add_argument('--data-dir', type=str, default='data/spectrograms')
    parser.add_argument('--metadata', type=str, default='data/metadata/metadata.csv')
    parser.add_argument('--n-folds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=3)
    parser.add_argument('--output-dir', type=str, default='loeo_results')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def train_fold(model, train_loader, val_loader, criterion, device, 
               epochs, patience, lr):
    """Train model for one fold."""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    early_stopping = EarlyStopping(patience=patience, mode='max')
    
    best_mag_acc = 0
    best_state = None
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for images, mag_labels, azi_labels in train_loader:
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            azi_labels = azi_labels.to(device)
            
            optimizer.zero_grad()
            mag_out, azi_out = model(images)
            loss, _, _ = criterion(mag_out, azi_out, mag_labels, azi_labels)
            loss.backward()
            optimizer.step()
        
        # Validate
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
        
        mag_acc = mag_correct / total
        azi_acc = azi_correct / total
        
        if mag_acc > best_mag_acc:
            best_mag_acc = mag_acc
            best_state = model.state_dict().copy()
        
        if early_stopping(mag_acc):
            break
    
    return best_state, best_mag_acc


def evaluate_fold(model, test_loader, device):
    """Evaluate model on test fold."""
    model.eval()
    
    mag_correct = 0
    azi_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, mag_labels, azi_labels in test_loader:
            images = images.to(device)
            mag_labels = mag_labels.to(device)
            azi_labels = azi_labels.to(device)
            
            mag_out, azi_out = model(images)
            mag_pred = torch.argmax(mag_out, dim=1)
            azi_pred = torch.argmax(azi_out, dim=1)
            
            mag_correct += (mag_pred == mag_labels).sum().item()
            azi_correct += (azi_pred == azi_labels).sum().item()
            total += images.size(0)
    
    return {
        'magnitude_accuracy': mag_correct / total,
        'azimuth_accuracy': azi_correct / total,
        'n_samples': total
    }


def main():
    args = parse_args()
    
    set_seed(args.seed)
    device = get_device()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'loeo_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logging(log_dir=output_dir)
    logger.info(f'LOEO Cross-Validation with {args.n_folds} folds')
    
    # Load metadata
    metadata = pd.read_csv(args.metadata)
    
    # Get unique events
    if 'event_id' in metadata.columns:
        event_col = 'event_id'
    elif 'earthquake_id' in metadata.columns:
        event_col = 'earthquake_id'
    else:
        # Create event ID from date
        metadata['event_id'] = metadata['date'].astype(str)
        event_col = 'event_id'
    
    events = metadata[event_col].unique()
    n_events = len(events)
    logger.info(f'Total events: {n_events}')
    
    # Shuffle events
    np.random.shuffle(events)
    
    # Class mappings
    mag_classes = sorted(metadata['magnitude_class'].unique())
    azi_classes = sorted(metadata['azimuth_class'].unique())
    mag_mapping = {c: i for i, c in enumerate(mag_classes)}
    azi_mapping = {c: i for i, c in enumerate(azi_classes)}
    
    # Transforms
    train_transform = get_transforms(is_training=True)
    val_transform = get_transforms(is_training=False)
    
    # Create full dataset
    full_dataset = EarthquakeDataset(
        metadata, args.data_dir, val_transform, mag_mapping, azi_mapping
    )
    
    # LOEO cross-validation
    fold_results = []
    events_per_fold = n_events // args.n_folds
    
    print('\n' + '=' * 70)
    print('LEAVE-ONE-EVENT-OUT CROSS-VALIDATION')
    print('=' * 70)
    
    for fold in range(args.n_folds):
        print(f'\n--- Fold {fold + 1}/{args.n_folds} ---')
        
        # Get test events for this fold
        start_idx = fold * events_per_fold
        if fold == args.n_folds - 1:
            end_idx = n_events
        else:
            end_idx = start_idx + events_per_fold
        
        test_events = set(events[start_idx:end_idx])
        
        # Split indices
        test_mask = metadata[event_col].isin(test_events)
        test_indices = metadata[test_mask].index.tolist()
        train_indices = metadata[~test_mask].index.tolist()
        
        # Further split train into train/val
        np.random.shuffle(train_indices)
        val_size = int(len(train_indices) * 0.1)
        val_indices = train_indices[:val_size]
        train_indices = train_indices[val_size:]
        
        logger.info(f'Fold {fold + 1}: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}')
        
        # Create dataloaders
        train_dataset = EarthquakeDataset(
            metadata.iloc[train_indices], args.data_dir, 
            train_transform, mag_mapping, azi_mapping
        )
        val_dataset = EarthquakeDataset(
            metadata.iloc[val_indices], args.data_dir,
            val_transform, mag_mapping, azi_mapping
        )
        test_dataset = EarthquakeDataset(
            metadata.iloc[test_indices], args.data_dir,
            val_transform, mag_mapping, azi_mapping
        )
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        
        # Create fresh model
        model = ConvNeXtMultiTask(
            variant=args.variant,
            pretrained=True,
            num_mag_classes=len(mag_classes),
            num_azi_classes=len(azi_classes)
        ).to(device)
        
        criterion = MultiTaskLoss()
        
        # Train
        best_state, _ = train_fold(
            model, train_loader, val_loader, criterion, device,
            args.epochs, args.patience, args.lr
        )
        
        # Load best model and evaluate
        model.load_state_dict(best_state)
        results = evaluate_fold(model, test_loader, device)
        results['fold'] = fold + 1
        results['n_train_events'] = n_events - len(test_events)
        results['n_test_events'] = len(test_events)
        
        fold_results.append(results)
        
        print(f'Fold {fold + 1} Results:')
        print(f'  Magnitude Accuracy: {results["magnitude_accuracy"]:.4f} ({results["magnitude_accuracy"]*100:.2f}%)')
        print(f'  Azimuth Accuracy:   {results["azimuth_accuracy"]:.4f} ({results["azimuth_accuracy"]*100:.2f}%)')
        
        # Save fold result
        with open(output_dir / f'fold_{fold + 1}.json', 'w') as f:
            json.dump(results, f, indent=2)
    
    # Aggregate results
    mag_accs = [r['magnitude_accuracy'] for r in fold_results]
    azi_accs = [r['azimuth_accuracy'] for r in fold_results]
    
    final_results = {
        'n_folds': args.n_folds,
        'magnitude_accuracy': {
            'mean': float(np.mean(mag_accs)),
            'std': float(np.std(mag_accs)),
            'min': float(np.min(mag_accs)),
            'max': float(np.max(mag_accs))
        },
        'azimuth_accuracy': {
            'mean': float(np.mean(azi_accs)),
            'std': float(np.std(azi_accs)),
            'min': float(np.min(azi_accs)),
            'max': float(np.max(azi_accs))
        },
        'per_fold_results': fold_results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(output_dir / 'loeo_final_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)
    
    # Print summary
    print('\n' + '=' * 70)
    print('LOEO CROSS-VALIDATION RESULTS')
    print('=' * 70)
    print(f'\nMagnitude Classification:')
    print(f'  Mean Accuracy: {np.mean(mag_accs)*100:.2f}% ± {np.std(mag_accs)*100:.2f}%')
    print(f'  Range: {np.min(mag_accs)*100:.2f}% - {np.max(mag_accs)*100:.2f}%')
    print(f'\nAzimuth Classification:')
    print(f'  Mean Accuracy: {np.mean(azi_accs)*100:.2f}% ± {np.std(azi_accs)*100:.2f}%')
    print(f'  Range: {np.min(azi_accs)*100:.2f}% - {np.max(azi_accs)*100:.2f}%')
    print(f'\nResults saved to: {output_dir}')


if __name__ == '__main__':
    main()
