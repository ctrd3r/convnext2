#!/usr/bin/env python3
"""
Training script for ConvNeXt earthquake precursor model.

Usage:
    python scripts/train.py --config configs/convnext_tiny.yaml
    python scripts/train.py --epochs 50 --batch-size 32 --lr 1e-4
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
import json

from src.model import ConvNeXtMultiTask, count_parameters
from src.dataset import EarthquakeDataset, get_transforms, create_dataloaders
from src.losses import MultiTaskLoss
from src.utils import (
    setup_logging, load_config, save_config, set_seed, 
    get_device, EarlyStopping, AverageMeter
)


def parse_args():
    parser = argparse.ArgumentParser(description='Train ConvNeXt model')
    
    # Config
    parser.add_argument('--config', type=str, help='Path to config file')
    
    # Model
    parser.add_argument('--variant', type=str, default='tiny',
                        choices=['tiny', 'small', 'base'])
    parser.add_argument('--pretrained', action='store_true', default=True)
    parser.add_argument('--dropout', type=float, default=0.5)
    
    # Data
    parser.add_argument('--data-dir', type=str, default='data/spectrograms')
    parser.add_argument('--metadata', type=str, default='data/metadata/metadata.csv')
    parser.add_argument('--image-size', type=int, default=224)
    
    # Training
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--warmup-epochs', type=int, default=5)
    parser.add_argument('--patience', type=int, default=10)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='experiments')
    parser.add_argument('--seed', type=int, default=42)
    
    return parser.parse_args()


def train_epoch(model, dataloader, optimizer, criterion, device, scaler=None):
    """Train for one epoch."""
    model.train()
    
    loss_meter = AverageMeter()
    mag_acc_meter = AverageMeter()
    azi_acc_meter = AverageMeter()
    
    pbar = tqdm(dataloader, desc='Training')
    
    for images, mag_labels, azi_labels in pbar:
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if scaler is not None:
            with torch.cuda.amp.autocast():
                mag_out, azi_out = model(images)
                loss, _, _ = criterion(mag_out, azi_out, mag_labels, azi_labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            mag_out, azi_out = model(images)
            loss, _, _ = criterion(mag_out, azi_out, mag_labels, azi_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Compute accuracy
        mag_pred = torch.argmax(mag_out, dim=1)
        azi_pred = torch.argmax(azi_out, dim=1)
        mag_acc = (mag_pred == mag_labels).float().mean()
        azi_acc = (azi_pred == azi_labels).float().mean()
        
        # Update meters
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        mag_acc_meter.update(mag_acc.item(), batch_size)
        azi_acc_meter.update(azi_acc.item(), batch_size)
        
        pbar.set_postfix({
            'loss': f'{loss_meter.avg:.4f}',
            'mag': f'{mag_acc_meter.avg:.4f}',
            'azi': f'{azi_acc_meter.avg:.4f}'
        })
    
    return {
        'loss': loss_meter.avg,
        'mag_acc': mag_acc_meter.avg,
        'azi_acc': azi_acc_meter.avg
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model."""
    model.eval()
    
    loss_meter = AverageMeter()
    mag_correct = 0
    azi_correct = 0
    total = 0
    
    for images, mag_labels, azi_labels in tqdm(dataloader, desc='Validating'):
        images = images.to(device)
        mag_labels = mag_labels.to(device)
        azi_labels = azi_labels.to(device)
        
        mag_out, azi_out = model(images)
        loss, _, _ = criterion(mag_out, azi_out, mag_labels, azi_labels)
        
        mag_pred = torch.argmax(mag_out, dim=1)
        azi_pred = torch.argmax(azi_out, dim=1)
        
        batch_size = images.size(0)
        loss_meter.update(loss.item(), batch_size)
        mag_correct += (mag_pred == mag_labels).sum().item()
        azi_correct += (azi_pred == azi_labels).sum().item()
        total += batch_size
    
    return {
        'loss': loss_meter.avg,
        'mag_acc': mag_correct / total,
        'azi_acc': azi_correct / total
    }


def main():
    args = parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        for key, value in config.items():
            if hasattr(args, key):
                setattr(args, key, value)
    
    # Setup
    set_seed(args.seed)
    device = get_device()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) / f'convnext_{args.variant}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(log_dir=output_dir)
    logger.info(f'Output directory: {output_dir}')
    
    # Save config
    save_config(vars(args), output_dir / 'config.json')
    
    # Load data
    logger.info('Loading data...')
    metadata = pd.read_csv(args.metadata)
    
    # Split data
    from sklearn.model_selection import train_test_split
    train_df, temp_df = train_test_split(
        metadata, test_size=0.3, 
        stratify=metadata['magnitude_class'],
        random_state=args.seed
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5,
        stratify=temp_df['magnitude_class'],
        random_state=args.seed
    )
    
    logger.info(f'Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_df, val_df, test_df,
        args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size
    )
    
    # Get class info
    mag_classes = sorted(metadata['magnitude_class'].unique())
    azi_classes = sorted(metadata['azimuth_class'].unique())
    
    # Create model
    logger.info(f'Creating ConvNeXt-{args.variant.upper()} model...')
    model = ConvNeXtMultiTask(
        variant=args.variant,
        pretrained=args.pretrained,
        num_mag_classes=len(mag_classes),
        num_azi_classes=len(azi_classes),
        dropout=args.dropout
    )
    model = model.to(device)
    
    params = count_parameters(model)
    logger.info(f'Parameters: {params["total"]:,} total, {params["trainable"]:,} trainable')
    
    # Loss function
    criterion = MultiTaskLoss(mag_weight=1.0, azi_weight=0.5)
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='max')
    
    # Training loop
    best_mag_acc = 0
    history = {'train': [], 'val': []}
    
    logger.info('Starting training...')
    
    for epoch in range(1, args.epochs + 1):
        print(f'\nEpoch {epoch}/{args.epochs}')
        print('-' * 50)
        
        # Train
        train_results = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        
        # Validate
        val_results = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log results
        logger.info(
            f'Epoch {epoch}: '
            f'Train Loss={train_results["loss"]:.4f}, '
            f'Val Mag={val_results["mag_acc"]:.4f}, '
            f'Val Azi={val_results["azi_acc"]:.4f}'
        )
        
        history['train'].append(train_results)
        history['val'].append(val_results)
        
        # Save best model
        if val_results['mag_acc'] > best_mag_acc:
            best_mag_acc = val_results['mag_acc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_mag_acc': val_results['mag_acc'],
                'val_azi_acc': val_results['azi_acc'],
                'config': vars(args)
            }, output_dir / 'best_model.pth')
            logger.info(f'✓ New best model saved! Mag Acc: {best_mag_acc:.4f}')
        
        # Early stopping
        if early_stopping(val_results['mag_acc']):
            logger.info(f'Early stopping at epoch {epoch}')
            break
    
    # Save final model
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'config': vars(args)
    }, output_dir / 'final_model.pth')
    
    # Save history
    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Final evaluation
    logger.info('\nEvaluating on test set...')
    model.load_state_dict(torch.load(output_dir / 'best_model.pth')['model_state_dict'])
    test_results = validate(model, test_loader, criterion, device)
    
    print('\n' + '=' * 50)
    print('FINAL TEST RESULTS')
    print('=' * 50)
    print(f'Magnitude Accuracy: {test_results["mag_acc"]:.4f} ({test_results["mag_acc"]*100:.2f}%)')
    print(f'Azimuth Accuracy:   {test_results["azi_acc"]:.4f} ({test_results["azi_acc"]*100:.2f}%)')
    
    logger.info(f'Training complete! Output: {output_dir}')


if __name__ == '__main__':
    main()
