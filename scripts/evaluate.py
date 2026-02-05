#!/usr/bin/env python3
"""
Evaluation script for ConvNeXt earthquake precursor model.

Usage:
    python scripts/evaluate.py --model models/best_model.pth --data data/test
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import json

from src.model import ConvNeXtPrecursorModel
from src.dataset import EarthquakeDataset, get_transforms
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Test data directory')
    parser.add_argument('--metadata', type=str, help='Metadata CSV')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--output', type=str, default='evaluation_results')
    return parser.parse_args()


def evaluate(model, dataloader, device, mag_classes, azi_classes):
    """Comprehensive evaluation."""
    model.eval()
    
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
    
    # Compute metrics
    results = {
        'magnitude': {
            'accuracy': accuracy_score(all_mag_labels, all_mag_preds),
            'f1_weighted': f1_score(all_mag_labels, all_mag_preds, average='weighted'),
            'f1_macro': f1_score(all_mag_labels, all_mag_preds, average='macro'),
            'precision': precision_score(all_mag_labels, all_mag_preds, average='weighted'),
            'recall': recall_score(all_mag_labels, all_mag_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_mag_labels, all_mag_preds).tolist(),
            'classification_report': classification_report(
                all_mag_labels, all_mag_preds, 
                target_names=mag_classes, output_dict=True
            )
        },
        'azimuth': {
            'accuracy': accuracy_score(all_azi_labels, all_azi_preds),
            'f1_weighted': f1_score(all_azi_labels, all_azi_preds, average='weighted'),
            'f1_macro': f1_score(all_azi_labels, all_azi_preds, average='macro'),
            'precision': precision_score(all_azi_labels, all_azi_preds, average='weighted'),
            'recall': recall_score(all_azi_labels, all_azi_preds, average='weighted'),
            'confusion_matrix': confusion_matrix(all_azi_labels, all_azi_preds).tolist(),
            'classification_report': classification_report(
                all_azi_labels, all_azi_preds,
                target_names=azi_classes, output_dict=True
            )
        }
    }
    
    return results, all_mag_labels, all_mag_preds, all_azi_labels, all_azi_preds


def plot_confusion_matrices(mag_labels, mag_preds, azi_labels, azi_preds,
                           mag_classes, azi_classes, output_dir):
    """Plot and save confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Magnitude
    mag_cm = confusion_matrix(mag_labels, mag_preds)
    sns.heatmap(mag_cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=mag_classes, yticklabels=mag_classes, ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title('Magnitude Classification')
    
    # Azimuth
    azi_cm = confusion_matrix(azi_labels, azi_preds)
    sns.heatmap(azi_cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=azi_classes, yticklabels=azi_classes, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Azimuth Classification')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f'Loading model from {args.model}...')
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint.get('config', {})
    
    from src.model import ConvNeXtMultiTask
    model = ConvNeXtMultiTask(
        variant=config.get('model_variant', config.get('variant', 'tiny')),
        pretrained=False,
        num_mag_classes=config.get('num_mag_classes', 4),
        num_azi_classes=config.get('num_azi_classes', 9)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Get class names
    mag_classes = ['Large', 'Medium', 'Moderate', 'Normal']
    azi_classes = ['E', 'N', 'NE', 'NW', 'Normal', 'S', 'SE', 'SW', 'W']
    
    # Load test data
    if args.metadata:
        metadata = pd.read_csv(args.metadata)
    else:
        # Try to find metadata
        metadata_path = Path(args.data).parent / 'metadata' / 'test_split.csv'
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
        else:
            raise ValueError('Please provide --metadata path')
    
    mag_mapping = {c: i for i, c in enumerate(mag_classes)}
    azi_mapping = {c: i for i, c in enumerate(azi_classes)}
    
    transform = get_transforms(is_training=False)
    dataset = EarthquakeDataset(metadata, args.data, transform, mag_mapping, azi_mapping)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f'Evaluating on {len(dataset)} samples...')
    
    # Evaluate
    results, mag_labels, mag_preds, azi_labels, azi_preds = evaluate(
        model, dataloader, device, mag_classes, azi_classes
    )
    
    # Print results
    print('\n' + '=' * 60)
    print('EVALUATION RESULTS')
    print('=' * 60)
    
    print('\nMagnitude Classification:')
    print(f"  Accuracy:  {results['magnitude']['accuracy']*100:.2f}%")
    print(f"  F1 Score:  {results['magnitude']['f1_weighted']:.4f}")
    print(f"  Precision: {results['magnitude']['precision']:.4f}")
    print(f"  Recall:    {results['magnitude']['recall']:.4f}")
    
    print('\nAzimuth Classification:')
    print(f"  Accuracy:  {results['azimuth']['accuracy']*100:.2f}%")
    print(f"  F1 Score:  {results['azimuth']['f1_weighted']:.4f}")
    print(f"  Precision: {results['azimuth']['precision']:.4f}")
    print(f"  Recall:    {results['azimuth']['recall']:.4f}")
    
    # Detailed reports
    print('\nMagnitude Classification Report:')
    print(classification_report(mag_labels, mag_preds, target_names=mag_classes))
    
    print('\nAzimuth Classification Report:')
    print(classification_report(azi_labels, azi_preds, target_names=azi_classes))
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Plot confusion matrices
    plot_confusion_matrices(
        mag_labels, mag_preds, azi_labels, azi_preds,
        mag_classes, azi_classes, output_dir
    )
    
    print(f'\nResults saved to {output_dir}')


if __name__ == '__main__':
    main()
