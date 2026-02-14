#!/usr/bin/env python3
"""
Evaluate Enhanced ConvNeXt Model
Compare with baseline and generate comprehensive report
"""

import os
import sys
import json
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from PIL import Image
import torchvision.transforms as transforms

from convnext_enhanced import EnhancedConvNeXt


class EvaluationDataset(torch.utils.data.Dataset):
    """Simple dataset for evaluation"""
    def __init__(self, metadata_file, transform=None):
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        
        self.mag_map = {'Small': 0, 'Medium': 1, 'Large': 2, 'VeryLarge': 3}
        self.azi_map = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7, 'Center': 8}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        img_path = row['file_path']
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        mag_label = self.mag_map[row['magnitude_class']]
        azi_label = self.azi_map[row['azimuth_class']]
        
        return img, mag_label, azi_label


def load_model(model_path, device):
    """Load trained model"""
    print(f"Loading model from: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Get config
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        # Default config
        config = {
            'model': {
                'num_magnitude_classes': 4,
                'num_azimuth_classes': 9,
                'dropout_rate': 0.4,
                'use_attention': True,
                'use_hierarchical_azimuth': True,
                'use_directional_features': True
            }
        }
    
    # Create model
    model = EnhancedConvNeXt(
        num_mag_classes=config['model']['num_magnitude_classes'],
        num_azi_classes=config['model']['num_azimuth_classes'],
        dropout=config['model']['dropout_rate'],
        use_attention=config['model'].get('use_attention', True),
        use_hierarchical_azimuth=config['model'].get('use_hierarchical_azimuth', True),
        use_directional_features=config['model'].get('use_directional_features', True)
    )
    
    # Load weights
    if 'ema_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['ema_state_dict'])
        print("✅ Loaded EMA model weights")
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("✅ Loaded model weights")
    else:
        print("⚠️  No weights found in checkpoint")
    
    model = model.to(device)
    model.eval()
    
    return model


def evaluate_model(model, dataloader, device):
    """Evaluate model and return predictions"""
    all_mag_preds = []
    all_mag_targets = []
    all_azi_preds = []
    all_azi_targets = []
    
    with torch.no_grad():
        for inputs, mag_targets, azi_targets in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device)
            
            mag_out, azi_out = model(inputs)
            
            mag_preds = torch.argmax(mag_out, dim=1).cpu().numpy()
            azi_preds = torch.argmax(azi_out, dim=1).cpu().numpy()
            
            all_mag_preds.extend(mag_preds)
            all_mag_targets.extend(mag_targets.numpy())
            all_azi_preds.extend(azi_preds)
            all_azi_targets.extend(azi_targets.numpy())
    
    return np.array(all_mag_preds), np.array(all_mag_targets), np.array(all_azi_preds), np.array(all_azi_targets)


def plot_confusion_matrices(mag_preds, mag_targets, azi_preds, azi_targets, output_dir):
    """Plot confusion matrices"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Magnitude confusion matrix
    mag_cm = confusion_matrix(mag_targets, mag_preds)
    sns.heatmap(mag_cm, annot=True, fmt='d', ax=axes[0], cmap='Blues',
                xticklabels=['Small', 'Medium', 'Large', 'VeryLarge'],
                yticklabels=['Small', 'Medium', 'Large', 'VeryLarge'])
    axes[0].set_title('Magnitude Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    # Azimuth confusion matrix
    azi_cm = confusion_matrix(azi_targets, azi_preds)
    sns.heatmap(azi_cm, annot=True, fmt='d', ax=axes[1], cmap='Greens',
                xticklabels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'C'],
                yticklabels=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'C'])
    axes[1].set_title('Azimuth Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Saved confusion matrices to: {output_dir / 'confusion_matrices.png'}")


def generate_report(mag_preds, mag_targets, azi_preds, azi_targets, output_dir, model_name="Enhanced"):
    """Generate comprehensive evaluation report"""
    
    # Calculate metrics
    mag_acc = np.mean(mag_preds == mag_targets)
    azi_acc = np.mean(azi_preds == azi_targets)
    
    mag_f1 = f1_score(mag_targets, mag_preds, average='weighted')
    azi_f1 = f1_score(azi_targets, azi_preds, average='weighted')
    
    # Print summary
    print("\n" + "="*80)
    print(f"{model_name} MODEL EVALUATION RESULTS")
    print("="*80)
    print(f"Magnitude Accuracy: {mag_acc*100:.2f}%")
    print(f"Magnitude F1-Score: {mag_f1*100:.2f}%")
    print(f"Azimuth Accuracy: {azi_acc*100:.2f}%")
    print(f"Azimuth F1-Score: {azi_f1*100:.2f}%")
    print(f"Combined Accuracy: {(mag_acc + azi_acc)/2*100:.2f}%")
    print("="*80)
    
    # Detailed classification reports
    print("\nMagnitude Classification Report:")
    print(classification_report(mag_targets, mag_preds,
                                target_names=['Small', 'Medium', 'Large', 'VeryLarge']))
    
    print("\nAzimuth Classification Report:")
    print(classification_report(azi_targets, azi_preds,
                                target_names=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Center']))
    
    # Save results
    results = {
        'model_name': model_name,
        'magnitude_accuracy': float(mag_acc),
        'magnitude_f1_score': float(mag_f1),
        'azimuth_accuracy': float(azi_acc),
        'azimuth_f1_score': float(azi_f1),
        'combined_accuracy': float((mag_acc + azi_acc) / 2)
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Saved results to: {output_dir / 'evaluation_results.json'}")
    
    return results


def compare_with_baseline(enhanced_results, baseline_results_path):
    """Compare enhanced model with baseline"""
    
    if not Path(baseline_results_path).exists():
        print(f"\n⚠️  Baseline results not found: {baseline_results_path}")
        return
    
    with open(baseline_results_path) as f:
        baseline = json.load(f)
    
    print("\n" + "="*80)
    print("COMPARISON WITH BASELINE")
    print("="*80)
    
    # Magnitude comparison
    baseline_mag = baseline.get('best_val_magnitude_acc', 0.6372)
    enhanced_mag = enhanced_results['magnitude_accuracy']
    mag_improvement = (enhanced_mag - baseline_mag) * 100
    
    print(f"\nMagnitude Accuracy:")
    print(f"  Baseline:  {baseline_mag*100:.2f}%")
    print(f"  Enhanced:  {enhanced_mag*100:.2f}%")
    print(f"  Improvement: {mag_improvement:+.2f}%")
    
    # Azimuth comparison
    baseline_azi = baseline.get('test_azimuth_acc', 0.1145)
    enhanced_azi = enhanced_results['azimuth_accuracy']
    azi_improvement = (enhanced_azi - baseline_azi) * 100
    
    print(f"\nAzimuth Accuracy:")
    print(f"  Baseline:  {baseline_azi*100:.2f}%")
    print(f"  Enhanced:  {enhanced_azi*100:.2f}%")
    print(f"  Improvement: {azi_improvement:+.2f}%")
    
    # Overall comparison
    baseline_combined = (baseline_mag + baseline_azi) / 2
    enhanced_combined = enhanced_results['combined_accuracy']
    combined_improvement = (enhanced_combined - baseline_combined) * 100
    
    print(f"\nCombined Accuracy:")
    print(f"  Baseline:  {baseline_combined*100:.2f}%")
    print(f"  Enhanced:  {enhanced_combined*100:.2f}%")
    print(f"  Improvement: {combined_improvement:+.2f}%")
    
    print("="*80)
    
    # Success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    criteria = {
        'Magnitude ≥ 70%': enhanced_mag >= 0.70,
        'Magnitude ≥ 75% (target)': enhanced_mag >= 0.75,
        'Magnitude ≥ 80% (stretch)': enhanced_mag >= 0.80,
        'Azimuth ≥ 50%': enhanced_azi >= 0.50,
        'Azimuth ≥ 65% (target)': enhanced_azi >= 0.65,
        'Azimuth ≥ 70% (stretch)': enhanced_azi >= 0.70,
        'Combined ≥ 60%': enhanced_combined >= 0.60,
        'Combined ≥ 70% (target)': enhanced_combined >= 0.70,
        'Combined ≥ 75% (stretch)': enhanced_combined >= 0.75,
    }
    
    for criterion, met in criteria.items():
        status = "✅" if met else "❌"
        print(f"{status} {criterion}")
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Enhanced ConvNeXt Model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--dataset', type=str, default='dataset_experiment_3',
                       help='Path to dataset')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Which split to evaluate')
    parser.add_argument('--output', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--baseline', type=str,
                       default='experiments_convnext/finetune_v3_gpu_20260214_143726/final_results.json',
                       help='Path to baseline results for comparison')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Output directory
    if args.output is None:
        model_dir = Path(args.model).parent
        output_dir = model_dir / 'evaluation'
    else:
        output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = load_model(args.model, device)
    
    # Create dataloader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    metadata_file = Path(args.dataset) / 'final_metadata' / f'{args.split}_exp3.csv'
    dataset = EvaluationDataset(metadata_file, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"\nEvaluating on {args.split} set ({len(dataset)} samples)...")
    
    # Evaluate
    mag_preds, mag_targets, azi_preds, azi_targets = evaluate_model(model, dataloader, device)
    
    # Generate report
    results = generate_report(mag_preds, mag_targets, azi_preds, azi_targets, output_dir)
    
    # Plot confusion matrices
    plot_confusion_matrices(mag_preds, mag_targets, azi_preds, azi_targets, output_dir)
    
    # Compare with baseline
    compare_with_baseline(results, args.baseline)
    
    print(f"\n✅ Evaluation complete! Results saved to: {output_dir}")


if __name__ == '__main__':
    main()
