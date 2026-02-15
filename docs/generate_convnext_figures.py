#!/usr/bin/env python3
"""
Generate Publication Figures for ConvNeXt Paper

This script generates all figures needed for the ConvNeXt publication:
1. Training curves (loss, accuracy)
2. Confusion matrices
3. Model comparison charts
4. Architecture diagram
5. Performance comparison

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

# Configuration
OUTPUT_DIR = Path("publication_convnext/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find latest ConvNeXt experiment
EXPERIMENT_DIRS = sorted(glob("experiments_convnext/convnext_tiny_*"))
if EXPERIMENT_DIRS:
    LATEST_EXP = Path(EXPERIMENT_DIRS[-1])
else:
    LATEST_EXP = None

print("=" * 60)
print("GENERATING CONVNEXT PUBLICATION FIGURES")
print("=" * 60)

if LATEST_EXP:
    print(f"Using experiment: {LATEST_EXP}")
else:
    print("WARNING: No ConvNeXt experiment found!")


def load_training_history():
    """Load training history from CSV"""
    if LATEST_EXP is None:
        return None
    
    history_path = LATEST_EXP / "training_history.csv"
    if history_path.exists():
        return pd.read_csv(history_path)
    return None


def load_training_summary():
    """Load training summary from JSON"""
    if LATEST_EXP is None:
        return None
    
    summary_path = LATEST_EXP / "training_summary.json"
    if summary_path.exists():
        with open(summary_path, 'r') as f:
            return json.load(f)
    return None


def plot_training_curves(history, output_path):
    """Generate training curves figure"""
    if history is None:
        print("No training history available")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    epochs = range(1, len(history) + 1)
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Training')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Validation')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Loss', fontsize=12)
    axes[0, 0].set_title('(a) Training and Validation Loss', fontsize=14)
    axes[0, 0].legend(fontsize=11)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Magnitude Accuracy
    axes[0, 1].plot(epochs, history['train_mag_acc'], 'b-', linewidth=2, label='Training')
    axes[0, 1].plot(epochs, history['val_mag_acc'], 'r-', linewidth=2, label='Validation')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Accuracy', fontsize=12)
    axes[0, 1].set_title('(b) Magnitude Classification Accuracy', fontsize=14)
    axes[0, 1].legend(fontsize=11)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Azimuth Accuracy
    axes[1, 0].plot(epochs, history['train_azi_acc'], 'b-', linewidth=2, label='Training')
    axes[1, 0].plot(epochs, history['val_azi_acc'], 'r-', linewidth=2, label='Validation')
    axes[1, 0].set_xlabel('Epoch', fontsize=12)
    axes[1, 0].set_ylabel('Accuracy', fontsize=12)
    axes[1, 0].set_title('(c) Azimuth Classification Accuracy', fontsize=14)
    axes[1, 0].legend(fontsize=11)
    axes[1, 0].grid(True, alpha=0.3)
    
    # F1 Scores
    axes[1, 1].plot(epochs, history['val_mag_f1'], 'g-', linewidth=2, label='Magnitude F1')
    axes[1, 1].plot(epochs, history['val_azi_f1'], 'm-', linewidth=2, label='Azimuth F1')
    axes[1, 1].set_xlabel('Epoch', fontsize=12)
    axes[1, 1].set_ylabel('F1 Score', fontsize=12)
    axes[1, 1].set_title('(d) Validation F1 Scores', fontsize=14)
    axes[1, 1].legend(fontsize=11)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_model_comparison(output_path):
    """Generate model comparison chart"""
    # Data from previous models + ConvNeXt (placeholder)
    models = ['VGG16', 'EfficientNet-B0', 'ConvNeXt-Tiny']
    
    # Load ConvNeXt results if available
    summary = load_training_summary()
    if summary and 'test_results' in summary:
        convnext_mag = summary['test_results']['magnitude_accuracy'] * 100
        convnext_azi = summary['test_results']['azimuth_accuracy'] * 100
    else:
        convnext_mag = 0
        convnext_azi = 0
    
    mag_acc = [98.68, 98.94, convnext_mag]
    azi_acc = [54.93, 83.92, convnext_azi]
    params = [138, 5.3, 28.6]  # in millions
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#3498db', '#2ecc71', '#e74c3c']
    
    # Magnitude Accuracy
    bars1 = axes[0].bar(models, mag_acc, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('(a) Magnitude Classification', fontsize=14)
    axes[0].set_ylim([0, 105])
    for bar, val in zip(bars1, mag_acc):
        if val > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Azimuth Accuracy
    bars2 = axes[1].bar(models, azi_acc, color=colors, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('(b) Azimuth Classification', fontsize=14)
    axes[1].set_ylim([0, 105])
    for bar, val in zip(bars2, azi_acc):
        if val > 0:
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f'{val:.1f}%', ha='center', fontsize=11, fontweight='bold')
    
    # Parameters
    bars3 = axes[2].bar(models, params, color=colors, edgecolor='black', linewidth=1.5)
    axes[2].set_ylabel('Parameters (Millions)', fontsize=12)
    axes[2].set_title('(c) Model Size', fontsize=14)
    for bar, val in zip(bars3, params):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val}M', ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_architecture_diagram(output_path):
    """Generate ConvNeXt architecture diagram"""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Colors
    stem_color = '#3498db'
    stage_colors = ['#2ecc71', '#f39c12', '#e74c3c', '#9b59b6']
    head_color = '#1abc9c'
    
    # Draw stem
    rect = plt.Rectangle((0.5, 3), 1.5, 2, facecolor=stem_color, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(1.25, 4, 'Stem\n4×4 conv\n96 ch', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Draw stages
    stage_info = [
        ('Stage 1', '3 blocks\n96 ch', 2.5),
        ('Stage 2', '3 blocks\n192 ch', 5),
        ('Stage 3', '9 blocks\n384 ch', 7.5),
        ('Stage 4', '3 blocks\n768 ch', 10),
    ]
    
    for i, (name, info, x) in enumerate(stage_info):
        rect = plt.Rectangle((x, 2.5), 2, 3, facecolor=stage_colors[i], edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x+1, 4, f'{name}\n{info}', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
        
        # Arrow
        if i < 3:
            ax.annotate('', xy=(x+2.3, 4), xytext=(x+2, 4),
                       arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Draw heads
    # Magnitude head
    rect = plt.Rectangle((12.5, 4.5), 2.5, 2), 
    ax.add_patch(plt.Rectangle((12.5, 4.5), 2.5, 2, facecolor=head_color, edgecolor='black', linewidth=2))
    ax.text(13.75, 5.5, 'Magnitude\nHead\n4 classes', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Azimuth head
    ax.add_patch(plt.Rectangle((12.5, 1.5), 2.5, 2, facecolor=head_color, edgecolor='black', linewidth=2))
    ax.text(13.75, 2.5, 'Azimuth\nHead\n9 classes', ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # Arrows to heads
    ax.annotate('', xy=(12.3, 5.5), xytext=(12, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.annotate('', xy=(12.3, 2.5), xytext=(12, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Input
    ax.text(0.2, 4, 'Input\n224×224×3', ha='center', va='center', fontsize=10, fontweight='bold')
    ax.annotate('', xy=(0.4, 4), xytext=(0.1, 4),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    # Title
    ax.text(8, 7.5, 'ConvNeXt-Tiny Architecture for Earthquake Precursor Detection',
           ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Legend
    ax.text(8, 0.5, 'Total Parameters: 28.6M | Pretrained: ImageNet-1K',
           ha='center', va='center', fontsize=11, style='italic')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved: {output_path}")


def copy_existing_figures():
    """Copy confusion matrices from experiment folder"""
    if LATEST_EXP is None:
        return
    
    # Copy confusion matrices
    cm_path = LATEST_EXP / "confusion_matrices.png"
    if cm_path.exists():
        import shutil
        shutil.copy(cm_path, OUTPUT_DIR / "convnext_confusion_matrices.png")
        print(f"Copied: confusion_matrices.png")
    
    # Copy training curves
    tc_path = LATEST_EXP / "training_curves.png"
    if tc_path.exists():
        import shutil
        shutil.copy(tc_path, OUTPUT_DIR / "convnext_training_curves_original.png")
        print(f"Copied: training_curves.png")


def main():
    """Main function"""
    print("\nGenerating figures...")
    
    # Load data
    history = load_training_history()
    summary = load_training_summary()
    
    if summary:
        print(f"\nTraining Summary:")
        print(f"  Model: {summary.get('model', 'N/A')}")
        print(f"  Best Epoch: {summary.get('best_epoch', 'N/A')}")
        if 'test_results' in summary:
            print(f"  Magnitude Accuracy: {summary['test_results']['magnitude_accuracy']*100:.2f}%")
            print(f"  Azimuth Accuracy: {summary['test_results']['azimuth_accuracy']*100:.2f}%")
    
    # Generate figures
    print("\n1. Training curves...")
    plot_training_curves(history, OUTPUT_DIR / "fig1_training_curves.png")
    
    print("\n2. Model comparison...")
    plot_model_comparison(OUTPUT_DIR / "fig2_model_comparison.png")
    
    print("\n3. Architecture diagram...")
    plot_architecture_diagram(OUTPUT_DIR / "fig3_architecture.png")
    
    print("\n4. Copying existing figures...")
    copy_existing_figures()
    
    print("\n" + "=" * 60)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 60)
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
