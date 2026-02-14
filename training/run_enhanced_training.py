#!/usr/bin/env python3
"""
Run Enhanced ConvNeXt Training
Simple script to start training with all improvements
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def check_requirements():
    """Check if all requirements are met"""
    print("Checking requirements...")
    
    # Check CUDA
    import torch
    if not torch.cuda.is_available():
        print("⚠️  WARNING: CUDA not available! Training will be very slow on CPU.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    else:
        print(f"✅ CUDA available")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Check dataset
    dataset_path = Path("dataset_experiment_3")
    if not dataset_path.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        sys.exit(1)
    print(f"✅ Dataset found: {dataset_path}")
    
    # Check checkpoint
    checkpoint_path = Path("experiments_convnext/finetune_v3_gpu_20260214_143726/checkpoint_latest.pth")
    if not checkpoint_path.exists():
        print(f"⚠️  Checkpoint not found: {checkpoint_path}")
        print("   Will train from scratch (slower convergence)")
    else:
        print(f"✅ Checkpoint found: {checkpoint_path}")
    
    # Check config
    config_path = Path("config_improved.json")
    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        sys.exit(1)
    print(f"✅ Config found: {config_path}")
    
    print("\n✅ All requirements met!\n")


def show_config_summary():
    """Show configuration summary"""
    with open("config_improved.json") as f:
        config = json.load(f)
    
    print("="*80)
    print("TRAINING CONFIGURATION SUMMARY")
    print("="*80)
    print(f"Model: Enhanced ConvNeXt with Attention")
    print(f"Batch Size: {config['training']['batch_size']}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Learning Rate: {config['optimizer']['base_lr']}")
    print(f"Warmup Epochs: {config['scheduler']['warmup_epochs']}")
    print(f"Task Weights: Magnitude={config['loss']['task_weights']['magnitude']}, Azimuth={config['loss']['task_weights']['azimuth']}")
    print(f"Mixed Precision: {config['training']['mixed_precision']}")
    print(f"Attention: {config['model']['use_attention']}")
    print(f"Hierarchical Azimuth: {config['model']['use_hierarchical_azimuth']}")
    print(f"Directional Features: {config['model']['use_directional_features']}")
    print("="*80)
    print()


def estimate_training_time():
    """Estimate training time"""
    import torch
    
    with open("config_improved.json") as f:
        config = json.load(f)
    
    epochs = config['training']['epochs']
    batch_size = config['training']['batch_size']
    
    # Rough estimates based on GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        if '3090' in gpu_name or '4090' in gpu_name:
            time_per_epoch = 8  # minutes
        elif '3080' in gpu_name or '4080' in gpu_name:
            time_per_epoch = 10
        elif '3070' in gpu_name:
            time_per_epoch = 12
        else:
            time_per_epoch = 15
    else:
        time_per_epoch = 120  # CPU is very slow
    
    total_time = epochs * time_per_epoch
    hours = total_time // 60
    minutes = total_time % 60
    
    print(f"Estimated Training Time: {hours}h {minutes}m")
    print(f"(~{time_per_epoch} min/epoch × {epochs} epochs)")
    print()


def main():
    print("\n" + "="*80)
    print("ENHANCED CONVNEXT TRAINING")
    print("="*80)
    print()
    
    # Check requirements
    check_requirements()
    
    # Show config
    show_config_summary()
    
    # Estimate time
    estimate_training_time()
    
    # Confirm
    print("Ready to start training!")
    print()
    response = input("Start training now? (y/n): ")
    if response.lower() != 'y':
        print("Training cancelled.")
        sys.exit(0)
    
    print("\n" + "="*80)
    print("STARTING TRAINING...")
    print("="*80)
    print()
    
    # Run training
    try:
        subprocess.run([
            sys.executable,
            "train_convnext_enhanced.py",
            "--config", "config_improved.json"
        ], check=True)
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\n\n❌ Training failed with error code {e.returncode}")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED!")
    print("="*80)
    print()
    print("Check the results in: experiments_convnext/enhanced_*/")
    print()


if __name__ == '__main__':
    main()
