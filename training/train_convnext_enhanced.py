#!/usr/bin/env python3
"""
Enhanced ConvNeXt Training Script
Includes all improvements: attention, hierarchical azimuth, advanced augmentation
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

# Import our enhanced modules
from convnext_enhanced import EnhancedConvNeXt, create_enhanced_model
from augmentations_advanced import AdvancedAugmentation, ProgressiveAugmentation, MixUp, CutMix
from earthquake_cnn_v3 import EMAModel, FocalLoss


class EnhancedDataset(torch.utils.data.Dataset):
    """Dataset with advanced augmentation support"""
    def __init__(self, metadata_file, transform=None, advanced_aug=None):
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        self.advanced_aug = advanced_aug
        
        # Map labels
        self.mag_map = {'Small': 0, 'Medium': 1, 'Large': 2, 'VeryLarge': 3}
        self.azi_map = {'N': 0, 'NE': 1, 'E': 2, 'SE': 3, 'S': 4, 'SW': 5, 'W': 6, 'NW': 7, 'Center': 8}
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        
        # Load image
        img_path = row['file_path']
        img = Image.open(img_path).convert('RGB')
        
        # Apply advanced augmentation first (before tensor conversion)
        if self.advanced_aug is not None:
            img = self.advanced_aug(img)
        
        # Apply standard transforms
        if self.transform is not None:
            img = self.transform(img)
        
        # Get labels
        mag_label = self.mag_map[row['magnitude_class']]
        azi_label = self.azi_map[row['azimuth_class']]
        
        return img, mag_label, azi_label


class EnhancedTrainer:
    """Enhanced trainer with all improvements"""
    def __init__(self, config_path=None, config_dict=None):
        # Load configuration
        if config_path:
            with open(config_path) as f:
                self.config = json.load(f)
        elif config_dict:
            self.config = config_dict
        else:
            raise ValueError("Must provide either config_path or config_dict")
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.exp_dir = Path(f"experiments_convnext/enhanced_{timestamp}")
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save configuration
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Setup logging
        self.log_file = self.exp_dir / 'training.log'
        self.history = {
            'train_loss': [], 'train_mag_acc': [], 'train_azi_acc': [],
            'val_loss': [], 'val_mag_acc': [], 'val_azi_acc': [],
            'learning_rates': []
        }
        
        self.log(f"Enhanced ConvNeXt Training")
        self.log(f"Experiment: {self.exp_dir}")
        self.log(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def log(self, message):
        """Log message to file and console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')
    
    def create_dataloaders(self):
        """Create train/val dataloaders with advanced augmentation"""
        self.log("Creating dataloaders...")
        
        dataset_path = Path(self.config['data']['dataset_path'])
        img_size = self.config['data']['image_size']
        
        # Advanced augmentation for training
        aug_config = self.config['augmentation']
        advanced_aug = AdvancedAugmentation(aug_config)
        
        # Progressive augmentation if schedule provided
        if 'progressive_schedule' in aug_config:
            schedule = aug_config['progressive_schedule']
            advanced_aug = ProgressiveAugmentation(aug_config, schedule)
        
        # Standard transforms
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        val_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create datasets
        train_dataset = EnhancedDataset(
            dataset_path / 'final_metadata' / 'train_exp3.csv',
            transform=train_transform,
            advanced_aug=advanced_aug
        )
        
        val_dataset = EnhancedDataset(
            dataset_path / 'final_metadata' / 'val_exp3.csv',
            transform=val_transform,
            advanced_aug=None
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data'].get('pin_memory', True),
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data']['num_workers'],
            pin_memory=self.config['data'].get('pin_memory', True)
        )
        
        self.log(f"Train samples: {len(train_dataset)}")
        self.log(f"Val samples: {len(val_dataset)}")
        self.log(f"Batch size: {self.config['training']['batch_size']}")
        
        return train_loader, val_loader, advanced_aug
    
    def create_model(self):
        """Create enhanced model"""
        self.log("Creating enhanced model...")
        
        model = create_enhanced_model(self.config['model'])
        model = model.to(self.device)
        
        # Load checkpoint if specified
        if 'checkpoint' in self.config and 'resume_from' in self.config['checkpoint']:
            checkpoint_path = self.config['checkpoint']['resume_from']
            if os.path.exists(checkpoint_path):
                self.log(f"Loading checkpoint: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                
                # Try to load model state
                if 'model_state_dict' in checkpoint:
                    try:
                        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                        self.log("[OK] Loaded model weights (non-strict)")
                    except:
                        self.log("[WARNING] Could not load checkpoint weights, starting fresh")
                else:
                    self.log("[WARNING] No model_state_dict in checkpoint")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.log(f"Total parameters: {total_params:,}")
        self.log(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def create_optimizer_scheduler(self, model):
        """Create optimizer and scheduler with warmup"""
        self.log("Creating optimizer and scheduler...")
        
        opt_config = self.config['optimizer']
        sched_config = self.config['scheduler']
        train_config = self.config['training']
        
        # Optimizer
        optimizer = optim.AdamW(
            model.parameters(),
            lr=opt_config['base_lr'],
            weight_decay=opt_config['weight_decay'],
            betas=tuple(opt_config['betas']),
            eps=opt_config['eps']
        )
        
        # Scheduler with warmup
        warmup_epochs = sched_config.get('warmup_epochs', 5)
        total_epochs = train_config['epochs']
        min_lr = sched_config.get('min_lr', 1e-6)
        
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                # Linear warmup
                return (epoch + 1) / warmup_epochs
            else:
                # Cosine annealing
                progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
                cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
                return max(min_lr / opt_config['base_lr'], cosine_decay)
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        self.log(f"Optimizer: AdamW (lr={opt_config['base_lr']}, wd={opt_config['weight_decay']})")
        self.log(f"Scheduler: Cosine Annealing with {warmup_epochs} epoch warmup")
        
        return optimizer, scheduler
    
    def create_criterion(self):
        """Create loss functions"""
        loss_config = self.config['loss']
        
        mag_criterion = FocalLoss(
            alpha=loss_config['focal_alpha'],
            gamma=loss_config['focal_gamma']
        )
        
        azi_criterion = FocalLoss(
            alpha=loss_config['focal_alpha'],
            gamma=loss_config['focal_gamma']
        )
        
        task_weights = loss_config['task_weights']
        
        self.log(f"Loss: Focal Loss (alpha={loss_config['focal_alpha']}, gamma={loss_config['focal_gamma']})")
        self.log(f"Task weights: Magnitude={task_weights['magnitude']}, Azimuth={task_weights['azimuth']}")
        
        return mag_criterion, azi_criterion, task_weights
    
    def train_epoch(self, model, train_loader, optimizer, mag_criterion, azi_criterion, 
                   task_weights, scaler, ema_model, mixup, cutmix, epoch):
        """Train one epoch"""
        model.train()
        
        total_loss = 0
        mag_correct = 0
        azi_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (inputs, mag_targets, azi_targets) in enumerate(pbar):
            inputs = inputs.to(self.device)
            mag_targets = mag_targets.to(self.device)
            azi_targets = azi_targets.to(self.device)
            
            # Apply MixUp or CutMix
            use_mixup = np.random.rand() < 0.5
            if use_mixup and mixup is not None:
                result = mixup(inputs, mag_targets, azi_targets)
                if len(result) > 3:
                    inputs, mag_targets, azi_targets, mag_targets_b, azi_targets_b, lam = result
                    use_mix = True
                else:
                    use_mix = False
            elif cutmix is not None:
                result = cutmix(inputs, mag_targets, azi_targets)
                if len(result) > 3:
                    inputs, mag_targets, azi_targets, mag_targets_b, azi_targets_b, lam = result
                    use_mix = True
                else:
                    use_mix = False
            else:
                use_mix = False
            
            optimizer.zero_grad()
            
            # Mixed precision forward pass
            with autocast():
                mag_out, azi_out = model(inputs)
                
                if use_mix:
                    loss_mag = lam * mag_criterion(mag_out, mag_targets) + (1 - lam) * mag_criterion(mag_out, mag_targets_b)
                    loss_azi = lam * azi_criterion(azi_out, azi_targets) + (1 - lam) * azi_criterion(azi_out, azi_targets_b)
                else:
                    loss_mag = mag_criterion(mag_out, mag_targets)
                    loss_azi = azi_criterion(azi_out, azi_targets)
                
                loss = task_weights['magnitude'] * loss_mag + task_weights['azimuth'] * loss_azi
            
            # Backward pass
            scaler.scale(loss).backward()
            
            # Gradient clipping
            if 'gradient_clip_norm' in self.config['optimizer']:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['optimizer']['gradient_clip_norm'])
            
            scaler.step(optimizer)
            scaler.update()
            
            # Update EMA
            if ema_model is not None:
                ema_model.update(model)
            
            # Calculate accuracy
            mag_pred = torch.argmax(mag_out, dim=1)
            azi_pred = torch.argmax(azi_out, dim=1)
            mag_correct += (mag_pred == mag_targets).sum().item()
            azi_correct += (azi_pred == azi_targets).sum().item()
            total_samples += inputs.size(0)
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mag_acc': f'{100*mag_correct/total_samples:.2f}%',
                'azi_acc': f'{100*azi_correct/total_samples:.2f}%'
            })
        
        avg_loss = total_loss / len(train_loader)
        mag_acc = mag_correct / total_samples
        azi_acc = azi_correct / total_samples
        
        return avg_loss, mag_acc, azi_acc
    
    def validate(self, model, val_loader, mag_criterion, azi_criterion, task_weights):
        """Validate model"""
        model.eval()
        
        total_loss = 0
        mag_correct = 0
        azi_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, mag_targets, azi_targets in tqdm(val_loader, desc="Validating"):
                inputs = inputs.to(self.device)
                mag_targets = mag_targets.to(self.device)
                azi_targets = azi_targets.to(self.device)
                
                with autocast():
                    mag_out, azi_out = model(inputs)
                    loss_mag = mag_criterion(mag_out, mag_targets)
                    loss_azi = azi_criterion(azi_out, azi_targets)
                    loss = task_weights['magnitude'] * loss_mag + task_weights['azimuth'] * loss_azi
                
                mag_pred = torch.argmax(mag_out, dim=1)
                azi_pred = torch.argmax(azi_out, dim=1)
                mag_correct += (mag_pred == mag_targets).sum().item()
                azi_correct += (azi_pred == azi_targets).sum().item()
                total_samples += inputs.size(0)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        mag_acc = mag_correct / total_samples
        azi_acc = azi_correct / total_samples
        
        return avg_loss, mag_acc, azi_acc
    
    def train(self):
        """Main training loop"""
        self.log("="*80)
        self.log("STARTING ENHANCED TRAINING")
        self.log("="*80)
        
        # Create dataloaders
        train_loader, val_loader, advanced_aug = self.create_dataloaders()
        
        # Create model
        model = self.create_model()
        
        # Create optimizer and scheduler
        optimizer, scheduler = self.create_optimizer_scheduler(model)
        
        # Create loss functions
        mag_criterion, azi_criterion, task_weights = self.create_criterion()
        
        # Create EMA model
        ema_model = EMAModel(model, decay=self.config['training']['ema_decay'])
        
        # Create MixUp and CutMix
        mixup = MixUp(alpha=self.config['augmentation']['mixup_alpha'], prob=0.5)
        cutmix = CutMix(alpha=self.config['augmentation']['cutmix_alpha'], prob=0.5)
        
        # Mixed precision scaler
        scaler = GradScaler()
        
        # Training loop
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(self.config['training']['epochs']):
            self.log(f"\n{'='*80}")
            self.log(f"Epoch {epoch+1}/{self.config['training']['epochs']}")
            self.log(f"{'='*80}")
            
            # Update progressive augmentation
            if isinstance(advanced_aug, ProgressiveAugmentation):
                advanced_aug.set_epoch(epoch)
            
            # Train
            train_loss, train_mag_acc, train_azi_acc = self.train_epoch(
                model, train_loader, optimizer, mag_criterion, azi_criterion,
                task_weights, scaler, ema_model, mixup, cutmix, epoch
            )
            
            # Validate
            val_loss, val_mag_acc, val_azi_acc = self.validate(
                model, val_loader, mag_criterion, azi_criterion, task_weights
            )
            
            # Update scheduler
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            
            # Log results
            self.log(f"\nTrain - Loss: {train_loss:.4f}, Mag Acc: {train_mag_acc*100:.2f}%, Azi Acc: {train_azi_acc*100:.2f}%")
            self.log(f"Val   - Loss: {val_loss:.4f}, Mag Acc: {val_mag_acc*100:.2f}%, Azi Acc: {val_azi_acc*100:.2f}%")
            self.log(f"LR: {current_lr:.6f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_mag_acc'].append(train_mag_acc)
            self.history['train_azi_acc'].append(train_azi_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_mag_acc'].append(val_mag_acc)
            self.history['val_azi_acc'].append(val_azi_acc)
            self.history['learning_rates'].append(current_lr)
            
            # Save history
            with open(self.exp_dir / 'history.json', 'w') as f:
                json.dump(self.history, f, indent=2)
            
            # Check if best model
            combined_acc = (val_mag_acc + val_azi_acc) / 2
            if combined_acc > best_val_acc:
                best_val_acc = combined_acc
                patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_mag_acc': val_mag_acc,
                    'val_azi_acc': val_azi_acc,
                    'config': self.config
                }, self.exp_dir / 'best_model.pth')
                
                self.log(f"[BEST] New best model! Combined Acc: {combined_acc*100:.2f}%")
            else:
                patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config['training']['save_interval'] == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'ema_state_dict': ema_model.ema_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'config': self.config
                }, self.exp_dir / f'checkpoint_epoch_{epoch+1}.pth')
            
            # Early stopping
            if patience_counter >= self.config['training']['early_stopping_patience']:
                self.log(f"\n[EARLY STOP] Triggered after {patience_counter} epochs without improvement")
                break
        
        # Save final results
        final_results = {
            'best_val_magnitude_acc': max(self.history['val_mag_acc']),
            'best_val_azimuth_acc': max(self.history['val_azi_acc']),
            'final_val_magnitude_acc': self.history['val_mag_acc'][-1],
            'final_val_azimuth_acc': self.history['val_azi_acc'][-1],
            'experiment_dir': str(self.exp_dir)
        }
        
        with open(self.exp_dir / 'final_results.json', 'w') as f:
            json.dump(final_results, f, indent=2)
        
        self.log("\n" + "="*80)
        self.log("TRAINING COMPLETE!")
        self.log("="*80)
        self.log(f"Best Val Magnitude Acc: {max(self.history['val_mag_acc'])*100:.2f}%")
        self.log(f"Best Val Azimuth Acc: {max(self.history['val_azi_acc'])*100:.2f}%")
        self.log(f"Results saved to: {self.exp_dir}")


def main():
    parser = argparse.ArgumentParser(description='Enhanced ConvNeXt Training')
    parser.add_argument('--config', type=str, default='config_improved.json',
                       help='Path to configuration file')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--lr', type=float, default=None,
                       help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = json.load(f)
    
    # Apply overrides
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.lr:
        config['optimizer']['base_lr'] = args.lr
    
    # Create trainer and train
    trainer = EnhancedTrainer(config_dict=config)
    trainer.train()


if __name__ == '__main__':
    main()
