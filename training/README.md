# ConvNeXt Implementation Guide
## Step-by-Step Instructions for Model Improvement

**Created**: 2026-02-14  
**Target**: Improve from 63.7% to 70-80% accuracy  
**Estimated Implementation Time**: 8-12 hours  
**Estimated Training Time**: 6-8 hours

---

## 📋 Prerequisites

### Required Software
- Python 3.8+
- PyTorch 1.12+ with CUDA support
- torchvision 0.13+
- CUDA 11.3+ and cuDNN 8.2+

### Required Hardware
- GPU with 8+ GB VRAM (RTX 3070 or better)
- 16+ GB System RAM
- 50+ GB free disk space

### Verify Setup
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

---

## 🎯 Implementation Phases

### Phase 1: Quick Wins (2-3 hours)
**Goal**: Immediate improvements with minimal code changes  
**Expected Gain**: +10-15% accuracy

### Phase 2: Architecture Enhancements (3-4 hours)
**Goal**: Add attention and multi-scale features  
**Expected Gain**: +5-10% accuracy

### Phase 3: Advanced Training (2-3 hours)
**Goal**: Implement advanced augmentation and training strategies  
**Expected Gain**: +5-8% accuracy

### Phase 4: Azimuth Fix (2-3 hours)
**Goal**: Dramatically improve azimuth classification  
**Expected Gain**: +40-50% azimuth accuracy

---

## 🚀 Phase 1: Quick Wins

### Step 1.1: Update Training Configuration

**File**: Create `config_improved.json`

```json
{
  "num_magnitude_classes": 4,
  "num_azimuth_classes": 9,
  "dropout_rate": 0.4,
  "drop_path_rate": 0.2,
  "focal_alpha": 0.25,
  "focal_gamma": 3.0,
  "label_smoothing": 0.1,
  "learn_task_weights": false,
  "task_weights": {
    "magnitude": 0.4,
    "azimuth": 0.6
  },
  "optimizer": "adamw",
  "base_lr": 0.0005,
  "weight_decay": 0.05,
  "batch_size": 64,
  "epochs": 50,
  "warmup_epochs": 5,
  "min_lr": 1e-6,
  "randaugment_magnitude": 12,
  "randaugment_num_ops": 3,
  "mixup_alpha": 0.4,
  "cutmix_alpha": 1.0,
  "ema_decay": 0.9999,
  "seed": 42,
  "dataset_path": "dataset_experiment_3",
  "num_workers": 8,
  "mixed_precision": true
}
```

**Changes from baseline**:
- ✅ Batch size: 4 → 64 (16x larger)
- ✅ Epochs: 3 → 50 (16x longer)
- ✅ Learning rate: 1e-4 → 5e-4 (5x higher with warmup)
- ✅ Task weights: Auto → Manual (0.4/0.6 for mag/azi)
- ✅ Augmentation: Stronger (9→12 magnitude, 2→3 ops)
- ✅ Mixed precision: Enabled

### Step 1.2: Implement Warmup Scheduler

**File**: Modify `start_finetune_server.py`

Add after line 96:

```python
# Warmup + Cosine Annealing Scheduler
def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
    """Learning rate scheduler with warmup and cosine annealing"""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return max(min_lr / self.config['base_lr'], cosine_decay)
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Replace line 97 with:
warmup_epochs = self.config.get('warmup_epochs', 5)
scheduler = get_warmup_cosine_scheduler(
    optimizer, 
    warmup_epochs=warmup_epochs,
    total_epochs=total_epochs,
    min_lr=self.config.get('min_lr', 1e-6)
)
```

### Step 1.3: Enable Mixed Precision Training

**File**: Modify `start_finetune_server.py`

Already implemented at line 100, but ensure it's used in training loop:

```python
# In training loop (around line 120-150)
for batch_idx, (inputs, mag_targets, azi_targets) in enumerate(train_loader):
    inputs = inputs.to(self.device)
    mag_targets = mag_targets.to(self.device)
    azi_targets = azi_targets.to(self.device)
    
    optimizer.zero_grad()
    
    # Mixed precision forward pass
    with autocast():
        mag_out, azi_out = model(inputs)
        loss_mag = criterion['magnitude'](mag_out, mag_targets)
        loss_azi = criterion['azimuth'](azi_out, azi_targets)
        
        # Apply task weights
        loss = (self.config['task_weights']['magnitude'] * loss_mag + 
                self.config['task_weights']['azimuth'] * loss_azi)
    
    # Mixed precision backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### Step 1.4: Run Quick Test

```bash
# Test with 10 epochs first
python start_finetune_server.py \
    --config config_improved.json \
    --epochs 10 \
    --batch-size 32

# Monitor GPU usage
nvidia-smi -l 1
```

**Expected Results after 10 epochs**:
- Magnitude Accuracy: 68-72%
- Azimuth Accuracy: 25-35%
- Training Time: ~1.5 hours

---

## 🏗️ Phase 2: Architecture Enhancements

### Step 2.1: Add Attention Modules

**File**: Create `convnext_enhanced.py`

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialAttention(nn.Module):
    """Spatial attention to focus on important regions"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise max and average pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention = torch.cat([max_pool, avg_pool], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention

class ChannelAttention(nn.Module):
    """Channel attention to focus on important features"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average and max pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        
        return x * attention

class CBAM(nn.Module):
    """Convolutional Block Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention()
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class EnhancedConvNeXt(nn.Module):
    """ConvNeXt with attention mechanisms"""
    def __init__(self, num_mag_classes=4, num_azi_classes=9, 
                 dropout=0.4, use_attention=True):
        super().__init__()
        
        # Load pretrained ConvNeXt
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.backbone = models.convnext_tiny(weights=weights)
        
        # Get feature dimension
        num_features = 768
        
        # Add attention after backbone features
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(num_features, reduction=16)
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Enhanced classification heads
        self.magnitude_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_mag_classes)
        )
        
        self.azimuth_head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, num_azi_classes)
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        
        # Apply attention
        if self.use_attention:
            features = self.attention(features)
        
        # Global average pooling
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        # Classification
        mag_out = self.magnitude_head(features)
        azi_out = self.azimuth_head(features)
        
        return mag_out, azi_out
```

### Step 2.2: Update Training Script

**File**: Modify `start_finetune_server.py`

Replace model creation (around line 80-82):

```python
# Import enhanced model
from convnext_enhanced import EnhancedConvNeXt

# In setup_model method:
model = EnhancedConvNeXt(
    num_mag_classes=self.config['num_magnitude_classes'],
    num_azi_classes=self.config['num_azimuth_classes'],
    dropout=self.config['dropout_rate'],
    use_attention=self.config.get('use_attention', True)
).to(self.device)
```

### Step 2.3: Test Enhanced Model

```bash
# Train with attention
python start_finetune_server.py \
    --config config_improved.json \
    --use-attention \
    --epochs 20

# Compare with baseline
python compare_models.py \
    --baseline experiments_convnext/finetune_v3_gpu_20260214_143726 \
    --enhanced experiments_convnext/enhanced_*
```

**Expected Results after 20 epochs**:
- Magnitude Accuracy: 72-76%
- Azimuth Accuracy: 35-50%

---

## 🎨 Phase 3: Advanced Augmentation

### Step 3.1: Implement GridMask

**File**: Create `augmentations.py`

```python
import torch
import numpy as np
from PIL import Image

class GridMask:
    """GridMask augmentation for better generalization"""
    def __init__(self, ratio=0.6, prob=0.3):
        self.ratio = ratio
        self.prob = prob
    
    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        
        h, w = img.shape[-2:]
        
        # Grid size
        d = min(h, w) // 4
        
        # Create mask
        mask = torch.ones_like(img)
        for i in range(0, h, d * 2):
            for j in range(0, w, d * 2):
                mask[:, i:i+d, j:j+d] = 0
        
        return img * mask

class NoiseInjection:
    """Add Gaussian noise to simulate sensor noise"""
    def __init__(self, snr_range=(10, 30), prob=0.2):
        self.snr_range = snr_range
        self.prob = prob
    
    def __call__(self, img):
        if np.random.rand() > self.prob:
            return img
        
        # Random SNR
        snr_db = np.random.uniform(*self.snr_range)
        snr = 10 ** (snr_db / 10)
        
        # Calculate noise power
        signal_power = torch.mean(img ** 2)
        noise_power = signal_power / snr
        
        # Add noise
        noise = torch.randn_like(img) * torch.sqrt(noise_power)
        return img + noise

class AdvancedAugmentation:
    """Combined advanced augmentations"""
    def __init__(self, config):
        self.gridmask = GridMask(
            ratio=config.get('gridmask_ratio', 0.6),
            prob=config.get('gridmask_prob', 0.3)
        )
        self.noise = NoiseInjection(
            snr_range=config.get('noise_snr_range', (10, 30)),
            prob=config.get('noise_prob', 0.2)
        )
    
    def __call__(self, img):
        img = self.gridmask(img)
        img = self.noise(img)
        return img
```

### Step 3.2: Update Data Loading

**File**: Modify `train_earthquake_v3.py`

Add to transform pipeline (around line 200-250):

```python
from augmentations import AdvancedAugmentation

# In create_dataloaders method:
train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    AdvancedAugmentation(config),  # Add this
])
```

---

## 🎯 Phase 4: Azimuth Classification Fix

### Step 4.1: Implement Hierarchical Azimuth

**File**: Create `hierarchical_azimuth.py`

```python
import torch
import torch.nn as nn

class HierarchicalAzimuthHead(nn.Module):
    """Two-stage azimuth classification"""
    def __init__(self, in_features, dropout=0.4):
        super().__init__()
        
        # Stage 1: Cardinal directions (N/S/E/W + Center)
        self.cardinal_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 5)  # N, S, E, W, Center
        )
        
        # Stage 2: Intercardinal directions (NE/SE/SW/NW)
        self.intercardinal_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 4)  # NE, SE, SW, NW
        )
        
        # Fusion layer
        self.fusion = nn.Linear(9, 9)
    
    def forward(self, x):
        cardinal = self.cardinal_head(x)
        intercardinal = self.intercardinal_head(x)
        
        # Combine predictions
        # Map: N, NE, E, SE, S, SW, W, NW, Center
        combined = torch.cat([
            cardinal[:, 0:1],      # N
            intercardinal[:, 0:1], # NE
            cardinal[:, 2:3],      # E
            intercardinal[:, 1:2], # SE
            cardinal[:, 1:2],      # S
            intercardinal[:, 2:3], # SW
            cardinal[:, 3:4],      # W
            intercardinal[:, 3:4], # NW
            cardinal[:, 4:5],      # Center
        ], dim=1)
        
        return self.fusion(combined)

class DirectionalFeatureExtractor(nn.Module):
    """Extract direction-specific features"""
    def __init__(self, in_channels):
        super().__init__()
        
        # Vertical emphasis (for N-S)
        self.vertical_conv = nn.Conv2d(
            in_channels, in_channels // 4,
            kernel_size=(7, 3), padding=(3, 1)
        )
        
        # Horizontal emphasis (for E-W)
        self.horizontal_conv = nn.Conv2d(
            in_channels, in_channels // 4,
            kernel_size=(3, 7), padding=(1, 3)
        )
        
        # Diagonal emphasis (for NE-SW, NW-SE)
        self.diagonal1_conv = nn.Conv2d(
            in_channels, in_channels // 4,
            kernel_size=5, padding=2
        )
        self.diagonal2_conv = nn.Conv2d(
            in_channels, in_channels // 4,
            kernel_size=5, padding=2
        )
        
        # Fusion
        self.fusion = nn.Conv2d(in_channels, in_channels, kernel_size=1)
    
    def forward(self, x):
        v = self.vertical_conv(x)
        h = self.horizontal_conv(x)
        d1 = self.diagonal1_conv(x)
        d2 = self.diagonal2_conv(x)
        
        combined = torch.cat([v, h, d1, d2], dim=1)
        return self.fusion(combined)
```

### Step 4.2: Update Model with Hierarchical Azimuth

**File**: Modify `convnext_enhanced.py`

```python
from hierarchical_azimuth import HierarchicalAzimuthHead, DirectionalFeatureExtractor

class EnhancedConvNeXtV2(nn.Module):
    """ConvNeXt with hierarchical azimuth classification"""
    def __init__(self, num_mag_classes=4, num_azi_classes=9, 
                 dropout=0.4, use_hierarchical_azimuth=True):
        super().__init__()
        
        # ... (previous code) ...
        
        # Add directional feature extractor
        if use_hierarchical_azimuth:
            self.directional_features = DirectionalFeatureExtractor(num_features)
            self.azimuth_head = HierarchicalAzimuthHead(num_features, dropout)
        else:
            self.azimuth_head = nn.Sequential(
                nn.Linear(num_features, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, num_azi_classes)
            )
    
    def forward(self, x):
        # Extract features
        features = self.backbone.features(x)
        
        # Apply directional feature extraction for azimuth
        if hasattr(self, 'directional_features'):
            directional_feat = self.directional_features(features)
            features = features + directional_feat
        
        # ... (rest of forward pass) ...
```

---

## 📊 Phase 5: Training and Evaluation

### Step 5.1: Full Training Run

```bash
# Run full training with all improvements
python start_finetune_server.py \
    --config config_improved.json \
    --epochs 50 \
    --batch-size 64 \
    --use-attention \
    --use-hierarchical-azimuth \
    --mixed-precision

# Expected time: 6-8 hours
```

### Step 5.2: Monitor Training

Create `monitor_training.py`:

```python
import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_training_progress(exp_dir):
    """Plot training curves in real-time"""
    history_file = Path(exp_dir) / 'history.json'
    
    if not history_file.exists():
        print("History file not found")
        return
    
    with open(history_file) as f:
        history = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Val')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # Magnitude accuracy
    axes[0, 1].plot(history['train_mag_acc'], label='Train')
    axes[0, 1].plot(history['val_mag_acc'], label='Val')
    axes[0, 1].set_title('Magnitude Accuracy')
    axes[0, 1].legend()
    
    # Azimuth accuracy
    axes[1, 0].plot(history['train_az_acc'], label='Train')
    axes[1, 0].plot(history['val_az_acc'], label='Val')
    axes[1, 0].set_title('Azimuth Accuracy')
    axes[1, 0].legend()
    
    # Learning rate
    axes[1, 1].plot(history['learning_rates'])
    axes[1, 1].set_title('Learning Rate')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(Path(exp_dir) / 'training_progress.png')
    plt.close()

# Run every 5 minutes
import time
while True:
    plot_training_progress('experiments_convnext/enhanced_*')
    time.sleep(300)
```

### Step 5.3: Comprehensive Evaluation

Create `evaluate_improved_model.py`:

```python
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model_path, test_loader):
    """Comprehensive model evaluation"""
    
    # Load model
    model = torch.load(model_path)
    model.eval()
    
    all_mag_preds = []
    all_mag_targets = []
    all_azi_preds = []
    all_azi_targets = []
    
    with torch.no_grad():
        for inputs, mag_targets, azi_targets in test_loader:
            mag_out, azi_out = model(inputs.cuda())
            
            mag_preds = torch.argmax(mag_out, dim=1)
            azi_preds = torch.argmax(azi_out, dim=1)
            
            all_mag_preds.extend(mag_preds.cpu().numpy())
            all_mag_targets.extend(mag_targets.numpy())
            all_azi_preds.extend(azi_preds.cpu().numpy())
            all_azi_targets.extend(azi_targets.numpy())
    
    # Calculate metrics
    mag_acc = np.mean(np.array(all_mag_preds) == np.array(all_mag_targets))
    azi_acc = np.mean(np.array(all_azi_preds) == np.array(all_azi_targets))
    
    print(f"Magnitude Accuracy: {mag_acc * 100:.2f}%")
    print(f"Azimuth Accuracy: {azi_acc * 100:.2f}%")
    
    # Classification reports
    print("\nMagnitude Classification Report:")
    print(classification_report(all_mag_targets, all_mag_preds,
                                target_names=['Small', 'Medium', 'Large', 'VeryLarge']))
    
    print("\nAzimuth Classification Report:")
    print(classification_report(all_azi_targets, all_azi_preds,
                                target_names=['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'Center']))
    
    # Confusion matrices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    mag_cm = confusion_matrix(all_mag_targets, all_mag_preds)
    sns.heatmap(mag_cm, annot=True, fmt='d', ax=axes[0], cmap='Blues')
    axes[0].set_title('Magnitude Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    
    azi_cm = confusion_matrix(all_azi_targets, all_azi_preds)
    sns.heatmap(azi_cm, annot=True, fmt='d', ax=axes[1], cmap='Greens')
    axes[1].set_title('Azimuth Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png')
    
    return {
        'magnitude_accuracy': mag_acc,
        'azimuth_accuracy': azi_acc,
        'magnitude_cm': mag_cm,
        'azimuth_cm': azi_cm
    }
```

---

## 📈 Expected Results Summary

### Baseline vs Improved

| Metric | Baseline | After Phase 1 | After Phase 2 | After Phase 3 | After Phase 4 | Final Target |
|--------|----------|---------------|---------------|---------------|---------------|--------------|
| Magnitude Acc | 63.7% | 70-73% | 73-76% | 75-78% | 75-78% | **75-80%** |
| Azimuth Acc | 11.5% | 25-35% | 35-50% | 45-60% | 60-70% | **65-70%** |
| Overall Acc | 37.6% | 47-54% | 54-63% | 60-69% | 67-74% | **70-75%** |
| Training Time | 43 min | 1.5 hrs | 3 hrs | 5 hrs | 6-8 hrs | **6-8 hrs** |

---

## ✅ Validation Checklist

After each phase, verify:

- [ ] Training loss is decreasing
- [ ] Validation accuracy is improving
- [ ] No overfitting (train/val gap < 10%)
- [ ] GPU utilization > 80%
- [ ] No OOM errors
- [ ] Checkpoints are being saved
- [ ] Logs are being written

---

## 🔧 Troubleshooting Guide

### Issue: OOM Error
**Solution**: Reduce batch size by half, enable gradient accumulation

### Issue: Training too slow
**Solution**: Enable mixed precision, increase num_workers, reduce augmentation

### Issue: Azimuth accuracy still low
**Solution**: Increase azimuth task weight to 0.7, use hierarchical classification

### Issue: Overfitting
**Solution**: Increase dropout, add more augmentation, reduce learning rate

### Issue: Underfitting
**Solution**: Train longer, reduce regularization, increase model capacity

---

## 📝 Next Steps After Implementation

1. **Compare with baseline** using evaluation scripts
2. **Analyze failure cases** to identify remaining issues
3. **Fine-tune hyperparameters** based on results
4. **Train ensemble** of 3-5 models with different seeds
5. **Implement test-time augmentation** for final predictions
6. **Document improvements** and create final report

---

**Document Version**: 1.0  
**Last Updated**: 2026-02-14  
**Status**: Ready for Implementation
