# ConvNeXt Model Architecture

## Detailed Architecture Documentation for Earthquake Precursor Detection

---

## 1. ConvNeXt Overview

ConvNeXt is a pure convolutional model that modernizes the standard ResNet architecture by incorporating design choices from Vision Transformers (ViT). It was introduced by Liu et al. in "A ConvNet for the 2020s" (CVPR 2022).

### 1.1 Key Design Principles

| Design Choice | Traditional CNN | ConvNeXt |
|---------------|-----------------|----------|
| Stem | 7×7 conv, stride 2 | 4×4 conv, stride 4 (patchify) |
| Normalization | Batch Norm | Layer Norm |
| Activation | ReLU | GELU |
| Block Design | Bottleneck | Inverted Bottleneck |
| Kernel Size | 3×3 | 7×7 depthwise |
| Stage Ratio | 3:4:6:3 | 3:3:9:3 |

### 1.2 Why ConvNeXt for Earthquake Precursors?

1. **Larger receptive field**: 7×7 kernels capture broader frequency patterns in spectrograms
2. **Better feature extraction**: Inverted bottleneck design preserves more information
3. **Stable training**: Layer Normalization handles varying signal amplitudes
4. **Modern efficiency**: Competitive with ViT while being simpler to implement

---

## 2. Architecture Details

### 2.1 ConvNeXt-Tiny Specifications

```
ConvNeXt-Tiny
├── Stem: Conv2d(3, 96, kernel=4, stride=4) + LayerNorm
├── Stage 1: 3 × ConvNeXt Block (96 channels)
├── Downsample: LayerNorm + Conv2d(96, 192, kernel=2, stride=2)
├── Stage 2: 3 × ConvNeXt Block (192 channels)
├── Downsample: LayerNorm + Conv2d(192, 384, kernel=2, stride=2)
├── Stage 3: 9 × ConvNeXt Block (384 channels)
├── Downsample: LayerNorm + Conv2d(384, 768, kernel=2, stride=2)
├── Stage 4: 3 × ConvNeXt Block (768 channels)
├── Global Average Pooling
└── Classifier (replaced with multi-task heads)
```

### 2.2 ConvNeXt Block

```python
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block Structure:
    
    Input (C channels)
        │
        ├── Depthwise Conv 7×7 (C → C)
        │
        ├── LayerNorm
        │
        ├── Pointwise Conv 1×1 (C → 4C)  # Expansion
        │
        ├── GELU Activation
        │
        ├── Pointwise Conv 1×1 (4C → C)  # Projection
        │
        ├── Layer Scale (learnable)
        │
        └── Stochastic Depth (drop path)
        │
    Output (C channels) + Residual Connection
    """
```

### 2.3 Parameter Count

| Component | Parameters |
|-----------|------------|
| Stem | 4,704 |
| Stage 1 | 442,368 |
| Stage 2 | 1,769,472 |
| Stage 3 | 15,925,248 |
| Stage 4 | 7,077,888 |
| Magnitude Head | 919,044 |
| Azimuth Head | 919,561 |
| **Total** | **28,615,789** |

---

## 3. Multi-Task Learning Design

### 3.1 Architecture Diagram

```
                    ┌─────────────────────────────────────┐
                    │         Input Spectrogram           │
                    │           224 × 224 × 3             │
                    └─────────────────┬───────────────────┘
                                      │
                    ┌─────────────────▼───────────────────┐
                    │         ConvNeXt Backbone           │
                    │         (Pretrained ImageNet)       │
                    │                                     │
                    │  Stem → Stage1 → Stage2 → Stage3 → Stage4
                    │                                     │
                    └─────────────────┬───────────────────┘
                                      │
                              Feature Vector
                                 768-dim
                                      │
                    ┌─────────────────┼───────────────────┐
                    │                 │                   │
          ┌─────────▼─────────┐       │       ┌──────────▼──────────┐
          │   Magnitude Head  │       │       │    Azimuth Head     │
          │                   │       │       │                     │
          │ LayerNorm(768)    │       │       │ LayerNorm(768)      │
          │ Dropout(0.5)      │       │       │ Dropout(0.5)        │
          │ Linear(768→512)   │       │       │ Linear(768→512)     │
          │ GELU              │       │       │ GELU                │
          │ Dropout(0.25)     │       │       │ Dropout(0.25)       │
          │ Linear(512→4)     │       │       │ Linear(512→9)       │
          │                   │       │       │                     │
          └─────────┬─────────┘       │       └──────────┬──────────┘
                    │                 │                   │
          ┌─────────▼─────────┐       │       ┌──────────▼──────────┐
          │  Magnitude Class  │       │       │   Azimuth Class     │
          │                   │       │       │                     │
          │  0: Large         │       │       │  0: E               │
          │  1: Medium        │       │       │  1: N               │
          │  2: Moderate      │       │       │  2: NE              │
          │  3: Normal        │       │       │  3: NW              │
          │                   │       │       │  4: Normal          │
          └───────────────────┘       │       │  5: S               │
                                      │       │  6: SE              │
                                      │       │  7: SW              │
                                      │       │  8: W               │
                                      │       └─────────────────────┘
```

### 3.2 Loss Function

```python
# Multi-task loss with magnitude prioritization
L_total = L_magnitude + 0.5 × L_azimuth

# Weighted Cross-Entropy for class imbalance
L_magnitude = CrossEntropyLoss(weight=mag_weights)
L_azimuth = CrossEntropyLoss(weight=azi_weights)

# Class weights (inverse frequency)
mag_weights = [w_Large, w_Medium, w_Moderate, w_Normal]
azi_weights = [w_E, w_N, w_NE, w_NW, w_Normal, w_S, w_SE, w_SW, w_W]
```

---

## 4. Training Configuration

### 4.1 Optimizer Settings

```python
optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=0.05,  # Higher than typical (ConvNeXt recommendation)
    betas=(0.9, 0.999)
)
```

### 4.2 Learning Rate Schedule

```python
# Cosine annealing with linear warmup
def lr_lambda(step):
    warmup_steps = len(train_loader) * 5  # 5 epochs warmup
    total_steps = len(train_loader) * 50  # 50 epochs total
    
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + cos(π * progress))  # Cosine decay
```

### 4.3 Data Augmentation Pipeline

```python
train_transform = Compose([
    Resize((224, 224)),
    RandomHorizontalFlip(p=0.5),
    RandomRotation(15),
    ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    RandomAffine(degrees=0, translate=(0.1, 0.1)),
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    RandomErasing(p=0.1),
])
```

---

## 5. Comparison with Other Architectures

### 5.1 Architecture Comparison

| Feature | VGG16 | EfficientNet-B0 | ConvNeXt-Tiny |
|---------|-------|-----------------|---------------|
| Year | 2014 | 2019 | 2022 |
| Parameters | 138M | 5.3M | 28.6M |
| Depth | 16 layers | 237 layers | 18 blocks |
| Normalization | Batch Norm | Batch Norm | Layer Norm |
| Activation | ReLU | Swish | GELU |
| Kernel Size | 3×3 | 3×3, 5×5 | 7×7 |
| Stem | 3×3 conv | 3×3 conv | 4×4 conv |
| Skip Connections | No | Yes | Yes |

### 5.2 Computational Efficiency

| Model | FLOPs | Memory | Inference (CPU) |
|-------|-------|--------|-----------------|
| VGG16 | 15.5G | 528 MB | 45 ms |
| EfficientNet-B0 | 0.4G | 20 MB | 18 ms |
| ConvNeXt-Tiny | 4.5G | 112 MB | ~30 ms |

---

## 6. Implementation Code

### 6.1 Model Definition

```python
class ConvNeXtMultiTask(nn.Module):
    def __init__(self, variant="tiny", pretrained=True, 
                 num_mag_classes=4, num_azi_classes=9, dropout=0.5):
        super().__init__()
        
        # Load pretrained ConvNeXt
        if variant == "tiny":
            weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = convnext_tiny(weights=weights)
            num_features = 768
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Magnitude classification head
        self.mag_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_mag_classes)
        )
        
        # Azimuth classification head
        self.azi_head = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.Dropout(dropout),
            nn.Linear(num_features, 512),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, num_azi_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        if features.dim() == 4:
            features = features.flatten(1)
        
        mag_out = self.mag_head(features)
        azi_out = self.azi_head(features)
        
        return mag_out, azi_out
```

---

## 7. References

1. Liu, Z., et al. (2022). A ConvNet for the 2020s. CVPR 2022.
2. He, K., et al. (2016). Deep residual learning for image recognition. CVPR 2016.
3. Dosovitskiy, A., et al. (2020). An image is worth 16x16 words. ICLR 2021.
