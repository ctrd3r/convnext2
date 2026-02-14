# Methodology

## Detailed Methodology for ConvNeXt Earthquake Precursor Detection

---

## 1. Data Collection

### 1.1 Geomagnetic Stations

Data was collected from Indonesian geomagnetic stations operated by BMKG:

| Station | Location | Coordinates | Equipment |
|---------|----------|-------------|-----------|
| SCN | Central Java | -7.05°, 110.44° | Fluxgate magnetometer |
| MLB | East Java | -7.97°, 112.63° | Fluxgate magnetometer |
| GTO | Gorontalo | 0.54°, 123.06° | Fluxgate magnetometer |
| TRD | Ternate | 0.78°, 127.37° | Fluxgate magnetometer |
| ... | ... | ... | ... |

### 1.2 Data Specifications

- **Sampling rate**: 1 Hz
- **Components**: H (horizontal), D (declination), Z (vertical)
- **Time period**: 2018-2025
- **Total events**: 256 earthquakes

### 1.3 Earthquake Selection Criteria

- Magnitude: M4.0 - M7.0+
- Depth: < 100 km (shallow earthquakes)
- Distance: < 500 km from station
- Data quality: Complete 24-hour record before event

---

## 2. Data Preprocessing

### 2.1 Signal Processing Pipeline

```
Raw Data (1 Hz) → Bandpass Filter → Hourly Segmentation → STFT → Spectrogram
```

### 2.2 Bandpass Filtering

- **Filter type**: Butterworth (4th order)
- **Passband**: 0.001 - 0.5 Hz (Pc3-Pc5 range)
- **Purpose**: Extract ULF precursor signals

```python
from scipy.signal import butter, filtfilt

def bandpass_filter(data, lowcut=0.001, highcut=0.5, fs=1.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)
```

### 2.3 Spectrogram Generation

- **Method**: Short-Time Fourier Transform (STFT)
- **Window**: Hanning, 256 samples
- **Overlap**: 128 samples (50%)
- **Output size**: 224 × 224 pixels

```python
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

def generate_spectrogram(signal, fs=1.0, nperseg=256, noverlap=128):
    f, t, Sxx = spectrogram(signal, fs, nperseg=nperseg, noverlap=noverlap)
    
    plt.figure(figsize=(10, 6))
    plt.pcolormesh(t, f, 10*np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Power [dB]')
    plt.savefig('spectrogram.png', dpi=224/10)  # 224x224 output
```

### 2.4 Three-Component Visualization

Each spectrogram combines H, D, Z components:
- **Red channel**: H component
- **Green channel**: D component
- **Blue channel**: Z component

---

## 3. Dataset Preparation

### 3.1 Class Definition

**Magnitude Classes:**
| Class | Range | Physical Interpretation |
|-------|-------|------------------------|
| Moderate | M4.0-4.9 | Light earthquakes |
| Medium | M5.0-5.9 | Moderate earthquakes |
| Large | M6.0-6.9 | Strong earthquakes |
| Normal | Non-precursor | Quiet geomagnetic days |

**Azimuth Classes:**
| Class | Angle Range | Direction |
|-------|-------------|-----------|
| N | 337.5° - 22.5° | North |
| NE | 22.5° - 67.5° | Northeast |
| E | 67.5° - 112.5° | East |
| SE | 112.5° - 157.5° | Southeast |
| S | 157.5° - 202.5° | South |
| SW | 202.5° - 247.5° | Southwest |
| W | 247.5° - 292.5° | West |
| NW | 292.5° - 337.5° | Northwest |
| Normal | N/A | Non-precursor |

### 3.2 Data Splitting

- **Training**: 70% (1,384 samples)
- **Validation**: 15% (284 samples)
- **Test**: 15% (304 samples)

**Stratification**: Maintained class proportions across splits

### 3.3 Class Imbalance Handling

1. **Inverse frequency weighting**:
```python
class_weights = total_samples / (n_classes * class_counts)
```

2. **Data augmentation** for minority classes

---

## 4. Model Architecture

### 4.1 ConvNeXt-Tiny Backbone

Based on Liu et al. (2022) "A ConvNet for the 2020s":

```
Input (224×224×3)
    │
    ├── Stem: Conv2d(3→96, k=4, s=4) + LayerNorm
    │
    ├── Stage 1: 3 × ConvNeXt Block (96 ch)
    ├── Downsample: LN + Conv2d(96→192, k=2, s=2)
    │
    ├── Stage 2: 3 × ConvNeXt Block (192 ch)
    ├── Downsample: LN + Conv2d(192→384, k=2, s=2)
    │
    ├── Stage 3: 9 × ConvNeXt Block (384 ch)
    ├── Downsample: LN + Conv2d(384→768, k=2, s=2)
    │
    ├── Stage 4: 3 × ConvNeXt Block (768 ch)
    │
    └── Global Average Pooling → 768-dim features
```

### 4.2 ConvNeXt Block

```
Input (C channels)
    │
    ├── Depthwise Conv 7×7 (C → C)
    ├── LayerNorm
    ├── Pointwise Conv 1×1 (C → 4C)
    ├── GELU
    ├── Pointwise Conv 1×1 (4C → C)
    ├── Layer Scale
    ├── Stochastic Depth
    │
    └── + Residual Connection
```

### 4.3 Multi-Task Heads

**Magnitude Head:**
```
768-dim → LayerNorm → Dropout(0.5) → Linear(512) → GELU → 
Dropout(0.25) → Linear(4) → Softmax
```

**Azimuth Head:**
```
768-dim → LayerNorm → Dropout(0.5) → Linear(512) → GELU → 
Dropout(0.25) → Linear(9) → Softmax
```

---

## 5. Training Procedure

### 5.1 Optimization

| Parameter | Value | Justification |
|-----------|-------|---------------|
| Optimizer | AdamW | Better weight decay handling |
| Learning Rate | 1e-4 | Standard for transfer learning |
| Weight Decay | 0.05 | ConvNeXt recommendation |
| Batch Size | 32 | Memory-efficient |
| Epochs | 50 | With early stopping |

### 5.2 Learning Rate Schedule

**Cosine annealing with linear warmup:**

```python
def lr_schedule(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps  # Linear warmup
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + cos(π * progress))  # Cosine decay
```

- Warmup: 5 epochs
- Total: 50 epochs

### 5.3 Loss Function

**Multi-task weighted loss:**
```
L_total = L_magnitude + 0.5 × L_azimuth
```

Where:
- L_magnitude = WeightedCrossEntropy(mag_pred, mag_true)
- L_azimuth = WeightedCrossEntropy(azi_pred, azi_true)

### 5.4 Data Augmentation

| Augmentation | Parameters | Purpose |
|--------------|------------|---------|
| Horizontal Flip | p=0.5 | Spatial invariance |
| Rotation | ±15° | Orientation invariance |
| Color Jitter | brightness=0.2, contrast=0.2 | Intensity invariance |
| Random Affine | translate=10% | Position invariance |
| Random Erasing | p=0.1 | Occlusion robustness |

### 5.5 Regularization

1. **Dropout**: 0.5 (heads), 0.25 (intermediate)
2. **Weight Decay**: 0.05
3. **Early Stopping**: 10 epochs patience
4. **Gradient Clipping**: max_norm=1.0

---

## 6. Validation Methods

### 6.1 LOEO (Leave-One-Event-Out)

**Purpose**: Validate temporal generalization

**Procedure**:
1. Group samples by earthquake event
2. Create 10 folds, each excluding different events
3. Train on 9 folds, test on 1 fold
4. Report mean ± std across folds

### 6.2 LOSO (Leave-One-Station-Out)

**Purpose**: Validate spatial generalization

**Procedure**:
1. Group samples by station
2. Create 9 folds (one per major station)
3. Train on 8 stations, test on 1 station
4. Report weighted mean across stations

### 6.3 Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| Accuracy | TP+TN / Total | Overall performance |
| Precision | TP / (TP+FP) | False positive rate |
| Recall | TP / (TP+FN) | False negative rate |
| F1 Score | 2×P×R / (P+R) | Balanced metric |
| Weighted F1 | Σ(w_i × F1_i) | Class-weighted |

---

## 7. Interpretability

### 7.1 Grad-CAM

**Gradient-weighted Class Activation Mapping:**

```python
# Forward pass
features = model.backbone(input)
output = model.head(features)

# Backward pass for target class
output[target_class].backward()

# Compute weights
weights = gradients.mean(dim=(2,3))

# Generate heatmap
cam = ReLU(Σ(weights × activations))
```

### 7.2 Feature Visualization

- t-SNE for feature space visualization
- Attention maps for spatial focus analysis
- Frequency band importance analysis

---

## 8. Implementation Details

### 8.1 Software Environment

```
Python: 3.10+
PyTorch: 2.0+
torchvision: 0.15+
NumPy: 1.24+
Pandas: 2.0+
scikit-learn: 1.3+
matplotlib: 3.7+
seaborn: 0.12+
```

### 8.2 Hardware

- **Training**: CPU (Intel Core i7) or GPU (NVIDIA RTX 3080)
- **Memory**: 16 GB RAM minimum
- **Storage**: 10 GB for dataset and models

### 8.3 Reproducibility

- Random seed: 42
- Deterministic operations enabled
- Full configuration saved with each experiment

---

*Methodology document for ConvNeXt earthquake precursor detection*
