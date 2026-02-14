# ConvNeXt Earthquake Precursor Detection

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning system for earthquake precursor detection using ConvNeXt architecture and geomagnetic spectrogram analysis.

## Overview

This repository contains the implementation of a ConvNeXt-based multi-task classification model for detecting earthquake precursors from geomagnetic field spectrograms. The model simultaneously predicts:

- **Magnitude Class**: Large (M≥6.0), Medium (5.0≤M<6.0), Moderate (4.0≤M<5.0), Normal
- **Azimuth Direction**: N, NE, E, SE, S, SW, W, NW, Normal

## Key Results

Using Leave-One-Event-Out (LOEO) 10-fold cross-validation:

| Task | Accuracy | F1 Score |
|------|----------|----------|
| Magnitude Classification | **97.53%** ± 0.96% | 97.14% |
| Azimuth Classification | **69.51%** ± 5.65% | 68.23% |

> **Note**: These results were achieved using EfficientNet-B0 backbone. ConvNeXt implementation is provided for comparison and further research.

## Model Architecture

ConvNeXt-Tiny with multi-task classification heads:

```
ConvNeXt-Tiny Backbone (28.6M params)
├── Patchify Stem (4×4 conv, stride 4)
├── Stage 1: 3× ConvNeXt Block (96 channels)
├── Stage 2: 3× ConvNeXt Block (192 channels)
├── Stage 3: 9× ConvNeXt Block (384 channels)
├── Stage 4: 3× ConvNeXt Block (768 channels)
└── Global Average Pooling

Multi-Task Heads
├── Magnitude Head: LayerNorm → Dropout → Linear(768,512) → GELU → Linear(512,4)
└── Azimuth Head: LayerNorm → Dropout → Linear(768,512) → GELU → Linear(512,9)
```

## Installation

```bash
# Clone repository
git clone https://github.com/ctrd3r/convnext2.git
cd convnext-earthquake-precursor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download pretrained models
python scripts/download_models.py
```

## Quick Start

### Inference on Single Image

```python
from src.model import ConvNeXtPrecursorModel
from src.inference import predict_spectrogram

# Load model
model = ConvNeXtPrecursorModel.load_pretrained('models/convnext_loeo_best.pth')

# Predict
result = predict_spectrogram(model, 'path/to/spectrogram.png')
print(f"Magnitude: {result['magnitude_class']} ({result['magnitude_prob']:.2%})")
print(f"Azimuth: {result['azimuth_class']} ({result['azimuth_prob']:.2%})")
```

### Training

```bash
# Standard training
python scripts/train.py --config configs/convnext_tiny.yaml

# LOEO cross-validation
python scripts/train_loeo.py --config configs/loeo_validation.yaml
```

### Evaluation

```bash
# Evaluate on test set
python scripts/evaluate.py --model models/convnext_loeo_best.pth --data data/test

# Generate GradCAM visualizations
python scripts/generate_gradcam.py --model models/convnext_loeo_best.pth --image sample.png
```

## Project Structure

```
convnext-earthquake-precursor/
├── configs/                    # Configuration files
│   ├── convnext_tiny.yaml
│   └── loeo_validation.yaml
├── data/                       # Dataset (not included, see Data section)
│   ├── spectrograms/
│   └── metadata/
├── models/                     # Pretrained models
│   └── README.md               # Instructions for downloading models
├── notebooks/                  # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_training_analysis.ipynb
│   └── 03_visualization.ipynb
├── scripts/                    # Training and evaluation scripts
│   ├── train.py
│   ├── train_loeo.py
│   ├── evaluate.py
│   ├── generate_gradcam.py
│   ├── prepare_data.py
│   └── download_models.py
├── src/                        # Source code
│   ├── __init__.py
│   ├── model.py
│   ├── dataset.py
│   ├── transforms.py
│   ├── losses.py
│   ├── inference.py
│   └── utils.py
├── tests/                      # Unit tests
│   ├── test_model.py
│   ├── test_dataset.py
│   └── test_inference.py
├── requirements.txt
├── setup.py
└── README.md
```

## Data

The model is trained on geomagnetic field spectrograms from Indonesian magnetometer stations:

- **Stations**: 9 stations across Indonesia (GTO, LUT, MLB, SBG, SCN, SKB, TRD, TRT, and 13 smaller stations)
- **Time Period**: 2018-2023
- **Events**: 301 earthquake events (M≥4.0)
- **Samples**: 1,972 spectrogram images (6-hour windows)

### Data Format

Spectrograms are 224×224 RGB images showing:
- X-axis: Time (6 hours)
- Y-axis: Frequency (0-0.5 Hz)
- Color: Power spectral density

### Generating Your Own Data

```bash
# Generate spectrograms from raw magnetometer data
python scripts/generate_spectrograms.py \
    --input data/raw/ \
    --output data/spectrograms/ \
    --window 6h \
    --overlap 0
```

## Training Details

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-4 |
| Weight Decay | 0.05 |
| Batch Size | 32 |
| Epochs | 50 (early stopping) |
| LR Schedule | Cosine with warmup |
| Warmup Epochs | 5 |

### Data Augmentation

- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness=0.2, contrast=0.2)
- Random affine translation (10%)
- Random erasing (p=0.1)

### Loss Function

Multi-task weighted cross-entropy:
```
L = L_magnitude + 0.5 × L_azimuth
```

Class weights computed using inverse frequency balancing.

## Validation Methods

### Leave-One-Event-Out (LOEO)

Tests temporal generalization by holding out all samples from specific earthquake events:

```
Fold 1: Train on events 2-301, Test on event 1
Fold 2: Train on events 1,3-301, Test on event 2
...
Fold 10: Train on events 1-291, Test on events 292-301
```

### Leave-One-Station-Out (LOSO)

Tests spatial generalization by holding out all samples from specific stations:

```
Fold 1: Train on 8 stations, Test on GTO
Fold 2: Train on 8 stations, Test on LUT
...
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{convnext_earthquake_2026,
  title={ConvNeXt-Based Multi-Task Learning for Earthquake Precursor Detection from Geomagnetic Spectrograms},
  author={Your Name et al.},
  journal={Journal Name},
  year={2026},
  volume={XX},
  pages={XX-XX},
  doi={XX.XXXX/XXXXX}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- ConvNeXt architecture: Liu et al., "A ConvNet for the 2020s" (CVPR 2022)
- Indonesian Agency for Meteorology, Climatology and Geophysics (BMKG) for magnetometer data
- PyTorch team for the deep learning framework

## Contact

For questions or collaboration, please open an issue or contact: your.email@example.com
