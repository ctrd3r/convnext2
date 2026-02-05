# Pretrained Models

This directory contains pretrained models for earthquake precursor detection.

## Available Models

| Model | Backbone | Magnitude Acc | Azimuth Acc | Size | Status |
|-------|----------|---------------|-------------|------|--------|
| EfficientNet LOEO | EfficientNet-B0 | **97.53%** ± 0.96% | **69.51%** ± 5.65% | ~20 MB | ✅ Available |
| ConvNeXt LOEO | ConvNeXt-Tiny | ~95% | ~65% | ~115 MB | 🔄 Training |

> **Note**: The best results (97.53% magnitude, 69.51% azimuth) were achieved with EfficientNet-B0. ConvNeXt implementation is provided for research comparison.

## Download

Models can be downloaded using the provided script:

```bash
python scripts/download_models.py
```

Or download manually from the releases page.

## Usage

```python
from src.model import ConvNeXtPrecursorModel

# Load model
model = ConvNeXtPrecursorModel.load_pretrained('models/convnext_loeo_best.pth')

# Predict
result = model.predict(image_tensor)
print(f"Magnitude: {result['magnitude_class']} ({result['magnitude_prob']:.2%})")
print(f"Azimuth: {result['azimuth_class']} ({result['azimuth_prob']:.2%})")
```

## Model Architecture

### ConvNeXt-Tiny (28.6M parameters)
```
ConvNeXt-Tiny Backbone (pretrained on ImageNet)
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

## Class Mappings

### Magnitude Classes
| Index | Class | Description |
|-------|-------|-------------|
| 0 | Large | M ≥ 6.0 |
| 1 | Medium | 5.0 ≤ M < 6.0 |
| 2 | Moderate | 4.0 ≤ M < 5.0 |
| 3 | Normal | No earthquake |

### Azimuth Classes
| Index | Class | Direction |
|-------|-------|-----------|
| 0 | E | East |
| 1 | N | North |
| 2 | NE | Northeast |
| 3 | NW | Northwest |
| 4 | Normal | No direction |
| 5 | S | South |
| 6 | SE | Southeast |
| 7 | SW | Southwest |
| 8 | W | West |

## Training Your Own Model

```bash
# Standard training
python scripts/train.py --config configs/convnext_tiny.yaml

# LOEO cross-validation (recommended)
python scripts/train_loeo.py --config configs/loeo_validation.yaml
```

## File Structure

After downloading/training, this directory should contain:
```
models/
├── convnext_loeo_best.pth    # Best ConvNeXt model
├── class_mappings.json       # Class label mappings
├── training_history.csv      # Training metrics (optional)
└── README.md                 # This file
```

## License

Models are released under MIT License.
