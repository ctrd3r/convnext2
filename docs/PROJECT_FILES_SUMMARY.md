# ConvNeXt Project Files Summary

This document lists all files related to the ConvNeXt earthquake precursor detection project.

## Project Overview

The ConvNeXt earthquake precursor detection project uses deep learning (ConvNeXt-Tiny architecture) to classify earthquake precursors from geomagnetic spectrograms. The model performs multi-task classification for:
- **Magnitude**: Large (M>=6.0), Medium (5.0-6.0), Moderate (4.0-5.0), Normal
- **Azimuth**: N, NE, E, SE, S, SW, W, NW, Normal

## Results

### LOEO 10-Fold Cross-Validation
- Magnitude Accuracy: 97.53% ± 0.96%
- Azimuth Accuracy: 69.51% ± 5.65%

### Baseline Finetuning (Latest)
- Best Validation Accuracy: 63.72%
- Test Accuracy: 63.88%
- Training Time: 42.9 minutes (3 epochs)

### Enhanced Model (Under Development)
- Target: 70-80% accuracy with GPU training
- Features: CBAM Attention, Hierarchical Azimuth, Focal Loss

---

## File Structure

### Core Source Files (src/)

| File | Description |
|------|-------------|
| `src/model.py` | ConvNeXtMultiTask model implementation |
| `src/dataset.py` | Dataset loading and preprocessing |
| `src/inference.py` | Inference utilities |
| `src/losses.py` | Loss functions |
| `src/transforms.py` | Data augmentation transforms |
| `src/utils.py` | Utility functions |

### Training Scripts (scripts/)

| File | Description |
|------|-------------|
| `scripts/train.py` | Standard training script |
| `scripts/train_loeo.py` | LOEO cross-validation training |
| `scripts/evaluate.py` | Model evaluation |
| `scripts/generate_gradcam.py` | GradCAM visualization |
| `scripts/prepare_data.py` | Data preparation |
| `scripts/download_models.py` | Model download utility |

### Enhanced Training (training/)

| File | Description |
|------|-------------|
| `training/convnext_enhanced.py` | Enhanced model with CBAM attention |
| `training/augmentations_advanced.py` | Advanced augmentations (GridMask, MixUp, CutMix) |
| `training/train_convnext_enhanced.py` | Enhanced training script with Focal Loss |
| `training/config_improved.json` | GPU-optimized configuration |
| `training/evaluate_enhanced_model.py` | Enhanced evaluation tools |
| `training/run_enhanced_training.py` | Easy launcher script |
| `training/README.md` | Implementation documentation |

### Configuration Files (configs/)

| File | Description |
|------|-------------|
| `configs/convnext_tiny.yaml` | ConvNeXt-Tiny configuration |
| `configs/loeo_validation.yaml` | LOEO validation configuration |

### Publication Materials (publication_convnext/)

| File | Description |
|------|-------------|
| `publication_convnext/MANUSCRIPT_DRAFT.md` | Manuscript draft |
| `publication_convnext/ABSTRACT.md` | Paper abstract |
| `publication_convnext/METHODOLOGY.md` | Methodology description |
| `publication_convnext/MODEL_ARCHITECTURE.md` | Architecture details |
| `publication_convnext/COMPARISON_WITH_OTHER_MODELS.md` | Model comparison |
| `publication_convnext/MCC_ANALYSIS.json` | MCC analysis results |
| `publication_convnext/TRAINING_REPORT.md` | Training report |
| `publication_convnext/figures/` | Publication figures |

### LOEO Results (loeo_convnext_results/)

| File | Description |
|------|-------------|
| `loeo_convnext_results/loeo_convnext_final_results.json` | Final LOEO results |
| `loeo_convnext_results/fold_*.json` | Per-fold results (10 folds) |

### Experiment Results (experiments_convnext/)

| Folder | Description |
|--------|-------------|
| `experiments_convnext/convnext_tiny_*/` | Standard experiments |
| `experiments_convnext/exp_v3_*/` | Version 3 experiments |
| `experiments_convnext/finetune_v3_gpu_*/` | GPU finetuning results (baseline) |
| `experiments_convnext/enhanced_*/` | Enhanced training experiments |

### Best Model Checkpoints

| Path | Description |
|------|-------------|
| `experiments_convnext/convnext_tiny_20260205_100924/best_model.pth` | Best LOEO model |
| `experiments_convnext/finetune_v3_gpu_20260214_143726/checkpoint_latest.pth` | Latest finetuned checkpoint |
| `convnext_production_model/best_convnext_model.pth` | Production model |

---

## Quick Start

### Standard Training
```bash
python scripts/train.py --config configs/convnext_tiny.yaml
```

### Enhanced Training (GPU Recommended)
```bash
cd training
python run_enhanced_training.py
```

### LOEO Cross-Validation
```bash
python scripts/train_loeo.py --config configs/loeo_validation.yaml
```

### Evaluation
```bash
python scripts/evaluate.py --model models/convnext_loeo_best.pth --data data/test
```

---

## Dependencies

See `requirements.txt` for full list:
- torch>=2.0.0
- torchvision>=0.15.0
- timm>=0.9.0
- pandas>=1.5.0
- numpy>=1.21.0
- Pillow>=9.0.0
- matplotlib>=3.5.0
- scikit-learn>=1.0.0

---

## Dataset

The model is trained on geomagnetic field spectrograms:
- **Source**: Indonesian magnetometer stations (BMKG)
- **Time Period**: 2018-2023
- **Events**: 301 earthquake events (M>=4.0)
- **Samples**: ~1,972 spectrogram images (6-hour windows)
- **Image Size**: 224x224 RGB

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{convnext_earthquake_2026,
  title={ConvNeXt-Based Multi-Task Learning for Earthquake Precursor Detection from Geomagnetic Spectrograms},
  author={Your Name et al.},
  journal={Journal of Geophysics},
  year={2026}
}
```

---

## License

MIT License - see LICENSE file for details.
