# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-02-05

### Added
- Initial release of ConvNeXt Earthquake Precursor Detection
- ConvNeXt-Tiny multi-task model implementation
- LOEO (Leave-One-Event-Out) cross-validation support
- LOSO (Leave-One-Station-Out) cross-validation support
- GradCAM visualization for model interpretability
- Comprehensive data augmentation pipeline
- Multi-task loss functions (Focal Loss, Label Smoothing)
- Jupyter notebooks for data exploration and analysis
- Unit tests for model, dataset, and inference modules
- Documentation and usage examples

### Model Performance
- Magnitude Classification: 97.53% ± 0.96% (LOEO)
- Azimuth Classification: 69.51% ± 5.65% (LOEO)

### Technical Details
- PyTorch 2.0+ compatible
- Mixed precision training support
- Cosine learning rate schedule with warmup
- Early stopping with patience
- Class-weighted loss for imbalanced data

## Future Plans

- [ ] Add more backbone options (ConvNeXt-Small, ConvNeXt-Base)
- [ ] Implement attention visualization
- [ ] Add real-time inference API
- [ ] Support for additional magnetometer stations
- [ ] Multi-GPU training support
