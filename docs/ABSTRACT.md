# Abstract

## Earthquake Precursor Detection using ConvNeXt: A Modern Convolutional Approach for ULF Geomagnetic Signal Classification

**Last Updated**: 6 February 2026

### Abstract

Earthquake precursor detection from Ultra-Low Frequency (ULF) geomagnetic signals remains a challenging task in seismology. This study presents a novel application of ConvNeXt, a modern convolutional neural network architecture that incorporates design principles from Vision Transformers while maintaining computational efficiency. We developed a multi-task learning framework for simultaneous earthquake magnitude classification (4 classes: Moderate, Medium, Large, Normal) and azimuth direction estimation (9 classes including Normal) from spectrogram representations of ULF signals.

The ConvNeXt-Tiny model (28.6M parameters) was trained and validated using Leave-One-Event-Out (LOEO) 10-fold cross-validation on a unified dataset of 1,972 spectrograms from Indonesian geomagnetic stations, covering earthquake events from 2018-2025. Key architectural innovations include patchify stem with 4×4 non-overlapping convolutions, inverted bottleneck design with 7×7 depthwise convolutions, and Layer Normalization for improved training stability.

**Results**: The ConvNeXt-Tiny model achieved **97.53% ± 0.96%** magnitude classification accuracy and **69.30% ± 5.74%** azimuth classification accuracy across 10 LOEO folds. These results are comparable to EfficientNet-B0 (97.53% ± 0.96% magnitude, 69.51% ± 5.65% azimuth) while offering a more modern architectural design. The model demonstrated consistent performance across all folds, with magnitude accuracy ranging from 95.56% to 98.15% and azimuth accuracy from 57.78% to 82.00%.

Notably, while the azimuth accuracy of 69.30% may appear moderate, it represents a **6.2-fold improvement** over random guessing (11.11% for 9 classes), with an estimated Matthews Correlation Coefficient (MCC) of 0.69, demonstrating that the model captures meaningful directional patterns in the ULF signals.

This work contributes to the growing body of research on deep learning applications for earthquake early warning systems and demonstrates the potential of modern CNN architectures for geophysical signal analysis.

### Keywords

ConvNeXt, earthquake precursor, ULF geomagnetic signals, deep learning, multi-task learning, spectrogram classification, seismology, early warning system, LOEO cross-validation

### Highlights

- First application of ConvNeXt architecture for earthquake precursor detection
- Multi-task learning for simultaneous magnitude and azimuth classification
- **97.53% ± 0.96%** magnitude accuracy via LOEO 10-fold cross-validation
- **69.30% ± 5.74%** azimuth accuracy, comparable to EfficientNet-B0
- Comprehensive comparison with VGG16, EfficientNet-B0 models
- Modern CNN design incorporating Vision Transformer principles

### Graphical Abstract

```
┌─────────────────────────────────────────────────────────────────┐
│                    EARTHQUAKE PRECURSOR DETECTION               │
│                      using ConvNeXt Architecture                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ULF Signal → Spectrogram → ConvNeXt → Magnitude + Azimuth     │
│                                                                 │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────────┐  │
│  │ H,D,Z   │ →  │ FFT +   │ →  │ConvNeXt │ →  │ M4.0-4.9    │  │
│  │Components│    │ STFT    │    │ Tiny    │    │ M5.0-5.9    │  │
│  │ 1 hour  │    │ 224×224 │    │ 28.6M   │    │ M6.0-6.9    │  │
│  └─────────┘    └─────────┘    └─────────┘    │ Normal      │  │
│                                               │ + Direction  │  │
│                                               └─────────────┘  │
│                                                                 │
│  Key Features:                                                  │
│  • Patchify stem (4×4 conv)                                    │
│  • Inverted bottleneck                                         │
│  • 7×7 depthwise convolutions                                  │
│  • Layer Normalization                                         │
│  • GELU activation                                             │
│                                                                 │
│  Results: Mag 97.53%±0.96% | Azi 69.30%±5.74% | LOEO 10-fold  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Word Count

- Abstract: ~280 words
- Full manuscript: ~5,000-6,000 words (target)
