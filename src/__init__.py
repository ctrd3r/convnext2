"""
ConvNeXt Earthquake Precursor Detection

A deep learning system for earthquake precursor detection using 
ConvNeXt architecture and geomagnetic spectrogram analysis.
"""

__version__ = "1.0.0"
__author__ = "Earthquake Prediction Research Team"

from .model import ConvNeXtPrecursorModel, ConvNeXtMultiTask
from .dataset import EarthquakeDataset, get_transforms
from .inference import PrecursorPredictor, predict_spectrogram
from .utils import load_config, setup_logging

__all__ = [
    "ConvNeXtPrecursorModel",
    "ConvNeXtMultiTask", 
    "EarthquakeDataset",
    "get_transforms",
    "PrecursorPredictor",
    "predict_spectrogram",
    "load_config",
    "setup_logging",
]
