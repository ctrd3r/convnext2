"""
Utility functions for ConvNeXt earthquake precursor detection.
"""

import os
import json
import yaml
import logging
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

import numpy as np
import torch


def setup_logging(
    log_dir: Optional[str] = None,
    log_level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        log_dir: Directory for log files
        log_level: Logging level
        log_file: Specific log file name
        
    Returns:
        Logger instance
    """
    logger = logging.getLogger('convnext_precursor')
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_dir or log_file:
        if log_file:
            log_path = Path(log_file)
        else:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_path = log_dir / f'training_{timestamp}.log'
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(log_level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    config_path = Path(config_path)
    
    if config_path.suffix in ['.yaml', '.yml']:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    elif config_path.suffix == '.json':
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    return config


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to file."""
    path = Path(path)
    
    if path.suffix in ['.yaml', '.yml']:
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Compute inverse frequency class weights.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Tensor of class weights
    """
    unique, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    n_classes = len(unique)
    
    weights = total / (n_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


class EarlyStopping:
    """Early stopping handler."""
    
    def __init__(
        self, 
        patience: int = 10, 
        min_delta: float = 0.0,
        mode: str = 'max'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta
        
        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_time(seconds: float) -> str:
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"
