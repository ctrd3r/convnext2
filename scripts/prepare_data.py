#!/usr/bin/env python3
"""
Data Preparation Script for ConvNeXt Earthquake Precursor Detection

This script prepares the dataset for training by:
1. Organizing spectrograms into the correct directory structure
2. Creating train/val/test splits
3. Generating metadata files

Usage:
    python scripts/prepare_data.py --input /path/to/raw/data --output data/
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm


def create_splits(metadata_df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Create stratified train/val/test splits.
    
    Args:
        metadata_df: DataFrame with metadata
        test_size: Fraction for test set
        val_size: Fraction for validation set
        random_state: Random seed
        
    Returns:
        train_df, val_df, test_df
    """
    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        metadata_df,
        test_size=test_size,
        stratify=metadata_df['magnitude_class'],
        random_state=random_state
    )
    
    # Second split: train vs val
    val_fraction = val_size / (1 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_fraction,
        stratify=train_val_df['magnitude_class'],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def create_loeo_folds(metadata_df, n_folds=10, random_state=42):
    """
    Create Leave-One-Event-Out folds.
    
    Args:
        metadata_df: DataFrame with 'event_id' column
        n_folds: Number of folds
        random_state: Random seed
        
    Returns:
        List of (train_df, test_df) tuples
    """
    np.random.seed(random_state)
    
    # Get unique events
    events = metadata_df['event_id'].unique()
    np.random.shuffle(events)
    
    # Split events into folds
    events_per_fold = len(events) // n_folds
    folds = []
    
    for i in range(n_folds):
        if i < n_folds - 1:
            test_events = events[i * events_per_fold:(i + 1) * events_per_fold]
        else:
            test_events = events[i * events_per_fold:]
        
        train_df = metadata_df[~metadata_df['event_id'].isin(test_events)]
        test_df = metadata_df[metadata_df['event_id'].isin(test_events)]
        
        folds.append((train_df, test_df))
    
    return folds


def prepare_dataset(input_dir, output_dir, metadata_file=None):
    """
    Prepare dataset for training.
    
    Args:
        input_dir: Directory containing raw spectrograms
        output_dir: Output directory for organized data
        metadata_file: Path to metadata CSV (optional)
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    (output_dir / 'spectrograms').mkdir(parents=True, exist_ok=True)
    (output_dir / 'metadata').mkdir(parents=True, exist_ok=True)
    
    # Load or create metadata
    if metadata_file and Path(metadata_file).exists():
        print(f"Loading metadata from {metadata_file}")
        metadata_df = pd.read_csv(metadata_file)
    else:
        print("Creating metadata from directory structure...")
        # Assume directory structure: input_dir/class/images
        records = []
        for img_path in input_dir.glob('**/*.png'):
            records.append({
                'filename': img_path.name,
                'original_path': str(img_path),
                'magnitude_class': 'Unknown',
                'azimuth_class': 'Unknown',
                'event_id': 0
            })
        metadata_df = pd.DataFrame(records)
    
    print(f"Total samples: {len(metadata_df)}")
    
    # Copy images to output directory
    print("Copying spectrograms...")
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        if 'original_path' in row and Path(row['original_path']).exists():
            src = Path(row['original_path'])
            dst = output_dir / 'spectrograms' / row['filename']
            if not dst.exists():
                shutil.copy(src, dst)
    
    # Create splits
    print("Creating train/val/test splits...")
    train_df, val_df, test_df = create_splits(metadata_df)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    
    # Save splits
    train_df.to_csv(output_dir / 'metadata' / 'train_split.csv', index=False)
    val_df.to_csv(output_dir / 'metadata' / 'val_split.csv', index=False)
    test_df.to_csv(output_dir / 'metadata' / 'test_split.csv', index=False)
    metadata_df.to_csv(output_dir / 'metadata' / 'unified_metadata.csv', index=False)
    
    # Create LOEO folds
    if 'event_id' in metadata_df.columns:
        print("Creating LOEO folds...")
        folds = create_loeo_folds(metadata_df)
        
        loeo_dir = output_dir / 'metadata' / 'loeo_folds'
        loeo_dir.mkdir(exist_ok=True)
        
        for i, (train_fold, test_fold) in enumerate(folds):
            train_fold.to_csv(loeo_dir / f'fold_{i+1}_train.csv', index=False)
            test_fold.to_csv(loeo_dir / f'fold_{i+1}_test.csv', index=False)
        
        print(f"  Created {len(folds)} LOEO folds")
    
    # Create class mappings
    mag_classes = sorted(metadata_df['magnitude_class'].unique())
    azi_classes = sorted(metadata_df['azimuth_class'].unique())
    
    mappings = {
        'magnitude': {str(i): c for i, c in enumerate(mag_classes)},
        'azimuth': {str(i): c for i, c in enumerate(azi_classes)}
    }
    
    with open(output_dir / 'metadata' / 'class_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=2)
    
    # Print class distribution
    print("\nClass Distribution:")
    print("\nMagnitude:")
    print(metadata_df['magnitude_class'].value_counts())
    print("\nAzimuth:")
    print(metadata_df['azimuth_class'].value_counts())
    
    print(f"\nDataset prepared successfully in {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset for ConvNeXt training')
    parser.add_argument('--input', '-i', required=True, help='Input directory with raw data')
    parser.add_argument('--output', '-o', default='data/', help='Output directory')
    parser.add_argument('--metadata', '-m', help='Path to metadata CSV file')
    
    args = parser.parse_args()
    
    prepare_dataset(args.input, args.output, args.metadata)


if __name__ == '__main__':
    main()
