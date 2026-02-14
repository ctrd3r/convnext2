#!/usr/bin/env python3
"""
Download pretrained models for ConvNeXt earthquake precursor detection.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model loeo_best
"""

import argparse
import os
import sys
from pathlib import Path
import hashlib
import urllib.request
from tqdm import tqdm

# Model URLs (replace with actual URLs when hosting)
MODEL_URLS = {
    'convnext_loeo_best': {
        'url': 'https://github.com/ctrd3r/convnext2/releases/download/v1.0/convnext_loeo_best.pth',
        'sha256': 'PLACEHOLDER_HASH',
        'size_mb': 115,
        'description': 'Best model from LOEO cross-validation (97.53% magnitude accuracy)'
    },
    'convnext_tiny_pretrained': {
        'url': 'https://github.com/ctrd3r/convnext2/releases/download/v1.0/convnext_tiny_pretrained.pth',
        'sha256': 'PLACEHOLDER_HASH',
        'size_mb': 115,
        'description': 'ConvNeXt-Tiny pretrained on earthquake spectrograms'
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path, expected_hash: str = None):
    """Download file with progress bar and optional hash verification."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f'Downloading to {output_path}...')
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, output_path, reporthook=t.update_to)
    
    # Verify hash
    if expected_hash and expected_hash != 'PLACEHOLDER_HASH':
        print('Verifying checksum...')
        sha256_hash = hashlib.sha256()
        with open(output_path, 'rb') as f:
            for byte_block in iter(lambda: f.read(4096), b''):
                sha256_hash.update(byte_block)
        
        if sha256_hash.hexdigest() != expected_hash:
            print(f'WARNING: Hash mismatch!')
            print(f'  Expected: {expected_hash}')
            print(f'  Got:      {sha256_hash.hexdigest()}')
        else:
            print('Checksum verified ✓')


def list_models():
    """List available models."""
    print('\nAvailable models:')
    print('-' * 60)
    for name, info in MODEL_URLS.items():
        print(f'\n{name}:')
        print(f'  Description: {info["description"]}')
        print(f'  Size: ~{info["size_mb"]} MB')


def main():
    parser = argparse.ArgumentParser(description='Download pretrained models')
    parser.add_argument('--model', type=str, help='Specific model to download')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    parser.add_argument('--list', action='store_true', help='List available models')
    args = parser.parse_args()
    
    if args.list:
        list_models()
        return
    
    output_dir = Path(args.output_dir)
    
    if args.model:
        # Download specific model
        if args.model not in MODEL_URLS:
            print(f'Unknown model: {args.model}')
            list_models()
            return
        
        info = MODEL_URLS[args.model]
        output_path = output_dir / f'{args.model}.pth'
        
        if output_path.exists():
            print(f'Model already exists: {output_path}')
            response = input('Overwrite? [y/N]: ')
            if response.lower() != 'y':
                return
        
        download_file(info['url'], output_path, info['sha256'])
        print(f'\nModel saved to: {output_path}')
    
    else:
        # Download all models
        print('Downloading all models...')
        
        for name, info in MODEL_URLS.items():
            output_path = output_dir / f'{name}.pth'
            
            if output_path.exists():
                print(f'\nSkipping {name} (already exists)')
                continue
            
            print(f'\n--- {name} ---')
            try:
                download_file(info['url'], output_path, info['sha256'])
            except Exception as e:
                print(f'Failed to download {name}: {e}')
                continue
        
        print('\nAll downloads complete!')
    
    # Create README in models directory
    readme_path = output_dir / 'README.md'
    if not readme_path.exists():
        with open(readme_path, 'w') as f:
            f.write('# Pretrained Models\n\n')
            f.write('This directory contains pretrained ConvNeXt models for earthquake precursor detection.\n\n')
            f.write('## Available Models\n\n')
            for name, info in MODEL_URLS.items():
                f.write(f'### {name}\n')
                f.write(f'- **Description**: {info["description"]}\n')
                f.write(f'- **Size**: ~{info["size_mb"]} MB\n\n')
            f.write('## Usage\n\n')
            f.write('```python\n')
            f.write('from src.model import ConvNeXtPrecursorModel\n\n')
            f.write('model = ConvNeXtPrecursorModel.load_pretrained("models/convnext_loeo_best.pth")\n')
            f.write('result = model.predict(image)\n')
            f.write('```\n')


if __name__ == '__main__':
    main()
