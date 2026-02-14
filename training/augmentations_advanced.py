#!/usr/bin/env python3
"""
Advanced Augmentation Pipeline for Seismic Spectrograms
Includes GridMask, noise injection, and spectrogram-specific augmentations
"""

import torch
import torch.nn as nn
import numpy as np
import random
from PIL import Image, ImageFilter


class GridMask:
    """GridMask augmentation for better generalization"""
    def __init__(self, ratio=0.6, prob=0.3):
        self.ratio = ratio
        self.prob = prob
    
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        
        if isinstance(img, Image.Image):
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            was_pil = True
        else:
            was_pil = False
        
        c, h, w = img.shape
        
        # Grid size
        d = min(h, w) // 4
        
        # Create mask
        mask = torch.ones_like(img)
        for i in range(0, h, d * 2):
            for j in range(0, w, d * 2):
                mask[:, i:min(i+d, h), j:min(j+d, w)] = 0
        
        result = img * mask
        
        if was_pil:
            result = Image.fromarray((result.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        
        return result


class NoiseInjection:
    """Add Gaussian noise to simulate sensor noise"""
    def __init__(self, snr_range=(10, 30), prob=0.2):
        self.snr_range = snr_range
        self.prob = prob
    
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        
        if isinstance(img, Image.Image):
            img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            was_pil = True
        else:
            was_pil = False
        
        # Random SNR
        snr_db = random.uniform(*self.snr_range)
        snr = 10 ** (snr_db / 10)
        
        # Calculate noise power
        signal_power = torch.mean(img ** 2)
        noise_power = signal_power / snr
        
        # Add noise
        noise = torch.randn_like(img) * torch.sqrt(noise_power)
        result = torch.clamp(img + noise, 0, 1)
        
        if was_pil:
            result = Image.fromarray((result.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
        
        return result


class FrequencyMask:
    """Mask random frequency bands (horizontal stripes)"""
    def __init__(self, max_mask_ratio=0.15, prob=0.3):
        self.max_mask_ratio = max_mask_ratio
        self.prob = prob
    
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        h, w = img_array.shape[:2]
        mask_height = int(h * random.uniform(0.05, self.max_mask_ratio))
        mask_start = random.randint(0, h - mask_height)
        
        img_array[mask_start:mask_start+mask_height, :] = 0
        
        if isinstance(img, Image.Image):
            return Image.fromarray(img_array)
        return img_array


class TimeMask:
    """Mask random time segments (vertical stripes)"""
    def __init__(self, max_mask_ratio=0.1, prob=0.3):
        self.max_mask_ratio = max_mask_ratio
        self.prob = prob
    
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        
        if isinstance(img, Image.Image):
            img_array = np.array(img)
        else:
            img_array = img
        
        h, w = img_array.shape[:2]
        mask_width = int(w * random.uniform(0.05, self.max_mask_ratio))
        mask_start = random.randint(0, w - mask_width)
        
        img_array[:, mask_start:mask_start+mask_width] = 0
        
        if isinstance(img, Image.Image):
            return Image.fromarray(img_array)
        return img_array


class GainAugmentation:
    """Adjust overall amplitude/gain"""
    def __init__(self, gain_range=(0.7, 1.3), prob=0.3):
        self.gain_range = gain_range
        self.prob = prob
    
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        
        gain = random.uniform(*self.gain_range)
        
        if isinstance(img, Image.Image):
            img_array = np.array(img).astype(np.float32)
            img_array = np.clip(img_array * gain, 0, 255).astype(np.uint8)
            return Image.fromarray(img_array)
        else:
            return torch.clamp(img * gain, 0, 1)


class SpectrogramBlur:
    """Apply Gaussian blur to simulate low-resolution sensors"""
    def __init__(self, kernel_size_range=(3, 7), prob=0.1):
        self.kernel_size_range = kernel_size_range
        self.prob = prob
    
    def __call__(self, img):
        if random.random() > self.prob:
            return img
        
        if not isinstance(img, Image.Image):
            return img
        
        kernel_size = random.choice(range(*self.kernel_size_range, 2))
        return img.filter(ImageFilter.GaussianBlur(radius=kernel_size//2))


class AdvancedAugmentation:
    """Combined advanced augmentations for seismic spectrograms"""
    def __init__(self, config=None):
        if config is None:
            config = {}
        
        self.augmentations = []
        
        # GridMask
        if config.get('gridmask_enabled', True):
            self.augmentations.append(GridMask(
                ratio=config.get('gridmask_ratio', 0.6),
                prob=config.get('gridmask_prob', 0.3)
            ))
        
        # Noise injection
        if config.get('noise_enabled', True):
            self.augmentations.append(NoiseInjection(
                snr_range=config.get('noise_snr_range', (10, 30)),
                prob=config.get('noise_prob', 0.2)
            ))
        
        # Frequency mask
        if config.get('freq_mask_enabled', True):
            self.augmentations.append(FrequencyMask(
                max_mask_ratio=config.get('freq_mask_ratio', 0.15),
                prob=config.get('freq_mask_prob', 0.3)
            ))
        
        # Time mask
        if config.get('time_mask_enabled', True):
            self.augmentations.append(TimeMask(
                max_mask_ratio=config.get('time_mask_ratio', 0.1),
                prob=config.get('time_mask_prob', 0.3)
            ))
        
        # Gain augmentation
        if config.get('gain_enabled', True):
            self.augmentations.append(GainAugmentation(
                gain_range=config.get('gain_range', (0.7, 1.3)),
                prob=config.get('gain_prob', 0.3)
            ))
        
        # Blur
        if config.get('blur_enabled', True):
            self.augmentations.append(SpectrogramBlur(
                kernel_size_range=config.get('blur_kernel_range', (3, 7)),
                prob=config.get('blur_prob', 0.1)
            ))
    
    def __call__(self, img):
        for aug in self.augmentations:
            img = aug(img)
        return img


class ProgressiveAugmentation:
    """Progressive augmentation that increases strength over epochs"""
    def __init__(self, base_config, schedule):
        """
        Args:
            base_config: Base augmentation configuration
            schedule: Dict mapping epoch ranges to strength multipliers
                     e.g., {'0-10': 0.3, '10-30': 0.5, '30-50': 0.7}
        """
        self.base_config = base_config
        self.schedule = schedule
        self.current_epoch = 0
        self.augmentation = AdvancedAugmentation(base_config)
    
    def set_epoch(self, epoch):
        """Update augmentation strength based on current epoch"""
        self.current_epoch = epoch
        
        # Find matching schedule
        strength = 1.0
        for epoch_range, s in self.schedule.items():
            start, end = map(int, epoch_range.split('-'))
            if start <= epoch < end:
                strength = s
                break
        
        # Update probabilities based on strength
        config = self.base_config.copy()
        for key in config:
            if key.endswith('_prob'):
                config[key] = config[key] * strength
        
        self.augmentation = AdvancedAugmentation(config)
    
    def __call__(self, img):
        return self.augmentation(img)


# MixUp and CutMix implementations
class MixUp:
    """MixUp augmentation"""
    def __init__(self, alpha=0.4, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_x, batch_y_mag, batch_y_azi):
        if random.random() > self.prob:
            return batch_x, batch_y_mag, batch_y_azi
        
        batch_size = batch_x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        index = torch.randperm(batch_size).to(batch_x.device)
        
        mixed_x = lam * batch_x + (1 - lam) * batch_x[index]
        
        # For classification, we keep original labels
        # In training loop, loss will be: lam * loss(pred, y_a) + (1-lam) * loss(pred, y_b)
        return mixed_x, batch_y_mag, batch_y_azi, batch_y_mag[index], batch_y_azi[index], lam


class CutMix:
    """CutMix augmentation"""
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch_x, batch_y_mag, batch_y_azi):
        if random.random() > self.prob:
            return batch_x, batch_y_mag, batch_y_azi
        
        batch_size = batch_x.size(0)
        lam = np.random.beta(self.alpha, self.alpha)
        
        index = torch.randperm(batch_size).to(batch_x.device)
        
        # Get random box
        _, _, h, w = batch_x.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix
        batch_x[:, :, bby1:bby2, bbx1:bbx2] = batch_x[index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (w * h))
        
        return batch_x, batch_y_mag, batch_y_azi, batch_y_mag[index], batch_y_azi[index], lam


if __name__ == '__main__':
    # Test augmentations
    print("Testing Advanced Augmentations...")
    
    # Create dummy image
    img = Image.new('RGB', (224, 224), color='white')
    
    # Test each augmentation
    print("\n1. Testing GridMask...")
    gridmask = GridMask(ratio=0.6, prob=1.0)
    img_grid = gridmask(img)
    print("✅ GridMask passed")
    
    print("\n2. Testing NoiseInjection...")
    noise = NoiseInjection(snr_range=(10, 30), prob=1.0)
    img_noise = noise(img)
    print("✅ NoiseInjection passed")
    
    print("\n3. Testing FrequencyMask...")
    freq_mask = FrequencyMask(max_mask_ratio=0.15, prob=1.0)
    img_freq = freq_mask(img)
    print("✅ FrequencyMask passed")
    
    print("\n4. Testing TimeMask...")
    time_mask = TimeMask(max_mask_ratio=0.1, prob=1.0)
    img_time = time_mask(img)
    print("✅ TimeMask passed")
    
    print("\n5. Testing GainAugmentation...")
    gain = GainAugmentation(gain_range=(0.7, 1.3), prob=1.0)
    img_gain = gain(img)
    print("✅ GainAugmentation passed")
    
    print("\n6. Testing AdvancedAugmentation...")
    config = {
        'gridmask_enabled': True,
        'noise_enabled': True,
        'freq_mask_enabled': True,
        'time_mask_enabled': True,
        'gain_enabled': True,
        'blur_enabled': True
    }
    aug = AdvancedAugmentation(config)
    img_aug = aug(img)
    print("✅ AdvancedAugmentation passed")
    
    print("\n7. Testing MixUp...")
    mixup = MixUp(alpha=0.4, prob=1.0)
    batch_x = torch.randn(4, 3, 224, 224)
    batch_y_mag = torch.randint(0, 4, (4,))
    batch_y_azi = torch.randint(0, 9, (4,))
    result = mixup(batch_x, batch_y_mag, batch_y_azi)
    print(f"MixUp output shapes: {result[0].shape}")
    print("✅ MixUp passed")
    
    print("\n8. Testing CutMix...")
    cutmix = CutMix(alpha=1.0, prob=1.0)
    result = cutmix(batch_x, batch_y_mag, batch_y_azi)
    print(f"CutMix output shapes: {result[0].shape}")
    print("✅ CutMix passed")
    
    print("\n✅ All augmentation tests passed!")
