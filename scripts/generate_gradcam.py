#!/usr/bin/env python3
"""
Generate GradCAM visualizations for ConvNeXt model.

Usage:
    python scripts/generate_gradcam.py --model models/best.pth --image sample.png
    python scripts/generate_gradcam.py --model models/best.pth --data-dir data/test --n-samples 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

from src.model import ConvNeXtMultiTask
from src.dataset import get_transforms


def get_gradcam(model, image_tensor, target_layer, target_class=None):
    """
    Generate GradCAM heatmap.
    
    Args:
        model: ConvNeXt model
        image_tensor: Input image tensor (1, 3, H, W)
        target_layer: Target layer for GradCAM
        target_class: Target class index (None for predicted class)
    
    Returns:
        GradCAM heatmap as numpy array
    """
    model.eval()
    
    # Hook for gradients
    gradients = []
    activations = []
    
    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])
    
    def forward_hook(module, input, output):
        activations.append(output)
    
    # Register hooks
    handle_backward = target_layer.register_full_backward_hook(backward_hook)
    handle_forward = target_layer.register_forward_hook(forward_hook)
    
    # Forward pass
    mag_out, azi_out = model(image_tensor)
    
    # Get target class
    if target_class is None:
        target_class = torch.argmax(mag_out, dim=1).item()
    
    # Backward pass
    model.zero_grad()
    one_hot = torch.zeros_like(mag_out)
    one_hot[0, target_class] = 1
    mag_out.backward(gradient=one_hot, retain_graph=True)
    
    # Remove hooks
    handle_backward.remove()
    handle_forward.remove()
    
    # Compute GradCAM
    grads = gradients[0].cpu().numpy()[0]
    acts = activations[0].cpu().detach().numpy()[0]
    
    # Global average pooling of gradients
    weights = np.mean(grads, axis=(1, 2))
    
    # Weighted combination of activation maps
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    for i, w in enumerate(weights):
        cam += w * acts[i]
    
    # ReLU
    cam = np.maximum(cam, 0)
    
    # Normalize
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    
    return cam


def overlay_heatmap(image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on image."""
    # Resize heatmap to image size
    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert to colormap
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay
    overlay = np.float32(heatmap_colored) * alpha + np.float32(image) * (1 - alpha)
    overlay = np.uint8(np.clip(overlay, 0, 255))
    
    return overlay


def visualize_gradcam(model, image_path, output_path, device, mag_classes):
    """Generate and save GradCAM visualization."""
    # Load and preprocess image
    transform = get_transforms(is_training=False)
    original_image = Image.open(image_path).convert('RGB')
    image_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # Get target layer (last convolutional layer)
    target_layer = model.backbone.features[-1]
    
    # Generate GradCAM
    heatmap = get_gradcam(model, image_tensor, target_layer)
    
    # Get prediction
    with torch.no_grad():
        mag_out, azi_out = model(image_tensor)
        mag_pred = torch.argmax(mag_out, dim=1).item()
        mag_prob = torch.softmax(mag_out, dim=1)[0, mag_pred].item()
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Spectrogram')
    axes[0].axis('off')
    
    # Heatmap
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    original_np = np.array(original_image)
    overlay = overlay_heatmap(original_np, heatmap)
    axes[2].imshow(overlay)
    axes[2].set_title(f'Prediction: {mag_classes[mag_pred]} ({mag_prob:.2%})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return mag_classes[mag_pred], mag_prob


def main():
    parser = argparse.ArgumentParser(description='Generate GradCAM visualizations')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint')
    parser.add_argument('--image', type=str, help='Single image path')
    parser.add_argument('--data-dir', type=str, help='Directory with images')
    parser.add_argument('--n-samples', type=int, default=10, help='Number of samples')
    parser.add_argument('--output', type=str, default='gradcam_results', help='Output directory')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print(f'Loading model from {args.model}...')
    checkpoint = torch.load(args.model, map_location=device)
    config = checkpoint.get('config', {})
    
    model = ConvNeXtMultiTask(
        variant=config.get('model_variant', config.get('variant', 'tiny')),
        pretrained=False,
        num_mag_classes=config.get('num_mag_classes', 4),
        num_azi_classes=config.get('num_azi_classes', 9)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    mag_classes = ['Large', 'Medium', 'Moderate', 'Normal']
    
    if args.image:
        # Single image
        output_path = output_dir / f'gradcam_{Path(args.image).stem}.png'
        pred, prob = visualize_gradcam(model, args.image, output_path, device, mag_classes)
        print(f'Prediction: {pred} ({prob:.2%})')
        print(f'Saved to: {output_path}')
    
    elif args.data_dir:
        # Multiple images
        data_dir = Path(args.data_dir)
        images = list(data_dir.glob('*.png')) + list(data_dir.glob('*.jpg'))
        
        if len(images) > args.n_samples:
            import random
            images = random.sample(images, args.n_samples)
        
        print(f'Processing {len(images)} images...')
        
        for img_path in images:
            output_path = output_dir / f'gradcam_{img_path.stem}.png'
            try:
                pred, prob = visualize_gradcam(model, img_path, output_path, device, mag_classes)
                print(f'{img_path.name}: {pred} ({prob:.2%})')
            except Exception as e:
                print(f'Error processing {img_path.name}: {e}')
        
        print(f'\nResults saved to: {output_dir}')
    
    else:
        print('Please provide --image or --data-dir')


if __name__ == '__main__':
    main()
