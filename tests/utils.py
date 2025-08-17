import hashlib
import os
import struct
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

OUTPUT_DIR = os.path.join(".", "images")  # Ensure this directory exists or create it
os.makedirs(OUTPUT_DIR, exist_ok=True)

def _to_normalized_array(obj):
    """Convert PIL Image or numpy array to normalized float32 array in [0,1] range."""
    if isinstance(obj, Image.Image):
        # Convert PIL Image to numpy array
        arr = np.array(obj.convert("RGB"))
        # Normalize to [0,1] range
        if arr.dtype != np.float32:
            if np.issubdtype(arr.dtype, np.integer):
                max_val = np.iinfo(arr.dtype).max
                arr = arr.astype(np.float32) / max_val
            else:
                arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
        return arr
    elif isinstance(obj, np.ndarray):
        return np.clip(obj.astype(np.float32), 0.0, 1.0)
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(obj)}")

def save_vpm_image(vpm, title: str, filename: str):
    """Save VPM as image with proper handling of both array and PIL Image types."""
    # Convert to normalized array for consistent processing
    arr = _to_normalized_array(vpm)
    
    # Handle 3D arrays (RGB) by converting to grayscale if needed
    if arr.ndim == 3:
        if arr.shape[2] == 3:
            # Convert RGB to grayscale
            arr = 0.2989 * arr[:,:,0] + 0.5870 * arr[:,:,1] + 0.1140 * arr[:,:,2]
        else:
            # Take first channel
            arr = arr[:,:,0]
    
    # Create and save image
    plt.figure(figsize=(6, 6))
    plt.imshow(arr, cmap='gray', vmin=0, vmax=1)
    plt.title(title)
    plt.colorbar(label='Normalized Score')
    plt.xlabel('Metrics (sorted)')
    plt.ylabel('Documents (sorted)')
    
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved VPM image: {filepath}")



def generate_synthetic_data(num_docs: int = 100, num_metrics: int = 50) -> Tuple[np.ndarray, List[str]]:
    """Generate synthetic score data for demonstration"""
    # Create realistic score distributions
    scores = np.zeros((num_docs, num_metrics))
    
    # Uncertainty: higher for early documents
    scores[:, 0] = np.linspace(0.9, 0.1, num_docs)
    
    # Size: random but correlated with uncertainty
    scores[:, 1] = 0.5 + 0.5 * np.random.rand(num_docs) - 0.3 * scores[:, 0]
    
    # Quality: higher for later documents
    scores[:, 2] = np.linspace(0.2, 0.9, num_docs)
    
    # Novelty: random
    scores[:, 3] = np.random.rand(num_docs)
    
    # Coherence: correlated with quality
    scores[:, 4] = scores[:, 2] * 0.7 + 0.3 * np.random.rand(num_docs)
    
    # Fill remaining metrics with random values
    for i in range(5, num_metrics):
        scores[:, i] = np.random.rand(num_docs)
    
    # Ensure values are in [0,1] range
    scores = np.clip(scores, 0, 1)
    
    # Create metric names
    metric_names = [
        "uncertainty", "size", "quality", "novelty", "coherence",
        "relevance", "diversity", "complexity", "readability", "accuracy"
    ]
    # Add numbered metrics for the rest
    for i in range(10, num_metrics):
        metric_names.append(f"metric_{i}")
    
    return scores[:num_docs, :num_metrics], metric_names[:num_metrics]



def save_vpm_montage(Y_list: list, title: str, filename: str):
    """
    Creates and saves a single image containing a montage of all VPM matrices.
    """
    num_timesteps = len(Y_list)
    
    # Calculate grid dimensions (e.g., 2x3 for 6 time steps)
    rows = int(np.ceil(np.sqrt(num_timesteps)))
    cols = int(np.ceil(num_timesteps / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    fig.suptitle(title, fontsize=16, y=1.02)
    
    for i, ax in enumerate(axes.flat):
        if i < num_timesteps:
            im = ax.imshow(Y_list[i], cmap='viridis', aspect='auto')
            ax.set_title(f"Time {i+1}")
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.axis('off')  # Turn off unused subplots
    
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Mass')
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(filename)
    plt.close()


def compute_task_hash(metric_names: List[str], weights: Dict[str,float]) -> int:
    h = hashlib.blake2s(digest_size=4)
    for name in metric_names:
        w = float(max(0.0, min(1.0, weights.get(name, 0.0))))
        h.update(name.encode('utf-8') + b'\0' + struct.pack('>f', w))
    return int.from_bytes(h.digest(), 'big')  # 32-bit
