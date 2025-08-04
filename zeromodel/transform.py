"""
Transformation Pipeline

This module provides functions for dynamically transforming visual policy maps
to prioritize specific metrics for different tasks. This enables the same
underlying data to be used for multiple decision contexts.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import quantize


def transform_vpm(vpm: np.ndarray, 
                 metric_names: List[str],
                 target_metrics: List[str]) -> np.ndarray:
    """
    Transform visual policy map to prioritize specific metrics.
    
    Args:
        vpm: Visual policy map (RGB image)
        metric_names: Original metric names
        target_metrics: Metrics to prioritize
    
    Returns:
        Transformed VPM with target metrics moved to front
    """
    height, width, _ = vpm.shape
    n_metrics = width * 3  # 3 metrics per pixel
    
    # Extract metrics from image
    metrics = np.zeros((height, n_metrics))
    for i in range(height):
        for j in range(n_metrics):
            pixel_x = j // 3
            channel = j % 3
            if pixel_x < width:
                metrics[i, j] = vpm[i, pixel_x, channel] / 255.0
    
    # Find metric indices
    metric_indices = []
    for m in target_metrics:
        try:
            idx = metric_names.index(m)
            metric_indices.append(idx)
        except ValueError:
            continue  # Skip metrics not found
    
    # Create new column order
    remaining_indices = [i for i in range(len(metric_names)) 
                        if i not in metric_indices]
    new_order = metric_indices + remaining_indices
    
    # Reorder columns
    reordered = metrics[:, new_order]
    
    # Sort rows by first target metric
    if len(metric_indices) > 0:
        sort_key = reordered[:, 0]
        sorted_indices = np.argsort(sort_key)[::-1]
        transformed = reordered[sorted_indices]
    else:
        transformed = reordered
    
    # Re-encode as image
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(height):
        for j in range(n_metrics):
            pixel_x = j // 3
            channel = j % 3
            if pixel_x < width:
                img[i, pixel_x, channel] = int(transformed[i, j] * 255)
    
    return img

def get_critical_tile(vpm: np.ndarray, tile_size: int = 3) -> bytes:
    """
    Extract critical tile from visual policy map.
    
    Args:
        vpm: Visual policy map (RGB image)
        tile_size: Size of tile to extract (default 3x3)
    
    Returns:
        Compact byte representation of the tile
    """
    height, width, _ = vpm.shape
    
    # Get top-left section (most relevant documents & metrics)
    # Convert to compact byte format
    tile_bytes = bytearray()
    tile_bytes.append(tile_size)  # Width
    tile_bytes.append(tile_size)  # Height
    tile_bytes.append(0)  # X offset
    tile_bytes.append(0)  # Y offset
    
    # Add pixel data (1 byte per channel)
    for i in range(min(tile_size, height)):
        for j in range(min(tile_size * 3, width * 3)):
            pixel_x = j // 3
            channel = j % 3
            if pixel_x < width:
                value = vpm[i, pixel_x, channel]
                tile_bytes.append(value)
            else:
                tile_bytes.append(0)  # Padding
    
    return bytes(tile_bytes)