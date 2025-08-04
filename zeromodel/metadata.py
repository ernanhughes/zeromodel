"""
Compact Metadata Handling

This module provides functions for encoding and decoding metadata in a compact
binary format that survives image processing operations. This is critical
for the self-describing nature of zeromodel maps.
"""

import struct
from typing import Dict, List, Tuple


def encode_metadata(task_weights: Dict[str, float], 
                   metric_names: List[str],
                   version: int = 1) -> bytes:
    """
    Encode metadata into compact binary format (<100 bytes).
    
    Args:
        task_weights: Weights for each metric in current task
        metric_names: All metric names
        version: Metadata format version
    
    Returns:
        Compact binary metadata
    """
    # 1. Version (1 byte)
    metadata = [version]
    
    # 2. Task ID hash (4 bytes)
    task_hash = 0
    for metric, weight in task_weights.items():
        if weight > 0:
            # Simple hash based on metric name and weight
            task_hash ^= hash(metric) ^ int(weight * 100)
    task_hash &= 0xFFFFFFFF  # Keep as 32-bit
    metadata.extend(task_hash.to_bytes(4, 'big'))
    
    # 3. Metric importance (4 bits per metric, 16 metrics per byte)
    importance_bytes = []
    for i in range(0, len(metric_names), 2):
        byte_val = 0
        if i < len(metric_names):
            metric = metric_names[i]
            weight_val = int(task_weights.get(metric, 0) * 15)  # 0-15 scale
            byte_val |= weight_val << 4
        if i+1 < len(metric_names):
            metric = metric_names[i+1]
            weight_val = int(task_weights.get(metric, 0) * 15)  # 0-15 scale
            byte_val |= weight_val
        importance_bytes.append(byte_val)
    
    return bytes(metadata + importance_bytes)

def decode_metadata(metadata_bytes: bytes, 
                  metric_names: List[str]) -> Dict[str, float]:
    """
    Decode compact metadata back to task weights.
    
    Args:
        metadata_bytes: Binary metadata
        metric_names: All metric names
    
    Returns:
        Task weights dictionary
    """
    if len(metadata_bytes) < 5:
        # Not enough data, return defaults
        return {m: 0.5 for m in metric_names}
    
    version = metadata_bytes[0]
    task_hash = int.from_bytes(metadata_bytes[1:5], 'big')
    
    weights = {}
    for i, metric in enumerate(metric_names):
        byte_idx = 5 + (i // 2)
        if byte_idx >= len(metadata_bytes):
            weights[metric] = 0.5
            continue
            
        shift = 4 if i % 2 == 0 else 0
        weight_val = (metadata_bytes[byte_idx] >> shift) & 0x0F
        weights[metric] = weight_val / 15.0
    
    return weights

def get_metadata_size(metric_count: int) -> int:
    """
    Calculate approximate metadata size in bytes.
    
    Args:
        metric_count: Number of metrics
    
    Returns:
        Estimated metadata size
    """
    # 5 bytes header + 1 byte per 2 metrics
    return 5 + (metric_count + 1) // 2