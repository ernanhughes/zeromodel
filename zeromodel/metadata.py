# zeromodel/metadata.py
"""
Compact Metadata Handling

This module provides functions for encoding and decoding metadata in a compact
binary format that survives image processing operations. This is critical
for the self-describing nature of zeromodel maps.
"""

import logging
from typing import Dict, List

# Create a logger for this module
logger = logging.getLogger(__name__)

def encode_metadata(task_weights: Dict[str, float], 
                   metric_names: List[str],
                   version: int = 1) -> bytes:
    """
    Encode metadata into compact binary format (<100 bytes).
    
    The encoded metadata includes:
    - A version number
    - A simple hash representing the task (based on metric names and weights)
    - The relative importance of each metric according to task_weights
    
    Args:
        task_weights: A dictionary mapping metric names (from metric_names) 
                      to their weights (floats, typically 0.0 to 1.0).
        metric_names: A list of all metric names. The order is significant.
        version: Metadata format version (integer).
    
    Returns:
        bytes: Compact binary representation of the metadata.
        
    Raises:
        ValueError: If inputs are invalid (e.g., None, negative version).
    """
    logger.debug(f"Encoding metadata: version={version}, metrics={len(metric_names)}")
    if task_weights is None:
        logger.error("task_weights cannot be None")
        raise ValueError("task_weights cannot be None")
    if metric_names is None:
        logger.error("metric_names cannot be None")
        raise ValueError("metric_names cannot be None")
    if version < 0:
        logger.error(f"Version must be non-negative, got {version}")
        raise ValueError(f"Version must be non-negative, got {version}")

    metadata = bytearray()

    # 1. Version (1 byte)
    metadata.append(version & 0xFF) # Ensure fits in 1 byte
    logger.debug(f"Encoded version: {version & 0xFF}")

    # 2. Task ID hash (4 bytes)
    task_hash = 0
    # Iterate through metric_names to ensure consistent order and only consider relevant metrics
    for metric in metric_names: 
        weight = task_weights.get(metric, 0.0)
        if weight > 0:
            # Simple hash based on metric name and weight
            # hash() can return negative values, ensure positive for XOR
            name_hash = hash(metric) & 0xFFFFFFFFFFFFFFFF # Treat as unsigned 64-bit if needed, or just use abs
            weight_int = int(weight * 1000) # Scale weight for hashing, using int for consistency
            task_hash ^= (abs(name_hash) ^ weight_int) # Use abs to handle negative hash
    task_hash &= 0xFFFFFFFF  # Keep as 32-bit unsigned
    metadata.extend(task_hash.to_bytes(4, 'big'))
    logger.debug(f"Encoded task hash: {task_hash:#010x}")

    # 3. Metric importance (4 bits per metric, 2 metrics per byte)
    # Assumes metric_names provides the order.
    # Pads with 0 (importance 0) if metric_names has an odd count.
    for i in range(0, len(metric_names), 2):
        byte_val = 0
        # Handle first metric in the pair (high 4 bits)
        if i < len(metric_names):
            metric1 = metric_names[i]
            # Get weight, defaulting to 0 if not found or if weight is None
            raw_weight1 = task_weights.get(metric1, 0.0) 
            # Clamp raw weight to [0.0, 1.0] to be safe
            clamped_weight1 = max(0.0, min(1.0, raw_weight1))
            # Scale to 0-15 integer
            weight_val1 = int(clamped_weight1 * 15) 
            byte_val |= (weight_val1 & 0x0F) << 4 # Mask to 4 bits, shift to high nibble
            logger.debug(f"Mapped metric '{metric1}' (weight {raw_weight1}) to nibble {weight_val1 & 0x0F}")

        # Handle second metric in the pair (low 4 bits)
        if i + 1 < len(metric_names):
            metric2 = metric_names[i+1]
            raw_weight2 = task_weights.get(metric2, 0.0)
            clamped_weight2 = max(0.0, min(1.0, raw_weight2))
            weight_val2 = int(clamped_weight2 * 15)
            byte_val |= (weight_val2 & 0x0F) # Mask to 4 bits, place in low nibble
            logger.debug(f"Mapped metric '{metric2}' (weight {raw_weight2}) to nibble {weight_val2 & 0x0F}")
            
        metadata.append(byte_val & 0xFF) # Ensure byte_val fits in a byte
    
    result_bytes = bytes(metadata)
    logger.info(f"Metadata encoded successfully. Size: {len(result_bytes)} bytes")
    return result_bytes

def decode_metadata(metadata_bytes: bytes, 
                  metric_names: List[str]) -> Dict[str, float]:
    """
    Decode compact binary metadata back to task weights.
    
    Args:
        metadata_bytes: Binary metadata produced by encode_metadata.
        metric_names: The list of metric names, used to map decoded weights back.
    
    Returns:
        Dict[str, float]: A dictionary mapping metric names to their decoded weights.
    """
    logger.debug(f"Decoding metadata: size={len(metadata_bytes)} bytes, expected metrics={len(metric_names)}")
    if not metadata_bytes:
        logger.warning("Empty metadata bytes provided. Returning default weights.")
        return {m: 0.5 for m in metric_names}
    if metric_names is None:
         logger.error("metric_names cannot be None for decoding.")
         # Returning empty dict might be better than defaulting here, but matching original logic somewhat.
         return {} 

    if len(metadata_bytes) < 5:
        # Not enough data for header, return defaults
        logger.warning("Metadata too short (<5 bytes). Returning default weights.")
        return {m: 0.5 for m in metric_names}
    
    version = metadata_bytes[0]
    # Optional: Check version if multiple versions are expected in the future
    if version != 1: # Assuming version 1 is expected for now
         logger.info(f"Decoded metadata version: {version}. Expected version 1. Proceeding.")
    else:
         logger.debug(f"Decoded metadata version: {version}")

    try:
        task_hash = int.from_bytes(metadata_bytes[1:5], 'big')
        logger.debug(f"Decoded task hash: {task_hash:#010x}")
    except (IndexError, ValueError) as e: # Catch potential errors in slicing/conversion
         logger.error(f"Error decoding task hash from metadata: {e}. Using default weights.")
         return {m: 0.5 for m in metric_names}

    weights = {}
    # Iterate through metric_names to assign weights in the correct order
    for i, metric in enumerate(metric_names):
        byte_idx = 5 + (i // 2) # Calculate which byte contains this metric's data
        if byte_idx >= len(metadata_bytes):
            # Ran out of metadata bytes, assign default weight
            logger.warning(f"Metadata exhausted before weight for metric '{metric}' (index {i}). Assigning default 0.5.")
            weights[metric] = 0.5
            continue
            
        # Determine if this metric's data is in the high or low nibble
        shift = 4 if i % 2 == 0 else 0 # High nibble (4 bits shifted) or low nibble (0 bits shifted)
        # Extract the 4-bit value
        weight_val_nibble = (metadata_bytes[byte_idx] >> shift) & 0x0F 
        # Convert 4-bit value (0-15) back to float weight (0.0-1.0)
        weight = weight_val_nibble / 15.0 
        weights[metric] = weight
        logger.debug(f"Decoded metric '{metric}': nibble {weight_val_nibble:#x} -> weight {weight:.4f}")
    
    logger.info("Metadata decoded successfully.")
    return weights

def get_metadata_size(metric_count: int) -> int:
    """
    Calculate approximate metadata size in bytes.
    
    The size consists of:
    - 1 byte for the version
    - 4 bytes for the task hash
    - N bytes for metric importance (2 metrics per byte)
    
    Args:
        metric_count: Number of metrics
    
    Returns:
        int: Estimated total metadata size in bytes.
    """
    if metric_count < 0:
        logger.warning(f"Negative metric_count {metric_count} provided. Returning size for 0 metrics.")
        metric_count = 0
    # 5 bytes header (version + task_hash) + ceiling division for metric bytes
    size = 5 + (metric_count + 1) // 2 
    logger.debug(f"Calculated metadata size for {metric_count} metrics: {size} bytes")
    return size
