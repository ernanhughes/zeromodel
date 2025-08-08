# zeromodel/transform.py
"""
Transformation Pipeline

This module provides functions for dynamically transforming visual policy maps
to prioritize specific metrics for different tasks. This enables the same
underlying data to be used for multiple decision contexts.
"""

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

# Create a logger for this module
logger = logging.getLogger(__name__)

def transform_vpm(vpm: np.ndarray, 
                 metric_names: List[str],
                 target_metrics: List[str]) -> np.ndarray:
    """
    Transform a Visual Policy Map (VPM) to prioritize specific metrics.
    
    This reorders the metrics (columns) in the VPM so that the target metrics
    appear first. It then sorts the documents (rows) based on the value in the
    first of these target metrics, descending. This makes the most relevant
    information appear in the top-left of the resulting image.

    Args:
        vpm: Visual policy map as an RGB image array of shape [height, width, 3].
             Values are expected to be in the range [0, 255].
        metric_names: List of original metric names corresponding to the columns
                      of the data *before* it was encoded into the VPM.
                      Length should match the total number of metrics represented.
        target_metrics: List of metric names to prioritize and move to the front.
                        Metrics not found in `metric_names` are ignored.

    Returns:
        np.ndarray: Transformed VPM (RGB image array of the same shape as input).
        
    Raises:
        ValueError: If inputs are invalid (e.g., None, incorrect shapes, mismatched dimensions).
    """
    logger.debug(f"Transforming VPM. Shape: {vpm.shape if vpm is not None else 'None'}, "
                 f"Target metrics: {target_metrics}")
    
    # Input validation
    if vpm is None:
        error_msg = "Input VPM cannot be None."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if vpm.ndim != 3 or vpm.shape[2] != 3:
        error_msg = f"VPM must be a 3D RGB array (H, W, 3), got shape {vpm.shape}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if metric_names is None:
        error_msg = "metric_names cannot be None."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if target_metrics is None: # Allow empty list, but not None
         target_metrics = []
         logger.info("target_metrics was None, treating as empty list.")

    height, width, channels = vpm.shape
    if channels != 3:
        # Redundant check, but good for clarity
        error_msg = f"VPM must have 3 channels (RGB), got {channels}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    total_metrics_in_vpm = width * 3
    if len(metric_names) != total_metrics_in_vpm:
        logger.warning(f"Mismatch: VPM width*3 ({total_metrics_in_vpm}) != len(metric_names) ({len(metric_names)}). "
                       f"Proceeding with VPM width*3 as metric count.")
        # Use VPM dimensions for processing, log warning about mismatch
        actual_metric_count = total_metrics_in_vpm
        # Truncate or pad metric_names conceptually for indexing, but warn
        if len(metric_names) < actual_metric_count:
             logger.info("metric_names list is shorter than VPM metrics. Padding conceptually for indexing.")
        # We'll use actual_metric_count for processing based on VPM
    else:
        actual_metric_count = len(metric_names)

    if actual_metric_count == 0:
        logger.info("No metrics to transform. Returning original VPM.")
        return vpm.copy() # Return a copy to avoid accidental mutation

    # 1. Extract metrics from the VPM image
    # Create an array to hold the normalized metric values [height, actual_metric_count]
    metrics_normalized = np.zeros((height, actual_metric_count), dtype=np.float64)
    for i in range(height):
        for j in range(actual_metric_count):
            pixel_x = j // 3
            channel = j % 3
            if pixel_x < width: # Should always be true given actual_metric_count calculation
                # Normalize from 0-255 uint8 to 0.0-1.0 float
                metrics_normalized[i, j] = vpm[i, pixel_x, channel] / 255.0 
    logger.debug(f"Extracted metrics array shape: {metrics_normalized.shape}")

    # 2. Determine the new order of metrics
    # Find indices of target metrics within the available metrics
    metric_indices_to_prioritize = []
    for m in target_metrics:
        try:
            # Find index in the provided metric_names list
            idx = metric_names.index(m)
            # Check if this index is valid for the actual data extracted from VPM
            if idx < actual_metric_count: 
                metric_indices_to_prioritize.append(idx)
            else:
                logger.warning(f"Target metric '{m}' (index {idx}) is beyond the metric count in VPM ({actual_metric_count}). Ignoring.")
        except ValueError:
            logger.warning(f"Target metric '{m}' not found in provided metric_names. Ignoring.")
            # continue # Skip metrics not found

    # Create the new column order: prioritized metrics first, then the rest
    remaining_indices = [i for i in range(actual_metric_count) 
                        if i not in metric_indices_to_prioritize]
    new_metric_order = metric_indices_to_prioritize + remaining_indices
    logger.debug(f"Calculated new metric order: {new_metric_order}")

    # 3. Reorder the columns (metrics) of the extracted data
    reordered_metrics_normalized = metrics_normalized[:, new_metric_order]
    logger.debug(f"Reordered metrics array shape: {reordered_metrics_normalized.shape}")

    # 4. Sort rows (documents) by the value in the first prioritized metric (descending)
    if len(metric_indices_to_prioritize) > 0:
        sort_key_column = 0 # First column after reordering is the first target metric
        sort_key_values = reordered_metrics_normalized[:, sort_key_column]
        # Get indices that would sort the array descending (highest values first)
        sorted_row_indices = np.argsort(sort_key_values)[::-1] 
        transformed_metrics_normalized = reordered_metrics_normalized[sorted_row_indices]
        logger.debug(f"Sorted rows by metric index {new_metric_order[0]} (original name: {metric_names[new_metric_order[0]] if new_metric_order[0] < len(metric_names) else 'N/A'})")
    else:
        logger.info("No valid target metrics found for sorting. Returning reordered metrics without row sorting.")
        transformed_metrics_normalized = reordered_metrics_normalized
        sorted_row_indices = np.arange(height) # Identity sort if no sorting

    # 5. Re-encode the transformed data back into an RGB image
    # Create output image array
    transformed_vpm = np.zeros_like(vpm, dtype=np.uint8) # Ensure correct dtype
    for i in range(height):
        for j in range(actual_metric_count):
            pixel_x = j // 3
            channel = j % 3
            if pixel_x < width: # Should always be true
                # Denormalize from 0.0-1.0 float back to 0-255 uint8
                value_0_255 = int(np.clip(transformed_metrics_normalized[i, j] * 255.0, 0, 255))
                transformed_vpm[i, pixel_x, channel] = value_0_255
    logger.info(f"VPM transformation complete. Output shape: {transformed_vpm.shape}")
    return transformed_vpm

def get_critical_tile(vpm: np.ndarray, tile_size: int = 3) -> bytes:
    """
    Extract a critical tile (top-left section) from a visual policy map.

    Args:
        vpm: Visual policy map as an RGB image array of shape [height, width, 3].
        tile_size: Desired size of the square tile (NxN pixels). Defaults to 3.

    Returns:
        bytes: Compact byte representation of the tile.
               Format: [width][height][x_offset][y_offset][pixel_data...]
               where pixel_data is R,G,B values for each pixel in row-major order.
               
    Raises:
        ValueError: If inputs are invalid (e.g., None VPM, negative tile_size).
    """
    logger.debug(f"Extracting critical tile. VPM shape: {vpm.shape if vpm is not None else 'None'}, tile_size: {tile_size}")
    
    # Input validation
    if vpm is None:
        error_msg = "Input VPM cannot be None for tile extraction."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if vpm.ndim != 3 or vpm.shape[2] != 3:
        error_msg = f"VPM must be a 3D RGB array (H, W, 3), got shape {vpm.shape}."
        logger.error(error_msg)
        raise ValueError(error_msg)
    if tile_size <= 0:
        error_msg = f"tile_size must be positive, got {tile_size}."
        logger.error(error_msg)
        raise ValueError(error_msg)

    vpm_height, vpm_width, _ = vpm.shape

    # Determine the actual tile dimensions (cannot exceed VPM dimensions)
    actual_tile_width = min(tile_size, vpm_width)
    actual_tile_height = min(tile_size, vpm_height)
    logger.debug(f"Actual tile dimensions: {actual_tile_width}x{actual_tile_height}")

    # Convert to compact byte format
    tile_bytes = bytearray()
    tile_bytes.append(actual_tile_width & 0xFF)   # Width (1 byte)
    tile_bytes.append(actual_tile_height & 0xFF)  # Height (1 byte)
    tile_bytes.append(0 & 0xFF)                   # X offset (1 byte, always 0 for top-left)
    tile_bytes.append(0 & 0xFF)                   # Y offset (1 byte, always 0 for top-left)
    logger.debug("Appended tile header bytes.")

    # Add pixel data (1 byte per channel, R,G,B for each pixel)
    # Iterate over the actual tile area within VPM bounds
    for y in range(actual_tile_height):
        for x in range(actual_tile_width):
            # VPM is [height, width, channels], so vpm[y, x, :] gives RGB for pixel (x,y)
            r_value = vpm[y, x, 0]
            g_value = vpm[y, x, 1]
            b_value = vpm[y, x, 2]
            tile_bytes.append(r_value & 0xFF)
            tile_bytes.append(g_value & 0xFF)
            tile_bytes.append(b_value & 0xFF)
            # logger.debug(f"Added pixel ({x},{y}): R={r_value}, G={g_value}, B={b_value}") # Very verbose

    result_bytes = bytes(tile_bytes)
    logger.info(f"Critical tile extracted successfully. Size: {len(result_bytes)} bytes.")
    return result_bytes

