# zeromodel/vpm_logic.py
"""
Visual Policy Map (VPM) Logic Engine.

Provides functions to perform logical operations on VPMs (as NumPy arrays),
enabling compositional reasoning and the creation of new symbolic belief states
from existing ones. This turns VPMs into a visual symbolic substrate.

Inspired by principles of Vector Symbolic Architectures (VSA) applied spatially.
"""

import numpy as np
import logging
from typing import Union
# If you want type hints for ZeroModel instances
# from .core import ZeroModel

logger = logging.getLogger(__name__)

# --- Core VPM Logic Operations ---

def vpm_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical OR operation on two VPMs.
    Result highlights areas relevant to EITHER input VPM.

    Args:
        a (np.ndarray): First VPM (2D or 3D array, values in [0, 1]).
        b (np.ndarray): Second VPM (same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the OR operation.
    """
    logger.debug(f"Performing VPM OR operation on shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        logger.error(f"VPM OR: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for OR. Got {a.shape} and {b.shape}")
    # Assuming normalized VPMs [0, 1], max represents OR (fuzzy union)
    result = np.maximum(a, b)
    logger.debug("VPM OR operation completed.")
    return result

def vpm_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical AND operation on two VPMs.
    Result highlights areas relevant to BOTH input VPMs.

    Args:
        a (np.ndarray): First VPM.
        b (np.ndarray): Second VPM (same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the AND operation.
    """
    logger.debug(f"Performing VPM AND operation on shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        logger.error(f"VPM AND: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for AND. Got {a.shape} and {b.shape}")
    # Assuming normalized VPMs [0, 1], min represents AND (fuzzy intersection)
    result = np.minimum(a, b)
    logger.debug("VPM AND operation completed.")
    return result

def vpm_not(a: np.ndarray) -> np.ndarray:
    """
    Performs a logical NOT operation on a VPM.
    Inverts the relevance/priority represented in the VPM.

    Args:
        a (np.ndarray): Input VPM.

    Returns:
        np.ndarray: The resulting inverted VPM.
    """
    logger.debug(f"Performing VPM NOT operation on shape {a.shape}")
    # Assuming normalized VPM [0, 1], 1 - value inverts it.
    result = 1.0 - a
    logger.debug("VPM NOT operation completed.")
    return result

def vpm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical difference operation (A - B) on two VPMs.
    Result highlights areas important to A but NOT to B.

    Args:
        a (np.ndarray): First VPM (minuend).
        b (np.ndarray): Second VPM (subtrahend, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the difference operation.
    """
    logger.debug(f"Performing VPM DIFF (A - B) operation on shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        logger.error(f"VPM DIFF: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for DIFF. Got {a.shape} and {b.shape}")
    # Subtract and clip to [0, 1] to ensure valid range
    result = np.clip(a - b, 0.0, 1.0)
    logger.debug("VPM DIFF operation completed.")
    return result

# --- Convenience Functions for Common Patterns ---

def create_interesting_map(quality_vpm: np.ndarray, novelty_vpm: np.ndarray, uncertainty_vpm: np.ndarray) -> np.ndarray:
    """
    Creates a composite 'interesting' VPM based on the logic:
    (Quality AND NOT Uncertainty) OR (Novelty AND NOT Uncertainty)

    Args:
        quality_vpm (np.ndarray): VPM representing quality.
        novelty_vpm (np.ndarray): VPM representing novelty.
        uncertainty_vpm (np.ndarray): VPM representing uncertainty.

    Returns:
        np.ndarray: The 'interesting' VPM.
    """
    logger.info("Creating 'interesting' composite VPM.")
    try:
        anti_uncertainty = vpm_not(uncertainty_vpm)
        good_map = vpm_and(quality_vpm, anti_uncertainty)
        exploratory_map = vpm_and(novelty_vpm, anti_uncertainty)
        interesting_map = vpm_or(good_map, exploratory_map)
        logger.info("'Interesting' VPM created successfully.")
        return interesting_map
    except Exception as e:
        logger.error(f"Failed to create 'interesting' VPM: {e}")
        raise

# --- Utility Functions ---

def query_top_left(vpm: np.ndarray, context_size: int = 1) -> float:
    """
    Queries the top-left region of a VPM for a relevance score.

    Args:
        vpm (np.ndarray): The VPM to query.
        context_size (int): The size of the top-left square region to consider (NxN).

    Returns:
        float: An aggregate relevance score (e.g., mean) from the top-left region.
    """
    logger.debug(f"Querying top-left region of VPM (shape: {vpm.shape}) with context size {context_size}")
    if vpm.ndim < 2:
        logger.error("VPM must be at least 2D for top-left query.")
        raise ValueError("VPM must be at least 2D.")
    height, width = vpm.shape[:2] # Handle both 2D and 3D (H, W) or (H, W, C)
    actual_context_h = min(context_size, height)
    actual_context_w = min(context_size, width)
    
    top_left_region = vpm[:actual_context_h, :actual_context_w]
    # Simple aggregation: mean. Could be max, weighted, etc.
    score = np.mean(top_left_region)
    logger.debug(f"Top-left query score (mean of {actual_context_h}x{actual_context_w} region): {score:.4f}")
    return score

# --- Example Usage (as comments or in a docstring) ---
"""
Example Usage:

# Assume you have prepared ZeroModel instances for different tasks
quality_model.prepare(data, "SELECT ... ORDER BY quality_metric DESC")
novelty_model.prepare(data, "SELECT ... ORDER BY novelty_metric DESC")
uncertainty_model.prepare(data, "SELECT ... ORDER BY uncertainty_metric ASC") # Low uncertainty is good

# Get their VPMs
quality_vpm = quality_model.encode() # Returns np.ndarray
novelty_vpm = novelty_model.encode()
uncertainty_vpm = uncertainty_model.encode()

# Compose a new logic
interesting_vpm = create_interesting_map(quality_vpm, novelty_vpm, uncertainty_vpm)

# Query the result
relevance_score = query_top_left(interesting_vpm)

if relevance_score > 0.7: # Define your threshold
    print("Found something interesting!")
    # Potentially extract tile, make decision, etc.
"""
