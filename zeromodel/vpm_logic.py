# zeromodel/vpm_logic.py
"""
Visual Policy Maps enable a new kind of symbolic mathematics.

Each VPM is a spatially organized array of scalar values encoding task-relevant priorities.
By composing them using logical operators (AND, OR, NOT, NAND, etc.), we form a new symbolic system
where reasoning becomes image composition, and meaning is distributed across space.

These operators allow tiny edge devices to perform sophisticated reasoning by querying
regions of interest in precomputed VPMs. Just like NAND gates enable classical computation,
VPM logic gates enable distributed visual intelligence.

This is not just fuzzy logic. This is **Visual Symbolic Math**.
"""

import numpy as np
import logging
from typing import Union, Optional

logger = logging.getLogger(__name__)


def vpm_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical OR operation (fuzzy union) on two VPMs.
    The result highlights areas relevant to EITHER input VPM by taking the element-wise maximum.
    Assumes VPMs are normalized to the range [0, 1].

    Args:
        a (np.ndarray): First VPM (2D or 3D array, values in [0, 1]).
        b (np.ndarray): Second VPM (same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the OR operation.
    
    Raises:
        ValueError: If VPMs have mismatched shapes.
    """
    logger.debug(f"Performing VPM OR operation on shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        logger.error(f"VPM OR: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for OR. Got {a.shape} and {b.shape}")
    result = np.maximum(a, b)
    logger.debug("VPM OR operation completed.")
    return result

def vpm_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical AND operation (fuzzy intersection) on two VPMs.
    The result highlights areas relevant to BOTH input VPMs by taking the element-wise minimum.
    Assumes VPMs are normalized to the range [0, 1].

    Args:
        a (np.ndarray): First VPM.
        b (np.ndarray): Second VPM (same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the AND operation.
    
    Raises:
        ValueError: If VPMs have mismatched shapes.
    """
    logger.debug(f"Performing VPM AND operation on shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        logger.error(f"VPM AND: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for AND. Got {a.shape} and {b.shape}")
    result = np.minimum(a, b)
    logger.debug("VPM AND operation completed.")
    return result

def vpm_not(a: np.ndarray) -> np.ndarray:
    """
    Performs a logical NOT operation on a VPM.
    Inverts the relevance/priority represented in the VPM by subtracting each value from 1.0.
    Assumes VPMs are normalized to the range [0, 1].

    Args:
        a (np.ndarray): Input VPM.

    Returns:
        np.ndarray: The resulting inverted VPM.
    """
    logger.debug(f"Performing VPM NOT operation on shape {a.shape}")
    result = 1.0 - a
    logger.debug("VPM NOT operation completed.")
    return result

def vpm_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical difference operation (A - B) on two VPMs.
    Result highlights areas important to A but NOT to B. This is functionally
    equivalent to `vpm_and(a, vpm_not(b))`.
    Assumes VPMs are normalized to the range [0, 1].

    Args:
        a (np.ndarray): First VPM (minuend).
        b (np.ndarray): Second VPM (subtrahend, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the difference operation.

    Raises:
        ValueError: If VPMs have mismatched shapes.
    """
    logger.debug(f"Performing VPM DIFF (A - B) operation on shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        logger.error(f"VPM DIFF: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for DIFF. Got {a.shape} and {b.shape}")
    # Subtract and clip to [0, 1] to ensure valid range
    result = np.clip(a - b, 0.0, 1.0)
    logger.debug("VPM DIFF operation completed.")
    return result

def vpm_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a simple additive operation on two VPMs (A + B), clipping the result
    to ensure it remains in the [0, 1] range. This operation can be used to
    amplify relevance when both VPMs are active.

    Args:
        a (np.ndarray): First VPM.
        b (np.ndarray): Second VPM (same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the additive operation.

    Raises:
        ValueError: If VPMs have mismatched shapes.
    """
    logger.debug(f"Performing VPM ADD (A + B) operation on shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        logger.error(f"VPM ADD: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for ADD. Got {a.shape} and {b.shape}")
    # Add and clip to [0, 1]
    result = np.clip(a + b, 0.0, 1.0)
    logger.debug("VPM ADD operation completed.")
    return result

def vpm_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical XOR (exclusive OR) operation on two VPMs.
    Result highlights areas relevant to A OR B, but NOT BOTH.
    Functionally equivalent to `vpm_or(vpm_diff(a, b), vpm_diff(b, a))`.

    Args:
        a (np.ndarray): First VPM.
        b (np.ndarray): Second VPM (same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the XOR operation.

    Raises:
        ValueError: If VPMs have mismatched shapes.
    """
    logger.debug(f"Performing VPM XOR operation on shapes {a.shape} and {b.shape}")
    if a.shape != b.shape:
        logger.error(f"VPM XOR: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for XOR. Got {a.shape} and {b.shape}")
    
    # Calculate (A AND NOT B) OR (B AND NOT A)
    a_and_not_b = vpm_diff(a, b)
    b_and_not_a = vpm_diff(b, a)
    result = vpm_or(a_and_not_b, b_and_not_a)
    
    logger.debug("VPM XOR operation completed.")
    return result

def vpm_nand(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a NAND operation: NOT(AND(a, b)).
    Universal gate for constructing any logic circuit.

    Returns:
        np.ndarray: Result of NAND.
    """
    return vpm_not(vpm_and(a, b))

def vpm_nor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a NOR operation: NOT(OR(a, b)).
    Also a universal logic gate.

    Returns:
        np.ndarray: Result of NOR.
    """
    return vpm_not(vpm_or(a, b))


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
        # The logic is:
        # "Good" items are high quality with low uncertainty.
        # "Exploratory" items are high novelty with low uncertainty.
        # An "interesting" item is either Good OR Exploratory.
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
    This provides a simple, aggregated measure of relevance for the entire VPM.

    Args:
        vpm (np.ndarray): The VPM to query.
        context_size (int): The size of the top-left square region to consider (NxN).
                           Must be a positive integer.

    Returns:
        float: An aggregate relevance score (mean) from the top-left region.

    Raises:
        ValueError: If VPM is not at least 2D or context_size is invalid.
    """
    logger.debug(f"Querying top-left region of VPM (shape: {vpm.shape}) with context size {context_size}")
    if vpm.ndim < 2:
        logger.error("VPM must be at least 2D for top-left query.")
        raise ValueError("VPM must be at least 2D.")
    if not isinstance(context_size, int) or context_size <= 0:
        logger.error(f"Invalid context_size: {context_size}. Must be a positive integer.")
        raise ValueError("context_size must be a positive integer.")
        
    height, width = vpm.shape[:2] # Handle both 2D and 3D (H, W) or (H, W, C)
    actual_context_h = min(context_size, height)
    actual_context_w = min(context_size, width)
    
    top_left_region = vpm[:actual_context_h, :actual_context_w]
    # Simple aggregation: mean. Could be max, weighted, etc.
    score = np.mean(top_left_region)
    logger.debug(f"Top-left query score (mean of {actual_context_h}x{actual_context_w} region): {score:.4f}")
    return score