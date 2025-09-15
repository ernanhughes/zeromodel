#  zeromodel/vpm/logic.py
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
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)


def normalize_vpm(vpm: np.ndarray) -> np.ndarray:
    """
    Ensures a VPM is in the normalized float [0.0, 1.0] range.
    Handles conversion from uint8, uint16, float16, float32, float64.
    """
    logger.debug(f"Normalizing VPM of dtype {vpm.dtype} and shape {vpm.shape}")
    if np.issubdtype(vpm.dtype, np.integer):
        # Integer types: normalize based on max value for the dtype
        dtype_info = np.iinfo(vpm.dtype)
        max_val = dtype_info.max
        min_val = dtype_info.min
        # Handle signed integers if necessary, but VPMs are typically unsigned
        if min_val < 0:
            logger.warning(
                f"VPM dtype {vpm.dtype} is signed. Normalizing assuming 0-min_val range."
            )
            range_val = max_val - min_val
            return ((vpm.astype(np.float64) - min_val) / range_val).astype(np.float32)
        else:
            # Unsigned integer
            return (vpm.astype(np.float64) / max_val).astype(np.float32)
    else:  # Floating point types
        # Assume already in [0, 1] or close enough. Clip for safety.
        return np.clip(vpm, 0.0, 1.0).astype(np.float32)


def denormalize_vpm(
    vpm: np.ndarray, output_type=np.uint8, assume_normalized: bool = True
) -> np.ndarray:
    """Convert a (normalized) VPM to a specified dtype.

    Args:
        vpm: Input VPM. If not already float in [0,1] set ``assume_normalized=False``.
        output_type: Target numpy dtype.
        assume_normalized: If False, will first run ``normalize_vpm``.
    """
    logger.debug(
        f"Denormalizing VPM to dtype {output_type} (assume_normalized={assume_normalized})"
    )
    data = vpm if assume_normalized else normalize_vpm(vpm)
    if np.issubdtype(output_type, np.integer):
        dtype_info = np.iinfo(output_type)
        max_val = dtype_info.max
        min_val = dtype_info.min
        scaled_vpm = np.clip(data * max_val, min_val, max_val)
        return scaled_vpm.astype(output_type)
    clipped_vpm = np.clip(data, 0.0, 1.0)
    return clipped_vpm.astype(output_type)


# ---------------- Internal Helpers ---------------- #
def _ensure_same_shape(a: np.ndarray, b: np.ndarray, op: str) -> None:
    if a.shape != b.shape:
        logger.error(f"VPM {op}: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(
            f"VPMs must have the same shape for {op.upper()}. Got {a.shape} and {b.shape}"
        )


def _normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return normalize_vpm(a), normalize_vpm(b)


def vpm_or(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical OR operation (fuzzy union) on two VPMs.
    The result highlights areas relevant to EITHER input VPM by taking the element-wise maximum.
    Assumes VPMs are normalized to the range [0, 1] (float).

    Args:
        a (np.ndarray): First VPM (normalized float).
        b (np.ndarray): Second VPM (normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the OR operation (normalized float).
    """
    logger.debug(f"Performing VPM OR operation on shapes {a.shape} and {b.shape}")
    _ensure_same_shape(a, b, "or")
    a_norm, b_norm = _normalize_pair(a, b)
    result = np.maximum(a_norm, b_norm)
    logger.debug("VPM OR operation completed.")
    return result  # Already normalized float32


def vpm_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical AND operation (fuzzy intersection) on two VPMs.
    The result highlights areas relevant to BOTH input VPMs by taking the element-wise minimum.
    Assumes VPMs are normalized to the range [0, 1] (float).

    Args:
        a (np.ndarray): First VPM (normalized float).
        b (np.ndarray): Second VPM (normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the AND operation (normalized float).
    """
    logger.debug(f"Performing VPM AND operation on shapes {a.shape} and {b.shape}")
    _ensure_same_shape(a, b, "and")
    a_norm, b_norm = _normalize_pair(a, b)
    result = np.minimum(a_norm, b_norm)
    logger.debug("VPM AND operation completed.")
    return result  # Already normalized float32


def vpm_not(a: np.ndarray) -> np.ndarray:
    """
    Performs a logical NOT operation on a VPM.
    Inverts the relevance/priority represented in the VPM.
    Assumes VPMs are normalized to the range [0, 1] (float).

    Args:
        a (np.ndarray): Input VPM (normalized float).

    Returns:
        np.ndarray: The resulting inverted VPM (normalized float).
    """
    logger.debug(
        f"Performing VPM NOT operation on shape {a.shape} with dtype {a.dtype}"
    )
    # Normalize input to ensure consistency
    a_norm = normalize_vpm(a)
    # Invert: 1.0 - value
    result = 1.0 - a_norm
    logger.debug("VPM NOT operation completed.")
    return result  # Already normalized float32


def vpm_subtract(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical difference operation (A - B) on two VPMs.
    Result highlights areas important to A but NOT to B.
    Functionally equivalent to `vpm_and(a, vpm_not(b))` but uses clipping.

    Args:
        a (np.ndarray): First VPM (minuend, normalized float).
        b (np.ndarray): Second VPM (subtrahend, normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the difference operation (normalized float).
    """
    logger.debug(
        f"Performing VPM SUBTRACT (A - B) operation on shapes {a.shape} and {b.shape}"
    )
    _ensure_same_shape(a, b, "subtract")
    a_norm, b_norm = _normalize_pair(a, b)
    # Subtract and clip to [0, 1] to ensure valid range
    result = np.clip(a_norm - b_norm, 0.0, 1.0)
    logger.debug("VPM SUBTRACT operation completed.")
    return result


def vpm_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a simple additive operation on two VPMs (A + B), clipping the result
    to ensure it remains in the valid range [0, 1].

    Args:
        a (np.ndarray): First VPM (normalized float).
        b (np.ndarray): Second VPM (normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the additive operation (normalized float).
    """
    logger.debug(
        f"Performing VPM ADD (A + B) operation on shapes {a.shape} and {b.shape}"
    )
    _ensure_same_shape(a, b, "add")
    a_norm, b_norm = _normalize_pair(a, b)
    # Add and clip to [0, 1]
    result = np.clip(a_norm + b_norm, 0.0, 1.0)
    logger.debug("VPM ADD operation completed.")
    return result


def vpm_xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a logical XOR (exclusive OR) operation on two VPMs.
    Result highlights areas relevant to A OR B, but NOT BOTH.
    Functionally equivalent to `vpm_or(vpm_diff(a, b), vpm_diff(b, a))`.

    Args:
        a (np.ndarray): First VPM (normalized float).
        b (np.ndarray): Second VPM (normalized float, same shape as a).

    Returns:
        np.ndarray: The resulting VPM from the XOR operation (normalized float).
    """
    logger.debug(f"Performing VPM XOR operation on shapes {a.shape} and {b.shape}")
    _ensure_same_shape(a, b, "xor")
    a_norm, b_norm = _normalize_pair(a, b)
    # Calculate (A AND NOT B) OR (B AND NOT A)
    a_and_not_b = vpm_subtract(a_norm, b_norm)  # Use normalized inputs
    b_and_not_a = vpm_subtract(b_norm, a_norm)
    result = vpm_or(a_and_not_b, b_and_not_a)  # vpm_or also uses normalized inputs
    logger.debug("VPM XOR operation completed.")
    return result


def vpm_nand(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a NAND operation: NOT(AND(a, b)).
    Universal gate for constructing any logic circuit.

    Returns:
        np.ndarray: Result of NAND (normalized float).
    """
    # Normalize inputs
    a_norm, b_norm = _normalize_pair(a, b)
    return vpm_not(vpm_and(a_norm, b_norm))


def vpm_nor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Performs a NOR operation: NOT(OR(a, b)).
    Also a universal logic gate.

    Returns:
        np.ndarray: Result of NOR (normalized float).
    """
    # Normalize inputs
    a_norm, b_norm = _normalize_pair(a, b)
    return vpm_not(vpm_or(a_norm, b_norm))


def vpm_resize(img, target_shape):
    """
    Drop-in replacement for scipy.ndimage.zoom(img, zoom=(h/w), order=1),
    for 2D or 3D (HWC) images using bilinear interpolation.
    """
    import numpy as np

    in_h, in_w = img.shape[:2]
    out_h, out_w = target_shape
    channels = img.shape[2] if img.ndim == 3 else 1

    scale_h = in_h / out_h
    scale_w = in_w / out_w

    # Match scipy.ndimage.zoom coordinate mapping
    row_idx = np.arange(out_h) * scale_h
    col_idx = np.arange(out_w) * scale_w

    row0 = np.floor(row_idx).astype(int)
    col0 = np.floor(col_idx).astype(int)
    row1 = np.clip(row0 + 1, 0, in_h - 1)
    col1 = np.clip(col0 + 1, 0, in_w - 1)

    wy = (row_idx - row0).reshape(-1, 1)
    wx = (col_idx - col0).reshape(1, -1)

    row0 = np.clip(row0, 0, in_h - 1)
    col0 = np.clip(col0, 0, in_w - 1)

    if img.ndim == 2:
        img = img[:, :, None]

    out = np.empty((out_h, out_w, channels), dtype=np.float32)

    for c in range(channels):
        I00 = img[row0[:, None], col0[None, :], c]
        I01 = img[row0[:, None], col1[None, :], c]
        I10 = img[row1[:, None], col0[None, :], c]
        I11 = img[row1[:, None], col1[None, :], c]

        top = I00 * (1 - wx) + I01 * wx
        bottom = I10 * (1 - wx) + I11 * wx
        out[..., c] = top * (1 - wy) + bottom * wy

    return out if channels > 1 else out[..., 0]


def vpm_concat_horizontal(vpm1: np.ndarray, vpm2: np.ndarray) -> np.ndarray:
    """
    Concatenate VPMs horizontally (side-by-side).
    Assumes VPMs are normalized floats. Handles height mismatch by cropping.

    Args:
        vpm1 (np.ndarray): Left VPM (normalized float).
        vpm2 (np.ndarray): Right VPM (normalized float).

    Returns:
        np.ndarray: Horizontally concatenated VPM (normalized float).
    """
    logger.debug(
        f"Horizontally concatenating VPMs of shapes {vpm1.shape} and {vpm2.shape}"
    )
    # Normalize inputs
    v1 = normalize_vpm(vpm1)
    v2 = normalize_vpm(vpm2)

    # Ensure same height by cropping the taller one
    min_height = min(v1.shape[0], v2.shape[0])
    v1_crop = v1[:min_height, :, :] if v1.ndim == 3 else v1[:min_height, :]
    v2_crop = v2[:min_height, :, :] if v2.ndim == 3 else v2[:min_height, :]

    # Concatenate along width axis (axis=1)
    try:
        result = np.concatenate((v1_crop, v2_crop), axis=1)
        logger.debug(f"Horizontal concatenation result shape: {result.shape}")
        return result  # Already normalized float32
    except ValueError as e:
        logger.error(f"Failed to concatenate VPMs horizontally: {e}")
        raise ValueError(f"VPMs could not be concatenated horizontally: {e}") from e


def vpm_concat_vertical(vpm1: np.ndarray, vpm2: np.ndarray) -> np.ndarray:
    """
    Concatenate VPMs vertically (stacked).
    Assumes VPMs are normalized floats. Handles width mismatch by cropping.

    Args:
        vpm1 (np.ndarray): Top VPM (normalized float).
        vpm2 (np.ndarray): Bottom VPM (normalized float).

    Returns:
        np.ndarray: Vertically concatenated VPM (normalized float).
    """
    logger.debug(
        f"Vertically concatenating VPMs of shapes {vpm1.shape} and {vpm2.shape}"
    )
    # Normalize inputs
    v1 = normalize_vpm(vpm1)
    v2 = normalize_vpm(vpm2)

    # Ensure same width by cropping the wider one
    min_width = min(v1.shape[1], v2.shape[1])
    v1_crop = v1[:, :min_width, :] if v1.ndim == 3 else v1[:, :min_width]
    v2_crop = v2[:, :min_width, :] if v2.ndim == 3 else v2[:, :min_width]

    # Concatenate along height axis (axis=0)
    try:
        result = np.concatenate((v1_crop, v2_crop), axis=0)
        logger.debug(f"Vertical concatenation result shape: {result.shape}")
        return result  # Already normalized float32
    except ValueError as e:
        logger.error(f"Failed to concatenate VPMs vertically: {e}")
        raise ValueError(f"VPMs could not be concatenated vertically: {e}") from e


def vpm_query_top_left(vpm: np.ndarray, context_size: int = 1) -> float:
    """
    Queries the top-left region of a VPM for a relevance score.
    This provides a simple, aggregated measure of relevance for the entire VPM.
    Now resolution-independent by using relative context_size.

    Args:
        vpm (np.ndarray): The VPM to query (assumed normalized float internally).
        context_size (int): The size of the top-left square region to consider (NxN).
                           Must be a positive integer. Interpreted relative to VPM size.

    Returns:
        float: An aggregate relevance score (mean) from the top-left region.
    """
    # Normalize input to ensure consistency for internal processing
    vpm_norm = normalize_vpm(vpm)
    logger.debug(
        f"Querying top-left region of VPM (shape: {vpm_norm.shape}) with context size {context_size}"
    )
    if vpm_norm.ndim < 2:
        logger.error("VPM must be at least 2D for top-left query.")
        raise ValueError("VPM must be at least 2D.")
    if not isinstance(context_size, int) or context_size <= 0:
        logger.error(
            f"Invalid context_size: {context_size}. Must be a positive integer."
        )
        raise ValueError("context_size must be a positive integer.")

    height, width = vpm_norm.shape[:2]  # Handle both 2D and 3D (H, W) or (H, W, C)
    # Make context_size relative and bounded
    actual_context_h = min(context_size, height)
    actual_context_w = min(context_size, width)

    top_left_region = vpm_norm[:actual_context_h, :actual_context_w]
    # Simple aggregation: mean. Could be max, weighted, etc.
    score = np.mean(top_left_region)
    logger.debug(
        f"Top-left query score (mean of {actual_context_h}x{actual_context_w} region): {score:.4f}"
    )
    return float(score)


def create_interesting_map(
    quality_vpm: np.ndarray, novelty_vpm: np.ndarray, uncertainty_vpm: np.ndarray
) -> np.ndarray:
    """
    Creates a composite 'interesting' VPM based on the logic:
    (Quality AND NOT Uncertainty) OR (Novelty AND NOT Uncertainty)

    Args:
        quality_vpm (np.ndarray): VPM representing quality (will be normalized).
        novelty_vpm (np.ndarray): VPM representing novelty (will be normalized).
        uncertainty_vpm (np.ndarray): VPM representing uncertainty (will be normalized).

    Returns:
        np.ndarray: The 'interesting' VPM (normalized float).
    """
    logger.info("Creating 'interesting' composite VPM.")
    try:
        # Normalize inputs implicitly via vpm_not/vpm_and/vpm_or
        anti_uncertainty = vpm_not(uncertainty_vpm)
        good_map = vpm_and(quality_vpm, anti_uncertainty)
        exploratory_map = vpm_and(novelty_vpm, anti_uncertainty)
        interesting_map = vpm_or(good_map, exploratory_map)
        logger.info("'Interesting' VPM created successfully.")
        return interesting_map
    except Exception as e:
        logger.error(f"Failed to create 'interesting' VPM: {e}")
        raise
