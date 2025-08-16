# zeromodel/logic.py
"""
Visual Policy Map (VPM) Logic Operations

Implements ZeroModel's symbolic visual logic engine - a complete system for
performing logical operations directly on VPMs using pixel-wise arithmetic.

Key Innovation:
Instead of running neural models, we run fuzzy logic on structured images.
This enables hardware-style reasoning where:
- AND = pixel-wise minimum (intersection)
- OR = pixel-wise maximum (union)  
- NOT = intensity inversion (255 - value)
- XOR = absolute difference (highlight differences)

These operations work the same whether the VPM came from a local IoT sensor
or a global index of 10¹² items - "symbolic math that works the same at any scale."

Why This Matters:
Traditional AI systems require complex query engines or retraining to combine conditions.
ZeroModel enables hardware-style logic operations directly on decision tiles:
| Operation | Visual Result | Use Case |
|----------|---------------|----------|
| AND | Intersection | Safety gates (safe AND relevant) |
| OR | Union | Alert systems (error OR warning) |
| NOT | Inversion | Anomaly detection |
| XOR | Difference | Change detection |

This is not just fuzzy logic. This is **Visual Symbolic Math**.
"""

import logging
from typing import Tuple, Union
import numpy as np
from PIL import Image

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


def denormalize_vpm(vpm: np.ndarray, output_type=np.uint8, assume_normalized: bool = True) -> np.ndarray:
    """Convert a (normalized) VPM to a specified dtype.

    Args:
        vpm: Input VPM. If not already float in [0,1] set ``assume_normalized=False``.
        output_type: Target numpy dtype.
        assume_normalized: If False, will first run ``normalize_vpm``.
    """
    logger.debug(f"Denormalizing VPM to dtype {output_type} (assume_normalized={assume_normalized})")
    data = vpm if assume_normalized else normalize_vpm(vpm)
    if np.issubdtype(output_type, np.integer):
        dtype_info = np.iinfo(output_type)
        max_val = dtype_info.max
        min_val = dtype_info.min
        scaled_vpm = np.clip(data * max_val, min_val, max_val)
        return scaled_vpm.astype(output_type)
    clipped_vpm = np.clip(data, 0.0, 1.0)
    return clipped_vpm.astype(output_type)


def vpm_resize(img, target_shape):
    """
    Drop-in replacement for scipy.ndimage.zoom(img, zoom=(h/w), order=1),
    for 2D or 3D (HWC) images using bilinear interpolation.
    """
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

# ---------------- Internal Helpers ---------------- #
def _ensure_same_shape(a: np.ndarray, b: np.ndarray, op: str) -> None:
    if a.shape != b.shape:
        logger.error(f"VPM {op}: Shape mismatch. a: {a.shape}, b: {b.shape}")
        raise ValueError(f"VPMs must have the same shape for {op.upper()}. Got {a.shape} and {b.shape}")


def _normalize_pair(a: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return normalize_vpm(a), normalize_vpm(b)


# =============================
# Core VPM Logic Operations
# =============================

def vpm_and(vpm1: Union[np.ndarray, Image.Image], 
                  vpm2: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Perform AND operation on two VPMs (pixel-wise minimum).
    
    This implements intersection logic - only areas that are strong in BOTH VPMs
    survive in the result. Perfect for safety gates and filtering.
    
    Args:
        vpm1: First VPM (numpy array or PIL Image)
        vpm2: Second VPM (numpy array or PIL Image)
        
    Returns:
        PIL Image with AND result
    """
    logger.debug("Performing VPM AND operation")
    
    # Convert to normalized arrays
    arr1 = _to_normalized_array(vpm1)
    arr2 = _to_normalized_array(vpm2)
    
    # Ensure same dimensions
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])
    
    # Pixel-wise minimum (logical AND for intensity)
    result = np.minimum(arr1[:h, :w], arr2[:h, :w])
    
    # Convert back to PIL Image
    return _array_to_pil(result)

def vpm_or(vpm1: Union[np.ndarray, Image.Image], 
                 vpm2: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Perform OR operation on two VPMs (pixel-wise maximum).
    
    This implements union logic - areas that are strong in EITHER VPM appear
    in the result. Perfect for alert systems and combining signals.
    
    Args:
        vpm1: First VPM (numpy array or PIL Image)
        vpm2: Second VPM (numpy array or PIL Image)
        
    Returns:
        PIL Image with OR result
    """
    logger.debug("Performing VPM OR operation")
    
    # Convert to normalized arrays
    arr1 = _to_normalized_array(vpm1)
    arr2 = _to_normalized_array(vpm2)
    
    # Ensure same dimensions
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])
    
    # Pixel-wise maximum (logical OR for intensity)
    result = np.maximum(arr1[:h, :w], arr2[:h, :w])
    
    # Convert back to PIL Image
    return _array_to_pil(result)

def vpm_not(vpm: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Perform NOT operation on a VPM (intensity inversion).
    
    This implements inversion logic - high values become low, low become high.
    Perfect for anomaly detection and finding what's missing.
    
    Args:
        vpm: Input VPM (numpy array or PIL Image)
        
    Returns:
        PIL Image with NOT result
    """
    logger.debug("Performing VPM NOT operation")
    
    # Convert to normalized array
    arr = _to_normalized_array(vpm)
    
    # Intensity inversion (255 - value)
    result = 1.0 - arr
    
    # Convert back to PIL Image
    return _array_to_pil(result)

def vpm_xor(vpm1: Union[np.ndarray, Image.Image], 
                  vpm2: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Perform XOR operation on two VPMs (absolute difference).
    
    This highlights areas where VPMs differ - perfect for change detection
    and identifying inconsistencies between models.
    
    Args:
        vpm1: First VPM (numpy array or PIL Image)
        vpm2: Second VPM (numpy array or PIL Image)
        
    Returns:
        PIL Image with XOR result
    """
    logger.debug("Performing VPM XOR operation")
    
    # Convert to normalized arrays
    arr1 = _to_normalized_array(vpm1)
    arr2 = _to_normalized_array(vpm2)
    
    # Ensure same dimensions
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])
    
    # Absolute difference (logical XOR for intensity)
    result = np.abs(arr1[:h, :w] - arr2[:h, :w])
    
    # Convert back to PIL Image
    return _array_to_pil(result)

# =============================
# Utility Functions
# =============================

def _to_normalized_array(obj: Union[np.ndarray, Image.Image]) -> np.ndarray:
    """
    Convert PIL Image or numpy array to normalized float32 array in [0,1] range.
    
    Args:
        obj: Input object (PIL Image or numpy array)
        
    Returns:
        Normalized numpy array in [0,1] range with float32 dtype
    """
    if isinstance(obj, Image.Image):
        # Convert PIL Image to numpy array
        arr = np.array(obj.convert("RGB"))
        
        # Handle different modes
        if arr.ndim == 3 and arr.shape[2] == 3:
            # RGB image - convert to grayscale if needed for consistency
            # Or keep as RGB for color operations
            pass
        elif arr.ndim == 2:
            # Grayscale - add channel dimension
            arr = arr[:, :, np.newaxis]
    elif isinstance(obj, np.ndarray):
        arr = obj
    else:
        raise TypeError(f"Expected PIL.Image or numpy.ndarray, got {type(obj)}")
    
    # Normalize to [0,1] range
    if arr.dtype != np.float32:
        if np.issubdtype(arr.dtype, np.integer):
            # Integer types: normalize based on max value
            max_val = np.iinfo(arr.dtype).max
            arr = arr.astype(np.float32) / max_val
        else:
            # Other float types: clip to [0,1]
            arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    
    return arr

def _array_to_pil(arr: np.ndarray) -> Image.Image:
    """
    Convert normalized numpy array to PIL Image.
    
    Args:
        arr: Normalized numpy array in [0,1] range with float32 dtype
        
    Returns:
        PIL Image in RGB mode
    """
    # Scale to [0,255] range and convert to uint8
    scaled = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    
    # Handle different array shapes
    if scaled.ndim == 2:
        # Grayscale - convert to RGB
        return Image.fromarray(scaled, mode="L").convert("RGB")
    elif scaled.ndim == 3:
        if scaled.shape[2] == 1:
            # Single channel - convert to grayscale then RGB
            return Image.fromarray(scaled[:, :, 0], mode="L").convert("RGB")
        elif scaled.shape[2] == 3:
            # RGB - direct conversion
            return Image.fromarray(scaled, mode="RGB")
        else:
            # Too many channels - take first 3
            rgb = scaled[:, :, :3]
            return Image.fromarray(rgb, mode="RGB")
    else:
        raise ValueError(f"Unsupported array shape: {arr.shape}")

# =============================
# Advanced VPM Operations
# =============================

def vpm_nand(vpm1: Union[np.ndarray, Image.Image], 
                   vpm2: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Perform NAND operation (NOT(AND)) on two VPMs.
    
    Universal gate for constructing any logic circuit.
    
    Args:
        vpm1: First VPM
        vpm2: Second VPM
        
    Returns:
        PIL Image with NAND result
    """
    logger.debug("Performing VPM NAND operation")
    
    # AND first, then NOT
    and_result = vpm_and(vpm1, vpm2)
    return vpm_not(and_result)

def vpm_nor(vpm1: Union[np.ndarray, Image.Image], 
                  vpm2: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Perform NOR operation (NOT(OR)) on two VPMs.
    
    Also a universal logic gate.
    
    Args:
        vpm1: First VPM
        vpm2: Second VPM
        
    Returns:
        PIL Image with NOR result
    """
    logger.debug("Performing VPM NOR operation")
    
    # OR first, then NOT
    or_result = vpm_or(vpm1, vpm2)
    return vpm_not(or_result)

def vpm_compare(vpm1: Union[np.ndarray, Image.Image], 
                vpm2: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Compare two VPMs and visualize differences.
    
    Useful for debugging and verifying VPM consistency.
    
    Args:
        vpm1: First VPM
        vpm2: Second VPM
        
    Returns:
        PIL Image showing differences (red channel highlights differences)
    """
    logger.debug("Comparing two VPMs")
    
    # Convert to normalized arrays
    arr1 = _to_normalized_array(vpm1)
    arr2 = _to_normalized_array(vpm2)
    
    # Ensure same dimensions
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])
    
    # Compute absolute difference
    diff = np.abs(arr1[:h, :w] - arr2[:h, :w])
    
    # Create visualization (red for differences)
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:, :, 0] = (diff * 255.0).astype(np.uint8)  # Red channel shows differences
    
    return Image.fromarray(vis, mode="RGB")

def vpm_query_top_left(vpm: Union[np.ndarray, Image.Image], 
                       context_size: int = 8) -> float:
    """
    Query the top-left region of a VPM for relevance score.
    
    This provides a simple, aggregated measure of relevance for the entire VPM.
    
    Args:
        vpm: Input VPM
        context_size: Size of the top-left square region to consider
        
    Returns:
        Aggregate relevance score (mean) from the top-left region
    """
    logger.debug(f"Querying top-left region with context size {context_size}")
    
    # Convert to normalized array
    arr = _to_normalized_array(vpm)
    
    # Ensure valid context size
    if context_size <= 0:
        raise ValueError("Context size must be positive")
    
    # Clip context size to actual dimensions
    h, w = arr.shape[:2]
    actual_context_h = min(context_size, h)
    actual_context_w = min(context_size, w)
    
    # Extract top-left region
    top_left_region = arr[:actual_context_h, :actual_context_w]
    
    # Compute mean relevance score
    score = float(np.mean(top_left_region))
    
    logger.debug(f"Top-left query score: {score:.4f}")
    return score

def create_interesting_vpm(
    quality_vpm: Union[np.ndarray, Image.Image],
    novelty_vpm: Union[np.ndarray, Image.Image],
    uncertainty_vpm: Union[np.ndarray, Image.Image]
) -> Image.Image:
    """
    Create a composite 'interesting' VPM based on:
    (Quality AND NOT Uncertainty) OR (Novelty AND NOT Uncertainty)
    
    This implements the core logic for finding documents that are either
    high-quality or novel, but not uncertain.
    
    Args:
        quality_vpm: VPM representing quality
        novelty_vpm: VPM representing novelty
        uncertainty_vpm: VPM representing uncertainty
        
    Returns:
        Composite 'interesting' VPM
    """
    logger.info("Creating 'interesting' composite VPM")
    
    try:
        # Create anti-uncertainty mask
        anti_uncertainty = vpm_not(uncertainty_vpm)
        
        # Create quality-focused map
        quality_map = vpm_and(quality_vpm, anti_uncertainty)
        
        # Create novelty-focused map
        novelty_map = vpm_and(novelty_vpm, anti_uncertainty)
        
        # Combine into interesting map
        interesting_map = vpm_or(quality_map, novelty_map)
        
        logger.info("'Interesting' VPM created successfully")
        return interesting_map
    except Exception as e:
        logger.error(f"Failed to create 'interesting' VPM: {e}")
        raise

# =============================
# Spatial Operations
# =============================

def vpm_concat_horizontal(
    vpm1: Union[np.ndarray, Image.Image],
    vpm2: Union[np.ndarray, Image.Image]
) -> Image.Image:
    """
    Concatenate VPMs horizontally (side-by-side).
    
    Args:
        vpm1: Left VPM
        vpm2: Right VPM
        
    Returns:
        Horizontally concatenated VPM
    """
    logger.debug("Concatenating VPMs horizontally")
    
    # Convert to normalized arrays
    arr1 = _to_normalized_array(vpm1)
    arr2 = _to_normalized_array(vpm2)
    
    # Ensure same height by cropping
    min_height = min(arr1.shape[0], arr2.shape[0])
    arr1_crop = arr1[:min_height, :, :]
    arr2_crop = arr2[:min_height, :, :]
    
    # Concatenate along width axis
    try:
        result = np.concatenate((arr1_crop, arr2_crop), axis=1)
        return _array_to_pil(result)
    except ValueError as e:
        logger.error(f"Failed to concatenate VPMs horizontally: {e}")
        raise

def vpm_concat_vertical(
    vpm1: Union[np.ndarray, Image.Image],
    vpm2: Union[np.ndarray, Image.Image]
) -> Image.Image:
    """
    Concatenate VPMs vertically (stacked).
    
    Args:
        vpm1: Top VPM
        vpm2: Bottom VPM
        
    Returns:
        Vertically concatenated VPM
    """
    logger.debug("Concatenating VPMs vertically")
    
    # Convert to normalized arrays
    arr1 = _to_normalized_array(vpm1)
    arr2 = _to_normalized_array(vpm2)
    
    # Ensure same width by cropping
    min_width = min(arr1.shape[1], arr2.shape[1])
    arr1_crop = arr1[:, :min_width, :]
    arr2_crop = arr2[:, :min_width, :]
    
    # Concatenate along height axis
    try:
        result = np.concatenate((arr1_crop, arr2_crop), axis=0)
        return _array_to_pil(result)
    except ValueError as e:
        logger.error(f"Failed to concatenate VPMs vertically: {e}")
        raise


# Add these to your zeromodel/logic.py file

def vpm_add(vpm1: Union[np.ndarray, Image.Image], 
            vpm2: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Perform ADD operation on two VPMs (pixel-wise addition with clipping).
    
    This implements additive composition - combining the strength of both VPMs.
    Perfect for ensemble methods and boosting signals.
    
    Args:
        vpm1: First VPM (numpy array or PIL Image)
        vpm2: Second VPM (numpy array or PIL Image)
        
    Returns:
        PIL Image with ADD result (clipped to [0, 255])
    """
    logger.debug("Performing VPM ADD operation")
    
    # Convert to normalized arrays
    arr1 = _to_normalized_array(vpm1)
    arr2 = _to_normalized_array(vpm2)
    
    # Ensure same dimensions
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])
    
    # Pixel-wise addition with clipping to [0, 1]
    result = np.clip(arr1[:h, :w] + arr2[:h, :w], 0.0, 1.0)
    
    # Convert back to PIL Image
    return _array_to_pil(result)

def vpm_subtract(vpm1: Union[np.ndarray, Image.Image], 
                 vpm2: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Perform SUBTRACT operation on two VPMs (pixel-wise subtraction with clipping).
    
    This implements difference detection - highlighting areas where vpm1 > vpm2.
    Perfect for change detection and anomaly identification.
    
    Args:
        vpm1: Minuend VPM (numpy array or PIL Image)
        vpm2: Subtrahend VPM (numpy array or PIL Image)
        
    Returns:
        PIL Image with SUBTRACT result (clipped to [0, 255])
    """
    logger.debug("Performing VPM SUBTRACT operation")
    
    # Convert to normalized arrays
    arr1 = _to_normalized_array(vpm1)
    arr2 = _to_normalized_array(vpm2)
    
    # Ensure same dimensions
    h = min(arr1.shape[0], arr2.shape[0])
    w = min(arr1.shape[1], arr2.shape[1])
    
    # Pixel-wise subtraction with clipping to [0, 1]
    result = np.clip(arr1[:h, :w] - arr2[:h, :w], 0.0, 1.0)
    
    # Convert back to PIL Image
    return _array_to_pil(result)

# =============================
# Export
# =============================

__all__ = [
    "vpm_add",
    "vpm_subtract",
    "vpm_and",
    "vpm_or",
    "vpm_not",
    "vpm_xor",
    "vpm_nand",
    "vpm_nor",
    "vpm_compare",
    "vpm_query_top_left",
    "create_interesting_vpm",
    "vpm_concat_horizontal",
    "vpm_concat_vertical",
    "_to_normalized_array",
    "_array_to_pil"
]