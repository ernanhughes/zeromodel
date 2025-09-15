#  zeromodel/utils.py
"""
Utility Functions

This module provides helper functions used throughout the zeromodel package.
"""
from __future__ import annotations

import hashlib
import io
from typing import Any, Union

import numpy as np
from PIL import Image

from zeromodel.constants import PRECISION_DTYPE_MAP

__all__ = ["quantize", "dct", "idct", "sha3", "png_to_gray_array", "to_png_bytes", "tile_to_pil"]


def quantize(value: Any, precision: int) -> Any:
    """Quantize values to specified bit precision (assumes input in [0,1]).

    Clamps input to [0,1] then scales to integer range. Chooses an appropriate
    unsigned integer dtype based on precision.

    Args:
        value: Scalar or ndarray of floats in any range (will be clipped to [0,1]).
        precision: Bit precision (4-32 typical). Values <1 raise, >64 truncated to 64.

    Returns:
        Quantized integer array / scalar of appropriate dtype.
    """
    if not isinstance(precision, int):
        raise TypeError("precision must be an int")
    if precision < 1:
        raise ValueError("precision must be >= 1")
    if precision > 64:
        precision = 64  # cap
    dtype = PRECISION_DTYPE_MAP(precision)
    max_val = (1 << precision) - 1 if precision < 64 else np.iinfo(dtype).max
    if isinstance(value, np.ndarray):
        clipped = np.clip(value, 0.0, 1.0)
        scaled = np.round(clipped * max_val)
        return scaled.astype(dtype)
    # Scalar path
    v = float(value)
    if v < 0.0:
        v = 0.0
    elif v > 1.0:
        v = 1.0
    return int(round(v * max_val))


def dct(matrix: np.ndarray, norm: str = "ortho", axis: int = -1) -> np.ndarray:
    """Compute a DCT-II along a chosen axis (minimal, SciPy-free).

    Based on the standard definition:
        X_n = sum_{k=0}^{N-1} x_k * cos[ pi/N * (k + 0.5) * n ]

    Orthonormal scaling (norm='ortho') matches scipy.fft.dct(type=2, norm='ortho').

    Complexity is O(N^2); intended for small edge scenarios.
    """
    x = np.asarray(matrix, dtype=np.float64)
    x = np.moveaxis(x, axis, -1)
    N = x.shape[-1]
    if N == 0:
        return matrix.copy()
    k = np.arange(N, dtype=np.float64)
    n = k  # reuse variable for clarity
    cos_table = np.cos(np.pi / N * (k + 0.5)[:, None] * n[None, :])  # shape (N,N)
    # Perform tensordot over last axis of x with first axis of cos_table
    out = np.tensordot(x, cos_table, axes=([-1], [0]))  # shape (..., N)
    if norm == "ortho":
        out[..., 0] *= np.sqrt(1.0 / N)
        out[..., 1:] *= np.sqrt(2.0 / N)
    out = np.moveaxis(out, -1, axis)
    return out.astype(np.float32, copy=False)


def idct(matrix: np.ndarray, norm: str = "ortho", axis: int = -1) -> np.ndarray:
    """Compute an IDCT (inverse of DCT-II) aka DCT-III along axis.

    For norm='ortho' this inverts ``dct(..., norm='ortho')`` numerically.
    Complexity O(N^2); intended for small inputs.
    """
    X = np.asarray(matrix, dtype=np.float64)
    X = np.moveaxis(X, axis, -1)
    N = X.shape[-1]
    if N == 0:
        return matrix.copy()
    n = np.arange(N, dtype=np.float64)
    k = n  # reuse
    cos_table = np.cos(np.pi / N * (n + 0.5)[:, None] * k[None, :])  # (N,N)
    Y = X.copy()
    if norm == "ortho":
        Y[..., 0] *= np.sqrt(1.0 / N)
        Y[..., 1:] *= np.sqrt(2.0 / N)
    else:
        # Undo scaling expected for unnormalized forward (approximate)
        Y[..., 0] *= 1.0 / (N / 2.0)
    out = np.tensordot(Y, cos_table.T, axes=([-1], [0]))  # shape (..., N)
    out = np.moveaxis(out, -1, axis)
    return out.astype(np.float32, copy=False)


def to_png_bytes(img: Union[np.ndarray, bytes, bytearray]) -> bytes:
    """Ensure we have real PNG bytes. If given a numpy image, encode it."""
    if isinstance(img, (bytes, bytearray)):
        return bytes(img)

    if not isinstance(img, np.ndarray):
        raise TypeError(f"Expected np.ndarray or bytes, got {type(img)}")

    arr = img
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # Infer mode
    if arr.ndim == 2:
        mode = "L"
    elif arr.ndim == 3 and arr.shape[2] == 3:
        mode = "RGB"
    elif arr.ndim == 3 and arr.shape[2] == 4:
        mode = "RGBA"
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    bio = io.BytesIO()
    Image.fromarray(arr, mode=mode).save(bio, format="PNG")
    return bio.getvalue()


def png_to_gray_array(png_bytes: bytes) -> np.ndarray:
    """
    Decode PNG to a 2D grayscale uint8 array (H, W).
    Guarantees a 2-D array so argmax/indices are (y, x).
    """
    with Image.open(io.BytesIO(png_bytes)) as im:
        im = im.convert("L")
        arr = np.array(im, dtype=np.uint8)
    return arr

def sha3(b):
    """
    Compute the SHA3-256 hash of input bytes and return as hex string.

    This is the core hashing function used throughout ZeroModel for:
    - Content addressing of VPM tiles
    - Provenance verification
    - Deterministic tile identification
    - Cryptographic integrity checks

    Args:
        b: Input bytes to hash

    Returns:
        Hexadecimal string representation of the SHA3-256 hash

    Example:
        >>> sha3_bytes(b"hello world")
        '944ad329d0fc15a38889e8d61a3d8e127506a0c8e67f8a8e1d3d6e9d3d0c6d0c'

    Note:
        SHA3-256 is used instead of SHA-256 for better resistance against length
        extension attacks and as part of the ZeroModel's commitment to modern
        cryptographic standards for provenance.
    """
    return hashlib.sha3_256(b).hexdigest()



def tile_to_pil(tile: np.ndarray) -> Image.Image:
    """
    Accepts HxW float32 in [0,1], HxW float32 any range, or HxWx3 uint8.
    Returns a PIL RGB image suitable for GIF encoding.
    """
    if tile.dtype == np.uint8:
        if tile.ndim == 2:
            return Image.fromarray(tile, mode="L").convert("RGB")
        if tile.ndim == 3 and tile.shape[2] in (3, 4):
            return Image.fromarray(tile[:, :, :3], mode="RGB")
    # float cases → normalize to 0..255
    arr = tile.astype(np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        arr = np.zeros_like(arr)
    else:
        vmin = float(np.nanpercentile(arr[finite], 1))
        vmax = float(np.nanpercentile(arr[finite], 99))
        if vmax <= vmin:  # degenerate
            vmax = vmin + 1.0
        arr = np.clip((arr - vmin) / (vmax - vmin), 0, 1) * 255.0
    u8 = arr.astype(np.uint8)
    if u8.ndim == 2:
        return Image.fromarray(u8, mode="L").convert("RGB")
    return Image.fromarray(u8[:, :, :3], mode="RGB")
