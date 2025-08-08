# zeromodel/utils.py
"""
Utility Functions

This module provides helper functions used throughout the zeromodel package.
"""

from typing import Any

import numpy as np

__all__ = [
    "quantize",
    "dct",
    "idct",
]


def _select_dtype_for_precision(precision: int):
    """Return an integer dtype able to hold the given precision (bits)."""
    if precision <= 8:
        return np.uint8
    if precision <= 16:
        return np.uint16
    if precision <= 32:
        return np.uint32
    return np.uint64


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
    dtype = _select_dtype_for_precision(precision)
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

def dct(matrix: np.ndarray, norm: str = 'ortho', axis: int = -1) -> np.ndarray:
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
    if norm == 'ortho':
        out[..., 0] *= np.sqrt(1.0 / N)
        out[..., 1:] *= np.sqrt(2.0 / N)
    out = np.moveaxis(out, -1, axis)
    return out.astype(np.float32, copy=False)

def idct(matrix: np.ndarray, norm: str = 'ortho', axis: int = -1) -> np.ndarray:
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
    if norm == 'ortho':
        Y[..., 0] *= np.sqrt(1.0 / N)
        Y[..., 1:] *= np.sqrt(2.0 / N)
    else:
        # Undo scaling expected for unnormalized forward (approximate)
        Y[..., 0] *= 1.0 / (N / 2.0)
    out = np.tensordot(Y, cos_table.T, axes=([-1], [0]))  # shape (..., N)
    out = np.moveaxis(out, -1, axis)
    return out.astype(np.float32, copy=False)