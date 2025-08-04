"""
Utility Functions

This module provides helper functions used throughout the zeromodel package.
"""

from typing import Any

import numpy as np


def quantize(value: Any, precision: int) -> Any:
    """
    Quantize values to specified bit precision.
    
    Args:
        value: Value or array to quantize
        precision: Bit precision (4-16)
    
    Returns:
        Quantized value(s)
    """
    max_val = (1 << precision) - 1
    if isinstance(value, np.ndarray):
        return np.round(value * max_val).astype(np.uint8)
    else:
        return int(value * max_val)

def dct(matrix: np.ndarray, norm: str = 'ortho', axis: int = -1) -> np.ndarray:
    """
    Discrete Cosine Transform (simplified implementation).
    
    Note: In production, use scipy.fft.dct for better performance.
    This is a minimal implementation for edge compatibility.
    
    Args:
        matrix: Input matrix
        norm: Normalization mode
        axis: Axis along which to compute DCT
    
    Returns:
        DCT of input
    """
    # Simplified DCT implementation for edge compatibility
    # In production, replace with scipy.fft.dct
    n = matrix.shape[axis]
    k = np.arange(n)
    result = np.zeros_like(matrix)
    
    for i in range(n):
        if axis == 0:
            result[i] = np.sum(matrix * np.cos(np.pi * k * i / n), axis=0)
        else:
            result[:, i] = np.sum(matrix * np.cos(np.pi * k * i / n), axis=1)
    
    if norm == 'ortho':
        result[0] *= np.sqrt(0.5)
        result[1:] *= 1.0
    
    return result

def idct(matrix: np.ndarray, norm: str = 'ortho', axis: int = -1) -> np.ndarray:
    """
    Inverse Discrete Cosine Transform (simplified implementation).
    
    Note: In production, use scipy.fft.idct for better performance.
    This is a minimal implementation for edge compatibility.
    
    Args:
        matrix: Input matrix
        norm: Normalization mode
        axis: Axis along which to compute IDCT
    
    Returns:
        IDCT of input
    """
    # Simplified IDCT implementation for edge compatibility
    # In production, replace with scipy.fft.idct
    n = matrix.shape[axis]
    k = np.arange(n)
    result = np.zeros_like(matrix)
    
    for i in range(n):
        if axis == 0:
            result[i] = 0.5 * matrix[0] + np.sum(matrix[1:] * np.cos(np.pi * k[1:] * i / n), axis=0)
        else:
            result[:, i] = 0.5 * matrix[:, 0] + np.sum(matrix[:, 1:] * np.cos(np.pi * k[1:] * i / n), axis=1)
    
    result *= 2.0 / n
    
    if norm == 'ortho':
        result[0] *= np.sqrt(0.5)
    
    return result