"""Explicit, shape-checked visual composition operators for VPM fields."""
from __future__ import annotations

from typing import Any

import numpy as np

from .artifact import VPMArtifact


def as_field(value: Any) -> np.ndarray:
    """Return a normalized float field from an artifact or array-like input."""
    if isinstance(value, VPMArtifact):
        arr = value.normalized_values
    else:
        arr = value
    field = np.asarray(arr, dtype=np.float64)
    if field.ndim != 2:
        raise ValueError("VPM composition expects a two-dimensional field")
    return np.clip(field, 0.0, 1.0)


def _pair(left: Any, right: Any) -> tuple[np.ndarray, np.ndarray]:
    a = as_field(left)
    b = as_field(right)
    if a.shape != b.shape:
        raise ValueError("VPM fields must have identical shapes; got %s and %s" % (a.shape, b.shape))
    return a, b


def vpm_and(left: Any, right: Any) -> np.ndarray:
    """Fuzzy visual AND: cell-wise minimum."""
    a, b = _pair(left, right)
    return np.minimum(a, b)


def vpm_or(left: Any, right: Any) -> np.ndarray:
    """Fuzzy visual OR: cell-wise maximum."""
    a, b = _pair(left, right)
    return np.maximum(a, b)


def vpm_not(value: Any) -> np.ndarray:
    """Fuzzy visual NOT: one minus normalized intensity."""
    return 1.0 - as_field(value)


def vpm_xor(left: Any, right: Any) -> np.ndarray:
    """Visual XOR/contrast: absolute cell-wise difference."""
    a, b = _pair(left, right)
    return np.abs(a - b)


def vpm_add(left: Any, right: Any, *, clip: bool = True) -> np.ndarray:
    """Cell-wise additive enrichment."""
    a, b = _pair(left, right)
    out = a + b
    return np.clip(out, 0.0, 1.0) if clip else out


def vpm_subtract(left: Any, right: Any, *, floor: bool = True) -> np.ndarray:
    """Cell-wise subtraction. With ``floor=True`` negative values become zero."""
    a, b = _pair(left, right)
    out = a - b
    return np.clip(out, 0.0, 1.0) if floor else out
