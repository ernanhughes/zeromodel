"""Hierarchical VPM pyramid helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List

import numpy as np

from zeromodel.analysis.compose import as_field


@dataclass(frozen=True)
class HierarchyLevel:
    level: int
    field: np.ndarray
    reduction: str
    source_shape: tuple[int, int]


def reduce_blocks(
    field: Any, *, factor: int = 2, reduction: str = "mean"
) -> np.ndarray:
    """Reduce a field by non-overlapping blocks, padding bottom/right as needed."""
    arr = as_field(field)
    if factor < 2:
        raise ValueError("factor must be >= 2")
    rows, cols = arr.shape
    pad_rows = (-rows) % factor
    pad_cols = (-cols) % factor
    if pad_rows or pad_cols:
        arr = np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode="constant")
    new_rows = arr.shape[0] // factor
    new_cols = arr.shape[1] // factor
    blocks = arr.reshape(new_rows, factor, new_cols, factor)
    if reduction == "mean":
        return blocks.mean(axis=(1, 3))
    if reduction == "max":
        return blocks.max(axis=(1, 3))
    if reduction == "sum":
        return np.clip(blocks.sum(axis=(1, 3)), 0.0, 1.0)
    raise ValueError("Unsupported reduction: %r" % reduction)


def build_pyramid(
    field: Any,
    *,
    max_levels: int | None = None,
    factor: int = 2,
    reduction: str = "mean",
) -> List[HierarchyLevel]:
    """Build a bounded hierarchy from a field.

    Level 0 is the original field; each later level reduces by ``factor`` until
    the field is 1x1 or ``max_levels`` is reached.
    """
    current = as_field(field)
    levels = [
        HierarchyLevel(
            level=0, field=current, reduction="source", source_shape=current.shape
        )
    ]
    level = 0
    while current.shape != (1, 1):
        if max_levels is not None and level + 1 >= max_levels:
            break
        reduced = reduce_blocks(current, factor=factor, reduction=reduction)
        level += 1
        levels.append(
            HierarchyLevel(
                level=level,
                field=reduced,
                reduction=reduction,
                source_shape=current.shape,
            )
        )
        current = reduced
    return levels
