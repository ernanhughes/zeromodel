"""PHOS packing and top-left concentration metrics.

PHOS is implemented here as a pure consumer layer around VPM artifacts. It does
not change artifact identity or source mapping; it creates derived visual fields
that can be measured, rendered, compared, or used by edge consumers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMArtifact


@dataclass(frozen=True)
class PHOSResult:
    raw: np.ndarray
    packed: np.ndarray
    order: Tuple[int, ...]
    top_left_fraction: float
    raw_concentration: float
    packed_concentration: float
    entropy: float
    improved: bool


def robust01(x: np.ndarray, p_lo: float = 10.0, p_hi: float = 90.0) -> np.ndarray:
    """Scale values to [0, 1] with percentile bounds to damp outliers."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.size == 0:
        return arr.copy()
    lo = float(np.nanpercentile(arr, p_lo))
    hi = float(np.nanpercentile(arr, p_hi))
    if hi - lo < 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)


def to_square(vec: np.ndarray, pad_value: float = 0.0) -> np.ndarray:
    """Pad a vector to the next square and reshape it row-major."""
    values = np.asarray(vec, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return np.zeros((1, 1), dtype=np.float64)
    side = int(np.ceil(np.sqrt(values.size)))
    pad = side * side - values.size
    if pad:
        values = np.concatenate([values, np.full(pad, float(pad_value), dtype=np.float64)])
    return values.reshape(side, side)


def top_left_concentration(field: np.ndarray, fraction: float = 0.25) -> float:
    """Return the fraction of total mass inside the top-left area."""
    arr = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    if arr.ndim != 2:
        raise ValueError("top_left_concentration expects a two-dimensional field")
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    side_rows, side_cols = arr.shape
    frac = max(float(fraction), 1e-12)
    rows = max(1, min(side_rows, int(round(np.sqrt(frac) * side_rows))))
    cols = max(1, min(side_cols, int(round(np.sqrt(frac) * side_cols))))
    return float(arr[:rows, :cols].sum() / total)


def image_entropy(field: np.ndarray) -> float:
    """Shannon entropy over normalized non-negative field mass."""
    arr = np.clip(np.asarray(field, dtype=np.float64), 0.0, None)
    total = float(arr.sum())
    if total <= 0:
        return 0.0
    p = (arr / total).reshape(-1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p + 1e-12)))


def phos_sort_pack(values: np.ndarray, *, robust: bool = True) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Sort descending and square-pack values so high mass concentrates first."""
    vec = np.asarray(values, dtype=np.float64).reshape(-1)
    vec01 = robust01(vec) if robust else np.clip(vec, 0.0, 1.0)
    order = tuple(int(i) for i in np.argsort(vec01)[::-1])
    packed = to_square(vec01[list(order)])
    return packed, order


def pack_artifact(artifact: VPMArtifact, *, top_left_fraction: float = 0.25, robust: bool = False) -> PHOSResult:
    """Create a PHOS-packed visual field from a VPM artifact."""
    raw = np.asarray(artifact.normalized_values, dtype=np.float64)
    raw_square = to_square(raw.reshape(-1))
    packed, order = phos_sort_pack(raw.reshape(-1), robust=robust)
    raw_conc = top_left_concentration(raw_square, top_left_fraction)
    packed_conc = top_left_concentration(packed, top_left_fraction)
    return PHOSResult(
        raw=raw_square,
        packed=packed,
        order=order,
        top_left_fraction=float(top_left_fraction),
        raw_concentration=raw_conc,
        packed_concentration=packed_conc,
        entropy=image_entropy(packed),
        improved=packed_conc > raw_conc,
    )


def guarded_pack_artifact(
    artifact: VPMArtifact,
    *,
    top_left_fractions: Sequence[float] = (0.25, 0.16, 0.36, 0.09),
    min_improvement: float = 0.02,
    robust: bool = False,
) -> PHOSResult:
    """Sweep PHOS packing fractions and choose the best honest improvement.

    Preference is given to the first result that improves top-left concentration
    by at least ``min_improvement``. If no candidate clears the guard, the most
    concentrated candidate is returned with ``improved=False``.
    """
    candidates = [
        pack_artifact(artifact, top_left_fraction=fraction, robust=robust)
        for fraction in top_left_fractions
    ]
    guarded = [
        item for item in candidates
        if item.packed_concentration >= item.raw_concentration * (1.0 + float(min_improvement))
    ]
    chosen = max(guarded or candidates, key=lambda item: item.packed_concentration)
    if not guarded:
        return PHOSResult(
            raw=chosen.raw,
            packed=chosen.packed,
            order=chosen.order,
            top_left_fraction=chosen.top_left_fraction,
            raw_concentration=chosen.raw_concentration,
            packed_concentration=chosen.packed_concentration,
            entropy=chosen.entropy,
            improved=False,
        )
    return chosen
