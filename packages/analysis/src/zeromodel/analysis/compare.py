"""Differential analysis helpers for VPM artifacts and fields."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from zeromodel.analysis.compose import as_field, vpm_add, vpm_and, vpm_subtract, vpm_xor


@dataclass(frozen=True)
class VPMComparison:
    diff: np.ndarray
    overlap: np.ndarray
    contrast: np.ndarray
    enriched: np.ndarray
    gain: float
    loss: float
    improvement_ratio: float


def compare_fields(baseline: Any, target: Any, *, overlap_weight: float = 0.25) -> VPMComparison:
    """Compare a baseline field with a target field.

    ``gain`` is positive target-minus-baseline mass; ``loss`` is negative mass;
    ``improvement_ratio`` is gain / (gain + loss).
    """
    base = as_field(baseline)
    tgt = as_field(target)
    if base.shape != tgt.shape:
        raise ValueError("VPM fields must have identical shapes; got %s and %s" % (base.shape, tgt.shape))

    signed = tgt - base
    gain = float(np.sum(np.clip(signed, 0.0, None)))
    loss = float(np.sum(np.clip(-signed, 0.0, None)))
    total = gain + loss + 1e-12
    overlap = vpm_and(base, tgt)
    contrast = vpm_xor(base, tgt)
    enriched = vpm_add(vpm_subtract(tgt, base), overlap * float(overlap_weight))
    return VPMComparison(
        diff=signed,
        overlap=overlap,
        contrast=contrast,
        enriched=enriched,
        gain=gain,
        loss=loss,
        improvement_ratio=float(gain / total),
    )
