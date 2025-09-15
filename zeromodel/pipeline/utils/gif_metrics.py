#  zeromodel/pipeline/utils/gif_metrics.py
"""
Metrics adapter for GIF logging.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def _gif_metrics(
    step: int, vpm: np.ndarray, tl_value: Optional[float] = None, tag: str = ""
) -> Dict[str, Any]:
    """
    Create metrics dictionary for GIF logging.

    This implements ZeroModel's "deterministic, reproducible provenance" principle:
    "A core tenet of ZeroModel is that the system's output should be inherently understandable."
    """
    v = vpm
    if v.ndim == 3:
        v = v[0]

    return {
        "step": step,
        "loss": float(-tl_value) if tl_value is not None else float(np.nan),
        "val_loss": float(np.nan),
        "acc": float(np.mean(v > np.percentile(v, 90))),
        "alerts": {},
        "tag": tag
    }