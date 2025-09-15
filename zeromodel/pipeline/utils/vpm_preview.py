#  zeromodel/pipeline/utils/vpm_preview.py
"""
Utilities for creating VPM previews for GIF logging.
"""
from __future__ import annotations


import numpy as np


def _vpm_preview_uint8(
    vpm: np.ndarray | None, p_lo: float = 1.0, p_hi: float = 99.0
) -> np.ndarray:
    """
    Convert a 2D or 3D VPM into a small RGB uint8 image for logging.

    Implements ZeroModel's "boring by design" principle:
    "It's just a PNG with a tiny header. Survives image pipelines, is easy to cache and diff."

    Args:
        vpm: Input VPM (2D or 3D). If None, returns a black placeholder.
        p_lo: Lower percentile for contrast stretch
        p_hi: Upper percentile for contrast stretch

    Returns:
        RGB preview image as uint8 array
    """
    if vpm is None:
        # Return black placeholder (32×32) if no VPM available
        return np.zeros((32, 32, 3), dtype=np.uint8)

    x = vpm
    if x.ndim == 3:
        # Choose first time frame by default
        x = x[0]
    elif x.ndim != 2:
        raise ValueError(f"VPM preview expects 2D (or 3D->2D slice), got {vpm.ndim}D")

    x = x.astype(np.float32)
    lo = np.percentile(x, p_lo)
    hi = np.percentile(x, p_hi)
    if hi <= lo:
        hi = lo + 1e-6
    y = np.clip((x - lo) / (hi - lo), 0, 1)  # [0..1]

    # Make a pleasant 3-channel preview with gamma correction
    r = (y**0.9) * 255.0
    g = (y**0.8) * 255.0
    b = (y**0.7) * 255.0
    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    return rgb


def _choose_best_frame(
    vpm: np.ndarray, Kc: int = 12, Kr: int = 48, alpha: float = 0.97
) -> int:
    """
    Choose the frame with highest top-left mass from a 3D VPM.

    This implements ZeroModel's "top-left rule" for signal concentration.
    """
    from zeromodel.vpm.stdm import top_left_mass

    if vpm.ndim != 3:
        return 0

    scores = [
        top_left_mass(vpm[t], Kr=Kr, Kc=Kc, alpha=alpha) for t in range(vpm.shape[0])
    ]
    return int(np.argmax(scores))