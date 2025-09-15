#  zeromodel/tools/overlay.py
from __future__ import annotations
import numpy as np
from PIL import Image

def _to_uint8(img):
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def _upsample_to(res_map, H, W, mode=Image.BILINEAR):
    """res_map: (h,w) -> (H,W)"""
    if res_map.ndim != 2:
        raise ValueError("res_map must be 2D (h,w)")
    h, w = res_map.shape
    if (h, w) == (H, W):
        return res_map
    im = Image.fromarray(_to_uint8(_normalize01(res_map) * 255))
    im = im.resize((W, H), resample=mode)
    return np.asarray(im).astype(np.float32) / 255.0

def _normalize01(x, eps=1e-8, robust=True):
    """Normalize to [0,1]; robust uses 2–98 percentile to reduce outlier dominance."""
    x = np.asarray(x, dtype=np.float32)
    if robust:
        lo, hi = np.percentile(x, 2.0), np.percentile(x, 98.0)
        if hi <= lo + eps:
            lo, hi = x.min(), x.max()
    else:
        lo, hi = x.min(), x.max()
    if hi <= lo + eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo + eps)

def _colormap_magma(x):
    """
    Tiny magma-like LUT (256x3) without external deps.
    Values in x are expected in [0,1]; returns uint8 RGB.
    """
    # Key control points (k, r,g,b) sampled from magma; linear interpolate
    # (to keep it short, we use a compact LUT blended from a few anchors)
    anchors = np.array([
        [0.00, 0.001, 0.000, 0.015],
        [0.10, 0.089, 0.023, 0.237],
        [0.25, 0.286, 0.059, 0.388],
        [0.50, 0.642, 0.121, 0.330],
        [0.75, 0.902, 0.349, 0.155],
        [0.90, 0.987, 0.703, 0.106],
        [1.00, 0.988, 0.998, 0.644],
    ], dtype=np.float32)
    xi = np.clip(x, 0.0, 1.0).ravel()
    # Find segments
    ks = anchors[:,0]
    r = np.interp(xi, ks, anchors[:,1])
    g = np.interp(xi, ks, anchors[:,2])
    b = np.interp(xi, ks, anchors[:,3])
    rgb = np.stack([r,g,b], axis=-1).reshape((*x.shape, 3))
    return _to_uint8(rgb)

def _apply_alpha_over(base_rgb_u8, heat_rgb_u8, alpha=0.4):
    base = base_rgb_u8.astype(np.float32)
    heat = heat_rgb_u8.astype(np.float32)
    out  = (1.0 - alpha) * base + alpha * heat
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)

def overlay_residual(frame_hwC, res_map_hw, *, alpha=0.4, thresh=None, robust=True):
    """
    Blend a colored residual heatmap over a base frame.

    Args:
      frame_hwC: np.ndarray [H,W,C], float in [0,1] or uint8.
      res_map_hw: np.ndarray [H,W] or [Hp,Wp] residual values (any real range).
      alpha: blend strength for heatmap overlay (0..1).
      thresh: optional float in [0,1]; values below are suppressed (after normalization).
      robust: use robust normalization (2–98 percentile) for residuals.
    Returns:
      PIL.Image (RGB)
    """
    if frame_hwC.ndim != 3 or frame_hwC.shape[2] not in (1,3,4):
        raise ValueError("frame_hwC must be HxWx{1,3,4}")
    H, W = frame_hwC.shape[:2]
    # Prepare base RGB
    if frame_hwC.shape[2] == 1:
        base_rgb = np.repeat(frame_hwC, 3, axis=2)
    else:
        base_rgb = frame_hwC[..., :3]
    base_rgb_u8 = _to_uint8(base_rgb)

    # Upsample & normalize residuals to [0,1]
    if res_map_hw.shape != (H, W):
        res_map = _upsample_to(res_map_hw, H, W)
    else:
        res_map = res_map_hw.astype(np.float32)
        res_map = _normalize01(res_map, robust=robust)

    # Optional threshold: keep only salient regions
    if thresh is not None:
        mask = (res_map >= float(thresh)).astype(np.float32)
        res_map = res_map * mask

    # Colorize
    heat_rgb_u8 = _colormap_magma(res_map)

    # Blend
    out_u8 = _apply_alpha_over(base_rgb_u8, heat_rgb_u8, alpha=alpha)
    return Image.fromarray(out_u8, mode="RGB")
