import numpy as np
from typing import Dict, Any, List
from PIL import Image
from zeromodel.pipeline.base import PipelineStage

# Minimal, using the overlay we shared earlier (inlined for convenience)
def _normalize01(x, eps=1e-8, robust=True):
    x = x.astype("float32")
    if robust:
        lo, hi = np.percentile(x, 2.0), np.percentile(x, 98.0)
        if hi <= lo + eps:
            lo, hi = x.min(), x.max()
    else:
        lo, hi = x.min(), x.max()
    if hi <= lo + eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - lo) / (hi - lo + eps)

def _to_uint8(img):
    if img.dtype == np.uint8: return img
    img = np.clip(img, 0.0, 1.0)
    return (img * 255.0 + 0.5).astype(np.uint8)

def _colormap_magma(x):
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
    ks = anchors[:,0]
    r = np.interp(xi, ks, anchors[:,1])
    g = np.interp(xi, ks, anchors[:,2])
    b = np.interp(xi, ks, anchors[:,3])
    rgb = np.stack([r,g,b], axis=-1).reshape((*x.shape, 3))
    return _to_uint8(rgb)

def overlay_residual(frame_hwC, res_map_hw, alpha=0.4):
    H, W = frame_hwC.shape[:2]
    base = frame_hwC[..., :3]
    base_u8 = _to_uint8(base)
    res = res_map_hw
    if res.shape != (H, W):
        res = np.array(Image.fromarray(_to_uint8(_normalize01(res)*255)).resize((W,H), Image.BILINEAR)) / 255.0
    else:
        res = _normalize01(res)
    heat = _colormap_magma(res)
    out = (1.0 - alpha) * base_u8.astype("float32") + alpha * heat.astype("float32")
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype("uint8"), mode="RGB")

class OverlayResidualStage(PipelineStage):
    """
    Blends residual heatmap on each frame; stores PIL.Images in context['frames_overlay'].
    """
    name = "OverlayResidual"


    def __init__(self, alpha: float = 0.4):
        self.alpha = float(alpha)

    def validate_params(self):
        pass

    def process(self, X, context: Dict[str, Any]):
        frames = context.get("frames_maestro", [])
        rendered = []
        for rec in frames:
            img = overlay_residual(rec["frame"], rec["residual"], alpha=self.alpha)
            rendered.append({**rec, "image": img})
        context["frames_overlay"] = rendered
        return X, context
