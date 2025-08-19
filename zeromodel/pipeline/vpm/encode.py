# zeromodel/pipeline/vpm/encode.py
from __future__ import annotations
import io
import json
import numpy as np
from typing import Any, Dict, Tuple
from PIL import Image
from zeromodel.pipeline.base import PipelineStage

class VPMEncode(PipelineStage):
    """
    Normalize/encode a numeric VPM to a displayable uint8 image.
    - mode: 'auto' (percentiles), 'minmax', or 'fixed'
    - channels: 1 (grayscale) or 3 (RGB)
    - write_png_to_context: if True, places PNG bytes in context['vpm_png_bytes']
    - store_ndarray: if True, stores uint8 array in context['vpm_uint8']
    """
    name = "vpm_encode"
    category = "vpm"

    def __init__(self, **params):
        super().__init__(**params)
        self.mode = params.get("mode", "auto")            # 'auto'|'minmax'|'fixed'
        self.channels = int(params.get("channels", 1))    # 1 or 3
        self.p_lo = float(params.get("p_lo", 0.5))        # for 'auto'
        self.p_hi = float(params.get("p_hi", 99.5))       # for 'auto'
        self.v_min = params.get("v_min")                  # for 'fixed'
        self.v_max = params.get("v_max")                  # for 'fixed'
        self.write_png_to_context = bool(params.get("write_png_to_context", True))
        self.store_ndarray = bool(params.get("store_ndarray", True))

    def validate_params(self):
        assert self.channels in (1, 3), "channels must be 1 or 3"
        assert self.mode in ("auto", "minmax", "fixed"), "mode invalid"
        if self.mode == "fixed":
            assert self.v_min is not None and self.v_max is not None and self.v_max > self.v_min

    def _normalize(self, X: np.ndarray) -> np.ndarray:
        Xf = X.astype(np.float32, copy=False)
        if self.mode == "auto":
            lo = float(np.percentile(Xf, self.p_lo))
            hi = float(np.percentile(Xf, self.p_hi))
        elif self.mode == "minmax":
            lo, hi = float(np.min(Xf)), float(np.max(Xf))
        else:  # fixed
            lo, hi = float(self.v_min), float(self.v_max)
        if hi <= lo:
            hi = lo + 1e-6
        Y = np.clip((Xf - lo) / (hi - lo), 0.0, 1.0)
        return (Y * 255.0 + 0.5).astype(np.uint8)

    def _to_image(self, U: np.ndarray) -> Image.Image:
        if U.ndim == 2 and self.channels == 1:
            return Image.fromarray(U, mode="L")
        if U.ndim == 2 and self.channels == 3:
            U3 = np.stack([U, U, U], axis=-1)
            return Image.fromarray(U3, mode="RGB")
        if U.ndim == 3 and U.shape[-1] == 3:
            return Image.fromarray(U, mode="RGB")
        raise ValueError(f"Unsupported shape for to_image: {U.shape}")

    def process(self, vpm: np.ndarray, context: Dict[str, Any] | None = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        context = self.get_context(context)

        # If VPM is time series, flatten with a grid is out-of-scope; encode first frame
        X = vpm
        if X.ndim == 3 and X.shape[-1] not in (1, 3):
            # shape (T,H,W) → just encode a per-frame montage later; here pick the mean frame
            X = X.mean(axis=0)
        if X.ndim == 4:
            X = X.mean(axis=0)  # (T,H,W,C) → (H,W,C)

        if X.ndim == 3 and X.shape[-1] in (1, 3):
            U = self._normalize(X[..., 0] if X.shape[-1] == 1 else X[..., :3])
        else:
            U = self._normalize(X)

        img = self._to_image(U)
        png_bytes = None
        if self.write_png_to_context:
            buf = io.BytesIO()
            img.save(buf, format="PNG", optimize=True)
            png_bytes = buf.getvalue()
            context["vpm_png_bytes"] = png_bytes

        if self.store_ndarray:
            if img.mode == "L":
                context["vpm_uint8"] = np.array(img, dtype=np.uint8)
            else:
                context["vpm_uint8"] = np.array(img.convert("RGB"), dtype=np.uint8)

        meta = {
            "mode": self.mode,
            "channels": self.channels,
            "input_shape": tuple(vpm.shape),
            "encoded_shape": tuple(img.size[::-1]) + ((3,) if img.mode == "RGB" else (1,)),
            "p_lo": self.p_lo,
            "p_hi": self.p_hi,
            "v_min": self.v_min,
            "v_max": self.v_max,
            "wrote_png_bytes": self.write_png_to_context,
            "stored_uint8": self.store_ndarray,
        }
        return vpm, meta
