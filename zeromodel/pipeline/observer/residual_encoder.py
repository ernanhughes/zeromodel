#  zeromodel/pipeline/observer/residual_encoder.py
from __future__ import annotations
import numpy as np
from typing import Dict, Any, List, Tuple
from zeromodel.pipeline.executor import PipelineStage

class ResidualEncode(PipelineStage):
    """
    Lightweight perceptual encoder for frames.

    Features:
      - residual: spatial gradient magnitude (edges)
      - temporal: difference vs. previous frame (motion)
      - phase: gradient orientation map

    Consumes context["frames_norm"] or context["frames_in"].
    Writes context["frames_encoded"].
    """

    def __init__(self, L: int = 8, **kwargs):
        super().__init__(L=L, **kwargs)
        self.buf: List[np.ndarray] = []
        self.L = int(L)

    def validate_params(self) -> None:
        if self.L <= 0:
            raise ValueError("L must be > 0")

    def _spatial_features(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute gradient magnitude (residual) and orientation (phase)."""
        gray = frame.mean(axis=2)  # grayscale
        gx = np.pad(np.diff(gray, axis=1), ((0,0),(1,0)))
        gy = np.pad(np.diff(gray, axis=0), ((1,0),(0,0)))
        res = np.sqrt(gx*gx + gy*gy).astype("float32")
        phase = np.arctan2(gy, gx).astype("float32")

        # robust normalize residual
        lo, hi = np.percentile(res, 2.0), np.percentile(res, 98.0)
        if hi > lo:
            res = ((res - lo) / (hi - lo)).clip(0, 1)
        else:
            res = np.zeros_like(res, dtype="float32")

        return res, phase

    def _temporal_features(self, frame: np.ndarray) -> np.ndarray:
        """Difference with previous frame in buffer."""
        if not self.buf:
            return np.zeros(frame.shape[:2], dtype="float32")
        prev = self.buf[-1].mean(axis=2)
        curr = frame.mean(axis=2)
        diff = np.abs(curr - prev).astype("float32")
        # normalize
        if diff.max() > 0:
            diff /= diff.max()
        return diff

    def process(self, X, context: Dict[str, Any]):
        frames = context.get("frames_norm") or context.get("frames_in") or []
        out = []
        for rec in frames:
            f = rec["frame"]

            # update buffer
            self.buf.append(f)
            if len(self.buf) > self.L:
                self.buf.pop(0)

            residual, phase = self._spatial_features(f)
            temporal = self._temporal_features(f)

            out.append({**rec,
                        "residual": residual,
                        "temporal": temporal,
                        "phase": phase})

        context["frames_encoded"] = out
        return X, context
