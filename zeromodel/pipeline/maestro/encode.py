# zeromodel/maestro/encode.py
"""
MAESTRO-ZM Encoding Stage for ZeroModel pipelines.

This stage converts input frames into a lightweight encoding signal
(residual + phase). Residual is computed as normalized gradient magnitude
(edge strength proxy), and phase is currently a placeholder (0).

Intended as a minimal bridge between raw frames and downstream encoders
or defect detectors.

Author: ZeroModel Project
"""

import logging
from typing import Any, Dict, List

import numpy as np
from zeromodel.pipeline.executor import PipelineStage

logger = logging.getLogger(__name__)


class Encode(PipelineStage):
    """
    Minimal encoder stage producing residual (gradient magnitude)
    and phase (currently 0) from input frames.

    Args:
        L: buffer length for temporal context (default: 8)

    Input Context:
        context["frames_norm"] or context["frames_in"]:
            list of {"frame": np.ndarray (H, W, C)}

    Output Context:
        context["frames_maestro"]:
            list of {"frame": np.ndarray, "residual": np.ndarray, "phase": int}
    """

    name = "encode"

    def __init__(self, L: int = 8, **kwargs):
        super().__init__(L=L, **kwargs)
        self.buf: List[np.ndarray] = []
        self.L = int(L)

    def validate_params(self) -> None:
        if self.L <= 0:
            raise ValueError("L must be > 0")

    def _compute_residual(self, frame_hwC: np.ndarray) -> np.ndarray:
        """
        Compute residual as gradient magnitude of grayscale frame.
        Normalized to [0,1] using robust percentiles.
        """
        gray = frame_hwC.mean(axis=2)
        gx = np.pad(np.diff(gray, axis=1), ((0, 0), (1, 0)))
        gy = np.pad(np.diff(gray, axis=0), ((1, 0), (0, 0)))
        res = np.sqrt(gx * gx + gy * gy).astype("float32")

        # Robust normalization
        lo, hi = np.percentile(res, 2.0), np.percentile(res, 98.0)
        if hi <= lo:
            return np.zeros_like(res, dtype="float32")

        normed = ((res - lo) / (hi - lo)).clip(0, 1)
        return normed.astype("float32")

    def process(self, X, context: Dict[str, Any]):
        frames = (
            context.get("frames_norm")
            or context.get("frames_in")
            or []
        )
        if not frames:
            logger.warning("[%s] No input frames provided", self.name)
            context["frames_maestro"] = []
            return X, context

        out = []
        for idx, rec in enumerate(frames):
            f = rec["frame"]

            # Maintain sliding buffer
            self.buf.append(f)
            if len(self.buf) > self.L:
                self.buf.pop(0)

            # Compute encoding
            res = self._compute_residual(f)
            out_rec = {**rec, "residual": res, "phase": 0}
            out.append(out_rec)

            logger.debug(
                "[%s] Frame %d encoded: residual shape=%s, phase=%d",
                self.name, idx, res.shape, out_rec["phase"]
            )

        context["frames_maestro"] = out
        logger.debug("[%s] Encoded %d frames", self.name, len(out))
        return X, context
