import numpy as np
from typing import Dict, Any, List
from zeromodel.pipeline.executor import PipelineStage

class Encode(PipelineStage):
    """
    Placeholder MAESTRO-ZM online encoder.
    Emits residual = gradient magnitude (dummy), phase=0.
    Consumes context["frames_norm"] or context["frames_in"].
    Writes context["frames_maestro"].
    """

    def __init__(self, L: int = 8, **kwargs):
        super().__init__(L=L, **kwargs)
        self.buf: List[np.ndarray] = []
        self.L = int(L)

    def validate_params(self) -> None:
        if self.L <= 0:
            raise ValueError("L must be > 0")

    def _residual_dummy(self, frame_hwC: np.ndarray) -> np.ndarray:
        gray = frame_hwC.mean(axis=2)
        gx = np.pad(np.diff(gray, axis=1), ((0,0),(1,0)))
        gy = np.pad(np.diff(gray, axis=0), ((1,0),(0,0)))
        res = np.sqrt(gx*gx + gy*gy).astype("float32")
        # robust normalize
        lo, hi = np.percentile(res, 2.0), np.percentile(res, 98.0)
        if hi <= lo: return np.zeros_like(res, dtype="float32")
        return ((res - lo) / (hi - lo)).astype("float32")

    def process(self, X, context: Dict[str, Any]):
        frames = context.get("frames_norm") or context.get("frames_in") or []
        out = []
        for rec in frames:
            f = rec["frame"]
            self.buf.append(f)
            if len(self.buf) > self.L:
                self.buf.pop(0)
            res = self._residual_dummy(f)
            out.append({**rec, "residual": res, "phase": 0})
        context["frames_maestro"] = out
        return X, context
