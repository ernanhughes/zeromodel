import numpy as np
from typing import Dict, Any, List, Tuple
from zeromodel.pipeline.executor import PipelineStage

class GroupNormalizeFrames(PipelineStage):
    """
    Per-frame, per-group z-score over channels.
    groups: list of [start, end) slices, e.g., [[0,8],[8,11]]
    Writes to context["frames_norm"].
    """

    def __init__(self, groups: List[List[int]], **kwargs):
        super().__init__(groups=groups, **kwargs)
        self.slices: List[Tuple[int,int]] = [(int(s), int(e)) for s, e in groups]

    def validate_params(self) -> None:
        if not self.slices or any(s < 0 or e <= s for s, e in self.slices):
            raise ValueError("groups must be non-empty [[start,end), ...] with start<end")

    def process(self, X, context: Dict[str, Any]):
        frames = context.get("frames_in", [])
        out = []
        for rec in frames:
            f = rec["frame"]  # (H,W,C)
            gframe = f.copy()
            for s, e in self.slices:
                g = gframe[..., s:e]
                mu = g.mean()
                sd = g.std() + 1e-6
                gframe[..., s:e] = (g - mu) / sd
            out.append({**rec, "frame": gframe})
        context["frames_norm"] = out
        return X, context
