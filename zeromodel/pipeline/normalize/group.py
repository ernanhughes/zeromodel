import numpy as np
from typing import Dict, Any, List, Tuple
from zeromodel.pipeline.executor import PipelineStage

class GroupNormalizeFrames(PipelineStage):
    """
    Per-frame z-score normalization over channel groups.
    Each group's mean/std are computed over spatial dims (H,W) **per channel**.
    Example groups: [[0,3], [3,5]] for RGB and 2 extra planes.
    Writes to context[output_key].
    """

    name = "group_normalize"
    category = "prep"


    def __init__(self, groups: List[List[int]], output_key: str = "frames_norm", eps: float = 1e-6, **kwargs):
        super().__init__(groups=groups, output_key=output_key, eps=eps, **kwargs)
        self.slices: List[Tuple[int,int]] = [(int(s), int(e)) for s, e in groups]
        self.output_key = output_key
        self.eps = eps

    def validate_params(self) -> None:
        if not self.slices or any(s < 0 or e <= s for s, e in self.slices):
            raise ValueError("groups must be non-empty [[start,end), ...] with start<end")

    def process(self, X, context: Dict[str, Any]):
        frames = context.get("frames_in", [])
        if not frames:
            context[self.output_key] = []
            return X, context

        # Validate against actual channel count
        H, W, C = frames[0]["frame"].shape
        for s, e in self.slices:
            if e > C:
                raise ValueError(f"group [{s},{e}) exceeds channels C={C}")

        out = []
        for rec in frames:
            f = rec["frame"].astype(np.float32, copy=False)  # ensure float
            gframe = f.copy()
            for s, e in self.slices:
                g = gframe[..., s:e]                    # (H,W,G)
                mu = g.mean(axis=(0,1), keepdims=True)  # per-channel
                sd = g.std(axis=(0,1), keepdims=True) + self.eps
                gframe[..., s:e] = (g - mu) / sd
            out.append({**rec, "frame": gframe})
        context[self.output_key] = out
        return X, context
