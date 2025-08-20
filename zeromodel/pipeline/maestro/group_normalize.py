# zeromodel/maestro/group_normalize.py
"""
Group-wise frame normalization stage for ZeroModel MAESTRO pipelines.

This stage performs z-score normalization over selected channel ranges,
either per-frame or across the entire batch. Useful for stabilizing
multi-channel inputs (e.g. RGB + metric channels) before encoding.

Author: ZeroModel Project
"""

import logging
from typing import Any, Dict, List, Tuple
import numpy as np
from zeromodel.pipeline.executor import PipelineStage

logger = logging.getLogger(__name__)


class GroupNormalizeFrames(PipelineStage):
    """
    Group-wise z-score normalization for frames.

    Args:
        groups: list of [start, end) slices, e.g. [[0,3],[3,8]].
            Each slice defines a contiguous channel group.
        mode: "frame" (default) → normalize per frame
              "batch" → normalize over all frames
        inplace: if True, modifies input frames in place (saves memory)

    Input Context:
        context["frames_in"]: list of {"frame": np.ndarray (H, W, C)}

    Output Context:
        context["frames_norm"]: list of {"frame": np.ndarray (H, W, C)}
    """

    name = "group_normalize"

    def __init__(self, groups: List[List[int]], mode: str = "frame", inplace: bool = False, **kwargs):
        super().__init__(groups=groups, mode=mode, inplace=inplace, **kwargs)
        self.slices: List[Tuple[int, int]] = [(int(s), int(e)) for s, e in groups]
        self.mode = mode
        self.inplace = inplace

    def validate_params(self) -> None:
        if not self.slices:
            raise ValueError("groups must be non-empty [[start,end), ...]")
        if self.mode not in ("frame", "batch"):
            raise ValueError("mode must be 'frame' or 'batch'")
        for s, e in self.slices:
            if s < 0 or e <= s:
                raise ValueError(
                    f"invalid slice ({s},{e}): must satisfy 0 <= start < end"
                )

    def _normalize_array(self, arr: np.ndarray, axis=None) -> np.ndarray:
        """Z-score normalize array with safe std."""
        mu = arr.mean(axis=axis, keepdims=True)
        sd = arr.std(axis=axis, keepdims=True) + 1e-6
        return (arr - mu) / sd

    def process(self, X, context: Dict[str, Any]):
        frames = context.get("frames_in", [])
        if not frames:
            logger.warning("[%s] No frames_in provided", self.name)
            context["frames_norm"] = []
            return X, context

        # Batch stats pre-compute (if mode=batch)
        if self.mode == "batch":
            batch_stats = {}
            for s, e in self.slices:
                stack = np.stack([rec["frame"][..., s:e] for rec in frames], axis=0)
                mu = stack.mean()
                sd = stack.std() + 1e-6
                batch_stats[(s, e)] = (mu, sd)
            logger.debug("[%s] Computed batch stats for %d groups", self.name, len(batch_stats))
        else:
            batch_stats = None

        out = []
        for idx, rec in enumerate(frames):
            f = rec["frame"]
            gframe = f if self.inplace else f.copy()

            for gi, (s, e) in enumerate(self.slices):
                g = gframe[..., s:e]

                if self.mode == "frame":
                    g_norm = self._normalize_array(g, axis=(0, 1))
                    mu, sd = g.mean(), g.std()
                else:  # batch
                    mu, sd = batch_stats[(s, e)]
                    g_norm = (g - mu) / sd

                gframe[..., s:e] = g_norm

                logger.debug(
                    "[%s] Frame %d, group %d: mean=%.4f, std=%.4f",
                    self.name, idx, gi, mu, sd
                )

            rec_out = dict(rec) if not self.inplace else rec
            rec_out["frame"] = gframe
            out.append(rec_out)

        context["frames_norm"] = out
        logger.debug("[%s] Completed normalization (%s mode, inplace=%s)",
                     self.name, self.mode, self.inplace)
        return X, context
