from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class TemporalDifference(PipelineStage):
    name = "temporal_diff"
    category = "temporal"

    def __init__(self, **params):
        super().__init__(**params)
        self.pad_mode = str(params.get("pad_mode", "edge"))  # how to handle first frame

    def validate_params(self):
        assert self.pad_mode in ("edge", "zero", "repeat_first")

    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        context = self._get_context(context)
        if vpm.ndim != 3:
            # no-op on non-temporal data
            return vpm, {"warning": "requires 3D VPM", "applied": False}

        T, N, M = vpm.shape
        diff = np.empty_like(vpm)
        diff[1:] = vpm[1:] - vpm[:-1]
        if self.pad_mode == "edge":
            diff[0] = diff[1]
        elif self.pad_mode == "zero":
            diff[0] = 0.0
        else:  # repeat_first
            diff[0] = vpm[0]
        return diff, {
            "pad_mode": self.pad_mode,
            "input_shape": vpm.shape,
            "output_shape": diff.shape,
            "applied": True,
        }
