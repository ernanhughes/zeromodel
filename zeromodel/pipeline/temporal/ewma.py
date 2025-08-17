from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class EWMASmoother(PipelineStage):
    name = "ewma"
    category = "temporal"

    def __init__(self, **params):
        super().__init__(**params)
        self.alpha = float(params.get("alpha", 0.2))  # 0<alpha<=1

    def validate_params(self):
        assert 0 < self.alpha <= 1

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None):
        context = self._get_context(context)
        if vpm.ndim != 3:
            return vpm, {"warning": "requires 3D VPM", "applied": False}
        T, N, M = vpm.shape
        out = np.empty_like(vpm)
        out[0] = vpm[0]
        a = self.alpha
        for t in range(1, T):
            out[t] = a * vpm[t] + (1 - a) * out[t - 1]
        return out, {
            "alpha": self.alpha,
            "input_shape": vpm.shape,
            "output_shape": out.shape,
            "applied": True,
        }
