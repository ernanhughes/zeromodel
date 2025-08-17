from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class StandardScaler(PipelineStage):
    name = "standard"
    category = "normalizer"

    def __init__(self, **params):
        super().__init__(**params)
        self.per_column = bool(params.get("per_column", True))
        self.eps = float(params.get("eps", 1e-12))

    def validate_params(self):
        assert self.eps > 0

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None):
        context = self._get_context(context)
        if vpm.ndim == 2:
            out = self._scale2d(vpm)
        elif vpm.ndim == 3:
            out = np.stack([self._scale2d(vpm[t]) for t in range(vpm.shape[0])], axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")
        return out, {"per_column": self.per_column, "eps": self.eps,
                     "input_shape": vpm.shape, "output_shape": out.shape}

    def _scale2d(self, X: np.ndarray) -> np.ndarray:
        if self.per_column:
            mu = X.mean(axis=0, keepdims=True)
            sd = X.std(axis=0, keepdims=True)
        else:
            mu = X.mean()
            sd = X.std()
        return (X - mu) / (sd + self.eps)
