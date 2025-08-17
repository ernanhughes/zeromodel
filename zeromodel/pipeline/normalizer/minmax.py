from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class MinMaxNormalizer(PipelineStage):
    name = "minmax"
    category = "normalizer"

    def __init__(self, **params):
        super().__init__(**params)
        self.eps = float(params.get("eps", 1e-12))
        self.per_column = bool(params.get("per_column", True))  # if True: per-feature (axis=0)

    def validate_params(self):
        assert self.eps > 0

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        context = self._get_context(context)
        if vpm.ndim == 2:
            out = self._norm2d(vpm)
        elif vpm.ndim == 3:
            out = np.stack([self._norm2d(vpm[t]) for t in range(vpm.shape[0])], axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")

        meta = {
            "per_column": self.per_column,
            "eps": self.eps,
            "input_shape": vpm.shape,
            "output_shape": out.shape,
        }
        return out, meta

    def _norm2d(self, X: np.ndarray) -> np.ndarray:
        if self.per_column:
            mn = X.min(axis=0, keepdims=True)
            mx = X.max(axis=0, keepdims=True)
        else:
            mn = X.min()
            mx = X.max()
        return (X - mn) / (mx - mn + self.eps)
