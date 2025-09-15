#  zeromodel/pipeline/selector/topk.py
from __future__ import annotations
from typing import Any, Dict

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class TopKSelector(PipelineStage):
    """
    Keep only top-K columns (features) by a simple score (mean/var/sum), zero others.
    Useful to hard-focus before STDM or after normalization.
    """

    name = "topk"
    category = "selector"

    def __init__(self, **params):
        super().__init__(**params)
        self.K = int(params.get("K", 12))
        self.metric = str(params.get("metric", "variance"))  # variance|mean|sum

    def validate_params(self):
        assert self.K >= 0
        assert self.metric in ("variance", "mean", "sum")

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None):
        context = self.get_context(context)

        def mask_cols(X: np.ndarray) -> np.ndarray:
            if self.metric == "variance":
                s = X.var(axis=0)
            elif self.metric == "mean":
                s = X.mean(axis=0)
            else:
                s = X.sum(axis=0)
            k = min(self.K, X.shape[1])
            keep = np.argsort(-s)[:k]
            out = np.zeros_like(X)
            out[:, keep] = X[:, keep]
            return out

        if vpm.ndim == 2:
            out = mask_cols(vpm)
        elif vpm.ndim == 3:
            out = np.stack([mask_cols(vpm[t]) for t in range(vpm.shape[0])], axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")

        return out, {
            "K": self.K,
            "metric": self.metric,
            "input_shape": vpm.shape,
            "output_shape": out.shape,
        }
