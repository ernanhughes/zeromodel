from typing import Any, Dict

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class WeightedSumCombiner(PipelineStage):
    """
    Combine K channels by weighted sum. If vpm is (H,W,K), outputs (H,W).
    If vpm is (T,H,W,K), outputs (T,H,W).
    """

    name = "weighted_sum"
    category = "combiner"

    def __init__(self, **params):
        super().__init__(**params)
        self.weights = np.array(params.get("weights", []), dtype=np.float64)
        self.normalize = bool(params.get("normalize", True))

    def validate_params(self):
        # empty weights means "uniform over last dim"
        if self.weights.size:
            assert np.all(self.weights >= 0)

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None):
        context = self._get_context(context)
        if vpm.ndim not in (3, 4):
            return vpm, {"warning": "requires (H,W,K) or (T,H,W,K)", "applied": False}

        if vpm.ndim == 3:
            H, W, K = vpm.shape
            w = self._get_w(K)
            out = (vpm * w[None, None, :]).sum(axis=-1)
        else:
            T, H, W, K = vpm.shape
            w = self._get_w(K)
            out = (vpm * w[None, None, None, :]).sum(axis=-1)

        return out, {
            "weights": w.tolist(),
            "normalize": self.normalize,
            "input_shape": vpm.shape,
            "output_shape": out.shape,
            "applied": True,
        }

    def _get_w(self, K: int) -> np.ndarray:
        if self.weights.size == 0:
            w = np.ones(K, dtype=np.float64)
        else:
            if self.weights.size != K:
                raise ValueError(f"weights length {self.weights.size} != channels {K}")
            w = self.weights.copy()
        if self.normalize:
            s = w.sum()
            if s > 0: w /= s
        return w
