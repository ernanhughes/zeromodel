from typing import Any, Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from zeromodel.pipeline.base import PipelineStage


class GaussianFilter(PipelineStage):
    name = "gaussian"
    category = "filter"

    def __init__(self, **params):
        super().__init__(**params)
        self.sigma = float(params.get("sigma", 1.0))

    def validate_params(self):
        assert self.sigma >= 0

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None):
        context = self._get_context(context)
        if vpm.ndim == 2:
            out = gaussian_filter(vpm, sigma=self.sigma)
        elif vpm.ndim == 3:
            out = np.stack(
                [
                    gaussian_filter(vpm[t], sigma=self.sigma)
                    for t in range(vpm.shape[0])
                ],
                axis=0,
            )
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")
        return out, {
            "sigma": self.sigma,
            "input_shape": vpm.shape,
            "output_shape": out.shape,
        }
