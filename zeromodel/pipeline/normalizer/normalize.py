# zeromodel/pipeline/stage/normalize.py
import numpy as np
from typing import Any, Dict, Tuple, Optional
from zeromodel.pipeline.base import PipelineStage
from zeromodel.normalizer import DynamicNormalizer
import logging

logger = logging.getLogger(__name__)

class NormalizeStage(PipelineStage):
    name = "normalize"
    category = "prep"

    def __init__(self, **params):
        super().__init__(**params)
        # optional: explicit metric_names (if not, expect provided in context)
        self.metric_names = params.get("metric_names")  # list[str] or None

    def validate_params(self):  # called by executor before process()
        pass

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self._get_context(context)

        if vpm.ndim != 2:
            raise ValueError(f"NormalizeStage expects a 2D score_matrix, got {vpm.ndim}D")

        # reconcile metric names
        metric_names = self.metric_names or ctx.get("metric_names")
        if metric_names is None:
            metric_names = [f"m{i}" for i in range(vpm.shape[1])]
        if len(metric_names) != vpm.shape[1]:
            # trim/extend to match
            if len(metric_names) > vpm.shape[1]:
                metric_names = metric_names[: vpm.shape[1]]
            else:
                metric_names = metric_names + [f"col_{i}" for i in range(len(metric_names), vpm.shape[1])]

        # align/create normalizer
        normalizer: Optional[DynamicNormalizer] = ctx.get("normalizer")
        if (normalizer is None) or (normalizer.metric_names != metric_names):
            normalizer = DynamicNormalizer(metric_names)
            ctx["normalizer"] = normalizer

        # update + normalize
        normalizer.update(vpm)
        norm = normalizer.normalize(vpm, as_float32=True)

        # publish basics
        ctx["metric_names"] = metric_names
        ctx["canonical_matrix"] = norm  # for later stages

        meta = {
            "metric_names": metric_names,
            "min": float(norm.min()),
            "max": float(norm.max()),
            "shape": tuple(norm.shape),
        }
        return norm, meta
