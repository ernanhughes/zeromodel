#  zeromodel/pipeline/amplifier/feature_engineer.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.nonlinear.feature_engineer import FeatureEngineer
from zeromodel.pipeline.base import PipelineStage


class FeatureEngineerStage(PipelineStage):
    name = "feature_engineering"
    category = "prep"

    def __init__(self, **params):
        super().__init__(**params)
        self.hint = params.get("nonlinearity_hint")

    def validate_params(self): 
        """No validation needed for this stage."""
        pass

    def process(self, vpm: np.ndarray, context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self.get_context(context)
        metric_names = ctx.get("metric_names", [f"m{i}" for i in range(vpm.shape[1])])

        fe: FeatureEngineer = ctx.get("feature_engineer") or FeatureEngineer()
        ctx["feature_engineer"] = fe

        processed, eff_names = fe.apply(self.hint, vpm, metric_names)
        ctx["metric_names"] = eff_names
        ctx["canonical_matrix"] = processed

        meta = {
            "added_metrics": int(processed.shape[1] - vpm.shape[1]),
            "shape": tuple(processed.shape),
            "hint": self.hint,
            "input_shape": tuple(vpm.shape),
            "stage": self.name
        }
        return processed, meta
