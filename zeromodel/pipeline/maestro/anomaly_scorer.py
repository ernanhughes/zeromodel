#  zeromodel/pipeline/maestro/anomaly_scorer.py
"""
Anomaly scoring stage for ZeroModel MAESTRO pipelines.

This stage computes simple global statistics over residual maps
(mean, std, energy) and records them for anomaly detection.

Author: ZeroModel Project
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List
import numpy as np

from zeromodel.pipeline.executor import PipelineStage

logger = logging.getLogger(__name__)


class AnomalyScorer(PipelineStage):
    """
    Compute per-frame anomaly scores from residuals.

    Input Context:
        context["frames_maestro"]: list of {"residual": np.ndarray, ...}

    Output Context:
        context["frames_scores"]: list of {"fname": str, "score": float}
    """

    name = "anomaly_scorer"

    def __init__(self, method: str = "mean", **kwargs):
        super().__init__(method=method, **kwargs)
        self.method = method

    def validate_params(self) -> None:
        if self.method not in {"mean", "std", "energy"}:
            raise ValueError("method must be one of: mean, std, energy")

    def process(self, X, context: Dict[str, Any]):
        frames = context.get("frames_maestro", [])
        if not frames:
            logger.warning("[%s] No frames_maestro found", self.name)
            context["frames_scores"] = []
            return X, context

        scores = []
        for rec in frames:
            res = rec["residual"]
            if self.method == "mean":
                score = float(res.mean())
            elif self.method == "std":
                score = float(res.std())
            else:  # energy
                score = float(np.sqrt((res**2).mean()))

            scores.append({"fname": rec.get("fname", "?"), "score": score})
            logger.debug("[%s] %s → score=%.4f", self.name, rec.get("fname"), score)

        context["frames_scores"] = scores
        return X, context
