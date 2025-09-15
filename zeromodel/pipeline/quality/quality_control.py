#  zeromodel/pipeline/quality/quality_control.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage
from zeromodel.vpm.explain import OcclusionVPMInterpreter


class QualityControl(PipelineStage):
    """Quality control stage using occlusion analysis."""

    name = "quality_control"
    category = "quality"

    def __init__(self, **params):
        super().__init__(**params)
        self.interpreter = OcclusionVPMInterpreter(
            patch_h=params.get("patch_h", 8),
            patch_w=params.get("patch_w", 8),
            stride=params.get("stride", 4),
            prior="top_left",  # Enforce top-left bias expectation
        )

    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Validate VPM quality using occlusion analysis.

        This ensures the VPM adheres to ZeroModel principles:
        "The top-left corner always contains the most decision-critical information."
        """
        context = self.get_context(context)

        # Create mock model and generate importance map
        class MockZeroModel:
            def __init__(self, matrix):
                self.sorted_matrix = matrix

        mock_model = MockZeroModel(vpm)
        importance_map, meta = self.interpreter.explain(mock_model)

        # Calculate quality metrics
        H, W = importance_map.shape
        center_h, center_w = H // 2, W // 2

        # Check top-left concentration (should be high)
        top_left_importance = importance_map[:center_h, :center_w].mean()

        # Check bottom-right concentration (should be low)
        bottom_right_importance = importance_map[center_h:, center_w:].mean()

        # Quality score: ratio of top-left to bottom-right importance
        quality_score = top_left_importance / (bottom_right_importance + 1e-8)

        # Flag if quality is below threshold
        quality_ok = (
            quality_score > 2.0
        )  # Expect top-left to be at least 2x more important

        metadata = {
            "quality_score": float(quality_score),
            "top_left_importance": float(top_left_importance),
            "bottom_right_importance": float(bottom_right_importance),
            "quality_ok": quality_ok,
            "importance_map_shape": importance_map.shape,
            "stage": "quality_control",
        }
        
        return vpm, metadata