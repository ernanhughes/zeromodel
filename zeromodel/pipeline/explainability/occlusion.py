# zeromodel/pipeline/stages/explainability/occlusion.py
from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage
from zeromodel.vpm.explain import OcclusionVPMInterpreter


class OcclusionExplainer(PipelineStage):
    """Occlusion-based explainability stage for ZeroModel."""

    name = "occlusion"
    category = "explainability"

    def __init__(self, **params):
        super().__init__(**params)
        # Pass parameters to the interpreter
        self.interpreter = OcclusionVPMInterpreter(**params)

    def validate_params(self):
        """Validate parameters for the occlusion explainer."""
        pass

    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Generate explainability map for a VPM.

        This implements ZeroModel's "inherently understandable" principle:
        "A core tenet of ZeroModel is that the system's output should be inherently understandable."
        """
        context = self._get_context(context)

        # Create a mock ZeroModel object with sorted_matrix
        class MockZeroModel:
            def __init__(self, matrix):
                self.sorted_matrix = matrix

        mock_model = MockZeroModel(vpm)

        # Generate importance map
        importance_map, meta = self.interpreter.explain(mock_model)

        # Add to context for downstream use
        context["explainability"] = {"importance_map": importance_map, "metadata": meta}

        # Return original VPM (explanation is metadata)
        return vpm, {
            "importance_map": importance_map.tolist(),
            "explainability_metadata": meta,
            "stage": "occlusion_explainer",
        }