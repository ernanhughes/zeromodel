#  zeromodel/pipeline/filter/morphological.py
"""
Morphological filter stage for ZeroModel.

This implements ZeroModel's "symbolic logic in the data" principle:
Instead of running a neural model, we run fuzzy logic on structured images.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
from scipy import ndimage

from zeromodel.pipeline.base import PipelineStage


class MorphologicalFilter(PipelineStage):
    """Morphological filter stage for ZeroModel."""

    name = "morphological"
    category = "filter"

    def __init__(self, **params):
        super().__init__(**params)
        self.operation = params.get("operation", "opening")
        self.kernel_size = params.get("kernel_size", 3)
        self.iterations = params.get("iterations", 1)

    def validate_params(self):
        """Validate morphological parameters."""
        if self.kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if self.iterations <= 0:
            raise ValueError("iterations must be positive")
        if self.operation not in ["opening", "closing", "erosion", "dilation"]:
            raise ValueError(
                "operation must be one of: opening, closing, erosion, dilation"
            )

    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply morphological operations to a VPM.

        This enhances or suppresses features based on their shape and size.
        """
        context = self.get_context(context)

        # Create structuring element (square kernel)
        kernel = np.ones((self.kernel_size, self.kernel_size))

        # Handle different VPM dimensions
        if vpm.ndim == 2:
            # Single matrix
            processed_vpm = self._apply_morphology(vpm, kernel)
        elif vpm.ndim == 3:
            # Time series - apply to each frame
            processed_frames = [
                self._apply_morphology(vpm[t], kernel) for t in range(vpm.shape[0])
            ]
            processed_vpm = np.stack(processed_frames, axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")

        metadata = {
            "operation": self.operation,
            "kernel_size": self.kernel_size,
            "iterations": self.iterations,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "morphology_applied": True,
        }

        return processed_vpm, metadata

    def _apply_morphology(self, matrix: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Apply morphological operation to a single matrix."""
        # Normalize to [0,1] for consistent processing
        matrix_min, matrix_max = matrix.min(), matrix.max()
        if matrix_max > matrix_min:
            normalized = (matrix - matrix_min) / (matrix_max - matrix_min)
        else:
            normalized = matrix.copy()

        # Apply morphological operation
        if self.operation == "opening":
            result = ndimage.morphology.binary_opening(
                normalized, structure=kernel, iterations=self.iterations
            )
        elif self.operation == "closing":
            result = ndimage.morphology.binary_closing(
                normalized, structure=kernel, iterations=self.iterations
            )
        elif self.operation == "erosion":
            result = ndimage.morphology.binary_erosion(
                normalized, structure=kernel, iterations=self.iterations
            )
        elif self.operation == "dilation":
            result = ndimage.morphology.binary_dilation(
                normalized, structure=kernel, iterations=self.iterations
            )

        # Restore original scale
        if matrix_max > matrix_min:
            result = result * (matrix_max - matrix_min) + matrix_min
        else:
            result = result * matrix_max
        
        return result