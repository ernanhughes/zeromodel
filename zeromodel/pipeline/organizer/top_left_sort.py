# zeromodel/pipeline/organizer/top_left_sort.py
"""
Top-left sorter for ZeroModel.

This implements ZeroModel's "top-left rule" for signal concentration:
The most important information is always in the top-left corner.
"""

from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class TopLeftSorter(PipelineStage):
    """Top-left sorter stage for ZeroModel."""
    
    name = "top_left_sort"
    category = "organizer"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.metric = params.get("metric", "variance")
        self.Kc = params.get("Kc", 12)
    
    def validate_params(self):
        """Validate stage parameters."""
        pass

    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reorder VPM to concentrate signal in top-left corner.
        
        This implements ZeroModel's "intelligence lives in the data structure" principle:
        The processing is minimal - the intelligence is in how the data is organized.
        """
        context = self._get_context(context)
        
        if vpm.ndim == 2:
            # Single matrix
            processed_vpm = self._sort_matrix(vpm)
        elif vpm.ndim == 3:
            # Time series - apply to each frame
            processed_frames = [self._sort_matrix(vpm[t]) for t in range(vpm.shape[0])]
            processed_vpm = np.stack(processed_frames, axis=0)
        else:
            raise ValueError(f"VPM must be 2D or 3D, got {vpm.ndim}D")
        
        metadata = {
            "metric": self.metric,
            "Kc": self.Kc,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "reordering_applied": True
        }
        
        return processed_vpm, metadata
    
    def _sort_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """Sort matrix to concentrate signal in top-left."""
        # Calculate column importance based on selected metric
        if self.metric == "variance":
            col_importance = np.var(matrix, axis=0)
        elif self.metric == "mean":
            col_importance = np.mean(matrix, axis=0)
        elif self.metric == "sum":
            col_importance = np.sum(matrix, axis=0)
        else:
            col_importance = np.var(matrix, axis=0)  # Default
        
        # Sort columns by importance (descending)
        col_order = np.argsort(-col_importance)
        matrix_sorted = matrix[:, col_order]
        
        # Calculate row importance based on top-Kc columns
        Kc_actual = min(self.Kc, matrix_sorted.shape[1])
        if Kc_actual > 0:
            row_importance = np.sum(matrix_sorted[:, :Kc_actual], axis=1)
        else:
            row_importance = np.sum(matrix_sorted, axis=1)
        
        # Sort rows by importance (descending)
        row_order = np.argsort(-row_importance)
        matrix_sorted = matrix_sorted[row_order, :]
        
        return matrix_sorted