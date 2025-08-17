# zeromodel/pipeline/combiner/and.py
"""
Logical AND combiner for ZeroModel.

This implements ZeroModel's "symbolic logic in the data" principle:
Instead of running a neural model, we run fuzzy logic on structured images.
"""

from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage


class AndCombiner(PipelineStage):
    """Logical AND combiner stage for ZeroModel."""
    
    name = "and"
    category = "combiner"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.threshold = params.get("threshold", 0.5)
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply logical AND to a VPM (assuming multiple channels represent different conditions).
        
        This creates a compound reasoning structure through pixel-wise arithmetic.
        """
        context = self._get_context(context)
        
        if vpm.ndim < 3:
            # Not enough dimensions for AND operation
            return vpm, {"warning": "VPM has <3 dimensions, skipping AND operation"}
        
        # Apply AND across channels (last dimension)
        # Convert to binary using threshold, then AND
        binary_vpm = (vpm > self.threshold).astype(float)
        processed_vpm = np.all(binary_vpm, axis=-1).astype(float)
        
        metadata = {
            "threshold": self.threshold,
            "input_shape": vpm.shape,
            "output_shape": processed_vpm.shape,
            "operation": "AND",
            "channels_combined": vpm.shape[-1]
        }
        
        return processed_vpm, metadata