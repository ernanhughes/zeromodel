#  zeromodel/pipeline/data/encoding.py
"""Encoding stages - convert to VPM."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.provenance.core import tensor_to_vpm
from zeromodel.pipeline.base import PipelineContext, PipelineStage


class EncodeVPMStage(PipelineStage):
    """Encode data as VPM image."""
    
    @property
    def name(self) -> str:
        return "encode_vpm"
    
    def __init__(self, min_size: tuple = (256, 256)):
        self.min_size = min_size
    
    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode data as VPM image."""
        data = context.data
        
        # Ensure data is 2D
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        
        # Create VPM
        vpm = tensor_to_vpm({"vpm": data}, min_size=self.min_size)
        
        metadata = {
            "vpm_size": (vpm.width, vpm.height),
            "original_shape": data.shape
        }
        
        return PipelineContext(vpm, {**context.metadata, **metadata})