# zeromodel/pipeline/data/input.py
"""
Input handling stages for ZeroModel pipeline.
"""

from typing import Dict, Any, Tuple
import numpy as np
from zeromodel.pipeline.base import PipelineStage

class LoadData(PipelineStage):
    """Load data from various sources."""
    
    name = "load_data"
    category = "input"
    
    def __init__(self, **params):
        super().__init__(**params)
        self.source_type = params.get("source_type", "array")
        self.source_path = params.get("source_path", None)
    
    def validate_params(self):
        """Validate input parameters."""
        if self.source_type not in ["array", "csv", "json"]:
            raise ValueError(f"Unsupported source type: {self.source_type}")
    
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Load data and return as VPM.
        For this stage, vpm parameter is ignored as we're loading fresh data.
        """
        context = self.get_context(context)
        
        # In a real implementation, load from source
        # For now, return the input as-is with metadata
        metadata = {
            "source_type": self.source_type,
            "source_path": self.source_path,
            "stage": self.name
        }
        
        return vpm, metadata