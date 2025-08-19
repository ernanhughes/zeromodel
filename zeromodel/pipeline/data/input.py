# zeromodel/pipeline/stages/input.py
"""Input handling stages - focused on one responsibility."""

from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineContext, PipelineStage


class LoadDataStage(PipelineStage):
    """Load data from various sources."""
    
    @property
    def name(self) -> str:
        return "load_data"
    
    def __init__(self, source_type: str, source_path: str):
        self.source_type = source_type
        self.source_path = source_path
    
    def validate_params(self):
        """Validate parameters for loading data."""
        if self.source_type not in ["csv", "json"]:
            raise ValueError(f"Unsupported source type: {self.source_type}")
        if not self.source_path:
            raise ValueError("source_path must be provided")

    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Load data and add source metadata."""
        if self.source_type == "csv":
            data = self._load_csv(self.source_path)
        elif self.source_type == "json":
            data = self._load_json(self.source_path)
        else:
            raise ValueError(f"Unsupported source type: {self.source_type}")
        
        metadata = {
            "source_type": self.source_type,
            "source_path": self.source_path,
            "data_shape": data.shape if hasattr(data, 'shape') else len(data)
        }
        
        return PipelineContext(data, {**context.metadata, **metadata})
    
    def _load_csv(self, path: str) -> Any:
        """Load CSV data."""
        import pandas as pd
        return pd.read_csv(path).values
    
    def _load_json(self, path: str) -> Any:
        """Load JSON data."""
        import json
        with open(path, 'r') as f:
            return json.load(f)