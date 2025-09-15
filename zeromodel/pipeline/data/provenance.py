#  zeromodel/pipeline/data/provenance.py
"""Provenance stages - add VPF."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from zeromodel.provenance.schema import create_vpf, embed_vpf
from zeromodel.pipeline.base import PipelineContext, PipelineStage


class AddProvenanceStage(PipelineStage):
    """Add provenance to VPM."""
    
    @property
    def name(self) -> str:
        return "add_provenance"
    
    def __init__(self, task: str, model_id: str = "zero-1.0"):
        self.task = task
        self.model_id = model_id
    
    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Add VPF to VPM."""
        vpm = context.data
        
        # Create VPF
        vpf = create_vpf(
            pipeline={"graph_hash": "sha3:stdm", "step": "signal-amplification"},
            model={"id": self.model_id, "assets": {}},
            determinism={"seed": 0, "rng_backends": ["numpy"]},
            params={"task": self.task},
            inputs={"task": self.task},
            metrics=context.metadata.get("metrics", {}),
            lineage={"parents": []},
        )
        
        # Embed VPF
        png_bytes = embed_vpf(vpm, vpf, mode="stripe")
        
        metadata = {
            "vpf_added": True,
            "task": self.task,
            "model_id": self.model_id
        }
        
        return PipelineContext(png_bytes, {**context.metadata, **metadata})