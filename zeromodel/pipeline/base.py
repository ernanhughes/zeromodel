# zeromodel/pipeline/core.py
"""
ZeroModel Pipeline Core - High Quality Implementation

This implements the core insight: small, focused components create superior results.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple, List
import logging
import time
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class PipelineContext:
    """Immutable context passed through pipeline stages."""
    data: Any
    metadata: Dict[str, Any]
    
    def update(self, **updates) -> 'PipelineContext':
        """Create new context with updates."""
        new_metadata = {**self.metadata, **updates}
        return PipelineContext(self.data, new_metadata)


class PipelineStage(ABC):
    """
    Base class for all ZeroModel pipeline stages.
    
    This implements ZeroModel's "intelligence lives in the data structure" principle:
    The processing is minimal - the intelligence is in how the data is organized.
    """
    
    name: str = "base"
    category: str = "base"
    
    def __init__(self, **params):
        self.params = params
    
    @abstractmethod
    def process(self, 
                vpm: np.ndarray, 
                context: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a VPM and return (transformed_vpm, metadata).
        
        Args:
            vpm: Input VPM as numpy array
            context: Optional context dictionary with pipeline state
            
        Returns:
            (transformed_vpm, metadata) - Enhanced VPM and diagnostic metadata
        """
        pass
    
    @abstractmethod
    def validate_params(self):
        """Validate stage parameters."""
        pass
    
    def _get_context(self, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get or create context dictionary."""
        if context is None:
            context = {}
        if 'provenance' not in context:
            context['provenance'] = []
        return context
    
    def _record_provenance(self, context: Dict[str, Any], stage_name: str, params: Dict[str, Any]):
        """Record stage execution in context provenance."""
        context['provenance'].append({
            'stage': stage_name,
            'params': params,
            'timestamp': np.datetime64('now')
        })

class Pipeline:
    """Compose stages into a complete pipeline."""
    
    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages
        self._validate_stages()
    
    def _validate_stages(self):
        """Ensure all stages have unique names."""
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("All pipeline stages must have unique names")
    
    def run(self, initial_data: Any, initial_metadata: Dict[str, Any] = None) -> Tuple[Any, Dict[str, Any]]:
        """
        Execute the pipeline with exceptional quality.
        
        The quality comes from:
        1. Small, focused stages
        2. Clear data flow
        3. Comprehensive logging
        4. Proper error handling
        """
        context = PipelineContext(
            data=initial_data,
            metadata=initial_metadata or {}
        )
        
        logger.info(f"Starting pipeline execution with {len(self.stages)} stages")
        start_time = time.time()
        
        try:
            for stage in self.stages:
                context = stage(context)
                
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {total_time:.3f}s")
            return context.data, context.metadata
            
        except Exception as e:
            total_time = time.time() - start_time
            logger.error(f"Pipeline failed after {total_time:.3f}s: {e}")
            raise