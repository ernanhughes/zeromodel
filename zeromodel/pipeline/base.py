# zeromodel/pipeline/base.py
"""
Base classes for ZeroModel pipeline stages.

Implements ZeroModel's "dumb pipe" communication model:
"Because the core representation is a standardized image (VPM tile),
the communication protocol becomes extremely simple and universally understandable."
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np

logger = logging.getLogger(__name__)


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
    def process(
        self, vpm: np.ndarray, context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
        if "provenance" not in context:
            context['provenance'] = []
        return context
   
