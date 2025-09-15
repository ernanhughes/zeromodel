#  zeromodel/organization/base.py
from __future__ import annotations
import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BaseOrganizationStrategy:
    """
    Abstract base class defining the interface for spatial organization strategies.
    
    Spatial organization strategies process matrices (typically document-metric data) 
    to optimize the arrangement of rows and columns based on specific algorithms.
    
    Attributes:
        name (str): Identifier for the strategy. Default is 'base'.
    
    Methods:
        set_task: Configures the strategy for a specific task.
        organize: Processes and reorganizes the input matrix.
    """

    name: str = "base"

    def set_task(self, spec: str) -> None:
        """
        Configure the strategy for a specific processing task.
        
        Args:
            spec: A string specification defining the task parameters or configuration.
        
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses.
        """
        logger.debug(f"Setting task for {self.name} strategy: {spec}")
        raise NotImplementedError("Subclasses must implement set_task method")

    def organize(
        self, matrix: np.ndarray, metric_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Reorganize a matrix according to the spatial strategy.
        
        Processes the input matrix to produce:
        1. A reordered matrix
        2. New metric (column) order
        3. New document (row) order
        4. Additional analytical information
        
        Args:
            matrix: Input data matrix (documents x metrics)
            metric_names: Names of the metrics corresponding to matrix columns
            
        Returns:
            Tuple containing:
                sorted_matrix: Reorganized matrix
                metric_order: New column order (as index array)
                doc_order: New row order (as index array)
                analysis_dict: Strategy-specific analytical metadata
            
        Raises:
            NotImplementedError: Must be implemented by concrete subclasses
        """
        logger.debug(
            f"Organizing matrix with shape {matrix.shape} "
            f"using {self.name} strategy. Metrics: {metric_names}"
        )
        raise NotImplementedError("Subclasses must implement organize method")