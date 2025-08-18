"""In-Memory Spatial Organization Strategy Module.

This module provides a lightweight strategy for organizing document-metric matrices
without requiring a database. It handles simple ordering specifications directly in
Python/Numpy for efficient in-memory processing.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base import BaseOrganizationStrategy

logger = logging.getLogger(__name__)


def _parse_spec(spec: str) -> List[Tuple[str, str]]:
    """
    Parse a simple metric ordering specification.
    
    Handles formats like:
    - "metric1 DESC"
    - "metric1 DESC, metric2 ASC"
    - "metric1" (defaults to DESC)
    
    Args:
        spec: Ordering specification string
        
    Returns:
        List of (metric_name, direction) tuples
    """
    if not spec or spec.strip().lower().startswith("select "):
        logger.debug("Skipping spec parsing: empty or SQL-like")
        return []
    
    priorities = []
    logger.debug("Parsing ordering spec: %s", spec)
    
    for token in spec.split(","):
        parts = token.strip().split()
        if not parts:
            continue
            
        metric = parts[0]
        direction = "DESC"  # Default direction
        
        # Check for explicit direction
        if len(parts) > 1 and parts[1].upper() in ("ASC", "DESC"):
            direction = parts[1].upper()
            
        priorities.append((metric, direction))
        logger.debug("Parsed ordering: %s %s", metric, direction)
        
    return priorities


class MemoryOrganizationStrategy(BaseOrganizationStrategy):
    """In-memory organization strategy without SQL dependencies.
    
    Processes ordering specifications directly in Python/Numpy for:
    - Document ordering based on metric priorities
    - Metric ordering (natural order only)
    
    Attributes:
        name: Strategy identifier ('memory')
        _spec: Current ordering specification
        _parsed_metric_priority: Parsed ordering rules
        _analysis: Cached analysis from last organization
    """

    name = "memory"

    def __init__(self):
        """Initialize in-memory organization strategy."""
        self._spec: str = ""
        self._parsed_metric_priority: Optional[List[Tuple[str, str]]] = None
        self._analysis: Optional[Dict[str, Any]] = None
        logger.debug("Initialized MemoryOrganizationStrategy")

    def set_task(self, spec: str):
        """
        Configure the ordering specification for document organization.
        
        Args:
            spec: Ordering specification string
        """
        self._spec = spec or ""
        logger.debug("Setting task with spec: %s", self._spec)
        self._parsed_metric_priority = _parse_spec(self._spec)

    def organize(self, matrix: np.ndarray, metric_names: List[str]):
        """
        Organize matrix based on configured ordering specification.
        
        Args:
            matrix: Input document-metric matrix
            metric_names: Names of metrics (matrix columns)
            
        Returns:
            Tuple of (reorganized matrix, metric order, document order, analysis)
        """
        logger.debug(
            "Starting memory organization. Matrix shape: %s, metrics: %s",
            matrix.shape, metric_names[:3] + ["..."] if len(metric_names) > 3 else metric_names
        )
        
        # Create metric name to index mapping
        name_to_idx = {n: i for i, n in enumerate(metric_names)}
        doc_indices = np.arange(matrix.shape[0])  # Default natural order
        
        # Process ordering specification if available
        if self._parsed_metric_priority:
            sort_keys = []
            logger.debug("Processing %d ordering rules", len(self._parsed_metric_priority))
            
            for metric, direction in self._parsed_metric_priority:
                idx = name_to_idx.get(metric)
                if idx is None:
                    logger.warning("Unknown metric '%s' in ordering spec", metric)
                    continue
                    
                column = matrix[:, idx]
                logger.debug("Processing metric '%s' (%s) with %d values", 
                            metric, direction, len(column))
                
                # Handle ordering direction
                if direction == "DESC":
                    if np.issubdtype(column.dtype, np.number):
                        # Numerical descending: negate values
                        sort_keys.append(-column)
                    else:
                        # Non-numerical descending: reverse sort order
                        sort_keys.append(np.argsort(column)[::-1])
                else:  # ASC
                    sort_keys.append(column)
            
            # Apply sorting if we have valid keys
            if sort_keys:
                logger.debug("Applying lexsort with %d keys", len(sort_keys))
                # Reverse because lexsort applies last key as primary
                doc_indices = np.lexsort(tuple(sort_keys[::-1]))
                logger.info("Sorted %d documents using %d ordering rules", 
                          len(doc_indices), len(sort_keys))
        
        # Metric order remains natural (no reordering)
        metric_order = list(range(matrix.shape[1]))
        
        # Determine primary ordering for analysis
        primary_metric, primary_direction = None, None
        for m, d in (self._parsed_metric_priority or []):
            if m in name_to_idx:
                primary_metric, primary_direction = m, d
                logger.debug("Primary ordering: %s %s", primary_metric, primary_direction)
                break
        
        # Prepare analysis metadata
        analysis = {
            "backend": self.name,
            "spec": self._spec,
            "applied_metric_priority": self._parsed_metric_priority or [],
            "doc_order": doc_indices.tolist(),
            "metric_order": metric_order,
        }
        
        if primary_metric is not None:
            analysis["ordering"] = {
                "primary_metric": primary_metric,
                "primary_metric_index": name_to_idx[primary_metric],
                "direction": primary_direction,
            }
            logger.info("Primary ordering: %s (%d) %s", 
                      primary_metric, name_to_idx[primary_metric], primary_direction)
        
        self._analysis = analysis.copy()
        logger.info(
            "Memory organization completed. %d documents reordered.",
            len(doc_indices)
        )
        
        return (
            matrix[doc_indices, :],  # Reordered matrix
            np.array(metric_order),   # Metric order (natural)
            doc_indices,              # Document order
            analysis                  # Analysis metadata
        )