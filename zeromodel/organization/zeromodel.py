#  zeromodel/organization/zeromodel.py
"""ZeroModel Spatial Organization Strategy Module.

This module implements a legacy-compatible spatial organization strategy based on the 
original ZeroModel approach. It handles both simple metric ordering specifications and 
SQL-like ORDER BY clauses for backward compatibility with older systems.

Key Features:
- Supports both simple ("metric DESC") and SQL-like ("ORDER BY metric DESC") syntax
- Gracefully handles all-NaN columns
- Preserves legacy analysis metadata format
- Implements ZeroModel design principles:
  - Intelligence in structure
  - Top-left rule
  - Constant time navigation
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseOrganizationStrategy

logger = logging.getLogger(__name__)

# Regular expression to parse SQL-like ORDER BY clauses
_ORDER_BY_RE = re.compile(
    r"order\s+by\s+((?:\"[^\"]+\"|'[^']+'|[\w\.])+)\s*(asc|desc)?\b",
    flags=re.IGNORECASE,
)

class ZeroModelOrganizationStrategy(BaseOrganizationStrategy):
    """
    Legacy-compatible organization strategy implementing ZeroModel principles.
    
    This strategy supports:
    - Simple specifications: "metric ASC|DESC"
    - SQL-like syntax: "ORDER BY metric DESC"
    - Default ordering: First metric descending
    
    Attributes:
        name: Strategy identifier ('zeromodel')
        _task: Current ordering specification
        _analysis: Cached analysis from last organization
    """

    name = "zeromodel"  # Unique identifier to avoid collision

    def __init__(self) -> None:
        """Initialize ZeroModel organization strategy."""
        self._task: Optional[str] = None
        self._analysis: Optional[Dict[str, Any]] = None
        logger.debug("[%s] Strategy initialized", self.name)

    def set_task(self, spec: str) -> None:
        """
        Configure the ordering specification.
        
        Args:
            spec: Ordering specification string (simple or SQL-like)
        """
        self._task = (spec or "").strip()
        logger.debug("[%s] Task set: %r", self.name, self._task)

    def organize(
        self, matrix: np.ndarray, metric_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Organize matrix based on ZeroModel principles.
        
        Args:
            matrix: Input document-metric matrix
            metric_names: Names of metrics (matrix columns)
            
        Returns:
            Tuple of (reorganized matrix, metric order, document order, analysis)
            
        Raises:
            ValueError: If matrix or metric_names are empty
        """
        # Validate inputs
        if matrix is None or matrix.size == 0:
            raise ValueError("Matrix cannot be empty")
        if not metric_names:
            raise ValueError("metric_names cannot be empty")
            
        logger.info(
            "[%s] Organizing matrix: shape=%s, metrics=%s", 
            self.name, matrix.shape, metric_names[:3] + ["..."] if len(metric_names) > 3 else metric_names
        )

        # Parse task to get primary metric, direction, and name
        idx, direction, primary_name = self._parse_task(self._task, metric_names)
        logger.debug(
            "[%s] Using primary metric: %s (%d) %s", 
            self.name, primary_name, idx, direction
        )

        # Extract the primary metric column
        col = matrix[:, idx]

        # Handle all-NaN column case
        if np.issubdtype(col.dtype, np.number) and np.isnan(col).all():
            logger.warning(
                "[%s] Column %r is all-NaN; using identity order", 
                self.name, primary_name
            )
            doc_order = np.arange(matrix.shape[0])
        else:
            # Numerical descending: use negative sort
            if direction == "DESC" and np.issubdtype(col.dtype, np.number):
                logger.debug("[%s] Using numerical descending sort", self.name)
                doc_order = np.argsort(-col, kind="stable")
            else:
                # Default ascending sort
                doc_order = np.argsort(col, kind="stable")
                # For non-numerical descending, reverse the order
                if direction == "DESC" and not np.issubdtype(col.dtype, np.number):
                    logger.debug("[%s] Reversing non-numerical sort order", self.name)
                    doc_order = doc_order[::-1]

        logger.info(
            "[%s] Sorted %d documents by %s (%s)", 
            self.name, len(doc_order), primary_name, direction
        )

        # Metric order remains natural (no reordering)
        metric_order = np.arange(len(metric_names), dtype=int)
        # Create sorted matrix view
        sorted_matrix = matrix[doc_order, :]

        # Prepare analysis metadata
        analysis: Dict[str, Any] = {
            "backend": self.name,
            "legacy_name": "memory",  # Backward compatibility
            "spec": self._task,
            "doc_order": doc_order.tolist(),
            "metric_order": metric_order.tolist(),
            "ordering": {
                "primary_metric": primary_name,
                "primary_metric_index": int(idx),
                "direction": direction,
                "source": "ORDER BY" if (_ORDER_BY_RE.search(self._task or "") is not None) else "simple_or_default",
            },
            "principles_applied": [
                "intelligence_in_structure",
                "top_left_rule",
                "constant_time_navigation",
            ],
        }

        # Store defensive copy to prevent external mutation
        self._analysis = {k: v.copy() if isinstance(v, dict) else v for k, v in analysis.items()}
        logger.info(
            "[%s] Organization completed. Analysis keys: %s", 
            self.name, list(analysis.keys())
        )
        
        return sorted_matrix, metric_order, doc_order, analysis

    def _parse_task(
        self, task: Optional[str], metric_names: List[str]
    ) -> Tuple[int, str, str]:
        """
        Parse ordering specification to primary metric, direction, and name.
        
        Supports:
        - SQL-like: "ORDER BY <ident> [ASC|DESC]"
        - Simple: "<metric> [ASC|DESC]"
        - Default: First metric descending
        
        Args:
            task: Ordering specification string
            metric_names: Available metric names
            
        Returns:
            Tuple: (metric_index, direction, metric_name)
        """
        logger.debug("[%s] Parsing task: %r", self.name, task)
        
        # Return default if no task specified
        if not task:
            logger.debug("[%s] Using default: first metric descending", self.name)
            return 0, "DESC", metric_names[0]

        # Try SQL-like ORDER BY syntax
        m = _ORDER_BY_RE.search(task)
        if m:
            raw_ident = m.group(1).strip()
            direction = (m.group(2) or "DESC").upper()
            
            # Clean and normalize identifier
            target = raw_ident.strip().strip('"').strip("'").split(".")[-1].strip().strip('"').strip("'")
            logger.debug("[%s] Parsed ORDER BY: %s %s", self.name, target, direction)
            
            # Find matching metric (case-insensitive)
            for i, name in enumerate(metric_names):
                if name.lower() == target.lower():
                    logger.debug("[%s] Matched metric: %s", self.name, name)
                    return i, direction, name

        # Try simple "metric DIRECTION" syntax
        tokens = task.split()
        if tokens:
            cand = tokens[0].strip().strip('"').strip("'")
            dir_token = tokens[1].upper() if len(tokens) > 1 and tokens[1].upper() in ("ASC", "DESC") else "DESC"
            logger.debug("[%s] Parsed simple: %s %s", self.name, cand, dir_token)
            
            for i, name in enumerate(metric_names):
                if name.lower() == cand.lower():
                    logger.debug("[%s] Matched metric: %s", self.name, name)
                    return i, dir_token, name

        # Fallback to default
        logger.warning(
            "[%s] No valid metric found in '%s'. Using first metric descending", 
            self.name, task
        )
        return 0, "DESC", metric_names[0]