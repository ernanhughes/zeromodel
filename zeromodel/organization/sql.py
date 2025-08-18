"""SQL-based spatial organization strategy module.

This module provides a strategy that uses SQL queries (via a database adapter)
to reorganize document-metric matrices. It leverages SQL's analytical capabilities
for sophisticated ordering and analysis of matrix data.
"""

import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseOrganizationStrategy

logger = logging.getLogger(__name__)


class SqlOrganizationStrategy(BaseOrganizationStrategy):
    """Organizes matrices using SQL queries executed via a database adapter.
    
    This strategy loads a document-metric matrix into a SQL database (via an adapter),
    executes a custom SQL query for analysis, and uses the results to reorganize
    the matrix rows (documents) and columns (metrics).

    Attributes:
        name: Strategy identifier ('sql')
        adapter: Database adapter for SQL execution (DuckDBAdapter-like interface)
        _sql_query: SQL query to be executed for analysis
        _analysis: Cached results from the last analysis query
    """

    name = "sql"

    def __init__(self, adapter):
        """
        Initialize SQL organization strategy with a database adapter.

        Args:
            adapter: Database adapter implementing the required interface
        """
        self.adapter = adapter
        self._sql_query: Optional[str] = None
        self._analysis: Optional[Dict[str, Any]] = None
        logger.debug("Initialized SQL organization strategy with adapter: %s", 
                    type(adapter).__name__)

    def set_task(self, spec: str):
        """
        Configure the SQL query for matrix organization.

        Args:
            spec: SQL query string defining the analysis and organization logic

        Raises:
            ValueError: If input is not a non-empty string
        """
        if not spec or not isinstance(spec, str):
            raise ValueError("SQL task spec must be a non-empty string.")
        self._sql_query = spec
        logger.debug("Set SQL organization task with query: %.100s...", spec)

    def organize(self, matrix: np.ndarray, metric_names: List[str]):
        """
        Reorganize matrix using SQL-based analysis.

        Steps:
        1. Load matrix into database via adapter
        2. Execute preconfigured SQL query
        3. Validate and apply document/metric ordering
        4. Parse ordering information from query
        5. Return reorganized matrix and metadata

        Args:
            matrix: Input document-metric matrix
            metric_names: Names of metrics (matrix columns)

        Returns:
            Tuple of (reorganized matrix, metric order, document order, analysis)

        Raises:
            RuntimeError: If SQL query hasn't been configured
        """
        if self._sql_query is None:
            raise RuntimeError("SQL task has not been set before organize().")

        logger.debug(
            "Starting SQL organization with matrix shape: %s and metrics: %s",
            matrix.shape, metric_names
        )

        # Prepare database and load data
        self.adapter.ensure_schema(metric_names)
        self.adapter.load_matrix(matrix, metric_names)
        logger.debug("Loaded matrix into database adapter")

        # Execute SQL analysis
        analysis = self.adapter.analyze_query(self._sql_query, metric_names)
        logger.debug("SQL analysis completed with keys: %s", list(analysis.keys()))

        # Process document ordering
        num_docs = matrix.shape[0]
        doc_order_list = analysis.get("doc_order", [])
        valid_doc_order = [idx for idx in doc_order_list if 0 <= idx < num_docs]
        # Ensure uniqueness but preserve order
        seen = set(); unique_doc_order = []
        for d in valid_doc_order:
            if d not in seen:
                unique_doc_order.append(d); seen.add(d)
        valid_doc_order = unique_doc_order

        if len(valid_doc_order) != len(doc_order_list):
            logger.warning(
                "Document order contains %d invalid indices (total docs: %d). Trimming invalid entries.",
                len(doc_order_list) - len(valid_doc_order), num_docs
            )
            
        if not valid_doc_order:
            valid_doc_order = list(range(num_docs))
            logger.info("No valid document order found. Using natural order.")
        
        sorted_by_docs = matrix[valid_doc_order, :]

        # Process metric ordering
        metric_count = matrix.shape[1]
        raw_metric_order = analysis.get("metric_order", list(range(metric_count)))
        valid_metric_order = [i for i in raw_metric_order if 0 <= i < metric_count]
        
        if len(valid_metric_order) != len(raw_metric_order):
            logger.warning(
                "Metric order contains %d invalid indices (total metrics: %d). Trimming invalid entries.",
                len(raw_metric_order) - len(valid_metric_order), metric_count
            )
        
        if len(valid_metric_order) != metric_count:
            remaining = [i for i in range(metric_count) if i not in valid_metric_order]
            valid_metric_order.extend(remaining)
            logger.info("Added %d missing metric indices to ordering", len(remaining))
        
        final_matrix = sorted_by_docs[:, valid_metric_order]

        # Parse ordering information from SQL query
        try:
            name_to_idx = {n: i for i, n in enumerate(metric_names)}
            ordering = analysis.get("ordering") or {}
            primary_name = ordering.get("primary_metric")
            primary_index = ordering.get("primary_metric_index")
            direction = ordering.get("direction")

            if primary_name is None:
                m = re.search(
                    r"order\s+by\s+([\w\.\"]+)\s*(asc|desc)?\b",
                    self._sql_query,
                    flags=re.IGNORECASE,
                )
                if m:
                    primary_name = m.group(1).split(".")[-1].strip('"')
                    direction = (m.group(2) or "DESC").upper()
                    logger.debug(
                        "Parsed ordering: metric='%s', direction=%s", 
                        primary_name, direction
                    )

            # Update analysis metadata
            if primary_name:
                if primary_index is None and primary_name in name_to_idx:
                    primary_index = name_to_idx[primary_name]
                
                analysis["ordering"] = {
                    "primary_metric": primary_name,
                    "primary_metric_index": int(primary_index) if primary_index is not None else 0,
                    "direction": direction or "DESC",
                }
                logger.info(
                    "Organization ordering: %s (%s) %s",
                    primary_name,
                    analysis["ordering"]["primary_metric_index"],
                    analysis["ordering"]["direction"]
                )
        except Exception as e:
            logger.warning("Failed to parse ordering info: %s", e, exc_info=True)

        self._analysis = analysis
        logger.info(
            "SQL organization completed. Final matrix shape: %s",
            final_matrix.shape
        )
        return (
            final_matrix,
            np.array(valid_metric_order),
            np.array(valid_doc_order),
            analysis,
        ) 