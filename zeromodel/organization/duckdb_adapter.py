"""DuckDB Adapter for Matrix Analysis.

This module provides an adapter class to interface with DuckDB for efficient analysis
of document-metric matrices. It handles:
- Schema management for metric columns
- Matrix loading via PyArrow integration
- Query execution with multiple result fetching strategies
- Resource optimization for in-memory analytics

Key Features:
- Automatic schema synchronization
- Zero-copy data transfers via PyArrow
- Fallback mechanisms for result fetching
- In-memory database with optimized settings
"""

import logging
import re
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)


class DuckDBAdapter:
    """Adapter for DuckDB database to analyze document-metric matrices.
    
    Manages an in-memory DuckDB instance optimized for matrix analysis operations.
    Provides methods for schema management, data loading, and SQL query execution.
    
    Attributes:
        _conn: DuckDB in-memory connection
        _metric_names: Current metric column names
        _matrix: Loaded document-metric matrix
        _registered: Flag indicating if data is registered in database
    """

    def __init__(self, metric_names: List[str]):
        """
        Initialize DuckDB adapter with metric schema.
        
        Args:
            metric_names: Initial list of metric column names
        """
        self._conn = duckdb.connect(database=":memory:")
        # Optimize for analytical workloads
        self._conn.execute("PRAGMA threads=8")
        self._conn.execute("PRAGMA memory_limit='1GB'")
        self._metric_names: List[str] = list(metric_names)
        self._matrix: Optional[np.ndarray] = None
        self._registered = False
        logger.debug(
            "DuckDB adapter initialized with %d metrics: %s",
            len(metric_names), metric_names[:3] + ["..."] if len(metric_names) > 3 else metric_names
        )

    # ---------------- Public API -----------------------
    def ensure_schema(self, metric_names: List[str]):
        """
        Synchronize database schema with current metric names.
        
        If metric names have changed, triggers re-registration on next load.
        
        Args:
            metric_names: Current metric names for column mapping
        """
        if list(metric_names) != self._metric_names:
            logger.debug(
                "Metric schema changed from %d to %d metrics. Resetting registration.",
                len(self._metric_names), len(metric_names))
            self._metric_names = list(metric_names)
            self._registered = False

    def load_matrix(self, matrix: np.ndarray, metric_names: List[str]):
        """
        Load document-metric matrix into DuckDB via PyArrow.
        
        Args:
            matrix: Document-metric data (documents x metrics)
            metric_names: Names for metric columns
            
        Raises:
            ValueError: If matrix shape doesn't match metric count
        """
        if matrix.shape[1] != len(metric_names):
            raise ValueError(
                f"Matrix columns ({matrix.shape[1]}) don't match "
                f"metric count ({len(metric_names)})")
                
        self._matrix = matrix
        self._metric_names = list(metric_names)
        n = matrix.shape[0]

        # Create Arrow table with row IDs and metric columns
        arrays = [pa.array(np.arange(n, dtype=np.int32), type=pa.int32())]
        names = ["row_id"]
        for j, name in enumerate(self._metric_names):
            arrays.append(pa.array(matrix[:, j], type=pa.float32()))
            names.append(name)
        table = pa.Table.from_arrays(arrays, names=names)

        # Unregister previous table if exists
        if self._registered:
            try:
                self._conn.unregister("virtual_index")
                logger.debug("Unregistered previous virtual_index table")
            except Exception as e:
                logger.debug("Unregister ignored (table may not exist): %s", e)

        # Register new table
        self._conn.register("virtual_index", table)
        self._registered = True
        logger.info(
            "Registered matrix: %d docs × %d metrics via Arrow",
            matrix.shape[0], matrix.shape[1]
        )

    def analyze_query(self, sql_query: str, metric_names: List[str]) -> Dict[str, Any]:
        """
        Execute SQL query and retrieve analysis results.
        
        Implements multiple fetching strategies:
        1. NumPy (fastest, zero-copy)
        2. PyArrow (efficient)
        3. Pandas (fallback)
        
        Args:
            sql_query: SQL query to execute
            metric_names: Current metric names for validation
            
        Returns:
            Analysis dictionary with:
            - doc_order: Document indices in sorted order
            - metric_order: Metric indices (default natural order)
            - original_query: Preserved input query
            
        Raises:
            RuntimeError: If no data loaded or all fetch methods fail
        """
        if not self._registered:
            raise RuntimeError("Call load_matrix() before analyze_query()")

        logger.debug("Analyzing SQL query: %.200s...", sql_query)
        
        # Normalize and modify query to include row_id
        q = sql_query.strip().rstrip(";")
        if re.match(r"select\s+\*\s+", q, re.IGNORECASE):
            q = re.sub(r"select\s+\*\s+", "SELECT row_id, ", q, flags=re.IGNORECASE, count=1)
        else:
            q = f"SELECT row_id FROM ({q}) AS user_sorted_view"
        logger.debug("Modified query: %s", q)

        # Execute query
        cur = self._conn.execute(q)
        logger.debug("Query executed successfully")

        # 1) Fast path: NumPy (optimal performance)
        try:
            npres = cur.fetchnumpy()
            idx = npres["row_id"].astype(np.int32, copy=False)
            logger.debug("Fetched %d rows via NumPy", len(idx))
            return {
                "doc_order": idx.tolist(),
                "metric_order": list(range(len(metric_names))),
                "original_query": sql_query,
            }
        except Exception as e_np:
            logger.debug(
                "NumPy fetch failed (common if non-NumPy types present), "
                "trying Arrow: %s", e_np
            )

        # 2) Arrow fallback (efficient for large datasets)
        try:
            arr_tbl = cur.arrow()
            col = arr_tbl["row_id"]
            # Handle chunked arrays
            if col.num_chunks > 1:
                col = col.combine_chunks()
            idx = col.to_numpy(zero_copy_only=False).astype(np.int32, copy=False)
            logger.debug("Fetched %d rows via Arrow", len(idx))
            return {
                "doc_order": idx.tolist(),
                "metric_order": list(range(len(metric_names))),
                "original_query": sql_query,
            }
        except Exception as e_arrow:
            logger.debug("Arrow fetch failed: %s", e_arrow)

        # 3) Pandas fallback (most compatible)
        try:
            import pandas as pd
            df = cur.df()
            idx = df["row_id"].to_numpy(dtype="int32", copy=False)
            logger.debug("Fetched %d rows via pandas", len(idx))
            return {
                "doc_order": idx.tolist(),
                "metric_order": list(range(len(metric_names))),
                "original_query": sql_query,
            }
        except Exception as e_pd:
            logger.error("All fetch methods failed: %s", e_pd)
            raise RuntimeError(
                "Unable to fetch results. Try installing pyarrow or pandas.\n"
                f"Original error: {e_pd}"
            ) from e_pd

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Expose DuckDB connection for advanced operations."""
        return self._conn