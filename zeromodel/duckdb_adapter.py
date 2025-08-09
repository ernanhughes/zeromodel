"""DuckDB adapter encapsulating schema management, data loading, and query analysis."""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import duckdb

logger = logging.getLogger(__name__)

class DuckDBAdapter:
    def __init__(self, metric_names: List[str]):
        self._conn = duckdb.connect(database=':memory:')
        self._create_table(metric_names)

    # ---------------- Internal helpers -----------------
    def _create_table(self, metric_names: List[str]):
        cols = ", ".join([f'"{c}" FLOAT' for c in metric_names])
        self._conn.execute(f"CREATE TABLE virtual_index (row_id INTEGER, {cols})")
        logger.debug("DuckDB virtual_index table created with %d metrics.", len(metric_names))

    # ---------------- Public API -----------------------
    def ensure_schema(self, metric_names: List[str]):
        try:
            cur = self._conn.execute("PRAGMA table_info(virtual_index)")
            info = cur.fetchall()
            expected = ["row_id"] + list(metric_names)
            current = [r[1] for r in info]
            if current != expected:
                logger.debug("Recreating DuckDB schema. Expected %s, found %s", expected, current)
                self._conn.execute("DROP TABLE IF EXISTS virtual_index")
                self._create_table(metric_names)
        except Exception as e:  # noqa: broad-except
            raise RuntimeError(f"Failed ensuring DuckDB schema: {e}") from e

    def load_matrix(self, matrix, metric_names: List[str]):
        try:
            self._conn.execute("DELETE FROM virtual_index")
            col_list = ", ".join([f'"{m}"' for m in metric_names])
            placeholders = ", ".join(["?"] * len(metric_names))
            sql = f"INSERT INTO virtual_index (row_id, {col_list}) VALUES (?, {placeholders})"
            for rid, row in enumerate(matrix):
                self._conn.execute(sql, [rid] + row.tolist())
            logger.debug("Loaded %d rows into DuckDB.", matrix.shape[0])
        except Exception as e:  # noqa: broad-except
            raise RuntimeError(f"Failed loading data into DuckDB: {e}") from e

    def analyze_query(self, sql_query: str, metric_names: List[str]) -> Dict[str, Any]:
        import re
        try:
            if sql_query.strip().upper().startswith("SELECT *"):
                modified = sql_query.replace("SELECT *", "SELECT row_id", 1)
            else:
                modified = f"SELECT row_id FROM ({sql_query}) AS user_sorted_view"
            result = self._conn.execute(modified).fetchall()
            doc_order = [r[0] for r in result]
            metric_order = list(range(len(metric_names)))
            primary_sort_metric_name = None
            primary_sort_metric_index = None
            match = re.search(r"ORDER\s+BY\s+(\w+)", sql_query, re.IGNORECASE)
            if match:
                candidate = match.group(1)
                if candidate in metric_names:
                    primary_sort_metric_name = candidate
                    primary_sort_metric_index = metric_names.index(candidate)
            return {
                'doc_order': doc_order,
                'metric_order': metric_order,
                'primary_sort_metric_name': primary_sort_metric_name,
                'primary_sort_metric_index': primary_sort_metric_index,
                'original_query': sql_query,
            }
        except Exception as e:  # noqa: broad-except
            raise ValueError(f"Invalid SQL query execution: {e}") from e

    @property
    def connection(self):  # Expose if low-level access needed
        return self._conn

__all__ = ["DuckDBAdapter"]
