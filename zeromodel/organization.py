"""Organization strategies for arranging documents and metrics.

Provides a pluggable abstraction so different ordering backends (SQL/DuckDB,
text specification, heuristic, etc.) can be swapped without changing the core
ZeroModel pipeline.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class BaseOrganizationStrategy:
    """Abstract base for spatial organization strategies."""
    name: str = "base"

    def set_task(self, spec: str):  # pragma: no cover - interface
        raise NotImplementedError

    def organize(self, matrix: np.ndarray, metric_names: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:  # pragma: no cover - interface
        """Return (sorted_matrix, metric_order, doc_order, analysis_dict)."""
        raise NotImplementedError

class MemoryOrganizationStrategy(BaseOrganizationStrategy):
    """In-memory (non-SQL) organization.

    Provides a lightweight default that keeps documents and metrics in their
    existing order (or applies a simple heuristic) without any DuckDB usage.
    The task spec can be a simple comma-separated list of metric names with
    optional DESC/ASC to define a priority ordering, e.g.:
        "uncertainty DESC, size ASC"
    If a full SQL statement is provided (starts with SELECT), higher-level
    code may decide to upgrade to a SQL strategy.
    """
    name = "memory"

    def __init__(self):
        self._spec: Optional[str] = None
        self._parsed_metric_priority: Optional[list[tuple[str, str]]] = None
        self._analysis: Optional[Dict[str, Any]] = None

    def set_task(self, spec: str):  # lenient acceptance
        self._spec = spec or ""
        self._parsed_metric_priority = self._parse_spec(self._spec)

    def _parse_spec(self, spec: str):
        if not spec:
            return []
        if spec.strip().lower().startswith("select "):
            # Not a memory-native spec; treat as opaque so caller may upgrade
            return []
        # Very small parser: split by comma, allow 'metric [ASC|DESC]' tokens
        priorities = []
        for token in spec.split(','):
            t = token.strip()
            if not t:
                continue
            parts = t.split()
            metric = parts[0]
            direction = 'DESC'
            if len(parts) > 1 and parts[1].upper() in ("ASC", "DESC"):
                direction = parts[1].upper()
            priorities.append((metric, direction))
        return priorities

    def organize(self, matrix: np.ndarray, metric_names: List[str]):
        # Build metric index map
        name_to_idx = {n: i for i, n in enumerate(metric_names)}
        doc_indices = np.arange(matrix.shape[0])
        # Apply simple multi-key sort if priorities defined
        if self._parsed_metric_priority:
            sort_keys = []
            for metric, direction in reversed(self._parsed_metric_priority):
                idx = name_to_idx.get(metric)
                if idx is None:
                    continue
                column = matrix[:, idx]
                # For descending we sort ascending on -column
                if direction == 'DESC' and np.issubdtype(column.dtype, np.number):
                    sort_keys.append(-column)
                else:
                    sort_keys.append(column)
            if sort_keys:
                stacked = np.lexsort(tuple(sort_keys))
                doc_indices = stacked
        final_matrix = matrix[doc_indices, :]
        metric_order = np.arange(matrix.shape[1])
        analysis = {
            "backend": self.name,
            "spec": self._spec,
            "applied_metric_priority": self._parsed_metric_priority or [],
            "doc_order": doc_indices.tolist(),
            "metric_order": metric_order.tolist(),
        }
        self._analysis = analysis
        return final_matrix, metric_order, doc_indices, analysis

class SqlOrganizationStrategy(BaseOrganizationStrategy):
    """SQL-based organization using a DuckDBAdapter-like object.

    Adapter must expose:
        ensure_schema(metric_names: List[str])
        load_matrix(matrix: np.ndarray, metric_names: List[str])
        analyze_query(sql_query: str, metric_names: List[str]) -> Dict[str, Any]
    """
    name = "sql"

    def __init__(self, adapter):
        self.adapter = adapter
        self._sql_query: Optional[str] = None
        self._analysis: Optional[Dict[str, Any]] = None

    def set_task(self, spec: str):
        if not spec or not isinstance(spec, str):
            raise ValueError("SQL task spec must be a non-empty string.")
        self._sql_query = spec

    def organize(self, matrix: np.ndarray, metric_names: List[str]):
        if self._sql_query is None:
            raise RuntimeError("SQL task has not been set before organize().")
        # Ensure schema reflects current metric names then load data
        self.adapter.ensure_schema(metric_names)
        self.adapter.load_matrix(matrix, metric_names)
        # Analyze query
        analysis = self.adapter.analyze_query(self._sql_query, metric_names)
        # Apply ordering
        doc_order_list = analysis.get("doc_order", [])
        num_docs = matrix.shape[0]
        valid_doc_order = [idx for idx in doc_order_list if 0 <= idx < num_docs]
        if not valid_doc_order:
            valid_doc_order = list(range(num_docs))
        sorted_by_docs = matrix[valid_doc_order, :]
        raw_metric_order = analysis.get("metric_order", list(range(matrix.shape[1])))
        metric_count = matrix.shape[1]
        valid_metric_order = [i for i in raw_metric_order if 0 <= i < metric_count]
        if len(valid_metric_order) != metric_count:
            remaining = [i for i in range(metric_count) if i not in valid_metric_order]
            valid_metric_order.extend(remaining)
        if not valid_metric_order:
            valid_metric_order = list(range(metric_count))
        final_matrix = sorted_by_docs[:, valid_metric_order]
        self._analysis = analysis
        return final_matrix, np.array(valid_metric_order), np.array(valid_doc_order), analysis

__all__ = [
    "BaseOrganizationStrategy",
    "MemoryOrganizationStrategy",
    "SqlOrganizationStrategy",
]
