import logging
import re
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseOrganizationStrategy

logger = logging.getLogger(__name__)


class SqlOrganizationStrategy(BaseOrganizationStrategy):
    """SQL-based organization using a DuckDBAdapter-like object."""

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

        self.adapter.ensure_schema(metric_names)
        self.adapter.load_matrix(matrix, metric_names)

        analysis = self.adapter.analyze_query(self._sql_query, metric_names)

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

        try:
            name_to_idx = {n: i for i, n in enumerate(metric_names)}
            ordering = analysis.get("ordering") or {}
            primary_name = ordering.get("primary_metric")
            primary_index = ordering.get("primary_metric_index")
            direction = ordering.get("direction")

            if primary_name is None:
                m = re.search(
                    r"order\s+by\s+([A-Za-z0-9_\"\.]+)\s*(ASC|DESC)?",
                    self._sql_query,
                    flags=re.IGNORECASE,
                )
                if m:
                    primary_name = m.group(1).strip().split(".")[-1].strip('"')
                    direction = (m.group(2) or "DESC").upper()

            if primary_name is not None:
                if primary_index is None and primary_name in name_to_idx:
                    primary_index = int(name_to_idx[primary_name])
                analysis["ordering"] = {
                    "primary_metric": primary_name,
                    "primary_metric_index": int(primary_index)
                    if primary_index is not None
                    else 0,
                    "direction": (direction or "DESC"),
                }
        except Exception as e:
            logger.debug("sql ordering resolution skipped: %s", e)

        self._analysis = analysis
        return (
            final_matrix,
            np.array(valid_metric_order),
            np.array(valid_doc_order),
            analysis,
        )
