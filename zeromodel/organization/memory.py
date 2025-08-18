import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .base import BaseOrganizationStrategy

logger = logging.getLogger(__name__)


def _parse_spec(spec: str):
    if not spec:
        return []
    if spec.strip().lower().startswith("select "):
        return []
    priorities = []
    for token in spec.split(","):
        t = token.strip()
        if not t:
            continue
        parts = t.split()
        metric = parts[0]
        direction = "DESC"
        if len(parts) > 1 and parts[1].upper() in ("ASC", "DESC"):
            direction = parts[1].upper()
        priorities.append((metric, direction))
    return priorities


class MemoryOrganizationStrategy(BaseOrganizationStrategy):
    """In-memory (non-SQL) organization."""

    name = "memory"

    def __init__(self):
        self._spec: Optional[str] = None
        self._parsed_metric_priority: Optional[list[tuple[str, str]]] = None
        self._analysis: Optional[Dict[str, Any]] = None

    def set_task(self, spec: str):
        self._spec = spec or ""
        self._parsed_metric_priority = _parse_spec(self._spec)

    def organize(self, matrix: np.ndarray, metric_names: List[str]):
        name_to_idx = {n: i for i, n in enumerate(metric_names)}
        doc_indices = np.arange(matrix.shape[0])

        if self._parsed_metric_priority:
            sort_keys = []
            for metric, direction in self._parsed_metric_priority:
                idx = name_to_idx.get(metric)
                if idx is None:
                    continue
                column = matrix[:, idx]
                if direction == "DESC" and np.issubdtype(column.dtype, np.number):
                    sort_keys.append(-column)
                else:
                    sort_keys.append(column)
            if sort_keys:
                doc_indices = np.lexsort(tuple(sort_keys[::-1]))

        final_matrix = matrix[doc_indices, :]
        metric_order = np.arange(matrix.shape[1])

        primary_metric, primary_direction = None, None
        if self._parsed_metric_priority:
            for m, d in self._parsed_metric_priority:
                if m in name_to_idx:
                    primary_metric, primary_direction = m, d
                    break
        if primary_metric is None and metric_names:
            primary_metric, primary_direction = metric_names[0], "DESC"

        analysis = {
            "backend": self.name,
            "spec": self._spec,
            "applied_metric_priority": self._parsed_metric_priority or [],
            "doc_order": doc_indices.tolist(),
            "metric_order": metric_order.tolist(),
        }
        if primary_metric is not None:
            try:
                analysis["ordering"] = {
                    "primary_metric": primary_metric,
                    "primary_metric_index": int(name_to_idx[primary_metric]),
                    "direction": primary_direction or "DESC",
                }
            except Exception as e:
                logger.debug("memory ordering resolution skipped: %s", e)

        self._analysis = analysis
        return final_matrix, metric_order, doc_indices, analysis
