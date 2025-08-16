# zeromodel/organization/zeromodel.py
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseOrganizationStrategy

logger = logging.getLogger(__name__)

class ZeroModelOrganizationStrategy(BaseOrganizationStrategy):
    """
    Legacy-compatible in-memory organization (ZeroModel-flavored).

    Keeps the old interface:
      - set_task(spec: str) -> None
      - organize(matrix, metric_names) -> (sorted_matrix, metric_order, doc_order, analysis)

    Lenient spec handling:
      - Accepts "", "metric ASC|DESC", or SQL-ish "ORDER BY metric DESC"
      - Defaults to first metric, DESC
    """

    name = "memory"  # keep legacy backend name

    def __init__(self) -> None:
        self._task: Optional[str] = None
        self._analysis: Optional[Dict[str, Any]] = None

    # ---- Legacy API ----
    def set_task(self, spec: str) -> None:
        # Be lenient: allow empty/None -> default behavior
        self._task = (spec or "").strip()
        logger.debug(f"[{self.name}] Task set: {self._task!r}")

    # ---- Legacy API ----
    def organize(
        self, matrix: np.ndarray, metric_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        if matrix is None or matrix.size == 0:
            raise ValueError("Matrix cannot be empty")
        if not metric_names:
            raise ValueError("metric_names cannot be empty")

        logger.debug(f"[{self.name}] Organizing matrix: shape={matrix.shape}")

        # Determine ordering (robust parsing)
        primary_metric_idx, direction, primary_name = self._parse_task(self._task, metric_names)

        # Sort docs by selected column; stable; numeric-aware for DESC
        col = matrix[:, primary_metric_idx]
        if direction == "DESC" and np.issubdtype(col.dtype, np.number):
            doc_order = np.argsort(-col, kind="stable")
        else:
            # ASC for numerics, and ASC for non-numerics; DESC non-numerics via reversed ASC
            doc_order = np.argsort(col, kind="stable")
            if direction == "DESC" and not np.issubdtype(col.dtype, np.number):
                doc_order = doc_order[::-1]

        metric_order = np.arange(len(metric_names), dtype=int)
        sorted_matrix = matrix[doc_order, :][:, metric_order]

        logger.debug(
            f"[{self.name}] Parsed ordering -> metric='{primary_name}', "
            f"direction={direction}, index={primary_metric_idx}"
        )
        logger.debug(f"[{self.name}] First 10 doc_order: {doc_order[:10].tolist()}")

        analysis: Dict[str, Any] = {
            "backend": self.name,
            "spec": self._task,
            "doc_order": doc_order.tolist(),
            "metric_order": metric_order.tolist(),
            "ordering": {
                "primary_metric": primary_name,
                "primary_metric_index": int(primary_metric_idx),  # absolute index
                "direction": direction,
            },
            # Extra, harmless metadata:
            "principles_applied": [
                "intelligence_in_structure",
                "top_left_rule",
                "constant_time_navigation",
            ],
        }

        self._analysis = analysis
        return sorted_matrix, metric_order, doc_order, analysis

    # ---------------- helpers ----------------
    def _parse_task(
        self, task: Optional[str], metric_names: List[str]
    ) -> Tuple[int, str, str]:
        """
        Parse "<metric> [ASC|DESC]" or SQL-ish 'ORDER BY <ident> [ASC|DESC]'.
        Handles quoted/qualified identifiers. Fallback to first metric, DESC.
        Returns (metric_index, direction, metric_name).
        """
        if not task:
            return 0, "DESC", metric_names[0]

        # SQL-ish ORDER BY
        m = re.search(
            r"order\s+by\s+([A-Za-z0-9_\"'.\s]+?)\s*(ASC|DESC)?\b",
            task,
            flags=re.IGNORECASE,
        )
        if m:
            raw_ident = m.group(1).strip()
            direction = (m.group(2) or "DESC").upper()
            target = raw_ident.strip().strip('"').strip("'").split(".")[-1].strip().strip('"').strip("'")
            for i, name in enumerate(metric_names):
                if name.lower() == target.lower():
                    return i, direction, name

        # Simple "<metric> [ASC|DESC]" form
        tokens = task.split()
        if tokens:
            cand = tokens[0].strip().strip('"').strip("'")
            dir_token = tokens[1].upper() if len(tokens) > 1 and tokens[1].upper() in ("ASC", "DESC") else "DESC"
            for i, name in enumerate(metric_names):
                if name.lower() == cand.lower():
                    return i, dir_token, name

        # Fallback
        return 0, "DESC", metric_names[0]
