import logging
from typing import Any, Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

class BaseOrganizationStrategy:
    """Abstract base for spatial organization strategies."""
    name: str = "base"

    def set_task(self, spec: str):  # pragma: no cover - interface
        raise NotImplementedError

    def organize(
        self, matrix: np.ndarray, metric_names: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:  # pragma: no cover - interface
        """Return (sorted_matrix, metric_order, doc_order, analysis_dict)."""
        raise NotImplementedError
