"""Feature engineering strategies for ZeroModel.

Encapsulates hint-based non-linear feature generation so the core model
remains focused on orchestration.
"""

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FeatureResult = Tuple[np.ndarray, List[str]]  # (augmented_matrix, new_metric_names)


class FeatureEngineer:
    """Applies optional non-linear feature transformations based on a hint string."""

    def __init__(self) -> None:
        # Registry pattern for easy extension
        self._strategies: Dict[
            str, Callable[[np.ndarray, List[str]], FeatureResult]
        ] = {
            "xor": self._xor_features,
            "radial": self._radial_features,
            "auto": self._auto_features,
        }

    # ------------------------ Public API ------------------------
    def apply(
        self, hint: Optional[str], data: np.ndarray, metric_names: List[str]
    ) -> FeatureResult:
        """Apply feature engineering based on hint.

        Args:
            hint: Optional string ('xor', 'radial', 'auto', ...)
            data: Normalized base matrix [docs x metrics]
            metric_names: Base metric names
        Returns:
            (processed_matrix, effective_metric_names)
        """
        if hint is None:
            logger.debug(
                "No feature engineering hint provided; returning original data."
            )
            return data, list(metric_names)
        key = hint.lower().strip()
        strategy = self._strategies.get(key)
        if strategy is None:
            logger.warning("Unknown nonlinearity_hint '%s'. No features added.", hint)
            return data, list(metric_names)
        try:
            augmented, new_names = strategy(data, metric_names)
            if augmented is data:  # No change
                return data, list(metric_names)
            return augmented, new_names
        except Exception as e:  # noqa: broad-except - defensive; fallback to original
            logger.error(
                "Feature engineering strategy '%s' failed: %s. Falling back to base data.",
                key,
                e,
            )
            return data, list(metric_names)

    # --------------------- Strategy Implementations ---------------------
    def _xor_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        if data.shape[1] < 2:
            logger.debug("Not enough metrics for xor features (<2).")
            return data, names
        m1, m2 = data[:, 0], data[:, 1]
        feats = [m1 * m2, np.abs(m1 - m2)]
        feat_names = [
            f"hint_product_{names[0]}_{names[1]}",
            f"hint_abs_diff_{names[0]}_{names[1]}",
        ]
        augmented = np.column_stack([data] + feats)
        return augmented, names + feat_names

    def _radial_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        if data.shape[1] < 2:
            logger.debug("Not enough metrics for radial features (<2).")
            return data, names
        x, y = data[:, 0], data[:, 1]
        cx = cy = 0.5
        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        angle = np.arctan2(y - cy, x - cx)
        feats = [distance, angle]
        feat_names = ["hint_radial_distance", "hint_radial_angle"]
        augmented = np.column_stack([data] + feats)
        return augmented, names + feat_names

    def _auto_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        n_orig = len(names)
        if n_orig == 0:
            return data, names
        engineered_features = []
        engineered_names: List[str] = []
        # Pairwise products among first min(3, n_orig)
        n_prod = min(3, n_orig)
        for i in range(n_prod):
            for j in range(i + 1, n_prod):
                if j < data.shape[1]:
                    engineered_features.append(data[:, i] * data[:, j])
                    engineered_names.append(f"auto_product_{names[i]}_{names[j]}")
        # Squares of first min(2, n_orig)
        n_sq = min(2, n_orig)
        for i in range(n_sq):
            if i < data.shape[1]:
                engineered_features.append(data[:, i] ** 2)
                engineered_names.append(f"auto_square_{names[i]}")
        if not engineered_features:
            logger.debug("Auto hint produced no additional features.")
            return data, names
        augmented = np.column_stack([data] + engineered_features)
        logger.info(
            "Auto hint added %d features (expected ~5 when n>=3).",
            len(engineered_features),
        )
        return augmented, names + engineered_names

__all__ = ["FeatureEngineer"]
