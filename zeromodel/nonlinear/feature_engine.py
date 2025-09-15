#  zeromodel/nonlinear/feature_engine.py
"""
Feature engineering strategies for ZeroModel.

Encapsulates hint-based non-linear feature generation so the core model
remains focused on orchestration while enabling spatial organization
of complex patterns.
"""
from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

FeatureResult = Tuple[np.ndarray, List[str]]  # (augmented_matrix, new_metric_names)


class FeatureEngineer:
    """Applies optional non-linear feature transformations based on a hint string."""

    def __init__(self) -> None:
        """
        Initialize feature engineer with strategy registry.

        The key insight: "Intelligence lives in the data structure, not in processing."
        Feature engineering transforms complex relationships into spatially
        organized patterns that can be easily navigated.
        """
        self._strategies: Dict[
            str, Callable[[np.ndarray, List[str]], FeatureResult]
        ] = {
            "xor": self._xor_features,
            "radial": self._radial_features,
            "product": self._product_features,
            "auto": self._auto_features,
            "none": self._identity_transform,  # Explicit no-op
        }

    # ------------------------ Public API ------------------------
    def apply(
        self, hint: Optional[str], data: np.ndarray, metric_names: List[str]
    ) -> FeatureResult:
        """Apply feature engineering based on hint.

        Args:
            hint: Optional string ('xor', 'radial', 'product', 'auto', 'none')
            data: Normalized base matrix [docs x metrics]
            metric_names: Base metric names

        Returns:
            (processed_matrix, effective_metric_names)

        Raises:
            ValueError: If inputs are invalid
        """
        # Validate inputs
        if data is None or not isinstance(data, np.ndarray):
            raise ValueError("Data must be a valid numpy array")
        if data.ndim != 2:
            raise ValueError(f"Data must be 2D (docs x metrics), got {data.ndim}D")
        if data.size == 0:
            raise ValueError("Data cannot be empty")
        if not metric_names:
            raise ValueError("metric_names cannot be empty")

        # Handle None hint
        if hint is None:
            logger.debug(
                "No feature engineering hint provided; returning original data."
            )
            return data, list(metric_names)

        key = hint.lower().strip()

        # Handle 'none' explicitly
        if key == "none":
            logger.debug("Explicit 'none' hint; returning original data.")
            return data, list(metric_names)

        strategy = self._strategies.get(key)
        if strategy is None:
            logger.warning("Unknown nonlinearity_hint '%s'. No features added.", hint)
            return data, list(metric_names)

        try:
            augmented, new_names = strategy(data, metric_names)

            # Verify output consistency
            if augmented.shape[0] != data.shape[0]:
                logger.error(
                    "Feature engineering changed document count from %d to %d. Using original.",
                    data.shape[0],
                    augmented.shape[0],
                )
                return data, list(metric_names)

            if augmented is data:  # No change
                return data, list(metric_names)

            logger.info(
                "Applied '%s' feature engineering: %d → %d metrics",
                key,
                len(metric_names),
                len(new_names),
            )
            return augmented, new_names

        except Exception as e:
            logger.error(
                "Feature engineering strategy '%s' failed: %s. Falling back to base data.",
                key,
                e,
            )
            return data, list(metric_names)

    # --------------------- Strategy Implementations ---------------------
    def _identity_transform(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        """Explicit identity transform for 'none' hint."""
        logger.debug("Applied identity transformation (no feature engineering).")
        return data, list(names)

    def _xor_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        """
        Generate features for XOR-like patterns.

        This is ZeroModel's "symbolic logic in the data" capability:
        - Creates features that make XOR patterns linearly separable
        - Enables spatial organization of non-linear relationships
        - The intelligence is in the data structure, not the processing
        """
        if data.shape[1] < 2:
            logger.debug("Not enough metrics for xor features (<2).")
            return data, names

        m1, m2 = data[:, 0], data[:, 1]

        # Key insight: XOR patterns become linearly separable with these features
        product = m1 * m2  # High when both high or both low
        abs_diff = np.abs(m1 - m2)  # High when different

        feats = [product, abs_diff]
        feat_names = [
            f"feature_xor_product_{names[0]}_{names[1]}",
            f"feature_xor_abs_diff_{names[0]}_{names[1]}",
        ]

        augmented = np.column_stack([data] + feats)
        logger.debug(
            "Applied XOR feature engineering for non-linear pattern separation."
        )
        return augmented, names + feat_names

    def _radial_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        """
        Generate radial/distance-based features.

        This enables ZeroModel's "top-left rule" for circular patterns:
        - Distance from center becomes a metric
        - Angle becomes a metric
        - Circular patterns become spatially organized
        """
        if data.shape[1] < 2:
            logger.debug("Not enough metrics for radial features (<2).")
            return data, names

        x, y = data[:, 0], data[:, 1]
        cx = cy = 0.5  # Assume normalized data [0,1]

        distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        angle = np.arctan2(y - cy, x - cx)

        feats = [distance, angle]
        feat_names = ["feature_radial_distance", "feature_radial_angle"]

        augmented = np.column_stack([data] + feats)
        logger.debug(
            "Applied radial feature engineering for circular pattern organization."
        )
        return augmented, names + feat_names

    def _product_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        """
        Generate pairwise product features.

        This implements ZeroModel's "spatial calculus" principle:
        - Products capture interaction effects
        - Enables organization of multiplicative relationships
        - The spatial layout becomes the index
        """
        if data.shape[1] < 2:
            logger.debug("Not enough metrics for product features (<2).")
            return data, names

        n_metrics = data.shape[1]
        max_metrics = min(4, n_metrics)  # Limit to avoid combinatorial explosion

        feats = []
        feat_names = []

        # Generate pairwise products
        for i in range(max_metrics):
            for j in range(i + 1, max_metrics):
                feats.append(data[:, i] * data[:, j])
                feat_names.append(f"feature_product_{names[i]}_{names[j]}")

        if not feats:
            return data, names

        augmented = np.column_stack([data] + feats)
        logger.debug(
            f"Applied product feature engineering: added {len(feats)} interaction terms."
        )
        return augmented, names + feat_names

    def _auto_features(self, data: np.ndarray, names: List[str]) -> FeatureResult:
        """
        Automated feature engineering for general non-linear patterns.

        This implements ZeroModel's "planet-scale navigation that feels flat":
        - Automatically generates features for common non-linear patterns
        - Enables handling of unknown pattern types
        - When the answer is always 40 steps away, size becomes irrelevant
        """
        n_orig = len(names)
        if n_orig == 0:
            return data, names

        engineered_features = []
        engineered_names: List[str] = []

        # 1. Pairwise products (capture interactions)
        n_prod = min(3, n_orig)
        for i in range(n_prod):
            for j in range(i + 1, n_prod):
                if j < data.shape[1]:
                    engineered_features.append(data[:, i] * data[:, j])
                    engineered_names.append(f"auto_product_{names[i]}_{names[j]}")

        # 2. Squares (capture non-linear effects)
        n_sq = min(2, n_orig)
        for i in range(n_sq):
            if i < data.shape[1]:
                engineered_features.append(data[:, i] ** 2)
                engineered_names.append(f"auto_square_{names[i]}")

        # 3. Absolute differences (capture dissimilarity)
        n_diff = min(3, n_orig)
        for i in range(n_diff):
            for j in range(i + 1, n_diff):
                if j < data.shape[1]:
                    engineered_features.append(np.abs(data[:, i] - data[:, j]))
                    engineered_names.append(f"auto_abs_diff_{names[i]}_{names[j]}")

        if not engineered_features:
            logger.debug("Auto hint produced no additional features.")
            return data, names

        augmented = np.column_stack([data] + engineered_features)
        logger.info("Auto feature engineering added %d features.", len(engineered_features))
        return augmented, names + engineered_names

__all__ = ["FeatureEngineer"]