# zeromodel/normalizer.py
"""
Dynamic Range Adaptation Module

Provides the DynamicNormalizer class which handles normalization of scores to
handle value drift over time. This is critical for long-term viability of the
zeromodel system as score distributions may change due to:
- Policy improvements
- New document types
- Shifting data distributions
"""

import logging
from typing import Dict, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class DynamicNormalizer:
    """
    Dynamic score normalizer with exponential smoothing adaptation.

    Maintains and updates min/max ranges for multiple metrics over time,
    using exponential smoothing to adapt to distribution changes while
    maintaining historical context.

    Key features:
    - Incremental range updates with adjustable smoothing factor
    - Robust handling of constant metrics and initialization edge cases
    - Configurable non-finite value handling
    - Thread-safe read operations

    Attributes:
        metric_names (List[str]): Names of tracked metrics (order preserved)
        alpha (float): Smoothing factor (0.0-1.0)
        min_vals (Dict[str, float]): Current minimum values per metric
        max_vals (Dict[str, float]): Current maximum values per metric
        allow_non_finite (bool): Whether to allow NaN/Inf values
    """

    def __init__(
        self,
        metric_names: List[str],
        alpha: float = 0.1,
        *,
        allow_non_finite: bool = False,
    ):
        """
        Initialize the dynamic normalizer.

        Args:
            metric_names: Names of metrics to track (order must match input matrices)
            alpha: Smoothing factor for range updates (0.0-1.0)
                   - 0.0: No adaptation (fixed ranges)
                   - 0.1: Gradual adaptation (recommended)
                   - 1.0: Instant adaptation (use current batch only)
            allow_non_finite: Allow NaN/Inf values in input (default: False)

        Raises:
            ValueError: On invalid metric_names or alpha
            TypeError: On non-iterable metric_names

        Example:
            >>> normalizer = DynamicNormalizer(['precision', 'recall'], alpha=0.2)
            >>> normalizer.metric_names
            ['precision', 'recall']
        """
        logger.debug(f"Initializing DynamicNormalizer for {len(metric_names)} metrics")
        if not metric_names:
            error_msg = "metric_names must contain at least one metric"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if not (0.0 <= alpha <= 1.0):
            error_msg = f"Alpha must be in [0.0, 1.0], got {alpha:.4f}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.metric_names = metric_names
        self.alpha = alpha
        self.allow_non_finite = allow_non_finite
        # Initialize with extreme values to be replaced on first update
        self.min_vals = {m: float("inf") for m in metric_names}
        self.max_vals = {m: float("-inf") for m in metric_names}
        # Avoid non-ASCII characters in logs for Windows consoles
        logger.info(
            f"Normalizer initialized with alpha={alpha:.2f} for metrics: {metric_names}"
        )

    def update(self, score_matrix: np.ndarray) -> None:
        """
        Update min/max ranges using exponential smoothing.

        Formula:
            min_new = (1-α)*min_old + α*min_current_batch
            max_new = (1-α)*max_old + α*max_current_batch

        Args:
            score_matrix: 2D array of shape (documents, metrics)
                          Must have columns matching metric_names order

        Raises:
            ValueError: On dimension mismatch or invalid data
            TypeError: On non-array input

        Example:
            >>> scores = np.array([[0.1, 0.8], [0.3, 0.9]])
            >>> normalizer.update(scores)
            INFO: Updated ranges for 2 documents
        """
        logger.debug(f"Processing update with matrix shape {score_matrix.shape}")

        # Handle empty batches gracefully
        if score_matrix.size == 0:
            logger.warning("Received empty score matrix - skipping update")
            return

        # Compute batch statistics
        batch_mins = np.min(score_matrix, axis=0)
        batch_maxs = np.max(score_matrix, axis=0)

        # Update ranges with exponential smoothing
        for idx, metric in enumerate(self.metric_names):
            current_min = batch_mins[idx]
            current_max = batch_maxs[idx]

            # Initialize if needed
            if np.isinf(self.min_vals[metric]):
                self.min_vals[metric] = current_min
                self.max_vals[metric] = current_max
                logger.debug(
                    f"Initialized {metric} range: [{current_min:.4f}, {current_max:.4f}]"
                )
                continue

            # Apply exponential smoothing
            prev_min = self.min_vals[metric]
            prev_max = self.max_vals[metric]

            self.min_vals[metric] = (
                1 - self.alpha
            ) * prev_min + self.alpha * current_min
            self.max_vals[metric] = (
                1 - self.alpha
            ) * prev_max + self.alpha * current_max

            logger.debug(
                f"Updated {metric}: min {prev_min:.4f}→{self.min_vals[metric]:.4f} "
                f"max {prev_max:.4f}→{self.max_vals[metric]:.4f} "
                f"(batch: [{current_min:.4f}, {current_max:.4f}])"
            )

        logger.info(f"Updated ranges using {score_matrix.shape[0]} documents")

    def normalize(
        self, score_matrix: np.ndarray, *, as_float32: bool = False
    ) -> np.ndarray:
        """
        Normalize scores to [0,1] range using current min/max values.

        Special cases:
        - Constant metrics → 0.5
        - Uninitialized metrics → 0.5 with warning
        - Values outside range → clamped to [0,1]

        Args:
            score_matrix: Input scores (shape: documents × metrics)
            as_float32: Return float32 instead of float64

        Returns:
            Normalized matrix with same shape as input

        Raises:
            ValueError: On dimension mismatch

        Example:
            >>> scores = np.array([[0.15], [0.25]])
            >>> normalizer.normalize(scores)
            array([[0.25], [0.75]])  # Assuming range [0.1, 0.3]
        """
        logger.debug(f"Normalizing matrix with shape {score_matrix.shape}")

        # Preallocate output array
        normalized = np.empty_like(score_matrix, dtype=np.float64)

        for idx, metric in enumerate(self.metric_names):
            min_val = self.min_vals[metric]
            max_val = self.max_vals[metric]
            col_data = score_matrix[:, idx]

            # Handle uninitialized and constant metrics
            if np.isinf(min_val) or (max_val - min_val) < 1e-12:
                if np.isinf(min_val):
                    logger.warning(
                        f"Using fallback 0.5 for uninitialized metric '{metric}'"
                    )
                else:
                    logger.debug(f"Constant metric '{metric}' - using 0.5")
                normalized[:, idx] = 0.5
                continue

            # Apply min-max normalization with clipping
            normalized_col = (col_data - min_val) / (max_val - min_val)
            np.clip(normalized_col, 0.0, 1.0, out=normalized_col)
            normalized[:, idx] = normalized_col

            logger.debug(
                f"Normalized '{metric}' using range [{min_val:.6f}, {max_val:.6f}]"
            )

        logger.info(f"Normalized {score_matrix.shape[0]} documents")

        if as_float32:
            return normalized.astype(np.float32)
        return normalized

    def get_ranges(self) -> Dict[str, Tuple[float, float]]:
        """
        Retrieve current normalization ranges.

        Returns:
            Dictionary mapping metric names to (min, max) tuples

        Example:
            >>> normalizer.get_ranges()
            {'precision': (0.15, 0.95), 'recall': (0.2, 0.8)}
        """
        ranges = {}
        for metric in self.metric_names:
            min_val = self.min_vals[metric]
            max_val = self.max_vals[metric]

            # Handle uninitialized state
            if np.isinf(min_val) or np.isinf(max_val):
                ranges[metric] = (float('nan'), float('nan'))
            else:
                ranges[metric] = (min_val, max_val)
                
        logger.debug("Returning current ranges")
        return ranges