# zeromodel/core.py
"""
Zero-Model Intelligence Encoder/Decoder with DuckDB SQL Processing.

This module provides the core functionality for transforming high-dimensional
policy evaluation data into spatially-optimized visual maps where the
intelligence is in the data structure itself, not in processing.
"""

import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
from zeromodel.normalizer import DynamicNormalizer
from zeromodel.config import init_config
from zeromodel.encoder import VPMEncoder
from zeromodel.feature_engineer import FeatureEngineer
from zeromodel.duckdb_adapter import DuckDBAdapter
from zeromodel.organization import SqlOrganizationStrategy, MemoryOrganizationStrategy
from zeromodel.timing import timeit

logger = logging.getLogger(__name__)  

precision_dtype_map = {
    'uint8': np.uint8,
    'uint16': np.uint16,
    'float16': np.float16,
    'float32': np.float32,
    'float64': np.float64,
}

init_config()

DATA_NOT_PROCESSED_ERR = "Data not processed yet. Call process() or prepare() first."

class ZeroModel:
    """
    Zero-Model Intelligence encoder/decoder with DuckDB SQL processing.

    This class transforms high-dimensional policy evaluation data into
    spatially-optimized visual maps where:
    - Position = Importance (top-left = most relevant)
    - Color = Value (darker = higher priority)
    - Structure = Task logic

    The intelligence is in the data structure itself, not in processing.
    """

    @timeit
    def __init__(
        self,
        metric_names: List[str],
        precision: int = 8,
        default_output_precision: str = "float32",
    ):
        """Initialize the ZeroModel core components."""
        logger.debug(
            "Initializing ZeroModel with metrics: %s, precision: %s, default_output_precision: %s",
            metric_names,
            precision,
            default_output_precision,
        )
        if not metric_names:
            raise ValueError("metric_names list cannot be empty.")
        valid_precisions = ["uint8", "uint16", "float16", "float32", "float64"]
        if default_output_precision not in valid_precisions:
            logger.warning(
                "Invalid default_output_precision '%s'. Must be one of %s. Defaulting to 'float32'.",
                default_output_precision,
                valid_precisions,
            )
            default_output_precision = "float32"

        # Core attributes
        self.metric_names = list(metric_names)
        self.effective_metric_names = list(metric_names)
        self.precision = max(4, min(16, precision))
        self.default_output_precision = default_output_precision
        self.sorted_matrix: Optional[np.ndarray] = None
        self.doc_order: Optional[np.ndarray] = None
        self.metric_order: Optional[np.ndarray] = None
        self.task: str = "default"
        self.task_config: Optional[Dict[str, Any]] = None

        # Components
        self.duckdb = DuckDBAdapter(self.effective_metric_names)
        self.normalizer = DynamicNormalizer(self.effective_metric_names)
        self._encoder = VPMEncoder(default_output_precision)
        self._feature_engineer = FeatureEngineer()
        # Default organization strategy is in-memory; may be upgraded later
        self._org_strategy = MemoryOrganizationStrategy()

        logger.info(
            "ZeroModel initialized with %d metrics. Default output precision: %s.",
            len(self.effective_metric_names),
            self.default_output_precision,
        )

    @timeit
    def encode(self, output_precision: Optional[str] = None) -> np.ndarray:
        if self.sorted_matrix is None:
            raise ValueError(DATA_NOT_PROCESSED_ERR)
        return self._encoder.encode(self.sorted_matrix, output_precision)

    def get_critical_tile(self, tile_size: int = 3, precision: Optional[str] = None) -> bytes:
        if self.sorted_matrix is None:
            raise ValueError(DATA_NOT_PROCESSED_ERR)
        return self._encoder.get_critical_tile(self.sorted_matrix, tile_size=tile_size, precision=precision)

    @timeit
    def get_decision(self, context_size: int = 3) -> Tuple[int, float]:
        """
        Get top decision with contextual understanding.
        NOTE: This method operates on the internal sorted_matrix (normalized float).
        It should produce a relevance score between 0.0 and 1.0.
        """
        logger.debug(f"Making decision with context size {context_size}")
        if self.sorted_matrix is None:
            error_msg = DATA_NOT_PROCESSED_ERR
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        if context_size <= 0:
            error_msg = f"context_size must be positive, got {context_size}."
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_docs, n_metrics = self.sorted_matrix.shape
        # Determine actual context window size
        actual_context_docs = min(context_size, n_docs)
        actual_context_metrics = min(context_size * 3, n_metrics)  # Width in metrics
        logger.debug(f"Actual decision context: {actual_context_docs} docs x {actual_context_metrics} metrics")

        # Get context window (top-left region) - operates on normalized float data
        context = self.sorted_matrix[:actual_context_docs, :actual_context_metrics]
        logger.debug(f"Context data shape for decision: {context.shape}")

        if context.size == 0:
            logger.warning("Empty context window. Returning default decision (0, 0.0).")
            top_doc_idx_in_original = int(self.doc_order[0]) if (self.doc_order is not None and len(self.doc_order) > 0) else 0
            return (top_doc_idx_in_original, 0.0)

        # Vectorized positional weight calculation
        # Rows (docs) indices
        row_indices = np.arange(actual_context_docs, dtype=np.float64).reshape(-1, 1)
        # Column (metric) indices -> convert to approximate pixel x-coordinate (metric groups of 3)
        col_indices = np.arange(actual_context_metrics, dtype=np.float64)
        pixel_x_coords = col_indices / 3.0
        # Broadcast to grid
        distances = np.sqrt(row_indices**2 + pixel_x_coords**2)
        weights = np.clip(1.0 - distances * 0.3, 0.0, None)

        sum_weights = weights.sum(dtype=np.float64)
        if sum_weights > 0.0:
            weighted_sum = np.sum(context * weights, dtype=np.float64)
            weighted_relevance = float(weighted_sum / sum_weights)
        else:
            logger.warning("Sum of weights is zero after vectorized computation. Assigning relevance score 0.0.")
            weighted_relevance = 0.0

        logger.debug(f"Calculated weighted relevance score: {weighted_relevance:.4f}")

        # Get top document index from the *original* order
        top_doc_idx_in_original = 0
        if self.doc_order is not None and len(self.doc_order) > 0:
            top_doc_idx_in_original = int(self.doc_order[0])
        else:
            logger.warning("doc_order is not available or empty. Defaulting top document index to 0.")
        
        logger.info(f"Decision made: Document index {top_doc_idx_in_original}, Relevance {weighted_relevance:.4f}")
        # Return index (int) and relevance (float, 0.0-1.0)
        return (top_doc_idx_in_original, weighted_relevance)


    @timeit
    def normalize(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Normalize the score matrix using the DynamicNormalizer.

        Args:
            score_matrix: 2D NumPy array of shape [documents x metrics].

        Returns:
            np.ndarray: Normalized score matrix.
        """
        logger.debug(f"Normalizing score matrix with shape {score_matrix.shape}")
        return self.normalizer.normalize(score_matrix)

    def _validate_prepare_inputs(self, score_matrix: np.ndarray, sql_query: str) -> None:
        """Validate inputs for prepare(); mutate effective metric names if needed."""
        original_metric_names = self.effective_metric_names
        if score_matrix is None:
            raise ValueError("score_matrix cannot be None.")
        if not isinstance(score_matrix, np.ndarray):
            raise ValueError(f"score_matrix must be a NumPy ndarray, got {type(score_matrix)}.")
        if score_matrix.ndim != 2:
            raise ValueError(f"score_matrix must be 2D, got shape {score_matrix.shape}.")
        if not sql_query or not isinstance(sql_query, str):
            raise ValueError("sql_query must be a non-empty string.")
        if not np.isfinite(score_matrix).all():
            nan_count = int(np.isnan(score_matrix).sum())
            pos_inf_count = int(np.isposinf(score_matrix).sum())
            neg_inf_count = int(np.isneginf(score_matrix).sum())
            raise ValueError(
                "score_matrix contains non-finite values: "
                f"NaN={nan_count}, +inf={pos_inf_count}, -inf={neg_inf_count}. "
                "Clean or impute these values before calling prepare()."
            )
        # Column count adaptation
        _, n_cols = score_matrix.shape
        if n_cols != len(original_metric_names):
            logger.warning(
                "Column count mismatch: %d (expected) vs %d (received).",
                len(original_metric_names), n_cols,
            )
            if n_cols > len(original_metric_names):
                added = [f"col_{i}" for i in range(len(original_metric_names), n_cols)]
                new_names = list(original_metric_names) + added
            else:
                new_names = list(original_metric_names[:n_cols])
            self.effective_metric_names = new_names
            self.normalizer = DynamicNormalizer(new_names)

    @timeit
    def prepare(self, score_matrix: np.ndarray, sql_query: str, nonlinearity_hint: Optional[str] = None) -> None:
        """Prepare model with data, optional feature engineering, and SQL-driven organization."""
        logger.info(
            f"Preparing ZeroModel with data shape {score_matrix.shape}, query: '{sql_query}', nonlinearity_hint: {nonlinearity_hint}"
        )
        self._validate_prepare_inputs(score_matrix, sql_query)
        original_metric_names = list(self.effective_metric_names)

        # --- 1. Dynamic Normalization ---
        try:
            logger.debug("Updating DynamicNormalizer with new data ranges.")
            self.normalizer.update(score_matrix)
            logger.debug("Normalizer updated. Applying normalization to the data.")
            normalized_data = self.normalizer.normalize(score_matrix)
            logger.debug("Data normalized successfully.")
        except Exception as e:
            logger.error(f"Failed during normalization step: {e}")
            raise RuntimeError(f"Error during data normalization: {e}") from e
        # --- End Dynamic Normalization ---

        # --- 2. Hint-Based Feature Engineering (delegated) ---
        processed_data, effective_metric_names = self._feature_engineer.apply(
            nonlinearity_hint, normalized_data, list(original_metric_names)
        )
        if processed_data is not normalized_data:
            logger.info(
                "Feature engineering added %d new metrics (total now %d)",
                processed_data.shape[1] - normalized_data.shape[1],
                processed_data.shape[1],
            )
        # --- End Feature Engineering ---

        # --- 3. Update Instance Metric Names (after feature engineering) ---
        # Crucially, update the instance's metric_names so get_metadata() and other
        # parts of the class that rely on self.metric_names are consistent with the
        # data that was actually processed and sorted.
        # Store the original names in case they are needed
        self._original_metric_names = self.effective_metric_names
        self.effective_metric_names = effective_metric_names
        logger.debug(f"Updated instance metric_names to reflect processed features: {len(self.effective_metric_names)} metrics.")
        # --- End Update Instance Metric Names ---

        # --- 4. Set organization task & apply via strategy ---
        try:
            # If user provided a full SQL SELECT, upgrade strategy to SQL-backed dynamically
            if isinstance(sql_query, str) and sql_query.strip().lower().startswith("select "):
                if not isinstance(self._org_strategy, SqlOrganizationStrategy):
                    self._org_strategy = SqlOrganizationStrategy(self.duckdb)
            self._org_strategy.set_task(sql_query)
            logger.debug("Organization task set on strategy (%s). Executing organize().", self._org_strategy.name)
            final_matrix, metric_order, doc_order, analysis = self._org_strategy.organize(
                processed_data, self.effective_metric_names
            )
            self.sorted_matrix = final_matrix
            self.metric_order = metric_order
            self.doc_order = doc_order
            self.task = self._org_strategy.name + "_task"
            self.task_config = {"sql_query": sql_query, "analysis": analysis}
            logger.debug("Organization strategy applied successfully.")
        except Exception as e:  # noqa: broad-except
            logger.error(f"Failed during organization strategy execution: {e}")
            raise RuntimeError(f"Error during organization strategy: {e}") from e
        # --- End Strategy Organization ---
            
        logger.info("ZeroModel preparation complete. Ready for encode/get_decision/etc.")

    # Legacy _set_sql_task and _apply_sql_organization removed in favor of strategy abstraction


    # Inside the get_metadata method
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata for the current encoding state."""
        logger.debug("Retrieving metadata.")
        metadata = {
            "task": self.task,
            "task_config": self.task_config,
            "metric_order": self.metric_order.tolist() if self.metric_order is not None else [],
            "doc_order": self.doc_order.tolist() if self.doc_order is not None else [],
            # --- Use the potentially updated self.metric_names ---
            "metric_names": self.effective_metric_names, # This should now be effective_metric_names after prepare()
            "precision": self.precision,
        }
        logger.debug(f"Metadata retrieved: {metadata}")
        return metadata

    def _apply_default_organization(self, score_matrix: np.ndarray):
        """
        Apply default ordering to the score matrix:
        - Documents in original order
        - Metrics in original order
        """
        logger.info("Applying default organization: preserving original order.")

        self.sorted_matrix = score_matrix.copy()
        self.metric_order = np.arange(score_matrix.shape[1])  # [0, 1, 2, ...]
        self.doc_order = np.arange(score_matrix.shape[0])     # [0, 1, 2, ...]

        # Optionally update metadata if needed
        self.metadata = {
            "task": "default",
            "precision": self.precision,
            "metric_names": self.effective_metric_names,
            "metric_order": self.metric_order.tolist(),
            "doc_order": self.doc_order.tolist()
        }


# --- Example usage or test code (optional, remove or comment out for library module) ---
# if __name__ == "__main__":
#     # This would typically be in a separate test or example script
#     pass
