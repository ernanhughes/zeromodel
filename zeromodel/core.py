# zeromodel/core.py
"""
Zero-Model Intelligence core with pluggable organization and VPM-IMG support.

This module provides the core functionality for transforming high-dimensional
policy evaluation data into spatially-optimized visual maps and a canonical
Pixel-Parametric Memory Image (VPM-IMG). Intelligence emerges from the data
layout and virtual views, not heavy processing.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from zeromodel.config import get_config, init_config
from zeromodel.constants import precision_dtype_map
from zeromodel.duckdb_adapter import DuckDBAdapter
from zeromodel.feature_engineer import FeatureEngineer
from zeromodel.normalizer import DynamicNormalizer
from zeromodel.organization import (MemoryOrganizationStrategy,
                                    SqlOrganizationStrategy)
from zeromodel.timing import timeit
from zeromodel.vpm.encoder import VPMEncoder
from zeromodel.vpm.image import VPMImageReader, VPMImageWriter

logger = logging.getLogger(__name__)

init_config()

DATA_NOT_PROCESSED_ERR = "Data not processed yet. Call process() or prepare() first."
PPM_IMAGE_NOT_READY_ERR = "PPM image not ready. Call prepare() first."


class ZeroModel:
    """
    Zero-Model Intelligence encoder/decoder with VPM-IMG support.

    Workflow:
    1. prepare() -> normalize, optional features, analyze org, write VPM-IMG
    2. compile_view()/extract_critical_tile() -> use VPM-IMG reader for virtual addressing
    """

    def __init__(self, metric_names: List[str]) -> None:
        logger.debug(
            "Initializing ZeroModel with metrics: %s, config: %s",
            metric_names,
            str(get_config("core")),
        )
        if not metric_names:
            raise ValueError("metric_names list cannot be empty.")

        # Core attributes
        self.metric_names = list(metric_names)
        self.effective_metric_names = list(metric_names)
        self.precision = get_config("core").get("precision", 8)
        if not (4 <= int(self.precision) <= 16):
            raise ValueError("Precision must be between 4 and 16.")
        self.default_output_precision = get_config("core").get(
            "default_output_precision", "float32"
        )
        if self.default_output_precision not in precision_dtype_map:
            raise ValueError(
                f"Invalid default_output_precision '{self.default_output_precision}'. "
                f"Must be one of {list(precision_dtype_map.keys())}."
            )

        # VPM-IMG state (canonical memory image)
        self.canonical_matrix: Optional[np.ndarray] = None  # docs x metrics (float)
        self.ppm_image_path: Optional[str] = None
        self._ppm_reader: Optional[VPMImageReader] = None

        # Legacy/compat state (virtual view matrix path)
        self.sorted_matrix: Optional[np.ndarray] = None
        self.doc_order: Optional[np.ndarray] = None
        self.metric_order: Optional[np.ndarray] = None
        self.task: str = "default"
        self.task_config: Optional[Dict[str, Any]] = None

        # Components
        self.duckdb = DuckDBAdapter(self.effective_metric_names)
        self.normalizer = DynamicNormalizer(self.effective_metric_names)
        self._encoder = VPMEncoder(get_config("core").get("default_output_precision", "float32"))
        self._feature_engineer = FeatureEngineer()
        self._org_strategy = MemoryOrganizationStrategy()

        logger.info(
            "ZeroModel initialized with %d metrics. Default output precision: %s.",
            len(self.effective_metric_names),
            self.default_output_precision,
        )

    def _get_ppm_reader(self) -> VPMImageReader:
        if self.ppm_image_path is None:
            raise ValueError(PPM_IMAGE_NOT_READY_ERR)
        if self._ppm_reader is None:
            self._ppm_reader = VPMImageReader(self.ppm_image_path)
        return self._ppm_reader

    @timeit
    def prepare(
        self,
        score_matrix: np.ndarray,
        sql_query: str,
        nonlinearity_hint: Optional[str] = None,
        ppm_output_path: Optional[str] = None,
    ) -> None:
        logger.info(
            "Preparing ZeroModel with data shape %s, query: '%s', nonlinearity_hint: %s",
            getattr(score_matrix, "shape", None),
            sql_query,
            nonlinearity_hint,
        )
        self._validate_prepare_inputs(score_matrix, sql_query)

        # 1) Normalize to canonical matrix (docs x metrics)
        try:
            self.normalizer.update(score_matrix)
            normalized_data = self.normalizer.normalize(score_matrix)
            self.canonical_matrix = normalized_data.astype(np.float32, copy=False)
        except Exception as e:  # noqa: broad-except
            logger.error("Normalization failed: %s", e)
            raise RuntimeError(f"Error during data normalization: {e}") from e

        # 2) Optional feature engineering
        original_metric_names = list(self.effective_metric_names)
        processed_data, effective_metric_names = self._feature_engineer.apply(
            nonlinearity_hint, self.canonical_matrix, original_metric_names
        )
        if processed_data is not self.canonical_matrix:
            logger.info(
                "Feature engineering added %d new metrics (total now %d)",
                processed_data.shape[1] - self.canonical_matrix.shape[1],
                processed_data.shape[1],
            )
            self.canonical_matrix = processed_data
        self.effective_metric_names = effective_metric_names

        # 3) Organization analysis (for metadata; we keep canonical unsorted)
        try:
            use_duckdb = bool(get_config("core").get("use_duckdb", False))
            if use_duckdb:
                self._org_strategy = SqlOrganizationStrategy(self.duckdb)
            else:
                self._org_strategy = MemoryOrganizationStrategy()
            self._org_strategy.set_task(sql_query)
            _, metric_order, doc_order, analysis = self._org_strategy.organize(
                self.canonical_matrix, self.effective_metric_names
            )
            self.metric_order = metric_order
            self.doc_order = doc_order
            self.task = self._org_strategy.name + "_task"
            self.task_config = {"sql_query": sql_query, "analysis": analysis}
        except Exception as e:  # noqa: broad-except
            logger.error("Organization analysis failed: %s", e)
            raise RuntimeError(f"Error during organization strategy: {e}") from e

        # Legacy: materialize a sorted view for backward-compat APIs
        if self.doc_order is not None and self.metric_order is not None:
            self.sorted_matrix = self.canonical_matrix[self.doc_order][:, self.metric_order]

        # 4) Write canonical memory to VPM-IMG (metrics x docs)
        try:
            if ppm_output_path is None:
                ppm_output_path = os.path.join(os.getcwd(), "zeromodel_canonical.ppm.png")
            mx_d = self.canonical_matrix.T  # (metrics x docs)
            # Ensure width meets VPM-IMG header minimum (META_MIN_COLS=12)
            MIN_PPM_WIDTH = 12
            if mx_d.shape[1] < MIN_PPM_WIDTH:
                pad = MIN_PPM_WIDTH - mx_d.shape[1]
                mx_d = np.pad(mx_d, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
            writer = VPMImageWriter(
                score_matrix=mx_d,
                metric_names=self.effective_metric_names,
                store_minmax=True,
                compression=6,
            )
            writer.write(ppm_output_path)
            self.ppm_image_path = ppm_output_path
            self._ppm_reader = None
            logger.info("VPM-IMG written to %s", ppm_output_path)
        except Exception as e:  # noqa: broad-except
            logger.error("VPM-IMG write failed: %s", e)
            raise RuntimeError(f"Error writing VPM-IMG: {e}") from e

        logger.info("ZeroModel preparation complete. VPM-IMG is ready.")

    # ---- VPM-IMG based operations ----
    def compile_view(
        self,
        *,
        metric_idx: Optional[int] = None,
        weights: Optional[Dict[int, float]] = None,
        top_k: Optional[int] = None,
    ) -> np.ndarray:
        if self.ppm_image_path is None:
            raise ValueError(PPM_IMAGE_NOT_READY_ERR)
        reader = self._get_ppm_reader()
        if metric_idx is not None:
            return reader.virtual_order(metric_idx=metric_idx, descending=True, top_k=top_k)
        if weights:
            return reader.virtual_order(weights=weights, descending=True, top_k=top_k)
        raise ValueError("Provide either 'metric_idx' or 'weights'.")

    def extract_critical_tile(
        self,
        *,
        metric_idx: Optional[int] = None,
        weights: Optional[Dict[int, float]] = None,
        size: int = 8,
    ) -> np.ndarray:
        if self.ppm_image_path is None:
            raise ValueError(PPM_IMAGE_NOT_READY_ERR)
        reader = self._get_ppm_reader()
        if metric_idx is not None:
            return reader.get_virtual_view(metric_idx=metric_idx, x=0, y=0, width=size, height=size)
        if weights:
            return reader.get_virtual_view(weights=weights, x=0, y=0, width=size, height=size)
        raise ValueError("Provide either 'metric_idx' or 'weights'.")

    def get_decision_by_metric(self, metric_idx: int, context_size: int = 8) -> Tuple[int, float]:
        if self.ppm_image_path is None:
            raise ValueError(PPM_IMAGE_NOT_READY_ERR)
        reader = self._get_ppm_reader()
        perm = reader.virtual_order(metric_idx=metric_idx, descending=True, top_k=context_size)
        if len(perm) == 0:
            return (0, 0.0)
        top_doc = int(perm[0])
        try:
            tile = reader.get_virtual_view(metric_idx=metric_idx, x=0, y=0, width=context_size, height=1)
            rel = float(np.mean(tile[0, :, 0]) / 65535.0) if tile.size > 0 else 0.0
        except Exception:
            rel = 0.0
        return (top_doc, rel)

    # ---- Legacy APIs retained for compatibility ----
    def encode(self, output_precision: Optional[str] = None) -> np.ndarray:
        if self.sorted_matrix is None:
            raise ValueError(DATA_NOT_PROCESSED_ERR)
        return self._encoder.encode(self.sorted_matrix, output_precision)

    def get_critical_tile(self, tile_size: int = 3, precision: Optional[str] = None) -> bytes:
        logger.warning("get_critical_tile is legacy. Prefer extract_critical_tile with VPM-IMG.")
        if self.sorted_matrix is None:
            raise ValueError(DATA_NOT_PROCESSED_ERR)
        return self._encoder.get_critical_tile(self.sorted_matrix, tile_size=tile_size, precision=precision)

    def get_decision(self, context_size: int = 3) -> Tuple[int, float]:
        logger.debug(f"Making decision with context size {context_size}")
        if self.sorted_matrix is None:
            raise ValueError(DATA_NOT_PROCESSED_ERR)

        if context_size <= 0:
            raise ValueError(f"context_size must be positive, got {context_size}.")

        n_docs, n_metrics = self.sorted_matrix.shape
        actual_context_docs = min(context_size, n_docs)
        actual_context_metrics = min(context_size * 3, n_metrics)
        context = self.sorted_matrix[:actual_context_docs, :actual_context_metrics]

        if context.size == 0:
            top_doc_idx_in_original = int(self.doc_order[0]) if (self.doc_order is not None and len(self.doc_order) > 0) else 0
            return (top_doc_idx_in_original, 0.0)

        row_indices = np.arange(actual_context_docs, dtype=np.float64).reshape(-1, 1)
        col_indices = np.arange(actual_context_metrics, dtype=np.float64)
        pixel_x_coords = col_indices / 3.0
        distances = np.sqrt(row_indices ** 2 + pixel_x_coords ** 2)
        weights = np.clip(1.0 - distances * 0.3, 0.0, None)
        sum_weights = weights.sum(dtype=np.float64)
        weighted_relevance = (
            float(np.sum(context * weights, dtype=np.float64) / sum_weights) if sum_weights > 0.0 else 0.0
        )
        top_doc_idx_in_original = int(self.doc_order[0]) if (self.doc_order is not None and len(self.doc_order) > 0) else 0
        return (top_doc_idx_in_original, weighted_relevance)

    # ---- Shared utilities from previous implementation ----
    def normalize(self, score_matrix: np.ndarray) -> np.ndarray:
        logger.debug(f"Normalizing score matrix with shape {score_matrix.shape}")
        # Return float32 to match the dtype used in canonical/sorted matrices
        return self.normalizer.normalize(score_matrix, as_float32=True)

    def _validate_prepare_inputs(self, score_matrix: np.ndarray, sql_query: str) -> None:
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
        _, n_cols = score_matrix.shape
        if n_cols != len(original_metric_names):
            logger.warning(
                "Column count mismatch: %d (expected) vs %d (received).",
                len(original_metric_names),
                n_cols,
            )
            if n_cols > len(original_metric_names):
                added = [f"col_{i}" for i in range(len(original_metric_names), n_cols)]
                new_names = list(original_metric_names) + added
            else:
                new_names = list(original_metric_names[:n_cols])
            self.effective_metric_names = new_names
            self.normalizer = DynamicNormalizer(new_names)

    def get_metadata(self) -> Dict[str, Any]:
        logger.debug("Retrieving metadata.")
        metadata = {
            "task": self.task,
            "task_config": self.task_config,
            "metric_order": self.metric_order.tolist() if self.metric_order is not None else [],
            "doc_order": self.doc_order.tolist() if self.doc_order is not None else [],
            "metric_names": self.effective_metric_names,
            "precision": self.precision,
            "default_output_precision": self.default_output_precision,
            "ppm_image_path": self.ppm_image_path,
        }
        logger.debug(f"Metadata retrieved: {metadata}")
        return metadata

