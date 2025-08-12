# zeromodel/core.py
"""
Zero-Model Intelligence core with pluggable organization and VPM-IMG support.

This module provides the core functionality for transforming high-dimensional
policy evaluation data into spatially-optimized visual maps and a canonical
Pixel-Parametric Memory Image (VPM-IMG). Intelligence emerges from the data
layout and virtual views, not heavy processing.
"""

import logging
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
from zeromodel.vpm.metadata import VPMMetadata, AggId

logger = logging.getLogger(__name__)

init_config()

DATA_NOT_PROCESSED_ERR = "Data not processed yet. Call process() or prepare() first."
VPM_IMAGE_NOT_READY_ERR = "VPM image not ready. Call prepare() first."


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
        self.vpm_image_path: Optional[str] = None
        self._vpm_reader: Optional[VPMImageReader] = None

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

    def _get_vpm_reader(self) -> VPMImageReader:
        if self.vpm_image_path is None:
            raise ValueError(VPM_IMAGE_NOT_READY_ERR)
        if self._vpm_reader is None:
            self._vpm_reader = VPMImageReader(self.vpm_image_path)
        return self._vpm_reader

    @timeit
    def prepare(
        self,
        score_matrix: np.ndarray,
        sql_query: str,
        nonlinearity_hint: Optional[str] = None,
        vpm_output_path: Optional[str] = None,
    ) -> VPMMetadata:
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

        logger.debug(
            "normalize: mins[max]=(%s..%s)[%d], dtype=%s",
            float(np.min(self.canonical_matrix)), float(np.max(self.canonical_matrix)),
            self.canonical_matrix.size, self.canonical_matrix.dtype
        )

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

        logger.debug(
            "org: metric_order len=%s, doc_order len=%s; top metric idx=%s, top doc idx=%s",
            None if self.metric_order is None else len(self.metric_order),
            None if self.doc_order is None else len(self.doc_order),
            None if self.metric_order is None else int(self.metric_order[0]),
            None if self.doc_order is None else int(self.doc_order[0]),
        )

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
            mx_d = self.canonical_matrix.T  # (metrics x docs)
            logical_docs = int(mx_d.shape[1])
            # Ensure width meets VPM-IMG header minimum (META_MIN_COLS=12)
            MIN_VPM_WIDTH = 12
            if mx_d.shape[1] < MIN_VPM_WIDTH:
                pad = MIN_VPM_WIDTH - mx_d.shape[1]
                mx_d = np.pad(mx_d, ((0, 0), (0, pad)), mode="constant", constant_values=0.0)
            # Build compact VMETA payload carrying the logical doc_count
            try:
                # Stable-ish 32-bit task hash from SQL query
                import zlib
                task_hash = zlib.crc32(sql_query.encode('utf-8')) & 0xFFFFFFFF
            except Exception:
                task_hash = 0
            try:
                tile_id = VPMMetadata.make_tile_id(f"{self.task}|{mx_d.shape}".encode('utf-8'))
            except Exception:
                tile_id = b"\x00" * 16
            vmeta = VPMMetadata.for_tile(
                level=0,
                metric_count=int(mx_d.shape[0]),
                doc_count=logical_docs,
                doc_block_size=1,
                agg_id=int(AggId.RAW),
                metric_weights=None,
                metric_names=self.effective_metric_names,
                task_hash=int(task_hash),
                tile_id=tile_id,
                parent_id=b"\x00" * 16,
            )

            if vpm_output_path:
                metadata_bytes = vmeta.to_bytes()

                logger.debug(
                    "vpm write: mx_d shape=%s (metrics x docs), pad_to_min_width=%s",
                    getattr(mx_d, "shape", None),
                    mx_d.shape[1] < 12
                )

                writer = VPMImageWriter(
                    score_matrix=mx_d,
                    metric_names=self.effective_metric_names,
                    metadata_bytes=metadata_bytes,
                    store_minmax=True,
                    compression=6,
                )
                writer.write(vpm_output_path)
                self.vpm_image_path = vpm_output_path
                self._vpm_reader = None
                logger.info("VPM-IMG written to %s", vpm_output_path)

            return vmeta
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
        if self.vpm_image_path is None:
            raise ValueError(VPM_IMAGE_NOT_READY_ERR)
        reader = self._get_vpm_reader()
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
        # If no VPM-IMG is present, fall back to encoding the in-memory sorted matrix.
        if self.vpm_image_path is None:
            logger.debug("extract_critical_tile(): no VPM-IMG available, using encoder fallback. size=%s", size)
            if self.sorted_matrix is None:
                raise ValueError(DATA_NOT_PROCESSED_ERR)
            # Encode full image (docs x width x 3) and slice the requested top-left tile
            try:
                img = self._encoder.encode(self.sorted_matrix, output_precision="uint16")
            except Exception as e:
                logger.error("Encoder fallback failed: %s", e)
                raise
            h = max(1, min(size, int(img.shape[0])))
            w = max(1, min(size, int(img.shape[1])))
            tile = img[:h, :w, :]
            logger.debug("encoder fallback tile: shape=%s dtype=%s", getattr(tile, "shape", None), getattr(tile, "dtype", None))
            return tile

        logger.debug("extract_critical_tile(): metric_idx=%s weights=%s size=%s", metric_idx, None if weights is None else dict(weights), size)
        reader = self._get_vpm_reader()
        if logger.isEnabledFor(logging.DEBUG):
            try:
                mb = reader.read_metadata_bytes()
                if mb:
                    md = VPMMetadata.from_bytes(mb)
                    D_eff = int(md.doc_count) or None
                    logger.debug(
                        "vpm reader: level=%s, M=%s, D_phys=%s, D_eff(logical)=%s, h_meta=%s",
                        reader.level, reader.M, reader.D, D_eff, reader.h_meta
                    )
            except Exception as e:
                logger.debug("vpm metadata read failed (non-fatal): %s", e)
        if metric_idx is not None:
            return reader.get_virtual_view(metric_idx=metric_idx, x=0, y=0, width=size, height=size)
        if weights:
            return reader.get_virtual_view(weights=weights, x=0, y=0, width=size, height=size)
        raise ValueError("Provide either 'metric_idx' or 'weights'.")

    def get_decision_by_metric(self, metric_idx: int, context_size: int = 8) -> Tuple[int, float]:
        # Fallback to in-memory path when no VPM-IMG is present
        if self.vpm_image_path is None:
            if self.sorted_matrix is None:
                raise ValueError(DATA_NOT_PROCESSED_ERR)
            n_docs, n_metrics = self.sorted_matrix.shape
            # Map original metric_idx to column in sorted_matrix via metric_order if available
            if self.metric_order is not None and 0 <= metric_idx < len(self.metric_order):
                try:
                    # Find position of metric_idx in the sorted order
                    pos_arr = np.where(self.metric_order == metric_idx)[0]
                    col_idx = int(pos_arr[0]) if pos_arr.size > 0 else int(min(metric_idx, n_metrics - 1))
                except Exception:
                    col_idx = int(min(metric_idx, n_metrics - 1))
            else:
                col_idx = int(min(metric_idx, n_metrics - 1))
            top_doc = 0  # sorted_matrix already reflects doc_order (top candidate first)
            h = int(max(1, min(context_size, n_docs)))
            try:
                rel = float(np.mean(self.sorted_matrix[:h, col_idx]))
            except Exception:
                rel = 0.0
            return (top_doc, rel)
        reader = self._get_vpm_reader()
        perm = reader.virtual_order(metric_idx=metric_idx, descending=True, top_k=context_size)
        if len(perm) == 0:
            return (0, 0.0)
        top_doc = int(perm[0])
        try:
            tile = reader.get_virtual_view(metric_idx=metric_idx, x=0, y=0, width=context_size, height=1)
            logger.debug(
                "tile: shape=%s dtype=%s R[min,max]=(%s,%s) G[min,max]=(%s,%s) B[min,max]=(%s,%s)",
                getattr(tile, "shape", None), getattr(tile, "dtype", None),
                int(tile[...,0].min()), int(tile[...,0].max()),
                int(tile[...,1].min()), int(tile[...,1].max()),
                int(tile[...,2].min()), int(tile[...,2].max())
            )
            rel = float(np.mean(tile[0, :, 0]) / 65535.0) if tile.size > 0 else 0.0
        except Exception:
            rel = 0.0
        return (top_doc, rel)

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
            "vpm_image_path": self.vpm_image_path,
        }
        logger.debug(f"Metadata retrieved: {metadata}")
        return metadata

