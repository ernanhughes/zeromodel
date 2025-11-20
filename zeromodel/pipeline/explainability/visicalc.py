# zeromodel/pipeline/explainability/visicalc.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage
from zeromodel.pipeline.utils.json_sanitize import dumps_safe

log = logging.getLogger(__name__)


class VisiCalcStage(PipelineStage):
    """
    VisiCalcStage

    Read-only analytics stage for a 2D score matrix (rows x metrics).

    It computes:
      - Global stats (mean/std/min/max/sparsity/entropy)
      - Frontier band coverage for a chosen metric
      - Frontier density per row-region (top/mid/bottom, etc.)
      - Per-metric stats (mean/std/min/max/quantiles, corr with frontier)
      - A dense feature vector + names for downstream critics/policies

    It does NOT modify the matrix; it only returns metadata and updates
    the shared context with a `visicalc` summary.
    """

    name = "visicalc"
    category = "analytics"

    def __init__(self, **params):
        super().__init__(**params)
        self.frontier_metric_index = params.get("frontier_metric_index", 0)  # index to split on
        self.frontier_low = float(params.get("frontier_low", 0.25))
        self.frontier_high = float(params.get("frontier_high", 0.75))
        self.row_region_splits = int(params.get("row_region_splits", 3))

    def validate_params(self):
        if self.frontier_low >= self.frontier_high:
            raise ValueError(
                f"VisiCalcStage: frontier_low ({self.frontier_low}) "
                f"must be < frontier_high ({self.frontier_high})"
            )
        if self.row_region_splits <= 0:
            raise ValueError("VisiCalcStage: row_region_splits must be > 0")

    def process(
        self,
        vpm: np.ndarray,
        context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        ctx = self.get_context(context)

        if vpm.ndim != 2:
            raise ValueError(
                f"VisiCalcStage expects a 2D matrix (rows x metrics), got {vpm.ndim}D"
            )

        num_rows, num_metrics = vpm.shape

        # -------------------------
        # Metric name reconciliation
        # -------------------------

        # -------------------------
        # Global stats
        # -------------------------
        global_mean = float(np.mean(vpm))
        global_std = float(np.std(vpm))
        global_min = float(np.min(vpm))
        global_max = float(np.max(vpm))

        sparsity_1e3 = float(np.mean(vpm <= 1e-3))
        sparsity_1e2 = float(np.mean(vpm <= 1e-2))

        # Histogram-based entropy (normalized to [0,1] over non-empty bins)
        hist, _ = np.histogram(vpm, bins=32, range=(0.0, 1.0), density=True)
        hist = hist.astype(np.float64)
        hist = hist[hist > 0]
        if hist.size > 0:
            entropy = float(-np.sum(hist * np.log(hist)) / np.log(hist.size))
        else:
            entropy = 0.0

        # -------------------------
        # Frontier metric + band
        # -------------------------

        frontier_values = vpm[:, self.frontier_metric_index]
        low = self.frontier_low
        high = self.frontier_high

        band_mask = (frontier_values >= low) & (frontier_values <= high)
        frontier_fraction = float(band_mask.mean()) if num_rows > 0 else 0.0
        num_frontier_rows = int(band_mask.sum())

        # global low / high fractions (below band / above band)
        if num_rows > 0:
            low_mask = frontier_values < low
            high_mask = frontier_values > high
            global_low_frac = float(low_mask.mean())
            global_high_frac = float(high_mask.mean())
        else:
            global_low_frac = 0.0
            global_high_frac = 0.0

        # -------------------------
        # Row-region splits
        # -------------------------
        splits = self.row_region_splits
        region_bounds: List[Tuple[int, int]] = []
        base = num_rows // splits
        remainder = num_rows % splits
        start = 0
        for i in range(splits):
            extra = 1 if i < remainder else 0
            end = start + base + extra
            region_bounds.append((start, end))
            start = end

        row_region_stats: Dict[str, Dict[str, Any]] = {}
        region_frontier_fractions: List[float] = []

        for i, (rs, re) in enumerate(region_bounds):
            if rs >= re:
                region_frontier_fractions.append(0.0)
                row_region_stats[f"region_{i}"] = {
                    "index": i,
                    "row_start": int(rs),
                    "row_end": int(re),
                    "num_rows": 0,
                    "frontier_fraction": 0.0,
                    "low_frac": 0.0,
                    "high_frac": 0.0,
                    "mean_frontier_value": None,
                }
                continue

            region_frontier_vals = frontier_values[rs:re]
            region_band_mask = band_mask[rs:re]
            region_frontier_fraction = float(region_band_mask.mean())

            # per-region low / high band coverage
            region_low_mask = region_frontier_vals < low
            region_high_mask = region_frontier_vals > high
            region_low_frac = float(region_low_mask.mean())
            region_high_frac = float(region_high_mask.mean())

            region_frontier_fractions.append(region_frontier_fraction)

            row_region_stats[f"region_{i}"] = {
                "index": i,
                "row_start": int(rs),
                "row_end": int(re),
                "num_rows": int(re - rs),
                "frontier_fraction": region_frontier_fraction,
                "low_frac": region_low_frac,
                "high_frac": region_high_frac,
                "mean_frontier_value": float(region_frontier_vals.mean()),
            }

        # -------------------------
        # Per-metric stats
        # -------------------------
        per_metric_stats: List[Dict[str, Any]] = []
        frontier_col = frontier_values
        frontier_std = float(np.std(frontier_col))

        for j in range(num_metrics):
            col = vpm[:, j]
            mean = float(np.mean(col))
            std = float(np.std(col))
            col_min = float(np.min(col))
            col_max = float(np.max(col))
            q25, q50, q75 = [float(q) for q in np.quantile(col, [0.25, 0.5, 0.75])]

            if j == self.frontier_metric_index:
                corr_frontier = 1.0
            else:
                if std > 1e-9 and frontier_std > 1e-9:
                    corr_frontier = float(np.corrcoef(col, frontier_col)[0, 1])
                else:
                    corr_frontier = 0.0

            per_metric_stats.append(
                {
                    "index": j,
                    "mean": mean,
                    "std": std,
                    "min": col_min,
                    "max": col_max,
                    "q25": q25,
                    "q50": q50,
                    "q75": q75,
                    "corr_frontier": corr_frontier,
                }
            )

        # -------------------------
        # Dense feature vector (+ names)
        # -------------------------
        feature_names: List[str] = []
        feature_values: List[float] = []

        def add_feature(name: str, value: float):
            feature_names.append(name)
            feature_values.append(float(value))

        # global
        add_feature("global_mean", global_mean)
        add_feature("global_std", global_std)
        add_feature("global_min", global_min)
        add_feature("global_max", global_max)
        add_feature("sparsity_le_1e-3", sparsity_1e3)
        add_feature("sparsity_le_1e-2", sparsity_1e2)
        add_feature("entropy", entropy)

        # frontier (global)
        add_feature("frontier_fraction", frontier_fraction)
        add_feature("global_low_frac", global_low_frac)
        add_feature("global_high_frac", global_high_frac)

        # row regions
        for i, frac in enumerate(region_frontier_fractions):
            add_feature(f"region_{i}_frontier_fraction", frac)

        # per-metric compressed view
        for pm in per_metric_stats:
            prefix = f"metric[{pm['index']}]"
            add_feature(f"{prefix}_mean", pm["mean"])
            add_feature(f"{prefix}_std", pm["std"])
            add_feature(f"{prefix}_corr_frontier", pm["corr_frontier"])

        visicalc_summary: Dict[str, Any] = {
            "shape": (num_rows, num_metrics),
            "global": {
                "mean": global_mean,
                "std": global_std,
                "min": global_min,
                "max": global_max,
                "sparsity_le_1e-3": sparsity_1e3,
                "sparsity_le_1e-2": sparsity_1e2,
                "entropy": entropy,
                "low_frac": global_low_frac,
                "high_frac": global_high_frac,
            },
            "frontier": {
                "metric_index": self.frontier_metric_index,
                "low": low,
                "high": high,
                "frontier_fraction": frontier_fraction,
                "num_frontier_rows": num_frontier_rows,
            },
            "row_regions": row_region_stats,
            "per_metric": per_metric_stats,
            "feature_names": feature_names,
            "feature_vector": feature_values,
            # convenience top-level aliases (for Stephanie)
            "frontier_metric_index": self.frontier_metric_index,
            "frontier_low": low,
            "frontier_high": high,
            "frontier_frac": frontier_fraction,
            "row_region_splits": splits,
        }

        # Expose in shared context for downstream stages
        ctx["visicalc"] = visicalc_summary

        meta: Dict[str, Any] = {
            "stage": self.name,
            "shape": (num_rows, num_metrics),
            "visicalc": visicalc_summary,
        }
        return vpm, meta


def format_visicalc_stats(
    stats: Dict[str, Any],
    max_metrics: int = 6,
) -> str:
    """
    Pretty-print VisiCalc metadata for debugging / tests.

    Accepts either:
      - full stage metadata (with 'visicalc' key), or
      - the nested 'visicalc' dict itself.
    """
    return dumps_safe(stats, indent=2)
