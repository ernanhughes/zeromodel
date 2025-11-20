from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np

from zeromodel.pipeline.base import PipelineStage

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
        self.frontier_metric = params.get("frontier_metric")  # name or None
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
        metric_names = ctx.get("metric_names") or [f"m{i}" for i in range(num_metrics)]
        if len(metric_names) != num_metrics:
            if len(metric_names) > num_metrics:
                metric_names = metric_names[:num_metrics]
            else:
                metric_names = metric_names + [
                    f"m{i}" for i in range(len(metric_names), num_metrics)
                ]
            ctx["metric_names"] = metric_names

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
        if self.frontier_metric is None:
            frontier_idx = 0
            frontier_name = metric_names[0]
        else:
            try:
                frontier_idx = metric_names.index(self.frontier_metric)
                frontier_name = self.frontier_metric
            except ValueError:
                frontier_idx = 0
                frontier_name = metric_names[0]
                log.warning(
                    "VisiCalcStage: frontier_metric '%s' not found, "
                    "falling back to '%s'",
                    self.frontier_metric,
                    frontier_name,
                )

        frontier_values = vpm[:, frontier_idx]
        low = self.frontier_low
        high = self.frontier_high

        band_mask = (frontier_values >= low) & (frontier_values <= high)
        frontier_fraction = float(band_mask.mean()) if num_rows > 0 else 0.0
        num_frontier_rows = int(band_mask.sum())

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
                    "row_start": int(rs),
                    "row_end": int(re),
                    "num_rows": 0,
                    "frontier_fraction": 0.0,
                    "mean_frontier_value": None,
                }
                continue

            region_mask = band_mask[rs:re]
            region_frontier_fraction = float(region_mask.mean())
            region_frontier_fractions.append(region_frontier_fraction)

            region_frontier_values = frontier_values[rs:re]
            row_region_stats[f"region_{i}"] = {
                "row_start": int(rs),
                "row_end": int(re),
                "num_rows": int(re - rs),
                "frontier_fraction": region_frontier_fraction,
                "mean_frontier_value": float(region_frontier_values.mean()),
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

            if j == frontier_idx:
                corr_frontier = 1.0
            else:
                if std > 1e-9 and frontier_std > 1e-9:
                    corr_frontier = float(np.corrcoef(col, frontier_col)[0, 1])
                else:
                    corr_frontier = 0.0

            per_metric_stats.append(
                {
                    "index": j,
                    "name": metric_names[j],
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

        # frontier
        add_feature("frontier_fraction", frontier_fraction)

        # row regions
        for i, frac in enumerate(region_frontier_fractions):
            add_feature(f"region_{i}_frontier_fraction", frac)

        # per-metric compressed view
        for pm in per_metric_stats:
            prefix = f"metric[{pm['name']}]"
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
            },
            "frontier": {
                "metric_name": frontier_name,
                "metric_index": frontier_idx,
                "low": low,
                "high": high,
                "frontier_fraction": frontier_fraction,
                "num_frontier_rows": num_frontier_rows,
            },
            "row_regions": row_region_stats,
            "per_metric": per_metric_stats,
            "feature_names": feature_names,
            "feature_vector": feature_values,
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
    if "visicalc" in stats:
        data = stats["visicalc"]
    else:
        data = stats

    lines: List[str] = []

    shape = data.get("shape", (None, None))
    lines.append(f"VisiCalc stats (shape={shape[0]}x{shape[1]}):")

    # global
    g = data.get("global", {})
    lines.append(
        "  global:"
        f" mean={g.get('mean', 0.0):.4f}"
        f" std={g.get('std', 0.0):.4f}"
        f" min={g.get('min', 0.0):.4f}"
        f" max={g.get('max', 0.0):.4f}"
        f" sparsity<=1e-3={g.get('sparsity_le_1e-3', 0.0):.3f}"
        f" entropy={g.get('entropy', 0.0):.3f}"
    )

    # frontier
    fr = data.get("frontier", {})
    lines.append(
        "  frontier:"
        f" metric={fr.get('metric_name')}[idx={fr.get('metric_index')}]"
        f" band=[{fr.get('low', 0.0):.2f},{fr.get('high', 1.0):.2f}]"
        f" fraction={fr.get('frontier_fraction', 0.0):.3f}"
        f" rows={fr.get('num_frontier_rows', 0)}"
    )

    # row regions
    rr = data.get("row_regions", {})
    lines.append(f"  row_regions: {len(rr)}")
    for name, r in rr.items():
        lines.append(
            f"    {name}: rows[{r['row_start']}:{r['row_end']})"
            f" n={r['num_rows']}"
            f" frontier_fraction={r['frontier_fraction']:.3f}"
        )

    # per metric (shortened)
    pm_list = data.get("per_metric", [])
    lines.append(f"  per_metric (first {min(max_metrics, len(pm_list))}):")
    for pm in pm_list[:max_metrics]:
        lines.append(
            f"    {pm['name']}:"
            f" mean={pm['mean']:.4f}"
            f" std={pm['std']:.4f}"
            f" min={pm['min']:.4f}"
            f" max={pm['max']:.4f}"
            f" corr_frontier={pm['corr_frontier']:.3f}"
        )

    fv = data.get("feature_vector") or []
    lines.append(f"  feature_vector: {len(fv)} dims")

    return "\n".join(lines)
