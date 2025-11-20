from __future__ import annotations

import logging
from typing import Any, Dict, Tuple, Optional, List

import numpy as np

from zeromodel.pipeline.base import PipelineStage

logger = logging.getLogger(__name__)


class VisiCalcStage(PipelineStage):
    """
    VisiCalcStage

    Lightweight analysis stage that computes:
      - frontier stats for a chosen metric (how many rows are in a mid-band)
      - row-region bands (top/mid/bottom, or arbitrary splits)
      - a compact feature vector summarizing the matrix & frontier pattern

    It is intentionally non-destructive:
      - input vpm shape is preserved
      - results are written into context["visicalc"]
      - meta contains only shallow, serializable stats
    """

    name = "visicalc"
    category = "analysis"

    def __init__(self, **params):
        super().__init__(**params)

        # Which metric to treat as "frontier" dimension
        self.frontier_metric: Optional[str] = params.get("frontier_metric")

        # Mid-band thresholds on that metric
        self.frontier_low: float = float(params.get("frontier_low", 0.25))
        self.frontier_high: float = float(params.get("frontier_high", 0.75))

        # How many row bands to summarize (e.g., 3 => top / mid / bottom)
        self.row_region_splits: int = int(params.get("row_region_splits", 3))

    def validate_params(self):
        if self.frontier_low >= self.frontier_high:
            raise ValueError(
                f"VisiCalcStage: frontier_low ({self.frontier_low}) must be < frontier_high ({self.frontier_high})"
            )
        if self.row_region_splits <= 0:
            raise ValueError(
                f"VisiCalcStage: row_region_splits must be positive, got {self.row_region_splits}"
            )

    def process(
        self,
        vpm: np.ndarray,
        context: Dict[str, Any] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Args:
            vpm: 2D array (rows x metrics)
            context: shared dict; we will write 'visicalc' into it

        Returns:
            (vpm, meta) — vpm is unchanged; meta is a small summary dict
        """
        ctx = self.get_context(context)

        if vpm.ndim != 2:
            raise ValueError(f"VisiCalcStage expects a 2D matrix, got {vpm.ndim}D")

        n_rows, n_cols = vpm.shape

        # ---- Metric name resolution ----
        metric_names: List[str] = ctx.get("metric_names") or [f"m{i}" for i in range(n_cols)]

        if len(metric_names) != n_cols:
            # Align length if upstream stages did something odd
            if len(metric_names) > n_cols:
                metric_names = metric_names[:n_cols]
            else:
                metric_names = metric_names + [f"col_{i}" for i in range(len(metric_names), n_cols)]
            ctx["metric_names"] = metric_names

        # Choose frontier metric index
        if self.frontier_metric is None:
            frontier_idx = 0
            frontier_name = metric_names[0]
        else:
            try:
                frontier_idx = metric_names.index(self.frontier_metric)
                frontier_name = self.frontier_metric
            except ValueError:
                logger.warning(
                    "VisiCalcStage: frontier_metric '%s' not found in metric_names; "
                    "defaulting to first metric.",
                    self.frontier_metric,
                )
                frontier_idx = 0
                frontier_name = metric_names[0]

        frontier_values = vpm[:, frontier_idx]

        # ---- Frontier band stats ----
        band_mask = (frontier_values >= self.frontier_low) & (frontier_values <= self.frontier_high)
        global_frontier_fraction = float(band_mask.mean()) if n_rows > 0 else 0.0

        frontier_stats = {
            "metric": frontier_name,
            "low": self.frontier_low,
            "high": self.frontier_high,
            "global_frontier_fraction": global_frontier_fraction,
            "row_count": int(n_rows),
        }

        # ---- Row region splits ----
        row_regions: Dict[str, Dict[str, Any]] = {}
        splits = min(self.row_region_splits, max(1, n_rows))  # cannot exceed n_rows if very small

        # Equal-ish partition of rows
        base = n_rows // splits
        rem = n_rows % splits
        start = 0
        for i in range(splits):
            extra = 1 if i < rem else 0
            end = start + base + extra
            if start >= end:  # safety
                break

            region_name = f"region_{i}"
            region_mask = band_mask[start:end]
            size = end - start
            mid_band_density = float(region_mask.mean()) if size > 0 else 0.0

            row_regions[region_name] = {
                "row_start": int(start),
                "row_end": int(end),
                "row_count": int(size),
                "mid_band_density": mid_band_density,
            }
            start = end

        # ---- Global scalar features ----
        # Simple summary: mean, std of whole matrix + frontier stats per region
        global_mean = float(vpm.mean()) if vpm.size > 0 else 0.0
        global_std = float(vpm.std()) if vpm.size > 0 else 0.0

        region_densities = [stats["mid_band_density"] for stats in row_regions.values()]
        features_list = [global_mean, global_std, global_frontier_fraction] + region_densities
        features = np.array(features_list, dtype=np.float32)

        # ---- Write into context ----
        ctx["visicalc"] = {
            "frontier_stats": frontier_stats,
            "row_regions": row_regions,
            "features": features,
        }

        # Small meta dict merged into pipeline metadata
        meta: Dict[str, Any] = {
            "stage": self.name,
            "shape": (n_rows, n_cols),
            "frontier_metric": frontier_name,
            "frontier_fraction": global_frontier_fraction,
            "row_region_count": len(row_regions),
            "frontier_stats": frontier_stats,
            "row_regions": row_regions,
            "features": features,
        }

        return vpm, meta
