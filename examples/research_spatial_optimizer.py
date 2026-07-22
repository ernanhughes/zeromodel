from __future__ import annotations

import json
from pathlib import Path

from zeromodel.analysis.spatial import (
    SpatialOptimizer,
    build_optimized_view,
    optimize_view_profile,
)
from zeromodel.core.artifact import ScoreTable
from zeromodel.core.bundle import to_bundle
from zeromodel.core.render import (
    write_png,
    write_svg,
)

OUT_DIR = Path(".zeromodel-demo")
OUT_DIR.mkdir(exist_ok=True)

source = ScoreTable(
    values=[
        [0.10, 0.50, 0.20],
        [0.95, 0.50, 0.25],
        [0.90, 0.50, 0.15],
        [0.05, 0.50, 0.20],
    ],
    row_ids=["background", "target_a", "target_b", "flat"],
    metric_ids=["target", "constant", "weak"],
    metadata={"kind": "spatial_optimizer_demo"},
)

optimizer = SpatialOptimizer(Kc=2, Kr=2, alpha=0.95, max_iters=40)
result = optimize_view_profile(source, name="optimized-target", optimizer=optimizer)
view = build_optimized_view(source, name="optimized-target", optimizer=optimizer)

bundle_path = to_bundle(view, OUT_DIR / "spatial_optimized.vpm")
png_path = write_png(view, OUT_DIR / "spatial_optimized.png")
svg_path = write_svg(view, OUT_DIR / "spatial_optimized.svg")
summary_path = OUT_DIR / "spatial_optimized_summary.json"
summary_path.write_text(
    json.dumps(
        {
            "artifact_id": view.artifact_id,
            "source_digest": source.digest,
            "top_left_row": view.cell(0, 0).row_id,
            "top_left_metric": view.cell(0, 0).metric_id,
            "optimization": result.to_dict(),
            "outputs": {
                "bundle": str(bundle_path),
                "png": str(png_path),
                "svg": str(svg_path),
            },
        },
        indent=2,
        sort_keys=True,
    ),
    encoding="utf-8",
)

print(json.dumps(json.loads(summary_path.read_text(encoding="utf-8")), indent=2, sort_keys=True))
