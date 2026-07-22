from __future__ import annotations

import json
from pathlib import Path

from zeromodel.analysis.manifold import build_decision_manifold
from zeromodel.analysis.spatial import SpatialOptimizer
from zeromodel.core.artifact import ScoreTable
from zeromodel.core.render import write_png, write_svg


OUT = Path(".zeromodel-demo")
OUT.mkdir(exist_ok=True)

ROW_IDS = ["forest", "crowd", "traffic", "meadow"]
METRIC_IDS = ["people", "trees", "risk"]


def panel(values, step):
    return ScoreTable(
        values=values,
        row_ids=ROW_IDS,
        metric_ids=METRIC_IDS,
        metadata={"step": step},
    )


panels = [
    panel(
        [
            [0.20, 0.60, 0.10],
            [1.00, 0.10, 0.10],
            [0.10, 0.20, 0.30],
            [0.00, 0.10, 0.10],
        ],
        step=0,
    ),
    panel(
        [
            [0.18, 0.62, 0.10],
            [0.96, 0.10, 0.12],
            [0.12, 0.20, 0.28],
            [0.02, 0.12, 0.10],
        ],
        step=1,
    ),
    panel(
        [
            [0.15, 0.55, 0.14],
            [0.25, 0.10, 0.30],
            [0.10, 0.18, 1.00],
            [0.00, 0.10, 0.12],
        ],
        step=2,
    ),
    panel(
        [
            [0.12, 0.52, 0.15],
            [0.20, 0.10, 0.28],
            [0.12, 0.16, 0.95],
            [0.00, 0.10, 0.10],
        ],
        step=3,
    ),
]

summary = build_decision_manifold(
    panels,
    optimizer=SpatialOptimizer(Kc=1, Kr=1, alpha=0.95, max_iters=30),
    name="scene-risk-shift",
    inflection_top_k=1,
)

for frame in summary.frames:
    write_png(frame.artifact, OUT / f"decision_manifold_frame_{frame.frame_index}.png")
    write_svg(frame.artifact, OUT / f"decision_manifold_frame_{frame.frame_index}.svg")

summary_path = OUT / "decision_manifold_summary.json"
summary_path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")

print(json.dumps({
    "frame_count": len(summary.frames),
    "inflection_indices": list(summary.inflection_indices),
    "mass_series": list(summary.mass_series),
    "curvature_series": list(summary.curvature_series),
    "summary": str(summary_path),
}, indent=2))
