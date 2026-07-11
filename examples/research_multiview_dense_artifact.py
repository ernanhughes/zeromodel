from __future__ import annotations

import json
from pathlib import Path

from zeromodel import ScoreTable, ViewProfile, build_views, to_bundle, write_png, write_svg


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / ".zeromodel-demo" / "multiview_dense_artifact"


def _dense_scene_table() -> ScoreTable:
    return ScoreTable(
        values=[
            [0.10, 0.96, 0.05, 0.72, 0.20],
            [0.94, 0.12, 0.08, 0.18, 0.35],
            [0.24, 0.07, 0.97, 0.08, 0.78],
            [0.07, 0.18, 0.04, 0.98, 0.10],
        ],
        row_ids=["forest", "crowd", "traffic", "meadow"],
        metric_ids=["people", "trees", "cars", "grass", "risk"],
        metadata={
            "kind": "dense_scene_scores",
            "note": "One dense table; multiple policy views.",
        },
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table = _dense_scene_table()
    profiles = [
        ViewProfile.from_metric("people", name="people"),
        ViewProfile.from_metric("trees", name="trees"),
        ViewProfile.from_metric("cars", name="cars"),
        ViewProfile.from_metric("risk", name="risk"),
        ViewProfile(
            name="safe-open-space",
            metric_weights={"grass": 1.0, "risk": -0.8, "cars": -0.4},
            metadata={"description": "Prefer grass/open space while suppressing risk and cars."},
        ),
    ]

    views = build_views(table, profiles)
    summary = {
        "source_digest": table.digest,
        "views": {},
    }

    for name, artifact in views.items():
        view_dir = OUT_DIR / name
        view_dir.mkdir(parents=True, exist_ok=True)
        to_bundle(artifact, view_dir / f"{name}.vpm")
        write_png(artifact.normalized_values, view_dir / f"{name}.png")
        write_svg(artifact.normalized_values, view_dir / f"{name}.svg")
        top_cell = artifact.cell(0, 0)
        summary["views"][name] = {
            "artifact_id": artifact.artifact_id,
            "recipe": artifact.recipe.name,
            "top_left_row_id": top_cell.row_id,
            "top_left_metric_id": top_cell.metric_id,
            "top_left_raw_value": top_cell.raw_value,
            "source_digest": artifact.source.digest,
        }

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
