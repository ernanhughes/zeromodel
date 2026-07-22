from __future__ import annotations

import zipfile

import pytest

from zeromodel.core import (
    ScoreTable,
    VPMValidationError,
    ViewProfile,
    ViewSet,
    build_view,
    build_views,
)
from zeromodel.core import from_bundle, png_bytes, svg_text, to_bundle


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
        metadata={"kind": "dense_scene_scores"},
    )


def test_view_profiles_preserve_source_mapping_and_identity() -> None:
    table = _dense_scene_table()
    people = build_view(table, ViewProfile.from_metric("people", name="people"))
    trees = build_view(table, ViewProfile.from_metric("trees", name="trees"))
    risk = build_view(table, ViewProfile.from_metric("risk", name="risk"))

    assert people.source.digest == table.digest
    assert people.cell(0, 0).row_id == "crowd"
    assert trees.cell(0, 0).row_id == "forest"
    assert risk.cell(0, 0).row_id == "traffic"
    assert len({people.artifact_id, trees.artifact_id, risk.artifact_id}) == 3


def test_view_validation_and_parent_provenance() -> None:
    table = _dense_scene_table()
    profile = ViewProfile(
        "people-only", {"people": 1.0}, include_unweighted_columns=False
    )
    view = build_view(table, profile)

    assert view.recipe.data["view_profile"]["visible_metric_ids"] == ("people",)
    assert view.column_order == (0, 1, 2, 3, 4)

    child = build_view(view, ViewProfile.from_metric("risk", name="risk"))
    assert child.provenance["parents"] == (view.artifact_id,)

    with pytest.raises(VPMValidationError, match="unknown metrics"):
        build_view(table, ViewProfile.from_metric("missing", name="missing"))
    with pytest.raises(VPMValidationError, match="Duplicate ViewProfile names"):
        ViewSet(
            [
                ViewProfile.from_metric("people", name="same"),
                ViewProfile.from_metric("trees", name="same"),
            ]
        )


def test_build_views_bundle_and_rendering(tmp_path) -> None:
    table = _dense_scene_table()
    views = build_views(
        table,
        [
            ViewProfile.from_metric("people", name="people"),
            ViewProfile.from_metric("trees", name="trees"),
            ViewProfile.from_metric("risk", name="risk"),
        ],
    )
    assert sorted(views) == ["people", "risk", "trees"]

    path = tmp_path / "artifact.vpm"
    to_bundle(views["risk"], path)
    assert zipfile.is_zipfile(path)
    assert from_bundle(path).artifact_id == views["risk"].artifact_id

    assert png_bytes(views["risk"]).startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in svg_text(views["risk"])
