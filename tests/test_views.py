from __future__ import annotations

import pytest

from zeromodel.core.artifact import ScoreTable
from zeromodel.core.views import (
    ViewProfile,
    ViewSet,
    build_view,
    build_views,
)
from zeromodel.core.artifact import VPMValidationError


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


def test_view_profiles_reorganize_same_dense_table_by_policy_lens() -> None:
    table = _dense_scene_table()

    people_view = build_view(table, ViewProfile.from_metric("people", name="people"))
    tree_view = build_view(table, ViewProfile.from_metric("trees", name="trees"))
    car_view = build_view(table, ViewProfile.from_metric("cars", name="cars"))
    risk_view = build_view(table, ViewProfile.from_metric("risk", name="risk"))

    assert people_view.source.digest == table.digest
    assert tree_view.source.digest == table.digest
    assert car_view.source.digest == table.digest
    assert risk_view.source.digest == table.digest

    assert people_view.cell(0, 0).row_id == "crowd"
    assert people_view.cell(0, 0).metric_id == "people"

    assert tree_view.cell(0, 0).row_id == "forest"
    assert tree_view.cell(0, 0).metric_id == "trees"

    assert car_view.cell(0, 0).row_id == "traffic"
    assert car_view.cell(0, 0).metric_id == "cars"

    assert risk_view.cell(0, 0).row_id == "traffic"
    assert risk_view.cell(0, 0).metric_id == "risk"

    assert len({people_view.artifact_id, tree_view.artifact_id, car_view.artifact_id, risk_view.artifact_id}) == 4


def test_negative_view_weight_makes_low_values_salient() -> None:
    table = ScoreTable(
        values=[[0.95], [0.20], [0.55]],
        row_ids=["slow", "fast", "medium"],
        metric_ids=["latency"],
    )

    view = build_view(table, ViewProfile("low-latency", {"latency": -1.0}))

    assert view.cell(0, 0).row_id == "fast"
    assert view.cell(0, 0).metric_id == "latency"


def test_include_unweighted_columns_false_records_hidden_columns() -> None:
    table = _dense_scene_table()
    profile = ViewProfile(
        "people-only",
        {"people": 1.0},
        include_unweighted_columns=False,
    )

    view = build_view(table, profile)

    assert view.recipe.data["view_profile"]["visible_metric_ids"] == ("people",)
    assert view.recipe.data["view_profile"]["hidden_metric_ids"] == ("trees", "cars", "grass", "risk")
    assert view.provenance["visible_metric_ids"] == ("people",)
    assert view.provenance["hidden_metric_ids"] == ("trees", "cars", "grass", "risk")
    # Source mapping remains complete and deterministic even when a renderer crops.
    assert view.column_order == (0, 1, 2, 3, 4)


def test_build_view_from_existing_artifact_preserves_parent_provenance() -> None:
    table = _dense_scene_table()
    base = build_view(table, ViewProfile.from_metric("risk", name="risk"))

    people = build_view(base, ViewProfile.from_metric("people", name="people"))

    assert people.source.digest == base.source.digest
    assert people.provenance["parents"] == (base.artifact_id,)
    assert people.provenance["view_profile"]["name"] == "people"


def test_build_views_returns_named_view_artifacts() -> None:
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
    assert views["people"].cell(0, 0).row_id == "crowd"
    assert views["trees"].cell(0, 0).row_id == "forest"
    assert views["risk"].cell(0, 0).row_id == "traffic"


def test_view_profile_rejects_unknown_metric() -> None:
    table = _dense_scene_table()
    with pytest.raises(VPMValidationError, match="unknown metrics"):
        build_view(table, ViewProfile.from_metric("missing", name="missing"))


def test_view_set_rejects_duplicate_profile_names() -> None:
    with pytest.raises(VPMValidationError, match="Duplicate ViewProfile names"):
        ViewSet([
            ViewProfile.from_metric("people", name="same"),
            ViewProfile.from_metric("trees", name="same"),
        ])
