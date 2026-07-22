from __future__ import annotations

from zeromodel.core.artifact import (
    LayoutRecipe,
    ScoreTable,
    build_vpm,
)
from zeromodel.analysis.phos import guarded_pack_artifact, pack_artifact


def _artifact():
    table = ScoreTable(
        values=[
            [0.0, 0.2, 0.1],
            [1.0, 0.4, 0.3],
            [0.8, 0.7, 0.9],
        ],
        row_ids=["a", "b", "c"],
        metric_ids=["m1", "m2", "m3"],
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "source",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe)


def test_guarded_pack_uses_first_ratio_guard_not_largest_window() -> None:
    artifact = _artifact()
    fractions = (0.09, 0.36)

    chosen = guarded_pack_artifact(
        artifact, top_left_fractions=fractions, min_improvement=0.0
    )
    first = pack_artifact(artifact, top_left_fraction=fractions[0])

    assert chosen.top_left_fraction == first.top_left_fraction
    assert chosen.improvement_ratio == first.improvement_ratio


def test_guarded_pack_fallback_marks_unimproved_candidate() -> None:
    artifact = _artifact()

    chosen = guarded_pack_artifact(
        artifact, top_left_fractions=(0.09, 0.36), min_improvement=100.0
    )

    assert chosen.improved is False
