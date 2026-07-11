from __future__ import annotations

import pytest

from zeromodel import ScoreTable, SpatialOptimizer, build_optimized_view, optimize_view_profile
from zeromodel.artifact import VPMValidationError


def _source_table() -> ScoreTable:
    return ScoreTable(
        values=[
            [0.10, 0.50, 0.20],
            [0.95, 0.50, 0.25],
            [0.90, 0.50, 0.15],
            [0.05, 0.50, 0.20],
        ],
        row_ids=["background", "target_a", "target_b", "flat"],
        metric_ids=["target", "constant", "weak"],
    )


def test_spatial_optimizer_emits_view_profile_that_improves_mass() -> None:
    source = _source_table()
    optimizer = SpatialOptimizer(Kc=2, Kr=2, alpha=0.95, max_iters=30)

    result = optimize_view_profile(source, name="optimized-target", optimizer=optimizer)

    assert result.profile.name == "optimized-target"
    assert result.optimized_mass >= result.baseline_mass
    assert result.metric_weights["target"] > result.metric_weights["constant"]
    assert result.canonical_metric_ids[0] == "target"


def test_spatial_optimizer_builds_view_with_same_source_digest() -> None:
    source = _source_table()
    optimizer = SpatialOptimizer(Kc=2, Kr=2, alpha=0.95, max_iters=30)

    view = build_optimized_view(source, name="optimized-target", optimizer=optimizer)

    assert view.source.digest == source.digest
    assert view.cell(0, 0).row_id == "target_a"
    assert view.cell(0, 0).metric_id == "target"
    assert view.provenance["kind"] == "spatial_optimized_view"


def test_spatial_optimizer_accepts_table_series() -> None:
    first = _source_table()
    second = ScoreTable(
        values=[
            [0.12, 0.50, 0.22],
            [0.93, 0.50, 0.24],
            [0.88, 0.50, 0.18],
            [0.08, 0.50, 0.21],
        ],
        row_ids=first.row_ids,
        metric_ids=first.metric_ids,
    )

    result = optimize_view_profile([first, second], optimizer=SpatialOptimizer(Kc=2, Kr=2, max_iters=20))

    assert result.metric_weights["target"] > 0.0
    assert result.optimized_mass >= result.baseline_mass


def test_spatial_optimizer_rejects_inconsistent_metric_ids() -> None:
    first = _source_table()
    second = ScoreTable(
        values=[[0.1, 0.2]],
        row_ids=["row"],
        metric_ids=["other", "target"],
    )

    with pytest.raises(VPMValidationError, match="identical metric_ids"):
        optimize_view_profile([first, second])


def test_spatial_optimizer_validates_parameters() -> None:
    with pytest.raises(VPMValidationError, match="alpha"):
        SpatialOptimizer(alpha=1.0)
