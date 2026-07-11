from __future__ import annotations

import numpy as np
import pytest

from zeromodel import (
    DecisionManifold,
    ScoreTable,
    SpatialOptimizer,
    VPMValidationError,
    build_decision_manifold,
    find_inflection_points,
)


def _panels():
    row_ids = ["forest", "crowd", "traffic", "meadow"]
    metric_ids = ["people", "trees", "risk"]
    return [
        ScoreTable(
            values=[
                [0.20, 0.60, 0.10],
                [1.00, 0.10, 0.10],
                [0.10, 0.20, 0.30],
                [0.00, 0.10, 0.10],
            ],
            row_ids=row_ids,
            metric_ids=metric_ids,
        ),
        ScoreTable(
            values=[
                [0.18, 0.62, 0.10],
                [0.96, 0.10, 0.12],
                [0.12, 0.20, 0.28],
                [0.02, 0.12, 0.10],
            ],
            row_ids=row_ids,
            metric_ids=metric_ids,
        ),
        ScoreTable(
            values=[
                [0.15, 0.55, 0.14],
                [0.25, 0.10, 0.30],
                [0.10, 0.18, 1.00],
                [0.00, 0.10, 0.12],
            ],
            row_ids=row_ids,
            metric_ids=metric_ids,
        ),
        ScoreTable(
            values=[
                [0.12, 0.52, 0.15],
                [0.20, 0.10, 0.28],
                [0.12, 0.16, 0.95],
                [0.00, 0.10, 0.10],
            ],
            row_ids=row_ids,
            metric_ids=metric_ids,
        ),
    ]


def _dominant_metric(frame):
    return max(frame.metric_weights.items(), key=lambda item: item[1])[0]


def test_decision_manifold_tracks_temporal_view_shift():
    optimizer = SpatialOptimizer(Kc=1, Kr=1, alpha=0.95, max_iters=30)
    summary = build_decision_manifold(
        _panels(),
        optimizer=optimizer,
        name="scene-shift",
        inflection_top_k=1,
    )

    assert len(summary.frames) == 4
    assert len(summary.transitions) == 3
    assert summary.inflection_indices == (2,)

    assert _dominant_metric(summary.frames[0]) == "people"
    assert _dominant_metric(summary.frames[1]) == "people"
    assert _dominant_metric(summary.frames[2]) == "risk"
    assert _dominant_metric(summary.frames[3]) == "risk"

    assert summary.frames[0].artifact.cell(0, 0).row_id == "crowd"
    assert summary.frames[2].artifact.cell(0, 0).row_id == "traffic"


def test_decision_manifold_metric_graph_and_serialization():
    summary = DecisionManifold(
        SpatialOptimizer(Kc=1, Kr=1, max_iters=20),
        inflection_top_k=2,
    ).build(_panels(), name="graph")

    assert summary.metric_ids == ("people", "trees", "risk")
    assert summary.metric_graph.shape == (3, 3)
    assert np.isfinite(summary.metric_graph).all()

    payload = summary.to_dict()
    assert payload["inflection_indices"] == list(summary.inflection_indices)
    assert len(payload["frames"]) == 4
    assert len(payload["transitions"]) == 3
    assert payload["metric_graph"][0][0] >= 0.0


def test_find_inflection_points_supports_threshold_and_top_k():
    summary = build_decision_manifold(
        _panels(),
        optimizer=SpatialOptimizer(Kc=1, Kr=1, max_iters=20),
        inflection_top_k=3,
    )
    selected = find_inflection_points(summary.transitions, top_k=1)
    assert selected == (2,)

    threshold_selected = find_inflection_points(summary.transitions, threshold=summary.transitions[1].curvature)
    assert 2 in threshold_selected


def test_decision_manifold_rejects_inconsistent_panels():
    panels = _panels()
    bad = ScoreTable(
        values=[[0.1, 0.2]],
        row_ids=["other"],
        metric_ids=["people", "trees"],
    )
    with pytest.raises(VPMValidationError):
        build_decision_manifold([panels[0], bad])
