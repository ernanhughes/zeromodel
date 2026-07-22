from __future__ import annotations

import numpy as np

from zeromodel.analysis.compare import compare_fields
from zeromodel.analysis.compose import (
    vpm_and,
    vpm_or,
    vpm_xor,
)
from zeromodel.analysis.controller import (
    Thresholds,
    VPMController,
)
from zeromodel.analysis.edge import TopLeftGate
from zeromodel.analysis.hierarchy import build_pyramid
from zeromodel.analysis.phos import guarded_pack_artifact
from zeromodel.core.artifact import (
    LayoutRecipe,
    ScoreTable,
    build_vpm,
)
from zeromodel.analysis.controller import Signal


def recipe() -> LayoutRecipe:
    return LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "overall-first",
            "row_order": {
                "kind": "lexicographic",
                "keys": [{"metric_id": "overall", "direction": "desc"}],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )


def artifact():
    table = ScoreTable(
        values=[[0.9, 0.2, 0.8], [0.4, 0.7, 0.3], [0.7, 0.1, 0.9]],
        row_ids=["a", "b", "c"],
        metric_ids=["overall", "coverage", "faithfulness"],
    )
    return build_vpm(table, recipe())


def test_phos_guarded_pack_measures_concentration() -> None:
    result = guarded_pack_artifact(artifact())
    assert result.packed.shape[0] == result.packed.shape[1]
    assert result.packed_concentration >= result.raw_concentration


def test_visual_logic_and_diff_are_shape_checked() -> None:
    a = np.array([[1.0, 0.0], [0.5, 0.2]])
    b = np.array([[0.2, 0.7], [0.5, 0.1]])
    assert np.allclose(vpm_and(a, b), np.minimum(a, b))
    assert np.allclose(vpm_or(a, b), np.maximum(a, b))
    assert np.allclose(vpm_xor(a, b), np.abs(a - b))
    comparison = compare_fields(a, b)
    assert comparison.gain > 0
    assert comparison.loss > 0
    assert 0 <= comparison.improvement_ratio <= 1


def test_hierarchy_builds_reduced_levels() -> None:
    levels = build_pyramid(np.ones((5, 3)), max_levels=3)
    assert [level.level for level in levels] == [0, 1, 2]
    assert levels[1].field.shape == (3, 2)


def test_edge_gate_evaluates_without_model() -> None:
    gate = TopLeftGate(threshold=0.2, rows=1, columns=1)
    result = gate.evaluate(artifact())
    assert result.accepted is True
    assert result.rows == 1
    assert result.columns == 1


def test_controller_emits_spinoff_signal() -> None:
    controller = VPMController(
        thresholds_code=Thresholds({"tests_pass_rate": 1.0}),
        thresholds_text=Thresholds({"coverage": 0.5}),
    )
    decision = controller.add_vpm_row(
        {"coverage": 0.7, "novelty": 0.9, "stickiness": 0.2},
        unit="doc:section",
    )
    assert decision.signal is Signal.SPINOFF
