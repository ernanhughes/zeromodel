from __future__ import annotations

import zipfile

import numpy as np

from zeromodel.v2 import (
    LayoutRecipe,
    ScoreTable,
    TopLeftGate,
    Thresholds,
    VPMController,
    build_pyramid,
    build_vpm,
    compare_fields,
    from_bundle,
    guarded_pack_artifact,
    pack_metrics,
    png_bytes,
    score_table_from_metric_rows,
    to_bundle,
    vpm_and,
    vpm_or,
    vpm_xor,
)
from zeromodel.v2.controller import Signal


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


def test_metric_packing_accepts_stephanie_aliases() -> None:
    packed = pack_metrics(
        {
            "claim_coverage": 0.8,
            "hallucination_rate": 0.1,
            "figure_results": {"overall_figure_score": 0.6},
        },
        metric_ids=["coverage", "no_halluc", "figure_ground"],
    )
    assert packed == {"coverage": 0.8, "no_halluc": 0.9, "figure_ground": 0.6}


def test_metric_rows_build_score_table() -> None:
    table = score_table_from_metric_rows(
        [{"overall": 0.9, "claim_coverage": 0.7}, {"overall": 0.4, "coverage": 0.6}],
        row_ids=["first", "second"],
        metric_ids=["overall", "coverage"],
    )
    assert table.row_ids == ("first", "second")
    assert table.metric_ids == ("overall", "coverage")
    assert table.values.shape == (2, 2)


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


def test_bundle_roundtrip_preserves_artifact_identity(tmp_path) -> None:
    art = artifact()
    path = tmp_path / "artifact.vpm"
    to_bundle(art, path)
    assert zipfile.is_zipfile(path)
    loaded = from_bundle(path)
    assert loaded.artifact_id == art.artifact_id


def test_png_renderer_emits_png_signature() -> None:
    data = png_bytes(artifact())
    assert data.startswith(b"\x89PNG\r\n\x1a\n")


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
