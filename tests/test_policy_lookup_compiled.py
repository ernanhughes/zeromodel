from __future__ import annotations

import pytest

from zeromodel import (
    LayoutRecipe,
    ScoreTable,
    VPMArtifact,
    VPMPolicyLookup,
    build_vpm,
    with_q_diagnostics,
)


def _artifact(values, row_ids, metric_ids, *, column_order=None):
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "compiled-policy-test",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": (
                {"kind": "source"}
                if column_order is None
                else {"kind": "explicit", "metric_ids": column_order}
            ),
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(
        ScoreTable(values=values, row_ids=row_ids, metric_ids=metric_ids),
        recipe,
    )


def test_choose_and_read_do_not_resolve_candidate_cells(monkeypatch) -> None:
    artifact = _artifact(
        [[1.0, 0.0], [0.0, 1.0]],
        ["state:left", "state:right"],
        ["LEFT", "RIGHT"],
    )
    reader = VPMPolicyLookup(artifact)

    def fail_cell(*args, **kwargs):
        raise AssertionError("compiled lookup must not call artifact.cell")

    monkeypatch.setattr(VPMArtifact, "cell", fail_cell)

    assert reader.choose("state:left") == "LEFT"
    assert reader.choose_many(["state:left", "state:right"]) == (
        "LEFT",
        "RIGHT",
    )
    decision = reader.read("state:right")
    assert decision.action == "RIGHT"
    assert decision.source_row_index == 1
    assert decision.source_metric_index == 1
    assert decision.view_row == 1
    assert decision.view_column == 1
    assert decision.candidates == {"LEFT": 0.0, "RIGHT": 1.0}


def test_normalized_lookup_compiles_rendered_view_values() -> None:
    artifact = _artifact(
        [[10.0, 100.0], [0.0, 99.0]],
        ["state:top", "state:bottom"],
        ["A", "B"],
    )

    raw = VPMPolicyLookup(artifact, value_source="raw")
    normalized = VPMPolicyLookup(artifact, value_source="normalized")

    assert raw.choose("state:top") == "B"
    assert normalized.choose("state:top") == "A"
    assert normalized.read("state:top").candidates == {
        "A": pytest.approx(1.0),
        "B": pytest.approx(1.0),
    }


def test_compiled_lookup_preserves_metric_id_tie_break() -> None:
    artifact = _artifact(
        [[1.0, 1.0]],
        ["state:tied"],
        ["B", "A"],
    )

    assert VPMPolicyLookup(
        artifact,
        tie_break="metric_order",
    ).choose("state:tied") == "B"
    assert VPMPolicyLookup(
        artifact,
        tie_break="metric_id",
    ).choose("state:tied") == "A"


def test_compiled_plan_retains_artifact_and_view_coordinates() -> None:
    artifact = _artifact(
        [[0.2, 0.8], [0.9, 0.1]],
        ["state:first", "state:second"],
        ["A", "B"],
        column_order=["B", "A"],
    )
    reader = VPMPolicyLookup(artifact)

    plan = reader.to_compiled_plan()

    assert plan["artifact_id"] == artifact.artifact_id
    assert plan["action_metric_ids"] == ["A", "B"]
    assert plan["winner_indices"] == [1, 0]
    assert plan["action_source_metric_indices"] == [0, 1]
    assert plan["action_view_columns"] == [1, 0]
    assert reader.read("state:first").view_column == 0


def test_compiled_reader_keeps_diagnostics_out_of_argmax() -> None:
    table = with_q_diagnostics(
        ScoreTable(
            values=[[2.0, 1.0], [0.0, 3.0]],
            row_ids=["side=left", "side=right"],
            metric_ids=["LEFT", "RIGHT"],
        ),
        action_metric_ids=["LEFT", "RIGHT"],
    )
    artifact = build_vpm(
        table,
        LayoutRecipe.from_dict(
            {
                "version": "vpm-layout/0",
                "name": "diagnostic-policy",
                "row_order": {"kind": "source", "tie_break": "row_id"},
                "column_order": {"kind": "source"},
                "normalization": {
                    "kind": "per_metric_minmax",
                    "clip": True,
                },
            }
        ),
    )
    reader = VPMPolicyLookup(artifact)

    assert reader.action_metric_ids == ("LEFT", "RIGHT")
    assert reader.evidence_metric_ids == ("criticality", "decision_margin")
    assert reader.choose("side=right") == "RIGHT"
    assert set(reader.read("side=right").evidence) == {
        "criticality",
        "decision_margin",
    }
