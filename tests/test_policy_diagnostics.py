from __future__ import annotations

import pytest

from zeromodel.analysis.policy_diagnostics import with_q_diagnostics
from zeromodel.core.artifact import (
    LayoutRecipe,
    ScoreTable,
    VPMValidationError,
    build_vpm,
)
from zeromodel.core.policy_lookup import VPMPolicyLookup

ACTIONS = ("LEFT", "RIGHT", "STAY", "FIRE")


def _artifact():
    table = ScoreTable(
        values=[
            [5.0, 1.0, 0.0, -1.0],
            [1.0, 0.5, 0.25, 0.0],
        ],
        row_ids=["state=critical", "state=close"],
        metric_ids=ACTIONS,
        metadata={"kind": "q_policy"},
    )
    enriched = with_q_diagnostics(table, action_metric_ids=ACTIONS)
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "source",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return enriched, build_vpm(enriched, recipe)


def test_q_diagnostics_are_exact_and_do_not_become_actions() -> None:
    table, artifact = _artifact()

    assert table.metric_ids == ACTIONS + ("criticality", "decision_margin")
    assert table.values[0, -2:].tolist() == pytest.approx([6.0, 4.0])
    assert table.values[1, -2:].tolist() == pytest.approx([1.0, 0.5])

    # Diagnostic metadata lets the reader safely separate actions from evidence
    # without requiring the caller to repeat the metric lists.
    reader = VPMPolicyLookup(artifact)
    decision = reader.read("state=critical")

    assert reader.action_metric_ids == ACTIONS
    assert reader.evidence_metric_ids == ("criticality", "decision_margin")
    assert decision.action == "LEFT"
    assert decision.evidence == pytest.approx(
        {"criticality": 6.0, "decision_margin": 4.0}
    )
    assert set(decision.candidates) == set(ACTIONS)
    assert decision.to_dict()["evidence"]["criticality"] == pytest.approx(6.0)


def test_q_diagnostics_reject_conflicts_and_missing_actions() -> None:
    table = ScoreTable([[1, 0]], ["s"], ["A", "B"])

    with pytest.raises(VPMValidationError, match="at least two"):
        with_q_diagnostics(table, action_metric_ids=["A"])

    with pytest.raises(VPMValidationError, match="Unknown action"):
        with_q_diagnostics(table, action_metric_ids=["A", "C"])

    conflict = ScoreTable(
        [[1, 0, 0]],
        ["s"],
        ["A", "B", "criticality"],
    )
    with pytest.raises(VPMValidationError, match="already exist"):
        with_q_diagnostics(conflict, action_metric_ids=["A", "B"])
