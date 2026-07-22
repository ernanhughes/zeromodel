from __future__ import annotations

import pytest

from zeromodel import LayoutRecipe, ScoreTable, SignReader, VPMPolicyLookup, build_vpm
from zeromodel.core.artifact import VPMValidationError


def _policy_artifact():
    table = ScoreTable(
        values=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.5, 1.0],
        ],
        row_ids=["state:left", "state:right", "state:aligned"],
        metric_ids=["LEFT", "RIGHT", "STAY", "FIRE"],
        metadata={"kind": "toy_policy"},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "policy-source-order",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe, provenance={"kind": "policy_lookup_test"})


def test_policy_lookup_reads_best_action_and_cell_proof() -> None:
    artifact = _policy_artifact()
    reader = VPMPolicyLookup(artifact)

    decision = reader.read("state:aligned")

    assert decision.artifact_id == artifact.artifact_id
    assert decision.row_id == "state:aligned"
    assert decision.action == "FIRE"
    assert decision.metric_id == "FIRE"
    assert decision.value == pytest.approx(1.0)
    assert decision.source_row_index == 2
    assert decision.source_metric_index == 3
    assert decision.view_row == 2
    assert decision.view_column == 3
    assert decision.candidates == {"LEFT": 0.0, "RIGHT": 0.0, "STAY": 0.5, "FIRE": 1.0}


def test_sign_reader_alias_is_blog_vocabulary_not_a_second_implementation() -> None:
    artifact = _policy_artifact()

    assert SignReader is VPMPolicyLookup
    assert SignReader(artifact).read("state:left").action == "LEFT"


def test_policy_lookup_can_limit_action_columns() -> None:
    artifact = _policy_artifact()
    reader = VPMPolicyLookup(artifact, action_metric_ids=["LEFT", "RIGHT", "STAY"])

    decision = reader.read("state:aligned")

    assert decision.action == "STAY"
    assert sorted(decision.candidates) == ["LEFT", "RIGHT", "STAY"]


def test_policy_lookup_rejects_unknown_state_or_action() -> None:
    artifact = _policy_artifact()
    reader = VPMPolicyLookup(artifact)

    with pytest.raises(VPMValidationError, match="Unknown policy row_id"):
        reader.read("missing")

    with pytest.raises(VPMValidationError, match="Unknown action metric ids"):
        VPMPolicyLookup(artifact, action_metric_ids=["JUMP"])
