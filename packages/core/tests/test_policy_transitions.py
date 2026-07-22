from __future__ import annotations

import pytest

from zeromodel.core import (
    ROW_UNION_TRANSITION_SCOPE,
    PolicyTransitionSpec,
    VPMValidationError,
)


def _spec() -> PolicyTransitionSpec:
    return PolicyTransitionSpec(
        allowed_row_transitions={
            "A": ("A", "B"),
            "B": ("B", "C"),
            "C": ("C",),
        },
        maximum_frame_gap=2,
    )


def test_transition_contract_distinguishes_possible_impossible_and_gap_unknown() -> (
    None
):
    spec = _spec()
    assert spec.classify(None, "A", frame_gap=1) == "initial"
    assert spec.classify("A", "B", frame_gap=1) == "possible"
    assert spec.classify("A", "C", frame_gap=1) == "impossible"
    assert spec.classify("A", "C", frame_gap=2) == "possible_with_gap"
    assert spec.classify("A", "C", frame_gap=3) == "unknown_due_to_gap"
    assert spec.classify("A", None, frame_gap=1) == "unknown_due_to_rejection"


def test_transition_identity_round_trip_and_validation() -> None:
    spec = _spec()
    loaded = PolicyTransitionSpec.from_dict(spec.to_dict())

    assert loaded.spec_id == spec.spec_id
    assert loaded.row_ids == ("A", "B", "C")
    assert loaded.transition_scope == ROW_UNION_TRANSITION_SCOPE

    with pytest.raises(VPMValidationError, match="unknown rows"):
        PolicyTransitionSpec(allowed_row_transitions={"A": ("B",)})
    with pytest.raises(
        VPMValidationError,
        match="action-conditioned transitions are not implemented",
    ):
        PolicyTransitionSpec.from_dict(
            {"allowed_row_transitions": {"A": ["A"]}, "action_conditioned": True}
        )
    with pytest.raises(VPMValidationError, match="unsupported transition_scope"):
        PolicyTransitionSpec(
            allowed_row_transitions={"A": ("A",)},
            transition_scope="action_conditioned",
        )
