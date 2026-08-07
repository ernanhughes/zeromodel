from __future__ import annotations

import pytest

from zeromodel.core import (
    DecisionAdjudication,
    DecisionAdjudicationOutcome,
    adjudicate_decision,
)


def test_exact_requires_state_and_action_match() -> None:
    result = adjudicate_decision(
        accepted=True,
        expected_state={"tank": 3, "target": 3},
        resolved_state={"tank": 3, "target": 3},
        expected_action="FIRE",
        selected_action="FIRE",
    )

    assert result.outcome is DecisionAdjudicationOutcome.EXACT
    assert result.state_match is True
    assert result.action_match is True
    assert result.action_correct is True
    assert result.exact is True


def test_action_equivalent_preserves_wrong_state_right_action_distinction() -> None:
    result = adjudicate_decision(
        accepted=True,
        expected_state={"tank": 3, "target": 3},
        resolved_state={"tank": 2, "target": 3},
        expected_action="LEFT",
        selected_action="LEFT",
    )

    assert result.outcome is DecisionAdjudicationOutcome.ACTION_EQUIVALENT
    assert result.state_match is False
    assert result.action_match is True
    assert result.action_correct is True
    assert result.action_equivalent is True


def test_action_changing_takes_precedence_over_exact_state() -> None:
    result = adjudicate_decision(
        accepted=True,
        expected_state="state-1",
        resolved_state="state-1",
        expected_action="FIRE",
        selected_action="STAY",
    )

    assert result.outcome is DecisionAdjudicationOutcome.ACTION_CHANGING
    assert result.state_match is True
    assert result.action_match is False
    assert result.action_correct is False
    assert result.action_changing is True


def test_rejected_is_mutually_exclusive_even_when_placeholders_match() -> None:
    result = adjudicate_decision(
        accepted=False,
        expected_state="state-1",
        resolved_state="state-1",
        expected_action="FIRE",
        selected_action="FIRE",
    )

    assert result.outcome is DecisionAdjudicationOutcome.REJECTED
    assert result.state_match is False
    assert result.action_match is False
    assert result.action_correct is False
    assert result.rejected is True


def test_accepted_missing_state_can_be_action_equivalent() -> None:
    result = adjudicate_decision(
        accepted=True,
        expected_state="state-1",
        resolved_state=None,
        expected_action="CONTINUE",
        selected_action="CONTINUE",
    )

    assert result.outcome is DecisionAdjudicationOutcome.ACTION_EQUIVALENT
    assert result.state_match is False
    assert result.action_match is True


def test_inconsistent_adjudication_cannot_be_constructed() -> None:
    with pytest.raises(ValueError, match="outcome does not match"):
        DecisionAdjudication(
            accepted=True,
            state_match=False,
            action_match=True,
            outcome=DecisionAdjudicationOutcome.EXACT,
        )


def test_outcome_values_are_stable_serialization_strings() -> None:
    assert DecisionAdjudicationOutcome.EXACT.value == "exact"
    assert (
        DecisionAdjudicationOutcome.ACTION_EQUIVALENT.value
        == "action_equivalent"
    )
    assert DecisionAdjudicationOutcome.ACTION_CHANGING.value == "action_changing"
    assert DecisionAdjudicationOutcome.REJECTED.value == "rejected"
