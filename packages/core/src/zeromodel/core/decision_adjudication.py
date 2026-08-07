"""Canonical decision adjudication for bounded ZeroModel decisions.

Decision adjudication separates two questions that ordinary action accuracy can
collapse:

* did the system resolve the expected state; and
* did the selected action match the expected action?

The resulting four-way taxonomy is intentionally small and domain-neutral. It
is not a lifecycle state machine and it does not decide whether an artifact or
provider should be promoted. Higher-level packages may consume adjudication
results as evidence for those decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class DecisionAdjudicationOutcome(str, Enum):
    """Mutually exclusive outcome for one bounded decision attempt."""

    EXACT = "exact"
    ACTION_EQUIVALENT = "action_equivalent"
    ACTION_CHANGING = "action_changing"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class DecisionAdjudication:
    """Derived state/action correctness for one decision attempt.

    ``accepted`` records whether the decision procedure returned an admissible
    result. For accepted decisions, ``state_match`` and ``action_match`` are
    evaluated independently and then collapsed into one mutually exclusive
    ``outcome``:

    * ``EXACT``: state and action both match;
    * ``ACTION_EQUIVALENT``: state differs but the action still matches;
    * ``ACTION_CHANGING``: the selected action differs from the expected one;
    * ``REJECTED``: no decision was accepted.

    The distinction is epistemic as well as behavioural: an
    ``ACTION_EQUIVALENT`` result is action-correct without being exact-state
    correct.
    """

    accepted: bool
    state_match: bool
    action_match: bool
    outcome: DecisionAdjudicationOutcome

    def __post_init__(self) -> None:
        if not isinstance(self.accepted, bool):
            raise TypeError("accepted must be bool")
        if not isinstance(self.state_match, bool):
            raise TypeError("state_match must be bool")
        if not isinstance(self.action_match, bool):
            raise TypeError("action_match must be bool")
        if not isinstance(self.outcome, DecisionAdjudicationOutcome):
            raise TypeError("outcome must be DecisionAdjudicationOutcome")

        expected = _outcome_from_flags(
            accepted=self.accepted,
            state_match=self.state_match,
            action_match=self.action_match,
        )
        if self.outcome is not expected:
            raise ValueError("decision adjudication outcome does not match its flags")

    @property
    def action_correct(self) -> bool:
        """Whether an accepted decision selected the expected action."""

        return self.accepted and self.action_match

    @property
    def exact(self) -> bool:
        """Whether the accepted decision matched both state and action."""

        return self.outcome is DecisionAdjudicationOutcome.EXACT

    @property
    def action_equivalent(self) -> bool:
        """Whether a non-exact accepted state produced the expected action."""

        return self.outcome is DecisionAdjudicationOutcome.ACTION_EQUIVALENT

    @property
    def action_changing(self) -> bool:
        """Whether an accepted decision changed the expected action."""

        return self.outcome is DecisionAdjudicationOutcome.ACTION_CHANGING

    @property
    def rejected(self) -> bool:
        """Whether the decision procedure refused or failed to return a result."""

        return self.outcome is DecisionAdjudicationOutcome.REJECTED


def adjudicate_decision(
    *,
    accepted: bool,
    expected_state: object,
    resolved_state: object,
    expected_action: object,
    selected_action: object,
) -> DecisionAdjudication:
    """Classify one bounded decision without imposing domain-specific DTOs.

    Equality is intentionally supplied by the caller's values. Core does not
    interpret state structure, policy semantics, provider confidence, or
    rejection reasons; owning packages retain those richer records.

    A rejected attempt is always ``REJECTED`` and carries ``False`` for both
    correctness flags, even if placeholder values happen to compare equal.
    For an accepted attempt, action mismatch takes precedence over state match:
    an exact state paired with the wrong action is ``ACTION_CHANGING``.
    """

    if not isinstance(accepted, bool):
        raise TypeError("accepted must be bool")

    if not accepted:
        return DecisionAdjudication(
            accepted=False,
            state_match=False,
            action_match=False,
            outcome=DecisionAdjudicationOutcome.REJECTED,
        )

    state_match = resolved_state == expected_state
    action_match = selected_action == expected_action
    outcome = _outcome_from_flags(
        accepted=True,
        state_match=state_match,
        action_match=action_match,
    )
    return DecisionAdjudication(
        accepted=True,
        state_match=state_match,
        action_match=action_match,
        outcome=outcome,
    )


def _outcome_from_flags(
    *,
    accepted: bool,
    state_match: bool,
    action_match: bool,
) -> DecisionAdjudicationOutcome:
    if not accepted:
        return DecisionAdjudicationOutcome.REJECTED
    if not action_match:
        return DecisionAdjudicationOutcome.ACTION_CHANGING
    if state_match:
        return DecisionAdjudicationOutcome.EXACT
    return DecisionAdjudicationOutcome.ACTION_EQUIVALENT


__all__ = [
    "DecisionAdjudication",
    "DecisionAdjudicationOutcome",
    "adjudicate_decision",
]
