from __future__ import annotations

from zeromodel.core import DecisionAdjudicationOutcome
from zeromodel.video.domains.video_action_set.provider_evaluation_case_dto import (
    CASE_OUTCOME_ACTION_EQUIVALENT,
    CASE_OUTCOME_REJECTED,
    ProviderEvaluationCaseContext,
    ProviderEvaluationCaseDTO,
    ProviderResponseEvidence,
)


POLICY_ID = "a" * 64
PROVIDER_ID = "b" * 64


def _decision(*, row_id: str, action: str) -> dict[str, object]:
    return {
        "artifact_id": POLICY_ID,
        "row_id": row_id,
        "action": action,
        "metric_id": action,
        "value": 1.0,
        "source_row_index": 0,
        "source_metric_index": 0,
        "view_row": 0,
        "view_column": 0,
        "candidates": {action: 1.0},
        "evidence": {},
    }


def test_provider_case_projects_core_action_equivalent_adjudication() -> None:
    case = ProviderEvaluationCaseDTO.build(
        case_ordinal=0,
        frame_id="frame-0",
        context=ProviderEvaluationCaseContext(
            policy_artifact_id=POLICY_ID,
            provider_configuration_id=PROVIDER_ID,
        ),
        expected_state={"tank": 3, "target": 3, "cooldown": 0},
        expected_decision=_decision(
            row_id="tank=3|target=3|cooldown=0",
            action="FIRE",
        ),
        accepted=True,
        predicted_state={"tank": 2, "target": 3, "cooldown": 0},
        predicted_decision=_decision(
            row_id="tank=2|target=3|cooldown=0",
            action="FIRE",
        ),
    )

    assert case.outcome == CASE_OUTCOME_ACTION_EQUIVALENT
    assert case.adjudication.outcome is DecisionAdjudicationOutcome.ACTION_EQUIVALENT
    assert case.adjudication.state_match is False
    assert case.adjudication.action_match is True
    assert case.exact_state_match is case.adjudication.state_match
    assert case.action_match is case.adjudication.action_match


def test_provider_rejection_projects_core_rejected_adjudication() -> None:
    case = ProviderEvaluationCaseDTO.build(
        case_ordinal=1,
        frame_id="frame-1",
        context=ProviderEvaluationCaseContext(
            policy_artifact_id=POLICY_ID,
            provider_configuration_id=PROVIDER_ID,
        ),
        expected_state={"tank": 3, "target": 3, "cooldown": 0},
        expected_decision=_decision(
            row_id="tank=3|target=3|cooldown=0",
            action="FIRE",
        ),
        accepted=False,
        evidence=ProviderResponseEvidence(rejection_reason="ambiguous observation"),
    )

    assert case.outcome == CASE_OUTCOME_REJECTED
    assert case.adjudication.outcome is DecisionAdjudicationOutcome.REJECTED
    assert case.adjudication.rejected is True
    assert case.exact_state_match is False
    assert case.action_match is False
