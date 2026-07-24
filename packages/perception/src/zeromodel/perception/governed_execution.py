"""Crash-recoverable governed rollback execution for Stage P17D.

The lifecycle store remains the sole authority for active-model state. The governance ledger
records the reviewed recommendation, disposition, and resulting execution receipt. This
module closes the process-crash gap between lifecycle mutation and receipt persistence by
reconciling an exact already-committed rollback transition after restart.
"""

from __future__ import annotations

from typing import Final

from .compatibility import ModelCompatibilityContractDTO, RollbackCompatibilityAssessmentDTO
from .disposition import OperationalRecommendationDispositionDTO, execute_approved_rollback
from .lifecycle import (
    ActiveModelPointerDTO,
    ModelLifecycleTransitionDTO,
    PerceptionModelLifecycleStore,
)
from .recommendation import OperationalRecommendationDTO
from .sql_governance import (
    GovernanceExecutionReceiptDTO,
    PerceptionSqlGovernanceError,
    SqlitePerceptionGovernanceLedgerStore,
)

GOVERNED_EXECUTION_SEMANTICS: Final = (
    "persisted_approval_with_exact_lifecycle_execution_or_restart_reconciliation"
)


class PerceptionGovernedExecutionError(ValueError):
    """Raised when durable execution cannot be proven safe or recoverable."""


def _selected_assessment(
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
) -> RollbackCompatibilityAssessmentDTO:
    assessment_id = disposition.selected_assessment_id
    target_id = disposition.selected_target_promoted_model_id
    if assessment_id is None or target_id is None:
        raise PerceptionGovernedExecutionError(
            "approved disposition lacks selected rollback evidence"
        )
    selected = next(
        (
            item
            for item in recommendation.assessed_candidates
            if item.assessment_id == assessment_id
            and item.target_promoted_model_id == target_id
        ),
        None,
    )
    if selected is None or selected.status != "compatible":
        raise PerceptionGovernedExecutionError(
            "approved disposition does not identify a compatible assessment"
        )
    return selected


def _require_persisted_chain(
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
) -> None:
    if governance_store.get_recommendation(recommendation.recommendation_id) != recommendation:
        raise PerceptionGovernedExecutionError(
            "execution requires the exact persisted recommendation"
        )
    if governance_store.get_disposition(disposition.disposition_id) != disposition:
        raise PerceptionGovernedExecutionError(
            "execution requires the exact persisted disposition"
        )
    if disposition.recommendation_id != recommendation.recommendation_id:
        raise PerceptionGovernedExecutionError(
            "persisted disposition does not belong to recommendation"
        )
    if disposition.status != "approved":
        raise PerceptionGovernedExecutionError("execution requires persisted approval")


def _existing_receipt(
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    disposition_id: str,
) -> GovernanceExecutionReceiptDTO | None:
    matches = tuple(
        item
        for item in governance_store.list_execution_receipts()
        if item.disposition_id == disposition_id
    )
    if len(matches) > 1:
        raise PerceptionGovernedExecutionError(
            "governance ledger contains multiple receipts for one disposition"
        )
    return matches[0] if matches else None


def _reconciled_transition(
    lifecycle_store: PerceptionModelLifecycleStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    pointer: ActiveModelPointerDTO,
) -> ModelLifecycleTransitionDTO:
    target_id = disposition.selected_target_promoted_model_id
    if target_id is None:
        raise PerceptionGovernedExecutionError("approved disposition lacks rollback target")
    expected_revision = recommendation.active_pointer_revision + 1
    if pointer.revision != expected_revision:
        raise PerceptionGovernedExecutionError(
            "active pointer is neither reviewed pre-state nor exact rollback post-state"
        )
    if pointer.active_promoted_model_id != target_id or pointer.last_transition_id is None:
        raise PerceptionGovernedExecutionError(
            "active pointer does not contain the reviewed rollback result"
        )
    transition = next(
        (
            item
            for item in lifecycle_store.list_transitions()
            if item.transition_id == pointer.last_transition_id
        ),
        None,
    )
    if transition is None:
        raise PerceptionGovernedExecutionError(
            "rollback post-state references an unknown lifecycle transition"
        )
    exact_match = all(
        (
            transition.transition_kind == "rollback",
            transition.sequence_number == expected_revision,
            transition.previous_promoted_model_id
            == recommendation.active_promoted_model_id,
            transition.next_promoted_model_id == target_id,
            transition.actor == disposition.reviewed_by,
            transition.reason == disposition.reason,
        )
    )
    if not exact_match:
        raise PerceptionGovernedExecutionError(
            "post-state transition does not exactly match the approved rollback"
        )
    return transition


def execute_or_reconcile_approved_rollback(
    lifecycle_store: PerceptionModelLifecycleStore,
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
) -> GovernanceExecutionReceiptDTO:
    """Execute once, or recover a missing receipt for one exact committed rollback."""

    _require_persisted_chain(governance_store, recommendation, disposition)
    assessment = _selected_assessment(recommendation, disposition)
    existing = _existing_receipt(governance_store, disposition.disposition_id)
    if existing is not None:
        return existing

    pointer = lifecycle_store.get_active_pointer()
    reviewed_pre_state = all(
        (
            pointer.pointer_id == recommendation.active_pointer_id,
            pointer.revision == recommendation.active_pointer_revision,
            pointer.active_promoted_model_id
            == recommendation.active_promoted_model_id,
        )
    )
    if reviewed_pre_state:
        try:
            executed_assessment, transition, resulting_pointer = execute_approved_rollback(
                lifecycle_store,
                recommendation,
                disposition,
                current_contract=current_contract,
                target_contract=target_contract,
            )
        except ValueError as error:
            raise PerceptionGovernedExecutionError(str(error)) from error
        if executed_assessment.assessment_id != assessment.assessment_id:
            raise PerceptionGovernedExecutionError(
                "executed compatibility assessment differs from approved evidence"
            )
    else:
        transition = _reconciled_transition(
            lifecycle_store,
            recommendation,
            disposition,
            pointer,
        )
        resulting_pointer = pointer

    receipt = GovernanceExecutionReceiptDTO.from_execution(
        disposition,
        assessment,
        transition,
        resulting_pointer,
    )
    try:
        governance_store.append_execution_receipt(receipt)
    except PerceptionSqlGovernanceError as error:
        raise PerceptionGovernedExecutionError(str(error)) from error
    return receipt
