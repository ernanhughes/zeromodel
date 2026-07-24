"""Crash-recoverable governed rollback execution for Stage P17D.

The lifecycle store remains the sole authority for active-model state. The governance ledger
records the reviewed recommendation, disposition, and resulting execution receipt. This
module closes the process-crash gap between lifecycle mutation and receipt persistence by
reconciling an exact already-committed rollback transition after restart.
"""

from __future__ import annotations

from typing import Final

from .compatibility import ModelCompatibilityContractDTO, RollbackCompatibilityAssessmentDTO
from .disposition import (
    OperationalRecommendationDispositionDTO,
    PerceptionOperationalDispositionError,
    execute_approved_rollback,
)
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


def _persisted_chain_matches(
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
) -> None:
    persisted_recommendation = governance_store.get_recommendation(
        recommendation.recommendation_id
    )
    if persisted_recommendation != recommendation:
        raise PerceptionGovernedExecutionError(
            "execution requires the exact persisted recommendation"
        )
    persisted_disposition = governance_store.get_disposition(disposition.disposition_id)
    if persisted_disposition != disposition:
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
    disposition: OperationalRecommendationDispositionDTO,
) -> GovernanceExecutionReceiptDTO | None:
    matches = tuple(
        item
        for item in governance_store.list_execution_receipts()
        if item.disposition_id == disposition.disposition_id
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
    if pointer.pointer_id != recommendation.active_pointer_id:
        raise PerceptionGovernedExecutionError(
            "active pointer identity differs from reviewed execution state"
        )
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
    if not all(
        (
            transition.transition_kind == "rollback",
            transition.sequence_number == expected_revision,
            transition.previous_promoted_model_id
            == recommendation.active_promoted_model_id,
            transition.next_promoted_model_id == target_id,
            transition.actor == disposition.reviewed_by,
            transition.reason == disposition.reason,
        )
    ):
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
    """Execute once, or recover the missing receipt for one exact committed rollback.

    The recommendation and disposition must already be persisted. Repeated calls are
    idempotent after a receipt exists. If lifecycle mutation committed before a process crash,
    the exact next pointer revision and rollback transition are reconciled into the receipt.
    Any other lifecycle movement is treated as stale or conflicting evidence.
    """

    _persisted_chain_matches(governance_store, recommendation, disposition)
    assessment = _selected_assessment(recommendation, disposition)
    existing = _existing_receipt(governance_store, disposition)
    if existing is not None:
        return existing

    pointer = lifecycle_store.get_active_pointer()
    if (
        pointer.pointer_id == recommendation.active_pointer_id
        and pointer.revision == recommendation.active_pointer_revision
        and pointer.active_promoted_model_id
        == recommendation.active_promoted_model_id
    ):
        try:
            executed_assessment, transition, resulting_pointer = execute_approved_rollback(
                lifecycle_store,
                recommendation,
                disposition,
                current_contract=current_contract,
                target_contract=target_contract,
            )
        except (PerceptionOperationalDispositionError, ValueError) as error:
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
