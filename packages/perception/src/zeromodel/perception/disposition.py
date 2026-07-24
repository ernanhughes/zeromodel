"""Operator disposition and governed execution linkage for Stage P17B.

Recommendations remain non-mutating. An operator may approve or reject a recommendation,
but execution revalidates the recommendation, active pointer revision, selected target, and
compatibility contracts against the current lifecycle state before any rollback occurs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .compatibility import (
    ModelCompatibilityContractDTO,
    RollbackCompatibilityAssessmentDTO,
    rollback_compatible_model,
)
from .lifecycle import (
    ActiveModelPointerDTO,
    ModelLifecycleTransitionDTO,
    PerceptionModelLifecycleStore,
)
from .recommendation import OperationalRecommendationDTO

OPERATIONAL_DISPOSITION_VERSION: Final = "perception-operational-disposition/1"
OPERATIONAL_DISPOSITION_SEMANTICS: Final = (
    "explicit_operator_review_of_non_mutating_operational_recommendation"
)
OPERATIONAL_DISPOSITION_STATUSES: Final = {"approved", "rejected"}


class PerceptionOperationalDispositionError(ValueError):
    """Raised when recommendation review or execution contracts are violated."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


@dataclass(frozen=True)
class OperationalRecommendationDispositionDTO:
    disposition_id: str
    recommendation_id: str
    recommendation_status: str
    active_pointer_id: str
    active_pointer_revision: int
    active_promoted_model_id: str
    selected_target_promoted_model_id: str | None
    selected_assessment_id: str | None
    status: str
    reviewed_by: str
    reason: str
    semantics: str = OPERATIONAL_DISPOSITION_SEMANTICS
    version: str = OPERATIONAL_DISPOSITION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.disposition_id,
                self.recommendation_id,
                self.recommendation_status,
                self.active_pointer_id,
                self.active_promoted_model_id,
                self.reviewed_by,
                self.reason,
            )
        ):
            raise PerceptionOperationalDispositionError(
                "disposition identities, reviewer, and reason must be non-empty"
            )
        if self.active_pointer_revision <= 0:
            raise PerceptionOperationalDispositionError(
                "disposition requires a non-zero pointer revision"
            )
        if self.status not in OPERATIONAL_DISPOSITION_STATUSES:
            raise PerceptionOperationalDispositionError("unsupported disposition status")
        selected = self.selected_target_promoted_model_id is not None
        if selected != (self.selected_assessment_id is not None):
            raise PerceptionOperationalDispositionError(
                "selected target and assessment must appear together"
            )
        if self.status == "approved" and self.recommendation_status != "rollback_candidate":
            raise PerceptionOperationalDispositionError(
                "only rollback-candidate recommendations may be approved"
            )
        if self.status == "approved" and not selected:
            raise PerceptionOperationalDispositionError(
                "approved rollback disposition requires a selected target"
            )
        if self.semantics != OPERATIONAL_DISPOSITION_SEMANTICS:
            raise PerceptionOperationalDispositionError("unsupported disposition semantics")
        if self.version != OPERATIONAL_DISPOSITION_VERSION:
            raise PerceptionOperationalDispositionError("unsupported disposition version")


def disposition_operational_recommendation(
    recommendation: OperationalRecommendationDTO,
    *,
    status: str,
    reviewed_by: str,
    reason: str,
) -> OperationalRecommendationDispositionDTO:
    """Record an explicit operator approval or rejection without mutating lifecycle state."""

    payload: Mapping[str, object] = {
        "active_pointer_id": recommendation.active_pointer_id,
        "active_pointer_revision": recommendation.active_pointer_revision,
        "active_promoted_model_id": recommendation.active_promoted_model_id,
        "reason": reason,
        "recommendation_id": recommendation.recommendation_id,
        "recommendation_status": recommendation.status,
        "reviewed_by": reviewed_by,
        "selected_assessment_id": recommendation.selected_assessment_id,
        "selected_target_promoted_model_id": recommendation.selected_target_promoted_model_id,
        "semantics": OPERATIONAL_DISPOSITION_SEMANTICS,
        "status": status,
        "version": OPERATIONAL_DISPOSITION_VERSION,
    }
    return OperationalRecommendationDispositionDTO(
        disposition_id=_digest(payload),
        recommendation_id=recommendation.recommendation_id,
        recommendation_status=recommendation.status,
        active_pointer_id=recommendation.active_pointer_id,
        active_pointer_revision=recommendation.active_pointer_revision,
        active_promoted_model_id=recommendation.active_promoted_model_id,
        selected_target_promoted_model_id=recommendation.selected_target_promoted_model_id,
        selected_assessment_id=recommendation.selected_assessment_id,
        status=status,
        reviewed_by=reviewed_by,
        reason=reason,
    )


def execute_approved_rollback(
    store: PerceptionModelLifecycleStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
) -> tuple[
    RollbackCompatibilityAssessmentDTO,
    ModelLifecycleTransitionDTO,
    ActiveModelPointerDTO,
]:
    """Execute an approved rollback only after revalidating all reviewed evidence."""

    if disposition.status != "approved":
        raise PerceptionOperationalDispositionError("rollback requires an approved disposition")
    if disposition.recommendation_id != recommendation.recommendation_id:
        raise PerceptionOperationalDispositionError(
            "disposition does not describe the supplied recommendation"
        )
    if recommendation.status != "rollback_candidate":
        raise PerceptionOperationalDispositionError(
            "only rollback-candidate recommendations may be executed"
        )
    pointer = store.get_active_pointer()
    if pointer.pointer_id != recommendation.active_pointer_id:
        raise PerceptionOperationalDispositionError(
            "active pointer identity changed after recommendation review"
        )
    if pointer.revision != recommendation.active_pointer_revision:
        raise PerceptionOperationalDispositionError(
            "active pointer revision changed after recommendation review"
        )
    if pointer.active_promoted_model_id != recommendation.active_promoted_model_id:
        raise PerceptionOperationalDispositionError(
            "active model changed after recommendation review"
        )
    target_id = recommendation.selected_target_promoted_model_id
    assessment_id = recommendation.selected_assessment_id
    if target_id is None or assessment_id is None:
        raise PerceptionOperationalDispositionError(
            "rollback recommendation lacks selected target evidence"
        )
    if current_contract.contract_id != recommendation.current_contract_id:
        raise PerceptionOperationalDispositionError(
            "current compatibility contract changed after recommendation review"
        )
    if target_contract.promoted_model_id != target_id:
        raise PerceptionOperationalDispositionError(
            "target compatibility contract does not describe recommended target"
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
        raise PerceptionOperationalDispositionError(
            "selected compatibility assessment is missing or no longer eligible"
        )
    return rollback_compatible_model(
        store,
        target_id,
        current_contract=current_contract,
        target_contract=target_contract,
        actor=disposition.reviewed_by,
        reason=disposition.reason,
    )
