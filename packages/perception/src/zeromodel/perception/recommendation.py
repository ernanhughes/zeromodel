"""Evidence-owned, non-mutating operational recommendations for Stage P17.

P17 converts governed P16 health evidence, immutable lifecycle history, and P16F
compatibility contracts into an inspectable recommendation artifact. Recommendations
never mutate lifecycle state and never treat insufficient evidence as grounds for action.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .compatibility import (
    ModelCompatibilityContractDTO,
    RollbackCompatibilityAssessmentDTO,
    assess_rollback_compatibility,
)
from .health import OperationalHealthReportDTO
from .lifecycle import ModelLifecycleSnapshotDTO

OPERATIONAL_RECOMMENDATION_VERSION: Final = "perception-operational-recommendation/1"
OPERATIONAL_RECOMMENDATION_SEMANTICS: Final = (
    "non_mutating_response_from_health_lifecycle_and_compatibility_evidence"
)
OPERATIONAL_RECOMMENDATION_STATUSES: Final = {
    "insufficient_evidence",
    "no_action",
    "investigate",
    "rollback_candidate",
}


class PerceptionOperationalRecommendationError(ValueError):
    """Raised when recommendation evidence is incomplete or inconsistent."""


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
class OperationalRecommendationDTO:
    recommendation_id: str
    health_report_id: str
    lifecycle_snapshot_id: str
    active_pointer_id: str
    active_pointer_revision: int
    active_promoted_model_id: str
    current_contract_id: str
    status: str
    selected_target_promoted_model_id: str | None
    selected_assessment_id: str | None
    assessed_candidates: tuple[RollbackCompatibilityAssessmentDTO, ...]
    rationale: str
    semantics: str = OPERATIONAL_RECOMMENDATION_SEMANTICS
    version: str = OPERATIONAL_RECOMMENDATION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.recommendation_id,
                self.health_report_id,
                self.lifecycle_snapshot_id,
                self.active_pointer_id,
                self.active_promoted_model_id,
                self.current_contract_id,
                self.rationale,
            )
        ):
            raise PerceptionOperationalRecommendationError(
                "recommendation identities and rationale must be non-empty"
            )
        if self.active_pointer_revision <= 0:
            raise PerceptionOperationalRecommendationError(
                "recommendation requires an active non-zero pointer revision"
            )
        if self.status not in OPERATIONAL_RECOMMENDATION_STATUSES:
            raise PerceptionOperationalRecommendationError(
                "unsupported operational recommendation status"
            )
        selected = self.selected_target_promoted_model_id is not None
        if selected != (self.selected_assessment_id is not None):
            raise PerceptionOperationalRecommendationError(
                "selected target and assessment must appear together"
            )
        if (self.status == "rollback_candidate") != selected:
            raise PerceptionOperationalRecommendationError(
                "only rollback_candidate recommendations may select a target"
            )
        if self.assessed_candidates != tuple(
            sorted(self.assessed_candidates, key=lambda item: item.target_promoted_model_id)
        ):
            raise PerceptionOperationalRecommendationError(
                "candidate assessments must be sorted by target model identity"
            )
        if self.semantics != OPERATIONAL_RECOMMENDATION_SEMANTICS:
            raise PerceptionOperationalRecommendationError(
                "unsupported recommendation semantics"
            )
        if self.version != OPERATIONAL_RECOMMENDATION_VERSION:
            raise PerceptionOperationalRecommendationError(
                "unsupported recommendation version"
            )


def recommend_operational_response(
    health: OperationalHealthReportDTO,
    lifecycle: ModelLifecycleSnapshotDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    candidate_contracts: Mapping[str, ModelCompatibilityContractDTO],
) -> OperationalRecommendationDTO:
    """Produce one deterministic recommendation without changing lifecycle state."""

    active_id = lifecycle.active_pointer.active_promoted_model_id
    if active_id is None:
        raise PerceptionOperationalRecommendationError(
            "recommendation requires an active promoted model"
        )
    if health.promoted_model_id != active_id:
        raise PerceptionOperationalRecommendationError(
            "health report does not describe the active promoted model"
        )
    if current_contract.promoted_model_id != active_id:
        raise PerceptionOperationalRecommendationError(
            "current compatibility contract does not describe the active model"
        )

    assessments: list[RollbackCompatibilityAssessmentDTO] = []
    selected_target: str | None = None
    selected_assessment: str | None = None

    if health.overall_status == "insufficient_evidence" or any(
        item.status == "insufficient_evidence" for item in health.findings
    ):
        status = "insufficient_evidence"
        rationale = "health evidence is insufficient; operational action is withheld"
    elif health.overall_status == "healthy":
        status = "no_action"
        rationale = "all governed health findings are adequately supported and healthy"
    else:
        most_recent: dict[str, int] = {}
        for transition in lifecycle.transitions:
            target = transition.next_promoted_model_id
            if target is not None and target != active_id:
                most_recent[target] = transition.sequence_number

        ordered_candidates = tuple(
            model_id
            for model_id, _ in sorted(
                most_recent.items(), key=lambda item: (-item[1], item[0])
            )
        )
        for model_id in ordered_candidates:
            contract = candidate_contracts.get(model_id)
            if contract is None:
                continue
            assessment = assess_rollback_compatibility(current_contract, contract)
            assessments.append(assessment)
            if selected_target is None and assessment.status == "compatible":
                selected_target = model_id
                selected_assessment = assessment.assessment_id

        if selected_target is None:
            status = "investigate"
            rationale = (
                "drift is adequately supported, but no compatible previously active model "
                "is available for operator consideration"
            )
        else:
            status = "rollback_candidate"
            rationale = (
                "drift is adequately supported and the most recently active compatible "
                "historical model is identified for operator review"
            )

    ordered_assessments = tuple(
        sorted(assessments, key=lambda item: item.target_promoted_model_id)
    )
    payload: Mapping[str, object] = {
        "active_pointer_id": lifecycle.active_pointer.pointer_id,
        "active_pointer_revision": lifecycle.active_pointer.revision,
        "active_promoted_model_id": active_id,
        "assessed_candidate_ids": [item.assessment_id for item in ordered_assessments],
        "current_contract_id": current_contract.contract_id,
        "health_report_id": health.report_id,
        "lifecycle_snapshot_id": lifecycle.snapshot_id,
        "rationale": rationale,
        "selected_assessment_id": selected_assessment,
        "selected_target_promoted_model_id": selected_target,
        "semantics": OPERATIONAL_RECOMMENDATION_SEMANTICS,
        "status": status,
        "version": OPERATIONAL_RECOMMENDATION_VERSION,
    }
    return OperationalRecommendationDTO(
        recommendation_id=_digest(payload),
        health_report_id=health.report_id,
        lifecycle_snapshot_id=lifecycle.snapshot_id,
        active_pointer_id=lifecycle.active_pointer.pointer_id,
        active_pointer_revision=lifecycle.active_pointer.revision,
        active_promoted_model_id=active_id,
        current_contract_id=current_contract.contract_id,
        status=status,
        selected_target_promoted_model_id=selected_target,
        selected_assessment_id=selected_assessment,
        assessed_candidates=ordered_assessments,
        rationale=rationale,
    )
