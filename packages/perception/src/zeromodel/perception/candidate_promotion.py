"""Governed promotion proposals for held-out validated candidates (Stage P18E).

P18E turns P18D-validated candidates into reviewable, content-addressed promotion
proposals. Human decisions are recorded separately. Even an approved decision is
explicitly not materialized: this stage never mutates production annotations,
relations, or transition expectations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final, Mapping

from .candidate_validation import (
    CandidateValidationReportDTO,
    CandidateValidationResultDTO,
)
from .transition_discovery import (
    RECURRENT_UNEXPLAINED_STATISTIC_KINDS,
    MissingComponentCandidateDTO,
    RecurrentUnexplainedStatisticDTO,
    TransitionDiscoveryReportDTO,
)

CANDIDATE_PROMOTION_PROPOSAL_VERSION: Final = (
    "perception-candidate-promotion-proposal/1"
)
CANDIDATE_PROMOTION_PROPOSAL_SET_VERSION: Final = (
    "perception-candidate-promotion-proposal-set/1"
)
CANDIDATE_PROMOTION_DECISION_VERSION: Final = (
    "perception-candidate-promotion-decision/1"
)
CANDIDATE_PROMOTION_REVIEW_VERSION: Final = "perception-candidate-promotion-review/1"
CANDIDATE_PROMOTION_SEMANTICS: Final = (
    "reviewable_authorization_without_automatic_semantic_or_production_materialization"
)
CANDIDATE_PROMOTION_PROPOSAL_STATUS: Final = "pending_review"
CANDIDATE_PROMOTION_MATERIALIZATION_STATUS: Final = "not_materialized"
CANDIDATE_PROMOTION_DECISIONS: Final = {
    "approved",
    "rejected",
    "deferred",
    "needs_semantic_annotation",
}
CANDIDATE_PROMOTION_REVIEW_STATUSES: Final = {
    "pending_review",
    "partially_reviewed",
    "review_complete",
}


class PerceptionCandidatePromotionError(ValueError):
    """Raised when governed promotion contracts are invalid."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    encoded = _canonical_json(payload)
    hasher = hashlib.sha256()
    hasher.update(len(encoded).to_bytes(8, "big"))
    hasher.update(encoded)
    return f"sha256:{hasher.hexdigest()}"


def _payload(value: object, identity_field: str) -> dict[str, object]:
    payload = asdict(value)  # type: ignore[arg-type]
    payload.pop(identity_field)
    return payload


def _unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise PerceptionCandidatePromotionError(f"{name} must be in [0, 1]")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerceptionCandidatePromotionError(f"{name} must be a positive integer")


def _non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PerceptionCandidatePromotionError(
            f"{name} must be a non-negative integer"
        )


def _ordered_unique(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and not values:
        raise PerceptionCandidatePromotionError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise PerceptionCandidatePromotionError(f"{name} must be unique and sorted")


def _ordered_with_repetition(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and not values:
        raise PerceptionCandidatePromotionError(f"{name} must be non-empty")
    if values != tuple(sorted(values)):
        raise PerceptionCandidatePromotionError(f"{name} must be sorted")


def _same_float(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


@dataclass(frozen=True)
class CandidatePromotionProposalDTO:
    """Reviewable proposal derived from one P18D-validated candidate."""

    proposal_id: str
    discovery_report_id: str
    validation_report_id: str
    validation_result_id: str
    candidate_id: str
    source_statistic_id: str
    validation_expectation_id: str
    candidate_kind: str
    field_schema_id: str
    field_ids: tuple[str, ...]
    proposed_expected_change: str
    minimum_mean_absolute_change: float
    minimum_changed_fraction: float
    minimum_signed_change_magnitude: float
    discovery_cohort_id: str
    validation_cohort_id: str
    discovery_occurrence_count: int
    discovery_observation_count: int
    discovery_recurrence_fraction: float
    dominant_direction: str
    direction_consistency: float
    validation_confirmation_count: int
    validation_observation_count: int
    validation_confirmation_fraction: float
    supporting_discovery_observation_ids: tuple[str, ...]
    supporting_discovery_interaction_ids: tuple[str, ...]
    supporting_discovery_transition_evidence_ids: tuple[str, ...]
    supporting_discovery_conformance_report_ids: tuple[str, ...]
    supporting_discovery_finding_ids: tuple[str, ...]
    validation_observation_ids: tuple[str, ...]
    validation_interaction_ids: tuple[str, ...]
    validation_transition_evidence_ids: tuple[str, ...]
    status: str = CANDIDATE_PROMOTION_PROPOSAL_STATUS
    materialization_status: str = CANDIDATE_PROMOTION_MATERIALIZATION_STATUS
    version: str = CANDIDATE_PROMOTION_PROPOSAL_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.proposal_id,
                self.discovery_report_id,
                self.validation_report_id,
                self.validation_result_id,
                self.candidate_id,
                self.source_statistic_id,
                self.validation_expectation_id,
                self.candidate_kind,
                self.field_schema_id,
                self.discovery_cohort_id,
                self.validation_cohort_id,
                self.dominant_direction,
            )
        ):
            raise PerceptionCandidatePromotionError(
                "promotion proposal identities must be non-empty"
            )
        if self.discovery_cohort_id == self.validation_cohort_id:
            raise PerceptionCandidatePromotionError(
                "promotion proposal discovery and validation cohorts must differ"
            )
        if self.candidate_kind not in RECURRENT_UNEXPLAINED_STATISTIC_KINDS:
            raise PerceptionCandidatePromotionError(
                f"unsupported promotion candidate_kind: {self.candidate_kind}"
            )
        _ordered_unique(
            "promotion proposal field_ids", self.field_ids, allow_empty=False
        )
        if self.candidate_kind == "field" and len(self.field_ids) != 1:
            raise PerceptionCandidatePromotionError(
                "field promotion proposals require exactly one field"
            )
        if self.candidate_kind == "cooccurrence_signature" and len(self.field_ids) < 2:
            raise PerceptionCandidatePromotionError(
                "signature promotion proposals require at least two fields"
            )
        if self.proposed_expected_change not in {"change", "increase", "decrease"}:
            raise PerceptionCandidatePromotionError(
                "unsupported proposed_expected_change"
            )
        for name in (
            "minimum_mean_absolute_change",
            "minimum_changed_fraction",
            "minimum_signed_change_magnitude",
            "discovery_recurrence_fraction",
            "direction_consistency",
            "validation_confirmation_fraction",
        ):
            _unit(name, getattr(self, name))
        if (
            self.proposed_expected_change == "change"
            and self.minimum_signed_change_magnitude != 0.0
        ):
            raise PerceptionCandidatePromotionError(
                "non-directional promotion proposal cannot require signed magnitude"
            )
        _positive_int("discovery_occurrence_count", self.discovery_occurrence_count)
        _positive_int("discovery_observation_count", self.discovery_observation_count)
        _positive_int("validation_observation_count", self.validation_observation_count)
        _non_negative_int(
            "validation_confirmation_count",
            self.validation_confirmation_count,
        )
        if self.discovery_occurrence_count > self.discovery_observation_count:
            raise PerceptionCandidatePromotionError(
                "discovery occurrence count exceeds observation count"
            )
        if self.validation_confirmation_count > self.validation_observation_count:
            raise PerceptionCandidatePromotionError(
                "validation confirmation count exceeds observation count"
            )
        if not _same_float(
            self.discovery_recurrence_fraction,
            self.discovery_occurrence_count / self.discovery_observation_count,
        ):
            raise PerceptionCandidatePromotionError(
                "discovery recurrence fraction disagrees with counts"
            )
        if not _same_float(
            self.validation_confirmation_fraction,
            self.validation_confirmation_count / self.validation_observation_count,
        ):
            raise PerceptionCandidatePromotionError(
                "validation confirmation fraction disagrees with counts"
            )
        _ordered_unique(
            "supporting discovery observation identities",
            self.supporting_discovery_observation_ids,
            allow_empty=False,
        )
        _ordered_unique(
            "supporting discovery interaction identities",
            self.supporting_discovery_interaction_ids,
            allow_empty=False,
        )
        _ordered_with_repetition(
            "supporting discovery transition identities",
            self.supporting_discovery_transition_evidence_ids,
            allow_empty=False,
        )
        _ordered_with_repetition(
            "supporting discovery conformance identities",
            self.supporting_discovery_conformance_report_ids,
            allow_empty=False,
        )
        _ordered_unique(
            "supporting discovery finding identities",
            self.supporting_discovery_finding_ids,
            allow_empty=False,
        )
        if not (
            len(self.supporting_discovery_observation_ids)
            == len(self.supporting_discovery_interaction_ids)
            == len(self.supporting_discovery_transition_evidence_ids)
            == len(self.supporting_discovery_conformance_report_ids)
            == self.discovery_occurrence_count
        ):
            raise PerceptionCandidatePromotionError(
                "supporting discovery lineage counts disagree with occurrences"
            )
        _ordered_unique(
            "validation observation identities",
            self.validation_observation_ids,
            allow_empty=False,
        )
        _ordered_unique(
            "validation interaction identities",
            self.validation_interaction_ids,
            allow_empty=False,
        )
        _ordered_with_repetition(
            "validation transition identities",
            self.validation_transition_evidence_ids,
            allow_empty=False,
        )
        if not (
            len(self.validation_observation_ids)
            == len(self.validation_interaction_ids)
            == len(self.validation_transition_evidence_ids)
            == self.validation_observation_count
        ):
            raise PerceptionCandidatePromotionError(
                "validation lineage counts disagree with observation count"
            )
        if self.status != CANDIDATE_PROMOTION_PROPOSAL_STATUS:
            raise PerceptionCandidatePromotionError(
                "unsupported candidate promotion proposal status"
            )
        if self.materialization_status != CANDIDATE_PROMOTION_MATERIALIZATION_STATUS:
            raise PerceptionCandidatePromotionError(
                "promotion proposals must remain not_materialized"
            )
        if self.version != CANDIDATE_PROMOTION_PROPOSAL_VERSION:
            raise PerceptionCandidatePromotionError(
                "unsupported candidate promotion proposal version"
            )
        if self.proposal_id != _digest(_payload(self, "proposal_id")):
            raise PerceptionCandidatePromotionError(
                "candidate promotion proposal identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class CandidatePromotionProposalSetDTO:
    proposal_set_id: str
    discovery_report_id: str
    validation_report_id: str
    discovery_cohort_id: str
    validation_cohort_id: str
    field_schema_id: str
    proposal_ids: tuple[str, ...]
    proposals: tuple[CandidatePromotionProposalDTO, ...]
    semantics: str = CANDIDATE_PROMOTION_SEMANTICS
    version: str = CANDIDATE_PROMOTION_PROPOSAL_SET_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.proposal_set_id,
                self.discovery_report_id,
                self.validation_report_id,
                self.discovery_cohort_id,
                self.validation_cohort_id,
                self.field_schema_id,
            )
        ):
            raise PerceptionCandidatePromotionError(
                "promotion proposal-set identities must be non-empty"
            )
        if self.discovery_cohort_id == self.validation_cohort_id:
            raise PerceptionCandidatePromotionError(
                "promotion proposal-set cohorts must differ"
            )
        _ordered_unique(
            "promotion proposal identities", self.proposal_ids, allow_empty=False
        )
        actual_ids = tuple(sorted(item.proposal_id for item in self.proposals))
        if actual_ids != self.proposal_ids:
            raise PerceptionCandidatePromotionError(
                "promotion proposal identities disagree with proposals"
            )
        if any(
            item.discovery_report_id != self.discovery_report_id
            or item.validation_report_id != self.validation_report_id
            or item.discovery_cohort_id != self.discovery_cohort_id
            or item.validation_cohort_id != self.validation_cohort_id
            or item.field_schema_id != self.field_schema_id
            for item in self.proposals
        ):
            raise PerceptionCandidatePromotionError(
                "promotion proposal lineage disagrees with proposal set"
            )
        if self.semantics != CANDIDATE_PROMOTION_SEMANTICS:
            raise PerceptionCandidatePromotionError(
                "unsupported candidate promotion semantics"
            )
        if self.version != CANDIDATE_PROMOTION_PROPOSAL_SET_VERSION:
            raise PerceptionCandidatePromotionError(
                "unsupported candidate promotion proposal-set version"
            )
        if self.proposal_set_id != _digest(_payload(self, "proposal_set_id")):
            raise PerceptionCandidatePromotionError(
                "candidate promotion proposal-set identity disagrees with canonical payload"
            )

    def proposal_for_candidate(
        self, candidate_id: str
    ) -> CandidatePromotionProposalDTO:
        for proposal in self.proposals:
            if proposal.candidate_id == candidate_id:
                return proposal
        raise KeyError(candidate_id)


@dataclass(frozen=True)
class CandidatePromotionDecisionDTO:
    decision_id: str
    proposal_id: str
    reviewer_id: str
    decision: str
    rationale: str
    semantic_name: str | None = None
    semantic_type: str | None = None
    semantic_role: str | None = None
    materialization_status: str = CANDIDATE_PROMOTION_MATERIALIZATION_STATUS
    version: str = CANDIDATE_PROMOTION_DECISION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (self.decision_id, self.proposal_id, self.reviewer_id, self.rationale)
        ):
            raise PerceptionCandidatePromotionError(
                "promotion decision identities and rationale must be non-empty"
            )
        if self.decision not in CANDIDATE_PROMOTION_DECISIONS:
            raise PerceptionCandidatePromotionError(
                f"unsupported candidate promotion decision: {self.decision}"
            )
        semantic_values = (self.semantic_name, self.semantic_type, self.semantic_role)
        if self.decision == "approved":
            if not self.semantic_name or not self.semantic_type:
                raise PerceptionCandidatePromotionError(
                    "approved promotion decisions require semantic_name and semantic_type"
                )
        elif any(value is not None for value in semantic_values):
            raise PerceptionCandidatePromotionError(
                "non-approved promotion decisions cannot carry semantic materialization"
            )
        if self.materialization_status != CANDIDATE_PROMOTION_MATERIALIZATION_STATUS:
            raise PerceptionCandidatePromotionError(
                "promotion decisions must remain not_materialized"
            )
        if self.version != CANDIDATE_PROMOTION_DECISION_VERSION:
            raise PerceptionCandidatePromotionError(
                "unsupported candidate promotion decision version"
            )
        if self.decision_id != _digest(_payload(self, "decision_id")):
            raise PerceptionCandidatePromotionError(
                "candidate promotion decision identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        proposal: CandidatePromotionProposalDTO,
        *,
        reviewer_id: str,
        decision: str,
        rationale: str,
        semantic_name: str | None = None,
        semantic_type: str | None = None,
        semantic_role: str | None = None,
    ) -> "CandidatePromotionDecisionDTO":
        values: dict[str, object] = {
            "proposal_id": proposal.proposal_id,
            "reviewer_id": reviewer_id,
            "decision": decision,
            "rationale": rationale,
            "semantic_name": semantic_name,
            "semantic_type": semantic_type,
            "semantic_role": semantic_role,
            "materialization_status": CANDIDATE_PROMOTION_MATERIALIZATION_STATUS,
            "version": CANDIDATE_PROMOTION_DECISION_VERSION,
        }
        return cls(decision_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CandidatePromotionReviewDTO:
    review_id: str
    status: str
    proposal_set_id: str
    proposal_ids: tuple[str, ...]
    decision_ids: tuple[str, ...]
    decisions: tuple[CandidatePromotionDecisionDTO, ...]
    pending_proposal_ids: tuple[str, ...]
    approved_proposal_ids: tuple[str, ...]
    rejected_proposal_ids: tuple[str, ...]
    deferred_proposal_ids: tuple[str, ...]
    semantic_annotation_required_proposal_ids: tuple[str, ...]
    semantics: str = CANDIDATE_PROMOTION_SEMANTICS
    version: str = CANDIDATE_PROMOTION_REVIEW_VERSION

    def __post_init__(self) -> None:
        if not self.review_id or not self.proposal_set_id:
            raise PerceptionCandidatePromotionError(
                "promotion review identities must be non-empty"
            )
        if self.status not in CANDIDATE_PROMOTION_REVIEW_STATUSES:
            raise PerceptionCandidatePromotionError(
                f"unsupported promotion review status: {self.status}"
            )
        _ordered_unique(
            "promotion review proposal_ids", self.proposal_ids, allow_empty=False
        )
        _ordered_unique("promotion review decision_ids", self.decision_ids)
        actual_decision_ids = tuple(sorted(item.decision_id for item in self.decisions))
        if actual_decision_ids != self.decision_ids:
            raise PerceptionCandidatePromotionError(
                "promotion review decision identities disagree with decisions"
            )
        decided_proposals = tuple(sorted(item.proposal_id for item in self.decisions))
        if len(decided_proposals) != len(set(decided_proposals)):
            raise PerceptionCandidatePromotionError(
                "promotion review allows one decision per proposal"
            )
        if not set(decided_proposals) <= set(self.proposal_ids):
            raise PerceptionCandidatePromotionError(
                "promotion decision references an unknown proposal"
            )
        categories = (
            self.pending_proposal_ids,
            self.approved_proposal_ids,
            self.rejected_proposal_ids,
            self.deferred_proposal_ids,
            self.semantic_annotation_required_proposal_ids,
        )
        for index, category in enumerate(categories):
            _ordered_unique(f"promotion review category {index}", category)
        flattened = tuple(item for category in categories for item in category)
        if len(flattened) != len(set(flattened)) or set(flattened) != set(
            self.proposal_ids
        ):
            raise PerceptionCandidatePromotionError(
                "promotion review categories must partition every proposal"
            )
        decision_map = {item.proposal_id: item.decision for item in self.decisions}
        expected = {
            "approved": set(self.approved_proposal_ids),
            "rejected": set(self.rejected_proposal_ids),
            "deferred": set(self.deferred_proposal_ids),
            "needs_semantic_annotation": set(
                self.semantic_annotation_required_proposal_ids
            ),
        }
        for decision, proposal_ids in expected.items():
            if {
                proposal_id
                for proposal_id, value in decision_map.items()
                if value == decision
            } != proposal_ids:
                raise PerceptionCandidatePromotionError(
                    "promotion review decision categories disagree with decisions"
                )
        expected_pending = set(self.proposal_ids) - set(decision_map)
        if expected_pending != set(self.pending_proposal_ids):
            raise PerceptionCandidatePromotionError(
                "promotion review pending proposals disagree with decisions"
            )
        expected_status = (
            "pending_review"
            if not self.decisions
            else "review_complete"
            if not self.pending_proposal_ids
            else "partially_reviewed"
        )
        if self.status != expected_status:
            raise PerceptionCandidatePromotionError(
                "promotion review status disagrees with decision coverage"
            )
        if self.semantics != CANDIDATE_PROMOTION_SEMANTICS:
            raise PerceptionCandidatePromotionError(
                "unsupported candidate promotion semantics"
            )
        if self.version != CANDIDATE_PROMOTION_REVIEW_VERSION:
            raise PerceptionCandidatePromotionError(
                "unsupported candidate promotion review version"
            )
        if self.review_id != _digest(_payload(self, "review_id")):
            raise PerceptionCandidatePromotionError(
                "candidate promotion review identity disagrees with canonical payload"
            )

    def decisions_for_status(
        self,
        decision: str,
    ) -> tuple[CandidatePromotionDecisionDTO, ...]:
        if decision not in CANDIDATE_PROMOTION_DECISIONS:
            raise PerceptionCandidatePromotionError(
                f"unsupported candidate promotion decision: {decision}"
            )
        return tuple(item for item in self.decisions if item.decision == decision)


def _candidate_and_statistic(
    discovery: TransitionDiscoveryReportDTO,
    result: CandidateValidationResultDTO,
) -> tuple[MissingComponentCandidateDTO, RecurrentUnexplainedStatisticDTO]:
    candidates = {item.candidate_id: item for item in discovery.candidates}
    statistics = {item.statistic_id: item for item in discovery.statistics}
    try:
        candidate = candidates[result.candidate_id]
    except KeyError as exc:
        raise PerceptionCandidatePromotionError(
            "validated result references an unknown discovery candidate"
        ) from exc
    try:
        statistic = statistics[candidate.source_statistic_id]
    except KeyError as exc:
        raise PerceptionCandidatePromotionError(
            "promotion candidate references an unknown discovery statistic"
        ) from exc
    expectation = result.expectation
    if (
        expectation.discovery_report_id != discovery.report_id
        or expectation.candidate_id != candidate.candidate_id
        or expectation.source_statistic_id != statistic.statistic_id
        or expectation.candidate_kind != candidate.candidate_kind
        or expectation.field_schema_id != discovery.field_schema_id
        or expectation.field_ids != candidate.field_ids
        or expectation.expected_change != candidate.proposed_expected_change
    ):
        raise PerceptionCandidatePromotionError(
            "validated result expectation disagrees with discovery lineage"
        )
    if (
        statistic.statistic_kind != candidate.candidate_kind
        or statistic.field_ids != candidate.field_ids
    ):
        raise PerceptionCandidatePromotionError(
            "promotion candidate disagrees with its source statistic"
        )
    return candidate, statistic


def _proposal(
    *,
    discovery: TransitionDiscoveryReportDTO,
    validation: CandidateValidationReportDTO,
    result: CandidateValidationResultDTO,
) -> CandidatePromotionProposalDTO:
    candidate, statistic = _candidate_and_statistic(discovery, result)
    expectation = result.expectation
    values: dict[str, object] = {
        "discovery_report_id": discovery.report_id,
        "validation_report_id": validation.report_id,
        "validation_result_id": result.result_id,
        "candidate_id": candidate.candidate_id,
        "source_statistic_id": statistic.statistic_id,
        "validation_expectation_id": expectation.expectation_id,
        "candidate_kind": candidate.candidate_kind,
        "field_schema_id": discovery.field_schema_id,
        "field_ids": candidate.field_ids,
        "proposed_expected_change": candidate.proposed_expected_change,
        "minimum_mean_absolute_change": expectation.minimum_mean_absolute_change,
        "minimum_changed_fraction": expectation.minimum_changed_fraction,
        "minimum_signed_change_magnitude": (
            expectation.minimum_signed_change_magnitude
        ),
        "discovery_cohort_id": discovery.cohort_id,
        "validation_cohort_id": validation.validation_cohort_id,
        "discovery_occurrence_count": statistic.occurrence_count,
        "discovery_observation_count": statistic.observation_count,
        "discovery_recurrence_fraction": statistic.recurrence_fraction,
        "dominant_direction": statistic.dominant_direction,
        "direction_consistency": statistic.direction_consistency,
        "validation_confirmation_count": result.confirmation_count,
        "validation_observation_count": result.observation_count,
        "validation_confirmation_fraction": result.confirmation_fraction,
        "supporting_discovery_observation_ids": (statistic.supporting_observation_ids),
        "supporting_discovery_interaction_ids": (statistic.supporting_interaction_ids),
        "supporting_discovery_transition_evidence_ids": (
            statistic.supporting_transition_evidence_ids
        ),
        "supporting_discovery_conformance_report_ids": (
            statistic.supporting_conformance_report_ids
        ),
        "supporting_discovery_finding_ids": statistic.supporting_finding_ids,
        "validation_observation_ids": validation.validation_observation_ids,
        "validation_interaction_ids": validation.validation_interaction_ids,
        "validation_transition_evidence_ids": (
            validation.validation_transition_evidence_ids
        ),
        "status": CANDIDATE_PROMOTION_PROPOSAL_STATUS,
        "materialization_status": CANDIDATE_PROMOTION_MATERIALIZATION_STATUS,
        "version": CANDIDATE_PROMOTION_PROPOSAL_VERSION,
    }
    return CandidatePromotionProposalDTO(
        proposal_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )


def propose_validated_candidate_promotions(
    discovery: TransitionDiscoveryReportDTO,
    validation: CandidateValidationReportDTO,
) -> CandidatePromotionProposalSetDTO:
    """Create reviewable proposals for P18D-validated candidates only."""

    if validation.discovery_report_id != discovery.report_id:
        raise PerceptionCandidatePromotionError(
            "validation report does not reference the supplied discovery report"
        )
    if validation.discovery_cohort_id != discovery.cohort_id:
        raise PerceptionCandidatePromotionError(
            "validation report discovery cohort disagrees with discovery report"
        )
    if validation.field_schema_id != discovery.field_schema_id:
        raise PerceptionCandidatePromotionError(
            "validation and discovery field schemas differ"
        )
    if validation.discovery_cohort_id == validation.validation_cohort_id:
        raise PerceptionCandidatePromotionError(
            "promotion requires disjoint discovery and validation cohorts"
        )
    validated_results = tuple(
        item for item in validation.results if item.status == "validated"
    )
    if not validated_results:
        raise PerceptionCandidatePromotionError(
            "candidate promotion requires at least one validated result"
        )
    proposals = tuple(
        sorted(
            (
                _proposal(
                    discovery=discovery,
                    validation=validation,
                    result=result,
                )
                for result in validated_results
            ),
            key=lambda item: item.proposal_id,
        )
    )
    values: dict[str, object] = {
        "discovery_report_id": discovery.report_id,
        "validation_report_id": validation.report_id,
        "discovery_cohort_id": discovery.cohort_id,
        "validation_cohort_id": validation.validation_cohort_id,
        "field_schema_id": discovery.field_schema_id,
        "proposal_ids": tuple(sorted(item.proposal_id for item in proposals)),
        "proposals": proposals,
        "semantics": CANDIDATE_PROMOTION_SEMANTICS,
        "version": CANDIDATE_PROMOTION_PROPOSAL_SET_VERSION,
    }
    identity_values = dict(values)
    identity_values["proposals"] = tuple(asdict(item) for item in proposals)
    return CandidatePromotionProposalSetDTO(
        proposal_set_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )


def review_candidate_promotion_proposals(
    proposal_set: CandidatePromotionProposalSetDTO,
    decisions: tuple[CandidatePromotionDecisionDTO, ...] = (),
) -> CandidatePromotionReviewDTO:
    """Build an immutable review ledger without applying approved proposals."""

    decision_ids = tuple(item.decision_id for item in decisions)
    if len(decision_ids) != len(set(decision_ids)):
        raise PerceptionCandidatePromotionError(
            "promotion decisions must have unique identities"
        )
    proposal_ids = tuple(item.proposal_id for item in decisions)
    if len(proposal_ids) != len(set(proposal_ids)):
        raise PerceptionCandidatePromotionError(
            "promotion review allows one decision per proposal"
        )
    unknown = set(proposal_ids) - set(proposal_set.proposal_ids)
    if unknown:
        raise PerceptionCandidatePromotionError(
            f"promotion decisions reference unknown proposals: {sorted(unknown)}"
        )
    ordered_decisions = tuple(sorted(decisions, key=lambda item: item.decision_id))
    by_decision = {
        decision: tuple(
            sorted(
                item.proposal_id
                for item in ordered_decisions
                if item.decision == decision
            )
        )
        for decision in CANDIDATE_PROMOTION_DECISIONS
    }
    pending = tuple(sorted(set(proposal_set.proposal_ids) - set(proposal_ids)))
    status = (
        "pending_review"
        if not ordered_decisions
        else "review_complete"
        if not pending
        else "partially_reviewed"
    )
    values: dict[str, object] = {
        "status": status,
        "proposal_set_id": proposal_set.proposal_set_id,
        "proposal_ids": proposal_set.proposal_ids,
        "decision_ids": tuple(sorted(item.decision_id for item in ordered_decisions)),
        "decisions": ordered_decisions,
        "pending_proposal_ids": pending,
        "approved_proposal_ids": by_decision["approved"],
        "rejected_proposal_ids": by_decision["rejected"],
        "deferred_proposal_ids": by_decision["deferred"],
        "semantic_annotation_required_proposal_ids": by_decision[
            "needs_semantic_annotation"
        ],
        "semantics": CANDIDATE_PROMOTION_SEMANTICS,
        "version": CANDIDATE_PROMOTION_REVIEW_VERSION,
    }
    identity_values = dict(values)
    identity_values["decisions"] = tuple(asdict(item) for item in ordered_decisions)
    return CandidatePromotionReviewDTO(
        review_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )
