"""World-action coupling: bounded influence of remembered futures on choice.

The expected transition must influence the representation used for action
selection, but transition plausibility must not be confused with action
utility. The baseline policy score stays the primary utility source: future
memory acts only as support / constraint / contradiction / out-of-distribution
evidence. Selection order among surviving candidates is always baseline
order, so future predictability alone can never manufacture utility.

No action is executed here. Prediction and enactment remain separate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Callable, Final, Mapping

from .expected_transition import ExpectedTransitionFieldDTO, ExpectedTransitionVPMDTO
from .fields import VPMFieldSchemaDTO, validate_source_for_schema
from .inference import (
    ActionCandidateDTO,
    BaselineNearestNeighborModelDTO,
    BaselinePredictionDTO,
    predict_baseline_action,
)
from .representation import SourceVPMDTO
from .transition_conformance import (
    TRANSITION_CONFORMANCE_STATUSES,
    RelationAnnotationDTO,
    TransitionExpectationDTO,
    _decide_transition_status,
)
from .transition_projection import TransitionModelDTO, project_expected_transition
from .expectations import PerceptionRegionAnnotationDTO

WORLD_ACTION_POLICY_VERSION: Final = "perception-world-action-policy/1"
WORLD_ACTION_CANDIDATE_VERSION: Final = "perception-world-action-candidate/1"
COUPLED_ACTION_PREDICTION_VERSION: Final = "perception-coupled-action-prediction/1"
EXPECTED_CONFORMANCE_VERSION: Final = "perception-expected-conformance/1"
EXPECTED_CONFORMANCE_FINDING_VERSION: Final = (
    "perception-expected-conformance-finding/1"
)
WORLD_ACTION_CANDIDATE_STATUSES: Final = frozenset(
    {
        "supported",
        "supported_with_low_future_confidence",
        "contradicted_by_transition_expectation",
        "future_out_of_distribution",
        "insufficient_future_evidence",
        "base_predictor_rejected",
    }
)
EXPECTED_CONFORMANCE_STATUSES: Final = frozenset(
    {"conformant", "contradicted", "not_applicable"}
)
# Mirrors the P18B nonconformant finding set: these projected outcomes
# contradict declared expectations and may veto a candidate.
EXPECTED_CONTRADICTION_STATUSES: Final = frozenset(
    {
        "missing_expected_change",
        "unexpected_change",
        "excessive_change",
        "insufficient_change",
        "wrong_change_direction",
    }
)


class PerceptionWorldActionError(ValueError):
    """Raised when world-action coupling violates its contract."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class WorldActionPolicyDTO:
    """Explicit, auditable gating policy for world-action coupling."""

    candidate_count: int = 3
    min_support: int = 5
    low_confidence_threshold: float = 0.35
    reject_on_contradiction: bool = True
    reject_on_ood: bool = True
    reject_on_insufficient: bool = False
    version: str = WORLD_ACTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.candidate_count <= 0:
            raise PerceptionWorldActionError("candidate_count must be positive")
        if self.min_support <= 0:
            raise PerceptionWorldActionError("min_support must be positive")
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise PerceptionWorldActionError(
                "low_confidence_threshold must be in [0, 1]"
            )
        if self.version != WORLD_ACTION_POLICY_VERSION:
            raise PerceptionWorldActionError("unsupported world action policy version")

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "candidate_count": self.candidate_count,
            "low_confidence_threshold": self.low_confidence_threshold,
            "min_support": self.min_support,
            "reject_on_contradiction": self.reject_on_contradiction,
            "reject_on_insufficient": self.reject_on_insufficient,
            "reject_on_ood": self.reject_on_ood,
            "version": self.version,
        }

    @property
    def policy_id(self) -> str:
        return _digest(_canonical_json(self.canonical_payload()))


@dataclass(frozen=True)
class ExpectedConformanceFindingDTO:
    """One declared expectation tested against a projected future."""

    finding_id: str
    expectation_id: str
    status: str
    detail: str
    version: str = EXPECTED_CONFORMANCE_FINDING_VERSION

    def __post_init__(self) -> None:
        if not self.finding_id or not self.expectation_id or not self.detail:
            raise PerceptionWorldActionError(
                "expected conformance finding identities and detail must be non-empty"
            )
        if self.status not in TRANSITION_CONFORMANCE_STATUSES:
            raise PerceptionWorldActionError(
                f"unsupported expected conformance finding status: {self.status!r}"
            )
        if self.status == "unexplained_change":
            raise PerceptionWorldActionError(
                "expected conformance tests declared targets only"
            )
        if self.version != EXPECTED_CONFORMANCE_FINDING_VERSION:
            raise PerceptionWorldActionError(
                "unsupported expected conformance finding version"
            )


@dataclass(frozen=True)
class ExpectedConformanceDTO:
    """Projected future tested against declared expectations pre-enactment."""

    conformance_id: str
    expected_transition_id: str
    status: str
    findings: tuple[ExpectedConformanceFindingDTO, ...]
    version: str = EXPECTED_CONFORMANCE_VERSION

    def __post_init__(self) -> None:
        if not self.conformance_id or not self.expected_transition_id:
            raise PerceptionWorldActionError(
                "expected conformance identities must be non-empty"
            )
        if self.status not in EXPECTED_CONFORMANCE_STATUSES:
            raise PerceptionWorldActionError(
                f"unsupported expected conformance status: {self.status!r}"
            )
        finding_ids = tuple(item.finding_id for item in self.findings)
        if finding_ids != tuple(sorted(set(finding_ids))):
            raise PerceptionWorldActionError(
                "expected conformance findings must be unique and sorted"
            )
        if self.version != EXPECTED_CONFORMANCE_VERSION:
            raise PerceptionWorldActionError("unsupported expected conformance version")


@dataclass(frozen=True)
class WorldActionCandidateDTO:
    """One candidate action coupled with its remembered expected future."""

    action_label: str
    base_score: float
    base_rank: int
    expected_transition_id: str | None
    transition_support: float | None
    transition_confidence: float | None
    expectation_conformance: str | None
    status: str
    reasons: tuple[str, ...] = ()
    version: str = WORLD_ACTION_CANDIDATE_VERSION

    def __post_init__(self) -> None:
        if not self.action_label:
            raise PerceptionWorldActionError("candidate action label must be non-empty")
        if self.base_rank < 0:
            raise PerceptionWorldActionError("base_rank must be >= 0")
        if self.status not in WORLD_ACTION_CANDIDATE_STATUSES:
            raise PerceptionWorldActionError(
                f"unsupported world action candidate status: {self.status!r}"
            )
        if (
            self.expectation_conformance is not None
            and self.expectation_conformance not in EXPECTED_CONFORMANCE_STATUSES
        ):
            raise PerceptionWorldActionError(
                "unsupported expectation conformance status"
            )
        if self.version != WORLD_ACTION_CANDIDATE_VERSION:
            raise PerceptionWorldActionError(
                "unsupported world action candidate version"
            )


@dataclass(frozen=True)
class CoupledActionPredictionDTO:
    """Bounded coupled result. The baseline prediction is preserved intact."""

    prediction_id: str
    source_vpm_id: str
    baseline: BaselinePredictionDTO
    candidates: tuple[WorldActionCandidateDTO, ...]
    selected_action: str | None
    accepted: bool
    policy_id: str
    baseline_model_id: str
    transition_model_id: str
    version: str = COUPLED_ACTION_PREDICTION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.prediction_id,
                self.source_vpm_id,
                self.policy_id,
                self.baseline_model_id,
                self.transition_model_id,
            )
        ):
            raise PerceptionWorldActionError(
                "coupled prediction identities must be non-empty"
            )
        if not self.candidates:
            raise PerceptionWorldActionError(
                "coupled prediction requires at least one candidate"
            )
        if self.baseline.source_vpm_id != self.source_vpm_id:
            raise PerceptionWorldActionError(
                "baseline source disagrees with coupled source"
            )
        if self.baseline.model_id != self.baseline_model_id:
            raise PerceptionWorldActionError(
                "baseline model disagrees with coupled predictor reference"
            )
        if self.accepted and self.selected_action is None:
            raise PerceptionWorldActionError(
                "accepted coupled prediction requires a selected action"
            )
        if self.selected_action is not None and self.selected_action not in {
            item.action_label for item in self.candidates
        }:
            raise PerceptionWorldActionError(
                "selected action is not among coupled candidates"
            )
        if self.version != COUPLED_ACTION_PREDICTION_VERSION:
            raise PerceptionWorldActionError(
                "unsupported coupled action prediction version"
            )


def _relation_field_ids(
    relation: RelationAnnotationDTO,
    annotations: Mapping[str, PerceptionRegionAnnotationDTO],
    known_fields: set[str],
) -> tuple[str, ...]:
    unknown = set(relation.member_annotation_ids) - set(annotations)
    if unknown:
        raise PerceptionWorldActionError(
            f"relation references unknown annotations: {sorted(unknown)}"
        )
    if relation.derived_field_ids:
        derived_unknown = set(relation.derived_field_ids) - known_fields
        if derived_unknown:
            raise PerceptionWorldActionError(
                f"relation references unknown derived fields: {sorted(derived_unknown)}"
            )
        return relation.derived_field_ids
    return tuple(
        sorted(
            {
                field_id
                for annotation_id in relation.member_annotation_ids
                for field_id in annotations[annotation_id].field_ids
            }
        )
    )


def _field_value_counts(field_schema: VPMFieldSchemaDTO) -> dict[str, int]:
    return {
        field.field_id: (field.x1 - field.x0)
        * (field.y1 - field.y0)
        * (field.channel_end - field.channel_start)
        for field in field_schema.fields
    }


def _resolve_conformance_targets(
    expectation: TransitionExpectationDTO,
    annotation_map: Mapping[str, PerceptionRegionAnnotationDTO],
    relation_map: Mapping[str, RelationAnnotationDTO],
    known_fields: set[str],
) -> set[str]:
    unknown_annotations = set(expectation.annotation_ids) - set(annotation_map)
    if unknown_annotations:
        raise PerceptionWorldActionError(
            f"expectation references unknown annotations: {sorted(unknown_annotations)}"
        )
    unknown_relations = set(expectation.relation_ids) - set(relation_map)
    if unknown_relations:
        raise PerceptionWorldActionError(
            f"expectation references unknown relations: {sorted(unknown_relations)}"
        )
    target_fields = {
        field_id
        for annotation_id in expectation.annotation_ids
        for field_id in annotation_map[annotation_id].field_ids
    }
    for relation_id in expectation.relation_ids:
        target_fields.update(
            _relation_field_ids(relation_map[relation_id], annotation_map, known_fields)
        )
    if not target_fields:
        raise PerceptionWorldActionError("transition expectation resolves to no fields")
    return target_fields


def _aggregate_expected_moments(
    projected: Mapping[str, ExpectedTransitionFieldDTO],
    target_fields: set[str],
    counts: Mapping[str, int],
) -> tuple[float, float, float]:
    total = sum(counts[field_id] for field_id in target_fields)

    def _weighted(attribute: str) -> float:
        return (
            sum(
                getattr(projected[field_id], attribute) * counts[field_id]
                for field_id in target_fields
            )
            / total
        )

    return (
        _weighted("expected_mean_absolute_change"),
        _weighted("expected_mean_signed_change"),
        _weighted("expected_changed_fraction"),
    )


def _not_applicable_conformance(
    expected: ExpectedTransitionVPMDTO,
) -> ExpectedConformanceDTO:
    payload = _digest(
        _canonical_json(
            {
                "expected_transition_id": expected.expected_transition_id,
                "status": "not_applicable",
                "version": EXPECTED_CONFORMANCE_VERSION,
            }
        )
    )
    return ExpectedConformanceDTO(
        conformance_id=payload,
        expected_transition_id=expected.expected_transition_id,
        status="not_applicable",
        findings=(),
    )


def check_expected_conformance(
    expected: ExpectedTransitionVPMDTO,
    field_schema: VPMFieldSchemaDTO,
    expectations: tuple[TransitionExpectationDTO, ...],
    annotations: tuple[PerceptionRegionAnnotationDTO, ...],
    relations: tuple[RelationAnnotationDTO, ...] = (),
) -> ExpectedConformanceDTO:
    """Test a projected future against declared expectations pre-enactment.

    Applies identical P18B declaration semantics (shared decision function)
    to projected aggregates. Per-action scoping lives with the caller, which
    passes only the expectations declared for the projected action.
    """
    if field_schema.field_schema_id != expected.field_schema_id:
        raise PerceptionWorldActionError(
            "field schema does not match expected transition"
        )
    if not expectations:
        return _not_applicable_conformance(expected)
    annotation_map = {item.annotation_id: item for item in annotations}
    if len(annotation_map) != len(annotations):
        raise PerceptionWorldActionError("annotations must have unique identities")
    relation_map = {item.relation_id: item for item in relations}
    if len(relation_map) != len(relations):
        raise PerceptionWorldActionError("relations must have unique identities")
    for annotation in annotations:
        if annotation.field_schema_id != field_schema.field_schema_id:
            raise PerceptionWorldActionError(
                "annotation field schema does not match expected transition"
            )
    projected = {item.field_id: item for item in expected.fields}
    if set(projected) != {field.field_id for field in field_schema.fields}:
        raise PerceptionWorldActionError(
            "expected transition fields do not cover the field schema"
        )
    counts = _field_value_counts(field_schema)
    findings: list[ExpectedConformanceFindingDTO] = []
    for expectation in sorted(expectations, key=lambda item: item.expectation_id):
        if expectation.field_schema_id != field_schema.field_schema_id:
            raise PerceptionWorldActionError(
                "expectation field schema does not match expected transition"
            )
        target_fields = _resolve_conformance_targets(
            expectation, annotation_map, relation_map, set(projected)
        )
        absolute, signed, fraction = _aggregate_expected_moments(
            projected, target_fields, counts
        )
        status, detail = _decide_transition_status(
            expectation,
            mean_absolute=absolute,
            mean_signed=signed,
            changed_fraction=fraction,
            has_changes=fraction > 0.0,
        )
        finding_payload = {
            "detail": detail,
            "expectation_id": expectation.expectation_id,
            "expected_transition_id": expected.expected_transition_id,
            "status": status,
            "version": EXPECTED_CONFORMANCE_FINDING_VERSION,
        }
        findings.append(
            ExpectedConformanceFindingDTO(
                finding_id=_digest(_canonical_json(finding_payload)),
                expectation_id=expectation.expectation_id,
                status=status,
                detail=detail,
            )
        )
    ordered = tuple(sorted(findings, key=lambda item: item.finding_id))
    overall = (
        "contradicted"
        if any(item.status in EXPECTED_CONTRADICTION_STATUSES for item in ordered)
        else "conformant"
    )
    conformance_payload = {
        "expected_transition_id": expected.expected_transition_id,
        "findings": [item.finding_id for item in ordered],
        "status": overall,
        "version": EXPECTED_CONFORMANCE_VERSION,
    }
    return ExpectedConformanceDTO(
        conformance_id=_digest(_canonical_json(conformance_payload)),
        expected_transition_id=expected.expected_transition_id,
        status=overall,
        findings=ordered,
    )


def _classify_candidate(
    *,
    action_label: str,
    base_score: float,
    base_rank: int,
    expected: ExpectedTransitionVPMDTO,
    conformance: ExpectedConformanceDTO,
    policy: WorldActionPolicyDTO,
) -> tuple[WorldActionCandidateDTO, bool]:
    """Classify one candidate; returns (candidate, vetoed)."""
    reasons: list[str] = [f"projection status: {expected.status}"]
    support = min(1.0, expected.support_count / max(1, 2 * policy.min_support))
    means_trusted = expected.status in {"supported", "ambiguous_future"}
    if means_trusted and conformance.status == "contradicted":
        if policy.reject_on_contradiction:
            return (
                WorldActionCandidateDTO(
                    action_label=action_label,
                    base_score=base_score,
                    base_rank=base_rank,
                    expected_transition_id=expected.expected_transition_id,
                    transition_support=support,
                    transition_confidence=expected.confidence,
                    expectation_conformance=conformance.status,
                    status="contradicted_by_transition_expectation",
                    reasons=tuple(
                        reasons + ["projected future contradicts declared expectations"]
                    ),
                ),
                True,
            )
        reasons.append("contradicted but policy keeps flagged")
    if expected.status == "supported":
        if expected.confidence < policy.low_confidence_threshold:
            status: str = "supported_with_low_future_confidence"
            reasons.append(
                f"low future confidence {expected.confidence:.3f} "
                f"< {policy.low_confidence_threshold:.3f}"
            )
        else:
            status = "supported"
            reasons.append("remembered consequence supports candidacy")
        vetoed = False
    elif expected.status == "out_of_distribution":
        status = "future_out_of_distribution"
        reasons.append("projected future is out of distribution")
        vetoed = policy.reject_on_ood
    elif expected.status in {
        "insufficient_examples",
        "unsupported_action",
        "ambiguous_future",
    }:
        status = "insufficient_future_evidence"
        reasons.append("insufficient future evidence; not silently trusted")
        vetoed = policy.reject_on_insufficient
    else:  # pragma: no cover - defensive
        status = "insufficient_future_evidence"
        reasons.append(f"unknown projection status {expected.status!r}")
        vetoed = True
    if conformance.status == "conformant":
        reasons.append("conforms to declared expectations")
    elif conformance.status == "contradicted":
        reasons.append("projected future contradicts declared expectations")
    return (
        WorldActionCandidateDTO(
            action_label=action_label,
            base_score=base_score,
            base_rank=base_rank,
            expected_transition_id=expected.expected_transition_id,
            transition_support=support,
            transition_confidence=expected.confidence,
            expectation_conformance=conformance.status,
            status=status,
            reasons=tuple(reasons),
        ),
        vetoed,
    )


def _project_and_classify_candidates(
    ranked: tuple[tuple[int, ActionCandidateDTO], ...],
    transition_model: TransitionModelDTO,
    source: SourceVPMDTO,
    field_schema: VPMFieldSchemaDTO,
    expectations_by_action: Mapping[str, tuple[TransitionExpectationDTO, ...]] | None,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...],
    relations: tuple[RelationAnnotationDTO, ...],
    policy: WorldActionPolicyDTO,
    field_weights: Mapping[str, float] | None,
    projector: Callable[..., ExpectedTransitionVPMDTO],
) -> tuple[list[WorldActionCandidateDTO], set[str]]:
    coupled: list[WorldActionCandidateDTO] = []
    vetoed_labels: set[str] = set()
    for base_rank, candidate in ranked:
        action_expectations = (
            expectations_by_action.get(candidate.action_label, ())
            if expectations_by_action is not None
            else ()
        )
        expected = projector(
            transition_model,
            source,
            candidate.action_label,
            field_schema,
            field_weights=field_weights,
            min_support=policy.min_support,
        )
        conformance = check_expected_conformance(
            expected, field_schema, action_expectations, annotations, relations
        )
        world_candidate, vetoed = _classify_candidate(
            action_label=candidate.action_label,
            base_score=candidate.score,
            base_rank=base_rank,
            expected=expected,
            conformance=conformance,
            policy=policy,
        )
        coupled.append(world_candidate)
        if vetoed:
            vetoed_labels.add(candidate.action_label)
    return coupled, vetoed_labels


def predict_action_with_future_memory(
    baseline_model: BaselineNearestNeighborModelDTO,
    transition_model: TransitionModelDTO,
    source: SourceVPMDTO,
    field_schema: VPMFieldSchemaDTO,
    *,
    expectations_by_action: Mapping[str, tuple[TransitionExpectationDTO, ...]]
    | None = None,
    annotations: tuple[PerceptionRegionAnnotationDTO, ...] = (),
    relations: tuple[RelationAnnotationDTO, ...] = (),
    policy: WorldActionPolicyDTO | None = None,
    field_weights: Mapping[str, float] | None = None,
    projector: Callable[..., ExpectedTransitionVPMDTO] = project_expected_transition,
    baseline_override: BaselinePredictionDTO | None = None,
) -> CoupledActionPredictionDTO:
    """Run baseline choice, then gate the top candidates with future memory.

    Preserves the complete baseline result, projects a structured future for
    each bounded candidate, evaluates transition support and optional declared
    expectations, and selects the highest-baseline-rank non-vetoed candidate
    (or abstains). Never executes the action.

    ``baseline_override`` optionally substitutes a precomputed ranking from
    the same predictor family (e.g. relevance-weighted distances over the
    same P3 memory) while the gate mechanics stay unchanged. The override
    must reference the same predictor model and source.
    """
    resolved = policy or WorldActionPolicyDTO()
    try:
        validate_source_for_schema(source, field_schema)
    except Exception as exc:
        raise PerceptionWorldActionError(str(exc)) from exc
    computed: BaselinePredictionDTO = predict_baseline_action(baseline_model, source)
    if baseline_override is None:
        baseline = computed
    else:
        if baseline_override.model_id != baseline_model.model_id:
            raise PerceptionWorldActionError(
                "baseline override references a different predictor model"
            )
        if baseline_override.source_vpm_id != source.source_vpm_id:
            raise PerceptionWorldActionError(
                "baseline override references a different source"
            )
        baseline = baseline_override
    ranked = tuple(enumerate(baseline.candidates))
    considered = ranked[: max(1, resolved.candidate_count)]
    coupled, vetoed_labels = _project_and_classify_candidates(
        considered,
        transition_model,
        source,
        field_schema,
        expectations_by_action,
        annotations,
        relations,
        resolved,
        field_weights,
        projector,
    )
    survivors = [item for item in coupled if item.action_label not in vetoed_labels]
    tail = tuple(
        WorldActionCandidateDTO(
            action_label=item.action_label,
            base_score=item.score,
            base_rank=base_rank,
            expected_transition_id=None,
            transition_support=None,
            transition_confidence=None,
            expectation_conformance=None,
            status="base_predictor_rejected",
            reasons=("outside coupled candidate bound; baseline rank preserved",),
        )
        for base_rank, item in ranked[max(1, resolved.candidate_count) :]
    )
    all_candidates = tuple(coupled) + tail
    if baseline.status == "accepted" and survivors:
        selected = survivors[0].action_label
        accepted = True
    else:
        selected, accepted = None, False
    payload = {
        "accepted": accepted,
        "baseline_override": baseline_override is not None,
        "baseline_prediction_id": baseline.prediction_id,
        "baseline_model_id": baseline_model.model_id,
        "candidates": [
            {
                "action_label": item.action_label,
                "base_rank": item.base_rank,
                "expected_transition_id": item.expected_transition_id,
                "status": item.status,
            }
            for item in all_candidates
        ],
        "policy_id": resolved.policy_id,
        "selected_action": selected,
        "source_vpm_id": source.source_vpm_id,
        "transition_model_id": transition_model.model_id,
        "version": COUPLED_ACTION_PREDICTION_VERSION,
    }
    return CoupledActionPredictionDTO(
        prediction_id=_digest(_canonical_json(payload)),
        source_vpm_id=source.source_vpm_id,
        baseline=baseline,
        candidates=all_candidates,
        selected_action=selected,
        accepted=accepted,
        policy_id=resolved.policy_id,
        baseline_model_id=baseline_model.model_id,
        transition_model_id=transition_model.model_id,
    )
