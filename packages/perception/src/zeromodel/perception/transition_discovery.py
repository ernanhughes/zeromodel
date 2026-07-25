"""Recurrent unexplained transition discovery for Stage P18C.

P18C materializes a declared discovery cohort from immutable P18A transition
artifacts and P18B conformance reports. It measures recurrence and exact
co-occurrence of unexplained fields, then emits unvalidated candidate components
and falsifiable change hypotheses. Recurrence is association evidence, not
semantic detection, validation, or causal proof.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final, Mapping

from .transition_conformance import TransitionConformanceReportDTO
from .transition_evidence import TransitionEvidenceVPMDTO, TransitionFieldEvidenceDTO

UNEXPLAINED_FIELD_OCCURRENCE_VERSION: Final = (
    "perception-unexplained-field-occurrence/1"
)
TRANSITION_DISCOVERY_OBSERVATION_VERSION: Final = (
    "perception-transition-discovery-observation/1"
)
TRANSITION_DISCOVERY_POLICY_VERSION: Final = (
    "perception-transition-discovery-policy/1"
)
UNEXPLAINED_FIELD_RECURRENCE_VERSION: Final = (
    "perception-unexplained-field-recurrence/1"
)
UNEXPLAINED_SIGNATURE_RECURRENCE_VERSION: Final = (
    "perception-unexplained-signature-recurrence/1"
)
MISSING_COMPONENT_CANDIDATE_VERSION: Final = (
    "perception-missing-component-candidate/1"
)
TRANSITION_DISCOVERY_REPORT_VERSION: Final = (
    "perception-transition-discovery-report/1"
)
TRANSITION_DISCOVERY_SEMANTICS: Final = (
    "recurrence_of_p18b_unexplained_fields_within_one_declared_discovery_cohort"
)
TRANSITION_DISCOVERY_REPORT_STATUSES: Final = {
    "insufficient_evidence",
    "no_candidates",
    "candidates_found",
}
MISSING_COMPONENT_CANDIDATE_KINDS: Final = {
    "field",
    "cooccurrence_signature",
}
MISSING_COMPONENT_HYPOTHESIS_STATUS: Final = "candidate_unvalidated"
PROPOSED_CHANGE_KINDS: Final = {"change", "increase", "decrease"}
DIRECTION_LABELS: Final = {"positive", "negative", "mixed", "neutral"}


class PerceptionTransitionDiscoveryError(ValueError):
    """Raised when recurrent-transition discovery contracts are invalid."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    value = _canonical_json(payload)
    hasher = hashlib.sha256()
    hasher.update(len(value).to_bytes(8, "big"))
    hasher.update(value)
    return f"sha256:{hasher.hexdigest()}"


def _payload(value: object, identity_field: str) -> dict[str, object]:
    payload = asdict(value)  # type: ignore[arg-type]
    payload.pop(identity_field)
    return payload


def _ordered(name: str, values: tuple[str, ...], *, allow_empty: bool = True) -> None:
    if not allow_empty and not values:
        raise PerceptionTransitionDiscoveryError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise PerceptionTransitionDiscoveryError(
            f"{name} must be unique and sorted"
        )


def _unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise PerceptionTransitionDiscoveryError(f"{name} must be in [0, 1]")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerceptionTransitionDiscoveryError(f"{name} must be a positive integer")


def _same_float(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


@dataclass(frozen=True)
class UnexplainedFieldOccurrenceDTO:
    """Exact P18A field evidence referenced by one P18B unexplained finding."""

    occurrence_id: str
    field_id: str
    finding_id: str
    mean_absolute_change: float
    mean_signed_change: float
    changed_fraction: float
    changed_value_count: int
    total_value_count: int
    version: str = UNEXPLAINED_FIELD_OCCURRENCE_VERSION

    def __post_init__(self) -> None:
        if not self.occurrence_id or not self.field_id or not self.finding_id:
            raise PerceptionTransitionDiscoveryError(
                "unexplained occurrence identities must be non-empty"
            )
        _unit("mean_absolute_change", self.mean_absolute_change)
        _unit("changed_fraction", self.changed_fraction)
        if not -1.0 <= self.mean_signed_change <= 1.0:
            raise PerceptionTransitionDiscoveryError(
                "mean_signed_change must be in [-1, 1]"
            )
        _positive_int("total_value_count", self.total_value_count)
        if (
            isinstance(self.changed_value_count, bool)
            or not isinstance(self.changed_value_count, int)
            or not 0 <= self.changed_value_count <= self.total_value_count
        ):
            raise PerceptionTransitionDiscoveryError(
                "changed_value_count must be within total_value_count"
            )
        if not _same_float(
            self.changed_fraction,
            self.changed_value_count / self.total_value_count,
        ):
            raise PerceptionTransitionDiscoveryError(
                "changed_fraction disagrees with occurrence counts"
            )
        if self.version != UNEXPLAINED_FIELD_OCCURRENCE_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported unexplained occurrence version"
            )
        if self.occurrence_id != _digest(_payload(self, "occurrence_id")):
            raise PerceptionTransitionDiscoveryError(
                "unexplained occurrence identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        finding_id: str,
        field: TransitionFieldEvidenceDTO,
    ) -> "UnexplainedFieldOccurrenceDTO":
        values: dict[str, object] = {
            "field_id": field.field_id,
            "finding_id": finding_id,
            "mean_absolute_change": field.mean_absolute_change,
            "mean_signed_change": field.mean_signed_change,
            "changed_fraction": field.changed_fraction,
            "changed_value_count": field.changed_value_count,
            "total_value_count": field.total_value_count,
            "version": UNEXPLAINED_FIELD_OCCURRENCE_VERSION,
        }
        return cls(occurrence_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class TransitionDiscoveryObservationDTO:
    """One interaction in an explicit discovery cohort, including empty evidence."""

    observation_id: str
    interaction_id: str
    cohort_id: str
    field_schema_id: str
    transition_evidence_id: str
    conformance_report_id: str
    unexplained_fields: tuple[UnexplainedFieldOccurrenceDTO, ...]
    version: str = TRANSITION_DISCOVERY_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.observation_id,
                self.interaction_id,
                self.cohort_id,
                self.field_schema_id,
                self.transition_evidence_id,
                self.conformance_report_id,
            )
        ):
            raise PerceptionTransitionDiscoveryError(
                "discovery observation identities must be non-empty"
            )
        field_ids = tuple(item.field_id for item in self.unexplained_fields)
        if field_ids != tuple(sorted(set(field_ids))):
            raise PerceptionTransitionDiscoveryError(
                "unexplained fields must be unique and sorted by field_id"
            )
        occurrence_ids = tuple(item.occurrence_id for item in self.unexplained_fields)
        _ordered("unexplained occurrence identities", occurrence_ids)
        if self.version != TRANSITION_DISCOVERY_OBSERVATION_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported discovery observation version"
            )
        if self.observation_id != _digest(_payload(self, "observation_id")):
            raise PerceptionTransitionDiscoveryError(
                "discovery observation identity disagrees with canonical payload"
            )

    @property
    def unexplained_field_ids(self) -> tuple[str, ...]:
        return tuple(item.field_id for item in self.unexplained_fields)

    @classmethod
    def create(
        cls,
        *,
        interaction_id: str,
        cohort_id: str,
        transition: TransitionEvidenceVPMDTO,
        conformance: TransitionConformanceReportDTO,
    ) -> "TransitionDiscoveryObservationDTO":
        if not interaction_id or not cohort_id:
            raise PerceptionTransitionDiscoveryError(
                "interaction_id and cohort_id must be non-empty"
            )
        if conformance.transition_evidence_id != transition.transition_evidence_id:
            raise PerceptionTransitionDiscoveryError(
                "conformance report does not reference the supplied transition"
            )
        if conformance.field_schema_id != transition.field_schema_id:
            raise PerceptionTransitionDiscoveryError(
                "conformance report field schema does not match transition"
            )
        field_map = {item.field_id: item for item in transition.fields}
        occurrences: list[UnexplainedFieldOccurrenceDTO] = []
        seen_fields: set[str] = set()
        for finding in conformance.findings:
            if finding.status != "unexplained_change":
                continue
            if len(finding.field_ids) != 1:
                raise PerceptionTransitionDiscoveryError(
                    "P18C requires field-addressable unexplained findings"
                )
            field_id = finding.field_ids[0]
            if field_id in seen_fields:
                raise PerceptionTransitionDiscoveryError(
                    "conformance report repeats an unexplained field"
                )
            try:
                field = field_map[field_id]
            except KeyError as exc:
                raise PerceptionTransitionDiscoveryError(
                    f"unexplained finding references unknown field: {field_id}"
                ) from exc
            if not (
                _same_float(
                    finding.observed_mean_absolute_change,
                    field.mean_absolute_change,
                )
                and _same_float(
                    finding.observed_mean_signed_change,
                    field.mean_signed_change,
                )
                and _same_float(
                    finding.observed_changed_fraction,
                    field.changed_fraction,
                )
            ):
                raise PerceptionTransitionDiscoveryError(
                    "unexplained finding metrics disagree with P18A field evidence"
                )
            seen_fields.add(field_id)
            occurrences.append(
                UnexplainedFieldOccurrenceDTO.create(
                    finding_id=finding.finding_id,
                    field=field,
                )
            )
        ordered = tuple(sorted(occurrences, key=lambda item: item.field_id))
        values: dict[str, object] = {
            "interaction_id": interaction_id,
            "cohort_id": cohort_id,
            "field_schema_id": transition.field_schema_id,
            "transition_evidence_id": transition.transition_evidence_id,
            "conformance_report_id": conformance.report_id,
            "unexplained_fields": ordered,
            "version": TRANSITION_DISCOVERY_OBSERVATION_VERSION,
        }
        identity_values = dict(values)
        identity_values["unexplained_fields"] = tuple(asdict(item) for item in ordered)
        return cls(
            observation_id=_digest(identity_values),
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class TransitionDiscoveryPolicyDTO:
    policy_id: str
    minimum_observation_count: int = 3
    minimum_field_occurrence_count: int = 2
    minimum_field_recurrence_fraction: float = 0.5
    minimum_signature_occurrence_count: int = 2
    minimum_signature_recurrence_fraction: float = 0.5
    minimum_direction_consistency: float = 0.75
    direction_epsilon: float = 0.01
    version: str = TRANSITION_DISCOVERY_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise PerceptionTransitionDiscoveryError("policy_id must be non-empty")
        for name in (
            "minimum_observation_count",
            "minimum_field_occurrence_count",
            "minimum_signature_occurrence_count",
        ):
            _positive_int(name, getattr(self, name))
        for name in (
            "minimum_field_recurrence_fraction",
            "minimum_signature_recurrence_fraction",
            "minimum_direction_consistency",
            "direction_epsilon",
        ):
            _unit(name, getattr(self, name))
        if self.version != TRANSITION_DISCOVERY_POLICY_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported transition discovery policy version"
            )
        if self.policy_id != _digest(_payload(self, "policy_id")):
            raise PerceptionTransitionDiscoveryError(
                "transition discovery policy identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        minimum_observation_count: int = 3,
        minimum_field_occurrence_count: int = 2,
        minimum_field_recurrence_fraction: float = 0.5,
        minimum_signature_occurrence_count: int = 2,
        minimum_signature_recurrence_fraction: float = 0.5,
        minimum_direction_consistency: float = 0.75,
        direction_epsilon: float = 0.01,
    ) -> "TransitionDiscoveryPolicyDTO":
        values: dict[str, object] = {
            "minimum_observation_count": minimum_observation_count,
            "minimum_field_occurrence_count": minimum_field_occurrence_count,
            "minimum_field_recurrence_fraction": minimum_field_recurrence_fraction,
            "minimum_signature_occurrence_count": minimum_signature_occurrence_count,
            "minimum_signature_recurrence_fraction": (
                minimum_signature_recurrence_fraction
            ),
            "minimum_direction_consistency": minimum_direction_consistency,
            "direction_epsilon": direction_epsilon,
            "version": TRANSITION_DISCOVERY_POLICY_VERSION,
        }
        return cls(policy_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class UnexplainedFieldRecurrenceDTO:
    statistic_id: str
    field_id: str
    observation_count: int
    occurrence_count: int
    recurrence_fraction: float
    mean_absolute_change: float
    mean_signed_change: float
    mean_changed_fraction: float
    positive_count: int
    negative_count: int
    neutral_count: int
    dominant_direction: str
    direction_consistency: float
    supporting_observation_ids: tuple[str, ...]
    supporting_interaction_ids: tuple[str, ...]
    supporting_transition_evidence_ids: tuple[str, ...]
    supporting_conformance_report_ids: tuple[str, ...]
    supporting_finding_ids: tuple[str, ...]
    version: str = UNEXPLAINED_FIELD_RECURRENCE_VERSION

    def __post_init__(self) -> None:
        if not self.statistic_id or not self.field_id:
            raise PerceptionTransitionDiscoveryError(
                "field recurrence identities must be non-empty"
            )
        _positive_int("observation_count", self.observation_count)
        _positive_int("occurrence_count", self.occurrence_count)
        if self.occurrence_count > self.observation_count:
            raise PerceptionTransitionDiscoveryError(
                "field occurrence_count exceeds observation_count"
            )
        _unit("recurrence_fraction", self.recurrence_fraction)
        _unit("mean_absolute_change", self.mean_absolute_change)
        _unit("mean_changed_fraction", self.mean_changed_fraction)
        _unit("direction_consistency", self.direction_consistency)
        if not -1.0 <= self.mean_signed_change <= 1.0:
            raise PerceptionTransitionDiscoveryError(
                "mean_signed_change must be in [-1, 1]"
            )
        if not _same_float(
            self.recurrence_fraction,
            self.occurrence_count / self.observation_count,
        ):
            raise PerceptionTransitionDiscoveryError(
                "field recurrence_fraction disagrees with counts"
            )
        if self.positive_count + self.negative_count + self.neutral_count != self.occurrence_count:
            raise PerceptionTransitionDiscoveryError(
                "field direction counts disagree with occurrence_count"
            )
        if self.dominant_direction not in DIRECTION_LABELS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported dominant_direction: {self.dominant_direction}"
            )
        for name in (
            "supporting_observation_ids",
            "supporting_interaction_ids",
            "supporting_transition_evidence_ids",
            "supporting_conformance_report_ids",
            "supporting_finding_ids",
        ):
            _ordered(name, getattr(self, name), allow_empty=False)
        if len(self.supporting_observation_ids) != self.occurrence_count:
            raise PerceptionTransitionDiscoveryError(
                "field supporting observation count disagrees with occurrence_count"
            )
        if self.version != UNEXPLAINED_FIELD_RECURRENCE_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported field recurrence version"
            )
        if self.statistic_id != _digest(_payload(self, "statistic_id")):
            raise PerceptionTransitionDiscoveryError(
                "field recurrence identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class UnexplainedSignatureRecurrenceDTO:
    statistic_id: str
    field_ids: tuple[str, ...]
    observation_count: int
    occurrence_count: int
    recurrence_fraction: float
    mean_absolute_change: float
    mean_signed_change: float
    mean_changed_fraction: float
    positive_count: int
    negative_count: int
    neutral_count: int
    dominant_direction: str
    direction_consistency: float
    supporting_observation_ids: tuple[str, ...]
    supporting_interaction_ids: tuple[str, ...]
    supporting_transition_evidence_ids: tuple[str, ...]
    supporting_conformance_report_ids: tuple[str, ...]
    supporting_finding_ids: tuple[str, ...]
    version: str = UNEXPLAINED_SIGNATURE_RECURRENCE_VERSION

    def __post_init__(self) -> None:
        if not self.statistic_id:
            raise PerceptionTransitionDiscoveryError(
                "signature recurrence identity must be non-empty"
            )
        _ordered("signature field_ids", self.field_ids, allow_empty=False)
        if len(self.field_ids) < 2:
            raise PerceptionTransitionDiscoveryError(
                "signature recurrence requires at least two fields"
            )
        _positive_int("observation_count", self.observation_count)
        _positive_int("occurrence_count", self.occurrence_count)
        if self.occurrence_count > self.observation_count:
            raise PerceptionTransitionDiscoveryError(
                "signature occurrence_count exceeds observation_count"
            )
        _unit("recurrence_fraction", self.recurrence_fraction)
        _unit("mean_absolute_change", self.mean_absolute_change)
        _unit("mean_changed_fraction", self.mean_changed_fraction)
        _unit("direction_consistency", self.direction_consistency)
        if not -1.0 <= self.mean_signed_change <= 1.0:
            raise PerceptionTransitionDiscoveryError(
                "mean_signed_change must be in [-1, 1]"
            )
        if not _same_float(
            self.recurrence_fraction,
            self.occurrence_count / self.observation_count,
        ):
            raise PerceptionTransitionDiscoveryError(
                "signature recurrence_fraction disagrees with counts"
            )
        if self.positive_count + self.negative_count + self.neutral_count != self.occurrence_count:
            raise PerceptionTransitionDiscoveryError(
                "signature direction counts disagree with occurrence_count"
            )
        if self.dominant_direction not in DIRECTION_LABELS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported dominant_direction: {self.dominant_direction}"
            )
        for name in (
            "supporting_observation_ids",
            "supporting_interaction_ids",
            "supporting_transition_evidence_ids",
            "supporting_conformance_report_ids",
            "supporting_finding_ids",
        ):
            _ordered(name, getattr(self, name), allow_empty=False)
        if len(self.supporting_observation_ids) != self.occurrence_count:
            raise PerceptionTransitionDiscoveryError(
                "signature supporting observation count disagrees with occurrence_count"
            )
        if self.version != UNEXPLAINED_SIGNATURE_RECURRENCE_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported signature recurrence version"
            )
        if self.statistic_id != _digest(_payload(self, "statistic_id")):
            raise PerceptionTransitionDiscoveryError(
                "signature recurrence identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class MissingComponentCandidateDTO:
    candidate_id: str
    candidate_kind: str
    source_statistic_id: str
    field_ids: tuple[str, ...]
    occurrence_count: int
    observation_count: int
    recurrence_fraction: float
    proposed_expected_change: str
    dominant_direction: str
    direction_consistency: float
    supporting_observation_ids: tuple[str, ...]
    hypothesis_status: str = MISSING_COMPONENT_HYPOTHESIS_STATUS
    version: str = MISSING_COMPONENT_CANDIDATE_VERSION

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source_statistic_id:
            raise PerceptionTransitionDiscoveryError(
                "missing-component candidate identities must be non-empty"
            )
        if self.candidate_kind not in MISSING_COMPONENT_CANDIDATE_KINDS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported candidate_kind: {self.candidate_kind}"
            )
        _ordered("candidate field_ids", self.field_ids, allow_empty=False)
        if self.candidate_kind == "field" and len(self.field_ids) != 1:
            raise PerceptionTransitionDiscoveryError(
                "field candidates require exactly one field"
            )
        if self.candidate_kind == "cooccurrence_signature" and len(self.field_ids) < 2:
            raise PerceptionTransitionDiscoveryError(
                "cooccurrence candidates require at least two fields"
            )
        _positive_int("occurrence_count", self.occurrence_count)
        _positive_int("observation_count", self.observation_count)
        if self.occurrence_count > self.observation_count:
            raise PerceptionTransitionDiscoveryError(
                "candidate occurrence_count exceeds observation_count"
            )
        _unit("recurrence_fraction", self.recurrence_fraction)
        _unit("direction_consistency", self.direction_consistency)
        if self.proposed_expected_change not in PROPOSED_CHANGE_KINDS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported proposed_expected_change: {self.proposed_expected_change}"
            )
        if self.dominant_direction not in DIRECTION_LABELS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported dominant_direction: {self.dominant_direction}"
            )
        _ordered(
            "candidate supporting_observation_ids",
            self.supporting_observation_ids,
            allow_empty=False,
        )
        if len(self.supporting_observation_ids) != self.occurrence_count:
            raise PerceptionTransitionDiscoveryError(
                "candidate supporting observations disagree with occurrence_count"
            )
        if self.hypothesis_status != MISSING_COMPONENT_HYPOTHESIS_STATUS:
            raise PerceptionTransitionDiscoveryError(
                "unsupported missing-component hypothesis status"
            )
        if self.version != MISSING_COMPONENT_CANDIDATE_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported missing-component candidate version"
            )
        if self.candidate_id != _digest(_payload(self, "candidate_id")):
            raise PerceptionTransitionDiscoveryError(
                "missing-component candidate identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class TransitionDiscoveryReportDTO:
    report_id: str
    status: str
    cohort_id: str
    field_schema_id: str
    policy_id: str
    observation_ids: tuple[str, ...]
    interaction_ids: tuple[str, ...]
    transition_evidence_ids: tuple[str, ...]
    conformance_report_ids: tuple[str, ...]
    field_statistics: tuple[UnexplainedFieldRecurrenceDTO, ...]
    signature_statistics: tuple[UnexplainedSignatureRecurrenceDTO, ...]
    candidates: tuple[MissingComponentCandidateDTO, ...]
    semantics: str = TRANSITION_DISCOVERY_SEMANTICS
    version: str = TRANSITION_DISCOVERY_REPORT_VERSION

    def __post_init__(self) -> None:
        if not all((self.report_id, self.cohort_id, self.field_schema_id, self.policy_id)):
            raise PerceptionTransitionDiscoveryError(
                "transition discovery report identities must be non-empty"
            )
        if self.status not in TRANSITION_DISCOVERY_REPORT_STATUSES:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported discovery report status: {self.status}"
            )
        for name in (
            "observation_ids",
            "interaction_ids",
            "transition_evidence_ids",
            "conformance_report_ids",
        ):
            _ordered(name, getattr(self, name), allow_empty=False)
        count = len(self.observation_ids)
        if not (
            len(self.interaction_ids)
            == len(self.transition_evidence_ids)
            == len(self.conformance_report_ids)
            == count
        ):
            raise PerceptionTransitionDiscoveryError(
                "report evidence identity counts must match observations"
            )
        field_statistic_ids = tuple(item.statistic_id for item in self.field_statistics)
        signature_statistic_ids = tuple(
            item.statistic_id for item in self.signature_statistics
        )
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        _ordered("field statistic identities", field_statistic_ids)
        _ordered("signature statistic identities", signature_statistic_ids)
        _ordered("candidate identities", candidate_ids)
        expected_status = (
            "candidates_found" if self.candidates else "no_candidates"
        )
        if self.status != "insufficient_evidence" and self.status != expected_status:
            raise PerceptionTransitionDiscoveryError(
                "discovery report status disagrees with candidates"
            )
        if self.status == "insufficient_evidence" and self.candidates:
            raise PerceptionTransitionDiscoveryError(
                "insufficient-evidence reports cannot contain candidates"
            )
        known_statistics = set(field_statistic_ids) | set(signature_statistic_ids)
        if any(item.source_statistic_id not in known_statistics for item in self.candidates):
            raise PerceptionTransitionDiscoveryError(
                "candidate references unknown recurrence statistic"
            )
        if self.semantics != TRANSITION_DISCOVERY_SEMANTICS:
            raise PerceptionTransitionDiscoveryError(
                "unsupported transition discovery semantics"
            )
        if self.version != TRANSITION_DISCOVERY_REPORT_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported transition discovery report version"
            )
        if self.report_id != _digest(_payload(self, "report_id")):
            raise PerceptionTransitionDiscoveryError(
                "transition discovery report identity disagrees with canonical payload"
            )

    def candidates_for_kind(
        self,
        candidate_kind: str,
    ) -> tuple[MissingComponentCandidateDTO, ...]:
        if candidate_kind not in MISSING_COMPONENT_CANDIDATE_KINDS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported candidate_kind: {candidate_kind}"
            )
        return tuple(
            item for item in self.candidates if item.candidate_kind == candidate_kind
        )


def _direction_counts(
    signed_values: tuple[float, ...],
    epsilon: float,
) -> tuple[int, int, int, str, float]:
    positive = sum(value > epsilon for value in signed_values)
    negative = sum(value < -epsilon for value in signed_values)
    neutral = len(signed_values) - positive - negative
    counts = {"positive": positive, "negative": negative, "neutral": neutral}
    maximum = max(counts.values())
    leaders = tuple(sorted(name for name, count in counts.items() if count == maximum))
    dominant = leaders[0] if len(leaders) == 1 else "mixed"
    return positive, negative, neutral, dominant, maximum / len(signed_values)


def _aggregate_occurrence_groups(
    groups: tuple[tuple[UnexplainedFieldOccurrenceDTO, ...], ...],
    epsilon: float,
) -> tuple[float, float, float, int, int, int, str, float]:
    total_values = 0
    absolute_total = 0.0
    signed_total = 0.0
    changed_values = 0
    per_observation_signed: list[float] = []
    for group in groups:
        group_total = sum(item.total_value_count for item in group)
        if group_total <= 0:
            raise PerceptionTransitionDiscoveryError(
                "recurrence evidence group has no measurable values"
            )
        total_values += group_total
        absolute_total += sum(
            item.mean_absolute_change * item.total_value_count for item in group
        )
        signed_value = sum(
            item.mean_signed_change * item.total_value_count for item in group
        ) / group_total
        signed_total += signed_value * group_total
        changed_values += sum(item.changed_value_count for item in group)
        per_observation_signed.append(signed_value)
    positive, negative, neutral, dominant, consistency = _direction_counts(
        tuple(per_observation_signed),
        epsilon,
    )
    return (
        absolute_total / total_values,
        signed_total / total_values,
        changed_values / total_values,
        positive,
        negative,
        neutral,
        dominant,
        consistency,
    )


def _support_values(
    observations: tuple[TransitionDiscoveryObservationDTO, ...],
    field_ids: tuple[str, ...],
) -> tuple[
    tuple[tuple[UnexplainedFieldOccurrenceDTO, ...], ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    field_set = set(field_ids)
    groups: list[tuple[UnexplainedFieldOccurrenceDTO, ...]] = []
    selected: list[TransitionDiscoveryObservationDTO] = []
    finding_ids: set[str] = set()
    for observation in observations:
        matches = tuple(
            item for item in observation.unexplained_fields if item.field_id in field_set
        )
        if len(matches) != len(field_ids):
            continue
        groups.append(matches)
        selected.append(observation)
        finding_ids.update(item.finding_id for item in matches)
    return (
        tuple(groups),
        tuple(sorted(item.observation_id for item in selected)),
        tuple(sorted(item.interaction_id for item in selected)),
        tuple(sorted(item.transition_evidence_id for item in selected)),
        tuple(sorted(item.conformance_report_id for item in selected)),
        tuple(sorted(finding_ids)),
    )


def _field_statistic(
    field_id: str,
    observations: tuple[TransitionDiscoveryObservationDTO, ...],
    policy: TransitionDiscoveryPolicyDTO,
) -> UnexplainedFieldRecurrenceDTO:
    groups, observation_ids, interaction_ids, transition_ids, report_ids, finding_ids = (
        _support_values(observations, (field_id,))
    )
    metrics = _aggregate_occurrence_groups(groups, policy.direction_epsilon)
    values: dict[str, object] = {
        "field_id": field_id,
        "observation_count": len(observations),
        "occurrence_count": len(groups),
        "recurrence_fraction": len(groups) / len(observations),
        "mean_absolute_change": metrics[0],
        "mean_signed_change": metrics[1],
        "mean_changed_fraction": metrics[2],
        "positive_count": metrics[3],
        "negative_count": metrics[4],
        "neutral_count": metrics[5],
        "dominant_direction": metrics[6],
        "direction_consistency": metrics[7],
        "supporting_observation_ids": observation_ids,
        "supporting_interaction_ids": interaction_ids,
        "supporting_transition_evidence_ids": transition_ids,
        "supporting_conformance_report_ids": report_ids,
        "supporting_finding_ids": finding_ids,
        "version": UNEXPLAINED_FIELD_RECURRENCE_VERSION,
    }
    return UnexplainedFieldRecurrenceDTO(
        statistic_id=_digest(values), **values  # type: ignore[arg-type]
    )


def _signature_statistic(
    field_ids: tuple[str, ...],
    supporting: tuple[TransitionDiscoveryObservationDTO, ...],
    observation_count: int,
    policy: TransitionDiscoveryPolicyDTO,
) -> UnexplainedSignatureRecurrenceDTO:
    groups = tuple(item.unexplained_fields for item in supporting)
    metrics = _aggregate_occurrence_groups(groups, policy.direction_epsilon)
    finding_ids = tuple(
        sorted(
            {
                occurrence.finding_id
                for observation in supporting
                for occurrence in observation.unexplained_fields
            }
        )
    )
    values: dict[str, object] = {
        "field_ids": field_ids,
        "observation_count": observation_count,
        "occurrence_count": len(supporting),
        "recurrence_fraction": len(supporting) / observation_count,
        "mean_absolute_change": metrics[0],
        "mean_signed_change": metrics[1],
        "mean_changed_fraction": metrics[2],
        "positive_count": metrics[3],
        "negative_count": metrics[4],
        "neutral_count": metrics[5],
        "dominant_direction": metrics[6],
        "direction_consistency": metrics[7],
        "supporting_observation_ids": tuple(
            sorted(item.observation_id for item in supporting)
        ),
        "supporting_interaction_ids": tuple(
            sorted(item.interaction_id for item in supporting)
        ),
        "supporting_transition_evidence_ids": tuple(
            sorted(item.transition_evidence_id for item in supporting)
        ),
        "supporting_conformance_report_ids": tuple(
            sorted(item.conformance_report_id for item in supporting)
        ),
        "supporting_finding_ids": finding_ids,
        "version": UNEXPLAINED_SIGNATURE_RECURRENCE_VERSION,
    }
    return UnexplainedSignatureRecurrenceDTO(
        statistic_id=_digest(values), **values  # type: ignore[arg-type]
    )


def _proposed_change(
    dominant_direction: str,
    direction_consistency: float,
    policy: TransitionDiscoveryPolicyDTO,
) -> str:
    if direction_consistency < policy.minimum_direction_consistency:
        return "change"
    if dominant_direction == "positive":
        return "increase"
    if dominant_direction == "negative":
        return "decrease"
    return "change"


def _candidate(
    *,
    candidate_kind: str,
    source_statistic_id: str,
    field_ids: tuple[str, ...],
    occurrence_count: int,
    observation_count: int,
    recurrence_fraction: float,
    dominant_direction: str,
    direction_consistency: float,
    supporting_observation_ids: tuple[str, ...],
    policy: TransitionDiscoveryPolicyDTO,
) -> MissingComponentCandidateDTO:
    values: dict[str, object] = {
        "candidate_kind": candidate_kind,
        "source_statistic_id": source_statistic_id,
        "field_ids": field_ids,
        "occurrence_count": occurrence_count,
        "observation_count": observation_count,
        "recurrence_fraction": recurrence_fraction,
        "proposed_expected_change": _proposed_change(
            dominant_direction,
            direction_consistency,
            policy,
        ),
        "dominant_direction": dominant_direction,
        "direction_consistency": direction_consistency,
        "supporting_observation_ids": supporting_observation_ids,
        "hypothesis_status": MISSING_COMPONENT_HYPOTHESIS_STATUS,
        "version": MISSING_COMPONENT_CANDIDATE_VERSION,
    }
    return MissingComponentCandidateDTO(
        candidate_id=_digest(values), **values  # type: ignore[arg-type]
    )


def discover_recurrent_unexplained_transitions(
    observations: tuple[TransitionDiscoveryObservationDTO, ...],
    policy: TransitionDiscoveryPolicyDTO | None = None,
) -> TransitionDiscoveryReportDTO:
    """Discover recurrent unexplained field hypotheses within one cohort."""

    if not observations:
        raise PerceptionTransitionDiscoveryError(
            "transition discovery requires at least one observation"
        )
    resolved = policy or TransitionDiscoveryPolicyDTO.create()
    observation_map = {item.observation_id: item for item in observations}
    interaction_map = {item.interaction_id: item for item in observations}
    transition_map = {item.transition_evidence_id: item for item in observations}
    report_map = {item.conformance_report_id: item for item in observations}
    if len(observation_map) != len(observations):
        raise PerceptionTransitionDiscoveryError(
            "discovery observations must have unique identities"
        )
    if len(interaction_map) != len(observations):
        raise PerceptionTransitionDiscoveryError(
            "discovery interactions must have unique identities"
        )
    if len(transition_map) != len(observations):
        raise PerceptionTransitionDiscoveryError(
            "discovery transitions must have unique identities"
        )
    if len(report_map) != len(observations):
        raise PerceptionTransitionDiscoveryError(
            "discovery conformance reports must have unique identities"
        )
    cohort_ids = {item.cohort_id for item in observations}
    schema_ids = {item.field_schema_id for item in observations}
    if len(cohort_ids) != 1:
        raise PerceptionTransitionDiscoveryError(
            "discovery observations must belong to one cohort"
        )
    if len(schema_ids) != 1:
        raise PerceptionTransitionDiscoveryError(
            "discovery observations must use one field schema"
        )
    ordered_observations = tuple(
        sorted(observations, key=lambda item: item.observation_id)
    )
    all_field_ids = tuple(
        sorted(
            {
                occurrence.field_id
                for observation in ordered_observations
                for occurrence in observation.unexplained_fields
            }
        )
    )
    field_statistics = tuple(
        sorted(
            (
                _field_statistic(field_id, ordered_observations, resolved)
                for field_id in all_field_ids
            ),
            key=lambda item: item.statistic_id,
        )
    )
    signature_support: dict[
        tuple[str, ...], list[TransitionDiscoveryObservationDTO]
    ] = {}
    for observation in ordered_observations:
        signature = observation.unexplained_field_ids
        if len(signature) >= 2:
            signature_support.setdefault(signature, []).append(observation)
    signature_statistics = tuple(
        sorted(
            (
                _signature_statistic(
                    signature,
                    tuple(supporting),
                    len(ordered_observations),
                    resolved,
                )
                for signature, supporting in signature_support.items()
            ),
            key=lambda item: item.statistic_id,
        )
    )

    candidates: list[MissingComponentCandidateDTO] = []
    evidence_ready = len(ordered_observations) >= resolved.minimum_observation_count
    if evidence_ready:
        for statistic in field_statistics:
            if (
                statistic.occurrence_count
                >= resolved.minimum_field_occurrence_count
                and statistic.recurrence_fraction
                >= resolved.minimum_field_recurrence_fraction
            ):
                candidates.append(
                    _candidate(
                        candidate_kind="field",
                        source_statistic_id=statistic.statistic_id,
                        field_ids=(statistic.field_id,),
                        occurrence_count=statistic.occurrence_count,
                        observation_count=statistic.observation_count,
                        recurrence_fraction=statistic.recurrence_fraction,
                        dominant_direction=statistic.dominant_direction,
                        direction_consistency=statistic.direction_consistency,
                        supporting_observation_ids=(
                            statistic.supporting_observation_ids
                        ),
                        policy=resolved,
                    )
                )
        for statistic in signature_statistics:
            if (
                statistic.occurrence_count
                >= resolved.minimum_signature_occurrence_count
                and statistic.recurrence_fraction
                >= resolved.minimum_signature_recurrence_fraction
            ):
                candidates.append(
                    _candidate(
                        candidate_kind="cooccurrence_signature",
                        source_statistic_id=statistic.statistic_id,
                        field_ids=statistic.field_ids,
                        occurrence_count=statistic.occurrence_count,
                        observation_count=statistic.observation_count,
                        recurrence_fraction=statistic.recurrence_fraction,
                        dominant_direction=statistic.dominant_direction,
                        direction_consistency=statistic.direction_consistency,
                        supporting_observation_ids=(
                            statistic.supporting_observation_ids
                        ),
                        policy=resolved,
                    )
                )
    ordered_candidates = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    if not evidence_ready:
        status = "insufficient_evidence"
    elif ordered_candidates:
        status = "candidates_found"
    else:
        status = "no_candidates"
    values: dict[str, object] = {
        "status": status,
        "cohort_id": next(iter(cohort_ids)),
        "field_schema_id": next(iter(schema_ids)),
        "policy_id": resolved.policy_id,
        "observation_ids": tuple(sorted(observation_map)),
        "interaction_ids": tuple(sorted(interaction_map)),
        "transition_evidence_ids": tuple(sorted(transition_map)),
        "conformance_report_ids": tuple(sorted(report_map)),
        "field_statistics": field_statistics,
        "signature_statistics": signature_statistics,
        "candidates": ordered_candidates,
        "semantics": TRANSITION_DISCOVERY_SEMANTICS,
        "version": TRANSITION_DISCOVERY_REPORT_VERSION,
    }
    identity_values = dict(values)
    identity_values["field_statistics"] = tuple(
        asdict(item) for item in field_statistics
    )
    identity_values["signature_statistics"] = tuple(
        asdict(item) for item in signature_statistics
    )
    identity_values["candidates"] = tuple(asdict(item) for item in ordered_candidates)
    return TransitionDiscoveryReportDTO(
        report_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )
