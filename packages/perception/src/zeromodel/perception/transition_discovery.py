"""Recurrent unexplained transition discovery for Stage P18C.

P18C aggregates immutable P18A evidence and P18B unexplained findings inside one
explicit discovery cohort. It produces thresholded, content-addressed candidate
components and falsifiable change hypotheses. Candidates remain unvalidated
associations; they are not semantic detections or causal conclusions.
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
RECURRENT_UNEXPLAINED_STATISTIC_VERSION: Final = (
    "perception-recurrent-unexplained-statistic/1"
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
RECURRENT_UNEXPLAINED_STATISTIC_KINDS: Final = {
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


def _ordered_unique(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and not values:
        raise PerceptionTransitionDiscoveryError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise PerceptionTransitionDiscoveryError(
            f"{name} must be unique and sorted"
        )


def _ordered_with_repetition(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and not values:
        raise PerceptionTransitionDiscoveryError(f"{name} must be non-empty")
    if values != tuple(sorted(values)):
        raise PerceptionTransitionDiscoveryError(f"{name} must be sorted")


def _unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise PerceptionTransitionDiscoveryError(f"{name} must be in [0, 1]")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerceptionTransitionDiscoveryError(
            f"{name} must be a positive integer"
        )


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
    """One interaction in a discovery cohort, including zero-finding controls."""

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
        if len(occurrence_ids) != len(set(occurrence_ids)):
            raise PerceptionTransitionDiscoveryError(
                "unexplained occurrence identities must be unique"
            )
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
        identity_values["unexplained_fields"] = tuple(
            asdict(item) for item in ordered
        )
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
class RecurrentUnexplainedStatisticDTO:
    statistic_id: str
    statistic_kind: str
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
    version: str = RECURRENT_UNEXPLAINED_STATISTIC_VERSION

    def __post_init__(self) -> None:
        if not self.statistic_id:
            raise PerceptionTransitionDiscoveryError(
                "recurrence statistic identity must be non-empty"
            )
        if self.statistic_kind not in RECURRENT_UNEXPLAINED_STATISTIC_KINDS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported statistic_kind: {self.statistic_kind}"
            )
        _ordered_unique("statistic field_ids", self.field_ids, allow_empty=False)
        if self.statistic_kind == "field" and len(self.field_ids) != 1:
            raise PerceptionTransitionDiscoveryError(
                "field statistics require exactly one field"
            )
        if self.statistic_kind == "cooccurrence_signature" and len(self.field_ids) < 2:
            raise PerceptionTransitionDiscoveryError(
                "cooccurrence statistics require at least two fields"
            )
        _positive_int("observation_count", self.observation_count)
        _positive_int("occurrence_count", self.occurrence_count)
        if self.occurrence_count > self.observation_count:
            raise PerceptionTransitionDiscoveryError(
                "occurrence_count exceeds observation_count"
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
                "recurrence_fraction disagrees with counts"
            )
        if (
            self.positive_count + self.negative_count + self.neutral_count
            != self.occurrence_count
        ):
            raise PerceptionTransitionDiscoveryError(
                "direction counts disagree with occurrence_count"
            )
        if self.dominant_direction not in DIRECTION_LABELS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported dominant_direction: {self.dominant_direction}"
            )
        _ordered_unique(
            "supporting_observation_ids",
            self.supporting_observation_ids,
            allow_empty=False,
        )
        _ordered_unique(
            "supporting_interaction_ids",
            self.supporting_interaction_ids,
            allow_empty=False,
        )
        _ordered_with_repetition(
            "supporting_transition_evidence_ids",
            self.supporting_transition_evidence_ids,
            allow_empty=False,
        )
        _ordered_with_repetition(
            "supporting_conformance_report_ids",
            self.supporting_conformance_report_ids,
            allow_empty=False,
        )
        _ordered_unique(
            "supporting_finding_ids",
            self.supporting_finding_ids,
            allow_empty=False,
        )
        if not (
            len(self.supporting_observation_ids)
            == len(self.supporting_interaction_ids)
            == len(self.supporting_transition_evidence_ids)
            == len(self.supporting_conformance_report_ids)
            == self.occurrence_count
        ):
            raise PerceptionTransitionDiscoveryError(
                "supporting evidence counts disagree with occurrence_count"
            )
        if self.version != RECURRENT_UNEXPLAINED_STATISTIC_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported recurrence statistic version"
            )
        if self.statistic_id != _digest(_payload(self, "statistic_id")):
            raise PerceptionTransitionDiscoveryError(
                "recurrence statistic identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class MissingComponentCandidateDTO:
    candidate_id: str
    source_statistic_id: str
    candidate_kind: str
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
                "candidate identities must be non-empty"
            )
        if self.candidate_kind not in RECURRENT_UNEXPLAINED_STATISTIC_KINDS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported candidate_kind: {self.candidate_kind}"
            )
        _ordered_unique("candidate field_ids", self.field_ids, allow_empty=False)
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
        _ordered_unique(
            "candidate supporting_observation_ids",
            self.supporting_observation_ids,
            allow_empty=False,
        )
        if len(self.supporting_observation_ids) != self.occurrence_count:
            raise PerceptionTransitionDiscoveryError(
                "candidate support count disagrees with occurrence_count"
            )
        if self.hypothesis_status != MISSING_COMPONENT_HYPOTHESIS_STATUS:
            raise PerceptionTransitionDiscoveryError(
                "unsupported candidate hypothesis status"
            )
        if self.version != MISSING_COMPONENT_CANDIDATE_VERSION:
            raise PerceptionTransitionDiscoveryError(
                "unsupported missing-component candidate version"
            )
        if self.candidate_id != _digest(_payload(self, "candidate_id")):
            raise PerceptionTransitionDiscoveryError(
                "candidate identity disagrees with canonical payload"
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
    statistics: tuple[RecurrentUnexplainedStatisticDTO, ...]
    candidates: tuple[MissingComponentCandidateDTO, ...]
    semantics: str = TRANSITION_DISCOVERY_SEMANTICS
    version: str = TRANSITION_DISCOVERY_REPORT_VERSION

    def __post_init__(self) -> None:
        if not all((self.report_id, self.cohort_id, self.field_schema_id, self.policy_id)):
            raise PerceptionTransitionDiscoveryError(
                "discovery report identities must be non-empty"
            )
        if self.status not in TRANSITION_DISCOVERY_REPORT_STATUSES:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported discovery report status: {self.status}"
            )
        _ordered_unique("observation_ids", self.observation_ids, allow_empty=False)
        _ordered_unique("interaction_ids", self.interaction_ids, allow_empty=False)
        _ordered_with_repetition(
            "transition_evidence_ids",
            self.transition_evidence_ids,
            allow_empty=False,
        )
        _ordered_with_repetition(
            "conformance_report_ids",
            self.conformance_report_ids,
            allow_empty=False,
        )
        count = len(self.observation_ids)
        if not (
            len(self.interaction_ids)
            == len(self.transition_evidence_ids)
            == len(self.conformance_report_ids)
            == count
        ):
            raise PerceptionTransitionDiscoveryError(
                "report evidence counts must match observations"
            )
        statistic_ids = tuple(item.statistic_id for item in self.statistics)
        candidate_ids = tuple(item.candidate_id for item in self.candidates)
        _ordered_unique("statistic identities", statistic_ids)
        _ordered_unique("candidate identities", candidate_ids)
        if self.status == "insufficient_evidence" and self.candidates:
            raise PerceptionTransitionDiscoveryError(
                "insufficient-evidence reports cannot contain candidates"
            )
        if self.status == "candidates_found" and not self.candidates:
            raise PerceptionTransitionDiscoveryError(
                "candidates_found reports require candidates"
            )
        if self.status == "no_candidates" and self.candidates:
            raise PerceptionTransitionDiscoveryError(
                "no_candidates reports cannot contain candidates"
            )
        known_statistics = set(statistic_ids)
        if any(
            item.source_statistic_id not in known_statistics
            for item in self.candidates
        ):
            raise PerceptionTransitionDiscoveryError(
                "candidate references an unknown recurrence statistic"
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
                "discovery report identity disagrees with canonical payload"
            )

    def candidates_for_kind(
        self,
        candidate_kind: str,
    ) -> tuple[MissingComponentCandidateDTO, ...]:
        if candidate_kind not in RECURRENT_UNEXPLAINED_STATISTIC_KINDS:
            raise PerceptionTransitionDiscoveryError(
                f"unsupported candidate_kind: {candidate_kind}"
            )
        return tuple(
            item for item in self.candidates if item.candidate_kind == candidate_kind
        )


def _direction_summary(
    values: tuple[float, ...],
    epsilon: float,
) -> tuple[int, int, int, str, float]:
    positive = sum(item > epsilon for item in values)
    negative = sum(item < -epsilon for item in values)
    neutral = len(values) - positive - negative
    counts = {"positive": positive, "negative": negative, "neutral": neutral}
    maximum = max(counts.values())
    leaders = tuple(sorted(name for name, count in counts.items() if count == maximum))
    dominant = leaders[0] if len(leaders) == 1 else "mixed"
    return positive, negative, neutral, dominant, maximum / len(values)


def _build_statistic(
    *,
    statistic_kind: str,
    field_ids: tuple[str, ...],
    supporting: tuple[TransitionDiscoveryObservationDTO, ...],
    observation_count: int,
    policy: TransitionDiscoveryPolicyDTO,
) -> RecurrentUnexplainedStatisticDTO:
    field_set = set(field_ids)
    total_values = 0
    absolute_total = 0.0
    signed_total = 0.0
    changed_values = 0
    observation_signed: list[float] = []
    finding_ids: set[str] = set()
    for observation in supporting:
        occurrences = tuple(
            item for item in observation.unexplained_fields if item.field_id in field_set
        )
        if len(occurrences) != len(field_ids):
            raise PerceptionTransitionDiscoveryError(
                "supporting observation does not contain every statistic field"
            )
        group_values = sum(item.total_value_count for item in occurrences)
        group_signed = sum(
            item.mean_signed_change * item.total_value_count
            for item in occurrences
        ) / group_values
        total_values += group_values
        absolute_total += sum(
            item.mean_absolute_change * item.total_value_count
            for item in occurrences
        )
        signed_total += group_signed * group_values
        changed_values += sum(item.changed_value_count for item in occurrences)
        observation_signed.append(group_signed)
        finding_ids.update(item.finding_id for item in occurrences)
    direction = _direction_summary(tuple(observation_signed), policy.direction_epsilon)
    values: dict[str, object] = {
        "statistic_kind": statistic_kind,
        "field_ids": field_ids,
        "observation_count": observation_count,
        "occurrence_count": len(supporting),
        "recurrence_fraction": len(supporting) / observation_count,
        "mean_absolute_change": absolute_total / total_values,
        "mean_signed_change": signed_total / total_values,
        "mean_changed_fraction": changed_values / total_values,
        "positive_count": direction[0],
        "negative_count": direction[1],
        "neutral_count": direction[2],
        "dominant_direction": direction[3],
        "direction_consistency": direction[4],
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
        "supporting_finding_ids": tuple(sorted(finding_ids)),
        "version": RECURRENT_UNEXPLAINED_STATISTIC_VERSION,
    }
    return RecurrentUnexplainedStatisticDTO(
        statistic_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )


def _proposed_change(
    statistic: RecurrentUnexplainedStatisticDTO,
    policy: TransitionDiscoveryPolicyDTO,
) -> str:
    if statistic.direction_consistency < policy.minimum_direction_consistency:
        return "change"
    if statistic.dominant_direction == "positive":
        return "increase"
    if statistic.dominant_direction == "negative":
        return "decrease"
    return "change"


def _candidate(
    statistic: RecurrentUnexplainedStatisticDTO,
    policy: TransitionDiscoveryPolicyDTO,
) -> MissingComponentCandidateDTO:
    values: dict[str, object] = {
        "source_statistic_id": statistic.statistic_id,
        "candidate_kind": statistic.statistic_kind,
        "field_ids": statistic.field_ids,
        "occurrence_count": statistic.occurrence_count,
        "observation_count": statistic.observation_count,
        "recurrence_fraction": statistic.recurrence_fraction,
        "proposed_expected_change": _proposed_change(statistic, policy),
        "dominant_direction": statistic.dominant_direction,
        "direction_consistency": statistic.direction_consistency,
        "supporting_observation_ids": statistic.supporting_observation_ids,
        "hypothesis_status": MISSING_COMPONENT_HYPOTHESIS_STATUS,
        "version": MISSING_COMPONENT_CANDIDATE_VERSION,
    }
    return MissingComponentCandidateDTO(
        candidate_id=_digest(values),
        **values,  # type: ignore[arg-type]
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
    if len({item.observation_id for item in observations}) != len(observations):
        raise PerceptionTransitionDiscoveryError(
            "discovery observations must have unique identities"
        )
    if len({item.interaction_id for item in observations}) != len(observations):
        raise PerceptionTransitionDiscoveryError(
            "discovery interactions must have unique identities"
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
    ordered = tuple(sorted(observations, key=lambda item: item.observation_id))
    all_field_ids = tuple(
        sorted(
            {
                occurrence.field_id
                for observation in ordered
                for occurrence in observation.unexplained_fields
            }
        )
    )
    statistics: list[RecurrentUnexplainedStatisticDTO] = []
    for field_id in all_field_ids:
        supporting = tuple(
            item for item in ordered if field_id in item.unexplained_field_ids
        )
        statistics.append(
            _build_statistic(
                statistic_kind="field",
                field_ids=(field_id,),
                supporting=supporting,
                observation_count=len(ordered),
                policy=resolved,
            )
        )
    signatures = tuple(
        sorted(
            {
                item.unexplained_field_ids
                for item in ordered
                if len(item.unexplained_field_ids) >= 2
            }
        )
    )
    for signature in signatures:
        supporting = tuple(
            item for item in ordered if item.unexplained_field_ids == signature
        )
        statistics.append(
            _build_statistic(
                statistic_kind="cooccurrence_signature",
                field_ids=signature,
                supporting=supporting,
                observation_count=len(ordered),
                policy=resolved,
            )
        )
    ordered_statistics = tuple(
        sorted(statistics, key=lambda item: item.statistic_id)
    )
    evidence_ready = len(ordered) >= resolved.minimum_observation_count
    candidates: list[MissingComponentCandidateDTO] = []
    if evidence_ready:
        for statistic in ordered_statistics:
            if statistic.statistic_kind == "field":
                qualifies = (
                    statistic.occurrence_count
                    >= resolved.minimum_field_occurrence_count
                    and statistic.recurrence_fraction
                    >= resolved.minimum_field_recurrence_fraction
                )
            else:
                qualifies = (
                    statistic.occurrence_count
                    >= resolved.minimum_signature_occurrence_count
                    and statistic.recurrence_fraction
                    >= resolved.minimum_signature_recurrence_fraction
                )
            if qualifies:
                candidates.append(_candidate(statistic, resolved))
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
        "observation_ids": tuple(sorted(item.observation_id for item in ordered)),
        "interaction_ids": tuple(sorted(item.interaction_id for item in ordered)),
        "transition_evidence_ids": tuple(
            sorted(item.transition_evidence_id for item in ordered)
        ),
        "conformance_report_ids": tuple(
            sorted(item.conformance_report_id for item in ordered)
        ),
        "statistics": ordered_statistics,
        "candidates": ordered_candidates,
        "semantics": TRANSITION_DISCOVERY_SEMANTICS,
        "version": TRANSITION_DISCOVERY_REPORT_VERSION,
    }
    identity_values = dict(values)
    identity_values["statistics"] = tuple(
        asdict(item) for item in ordered_statistics
    )
    identity_values["candidates"] = tuple(
        asdict(item) for item in ordered_candidates
    )
    return TransitionDiscoveryReportDTO(
        report_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )
