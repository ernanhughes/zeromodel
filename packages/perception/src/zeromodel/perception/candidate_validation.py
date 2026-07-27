"""Held-out validation for P18C missing-component candidates (Stage P18D).

P18D derives explicit validation expectations from P18C discovery statistics and
evaluates them against a disjoint cohort of immutable P18A transition evidence.
Discovery and validation lineage must not overlap. Candidate outcomes preserve
validated, rejected, inconclusive, and insufficient-evidence states.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final, Mapping

from .transition_discovery import (
    MissingComponentCandidateDTO,
    RecurrentUnexplainedStatisticDTO,
    TransitionDiscoveryReportDTO,
)
from .transition_evidence import TransitionEvidenceVPMDTO, TransitionFieldEvidenceDTO

HELD_OUT_TRANSITION_OBSERVATION_VERSION: Final = (
    "perception-held-out-transition-observation/1"
)
CANDIDATE_VALIDATION_POLICY_VERSION: Final = "perception-candidate-validation-policy/1"
CANDIDATE_VALIDATION_EXPECTATION_VERSION: Final = (
    "perception-candidate-validation-expectation/1"
)
CANDIDATE_VALIDATION_FINDING_VERSION: Final = (
    "perception-candidate-validation-finding/1"
)
CANDIDATE_VALIDATION_RESULT_VERSION: Final = "perception-candidate-validation-result/1"
CANDIDATE_VALIDATION_REPORT_VERSION: Final = "perception-candidate-validation-report/1"
CANDIDATE_VALIDATION_SEMANTICS: Final = (
    "held_out_validation_of_p18c_candidates_against_disjoint_p18a_transition_evidence"
)
CANDIDATE_VALIDATION_FINDING_STATUSES: Final = {
    "confirmed",
    "missing_change",
    "insufficient_change",
    "wrong_change_direction",
    "inconclusive_direction",
}
CANDIDATE_VALIDATION_RESULT_STATUSES: Final = {
    "validated",
    "rejected",
    "inconclusive",
    "insufficient_validation_evidence",
}
CANDIDATE_VALIDATION_REPORT_STATUSES: Final = {
    "all_validated",
    "mixed_outcomes",
    "none_validated",
    "insufficient_evidence",
}

_REJECTION_FINDINGS: Final = {
    "missing_change",
    "insufficient_change",
    "wrong_change_direction",
}


class PerceptionCandidateValidationError(ValueError):
    """Raised when held-out candidate-validation contracts are invalid."""


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


def _unit(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise PerceptionCandidateValidationError(f"{name} must be in [0, 1]")


def _positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerceptionCandidateValidationError(f"{name} must be a positive integer")


def _ordered_unique(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and not values:
        raise PerceptionCandidateValidationError(f"{name} must be non-empty")
    if values != tuple(sorted(set(values))):
        raise PerceptionCandidateValidationError(f"{name} must be unique and sorted")


def _ordered_with_repetition(
    name: str,
    values: tuple[str, ...],
    *,
    allow_empty: bool = True,
) -> None:
    if not allow_empty and not values:
        raise PerceptionCandidateValidationError(f"{name} must be non-empty")
    if values != tuple(sorted(values)):
        raise PerceptionCandidateValidationError(f"{name} must be sorted")


def _same_float(left: float, right: float) -> bool:
    return abs(left - right) <= 1e-12


@dataclass(frozen=True)
class HeldOutTransitionObservationDTO:
    """One validation interaction materialized from immutable P18A evidence."""

    observation_id: str
    interaction_id: str
    cohort_id: str
    field_schema_id: str
    transition_evidence_id: str
    fields: tuple[TransitionFieldEvidenceDTO, ...]
    version: str = HELD_OUT_TRANSITION_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.observation_id,
                self.interaction_id,
                self.cohort_id,
                self.field_schema_id,
                self.transition_evidence_id,
            )
        ):
            raise PerceptionCandidateValidationError(
                "held-out observation identities must be non-empty"
            )
        field_ids = tuple(item.field_id for item in self.fields)
        _ordered_unique("held-out field_ids", field_ids, allow_empty=False)
        if self.version != HELD_OUT_TRANSITION_OBSERVATION_VERSION:
            raise PerceptionCandidateValidationError(
                "unsupported held-out observation version"
            )
        if self.observation_id != _digest(_payload(self, "observation_id")):
            raise PerceptionCandidateValidationError(
                "held-out observation identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        interaction_id: str,
        cohort_id: str,
        transition: TransitionEvidenceVPMDTO,
    ) -> "HeldOutTransitionObservationDTO":
        if not interaction_id or not cohort_id:
            raise PerceptionCandidateValidationError(
                "interaction_id and cohort_id must be non-empty"
            )
        fields = tuple(sorted(transition.fields, key=lambda item: item.field_id))
        values: dict[str, object] = {
            "interaction_id": interaction_id,
            "cohort_id": cohort_id,
            "field_schema_id": transition.field_schema_id,
            "transition_evidence_id": transition.transition_evidence_id,
            "fields": fields,
            "version": HELD_OUT_TRANSITION_OBSERVATION_VERSION,
        }
        identity_values = dict(values)
        identity_values["fields"] = tuple(asdict(item) for item in fields)
        return cls(
            observation_id=_digest(identity_values),
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class CandidateValidationPolicyDTO:
    policy_id: str
    minimum_validation_observation_count: int = 3
    minimum_confirmation_fraction: float = 2 / 3
    minimum_rejection_fraction: float = 2 / 3
    minimum_magnitude_retention_fraction: float = 0.5
    direction_epsilon: float = 0.01
    version: str = CANDIDATE_VALIDATION_POLICY_VERSION

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise PerceptionCandidateValidationError("policy_id must be non-empty")
        _positive_int(
            "minimum_validation_observation_count",
            self.minimum_validation_observation_count,
        )
        for name in (
            "minimum_confirmation_fraction",
            "minimum_rejection_fraction",
            "minimum_magnitude_retention_fraction",
            "direction_epsilon",
        ):
            _unit(name, getattr(self, name))
        if self.version != CANDIDATE_VALIDATION_POLICY_VERSION:
            raise PerceptionCandidateValidationError(
                "unsupported candidate validation policy version"
            )
        if self.policy_id != _digest(_payload(self, "policy_id")):
            raise PerceptionCandidateValidationError(
                "candidate validation policy identity disagrees with canonical payload"
            )

    @classmethod
    def create(
        cls,
        *,
        minimum_validation_observation_count: int = 3,
        minimum_confirmation_fraction: float = 2 / 3,
        minimum_rejection_fraction: float = 2 / 3,
        minimum_magnitude_retention_fraction: float = 0.5,
        direction_epsilon: float = 0.01,
    ) -> "CandidateValidationPolicyDTO":
        values: dict[str, object] = {
            "minimum_validation_observation_count": (
                minimum_validation_observation_count
            ),
            "minimum_confirmation_fraction": minimum_confirmation_fraction,
            "minimum_rejection_fraction": minimum_rejection_fraction,
            "minimum_magnitude_retention_fraction": (
                minimum_magnitude_retention_fraction
            ),
            "direction_epsilon": direction_epsilon,
            "version": CANDIDATE_VALIDATION_POLICY_VERSION,
        }
        return cls(policy_id=_digest(values), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class CandidateValidationExpectationDTO:
    expectation_id: str
    discovery_report_id: str
    candidate_id: str
    source_statistic_id: str
    policy_id: str
    candidate_kind: str
    field_schema_id: str
    field_ids: tuple[str, ...]
    expected_change: str
    minimum_mean_absolute_change: float
    minimum_changed_fraction: float
    minimum_signed_change_magnitude: float
    version: str = CANDIDATE_VALIDATION_EXPECTATION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.expectation_id,
                self.discovery_report_id,
                self.candidate_id,
                self.source_statistic_id,
                self.policy_id,
                self.candidate_kind,
                self.field_schema_id,
            )
        ):
            raise PerceptionCandidateValidationError(
                "candidate expectation identities must be non-empty"
            )
        _ordered_unique(
            "candidate expectation field_ids", self.field_ids, allow_empty=False
        )
        if self.expected_change not in {"change", "increase", "decrease"}:
            raise PerceptionCandidateValidationError(
                f"unsupported candidate expected_change: {self.expected_change}"
            )
        for name in (
            "minimum_mean_absolute_change",
            "minimum_changed_fraction",
            "minimum_signed_change_magnitude",
        ):
            _unit(name, getattr(self, name))
        if (
            self.expected_change == "change"
            and self.minimum_signed_change_magnitude != 0.0
        ):
            raise PerceptionCandidateValidationError(
                "non-directional candidate expectation cannot require signed magnitude"
            )
        if self.version != CANDIDATE_VALIDATION_EXPECTATION_VERSION:
            raise PerceptionCandidateValidationError(
                "unsupported candidate validation expectation version"
            )
        if self.expectation_id != _digest(_payload(self, "expectation_id")):
            raise PerceptionCandidateValidationError(
                "candidate validation expectation identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class CandidateValidationFindingDTO:
    finding_id: str
    expectation_id: str
    validation_observation_id: str
    interaction_id: str
    transition_evidence_id: str
    field_ids: tuple[str, ...]
    status: str
    observed_mean_absolute_change: float
    observed_mean_signed_change: float
    observed_changed_fraction: float
    observed_changed_value_count: int
    observed_total_value_count: int
    detail: str
    version: str = CANDIDATE_VALIDATION_FINDING_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.finding_id,
                self.expectation_id,
                self.validation_observation_id,
                self.interaction_id,
                self.transition_evidence_id,
                self.detail,
            )
        ):
            raise PerceptionCandidateValidationError(
                "candidate validation finding identities and detail must be non-empty"
            )
        _ordered_unique(
            "candidate finding field_ids", self.field_ids, allow_empty=False
        )
        if self.status not in CANDIDATE_VALIDATION_FINDING_STATUSES:
            raise PerceptionCandidateValidationError(
                f"unsupported candidate validation finding status: {self.status}"
            )
        _unit(
            "observed_mean_absolute_change",
            self.observed_mean_absolute_change,
        )
        _unit("observed_changed_fraction", self.observed_changed_fraction)
        if not -1.0 <= self.observed_mean_signed_change <= 1.0:
            raise PerceptionCandidateValidationError(
                "observed_mean_signed_change must be in [-1, 1]"
            )
        _positive_int(
            "observed_total_value_count",
            self.observed_total_value_count,
        )
        if (
            isinstance(self.observed_changed_value_count, bool)
            or not isinstance(self.observed_changed_value_count, int)
            or not 0
            <= self.observed_changed_value_count
            <= self.observed_total_value_count
        ):
            raise PerceptionCandidateValidationError(
                "observed_changed_value_count must be within observed_total_value_count"
            )
        if not _same_float(
            self.observed_changed_fraction,
            self.observed_changed_value_count / self.observed_total_value_count,
        ):
            raise PerceptionCandidateValidationError(
                "observed_changed_fraction disagrees with counts"
            )
        if self.version != CANDIDATE_VALIDATION_FINDING_VERSION:
            raise PerceptionCandidateValidationError(
                "unsupported candidate validation finding version"
            )
        if self.finding_id != _digest(_payload(self, "finding_id")):
            raise PerceptionCandidateValidationError(
                "candidate validation finding identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class CandidateValidationResultDTO:
    result_id: str
    candidate_id: str
    expectation: CandidateValidationExpectationDTO
    status: str
    observation_count: int
    confirmation_count: int
    rejection_count: int
    inconclusive_count: int
    confirmation_fraction: float
    rejection_fraction: float
    findings: tuple[CandidateValidationFindingDTO, ...]
    version: str = CANDIDATE_VALIDATION_RESULT_VERSION

    def __post_init__(self) -> None:
        if not self.result_id or not self.candidate_id:
            raise PerceptionCandidateValidationError(
                "candidate validation result identities must be non-empty"
            )
        if self.candidate_id != self.expectation.candidate_id:
            raise PerceptionCandidateValidationError(
                "validation result candidate disagrees with expectation"
            )
        if self.status not in CANDIDATE_VALIDATION_RESULT_STATUSES:
            raise PerceptionCandidateValidationError(
                f"unsupported candidate validation result status: {self.status}"
            )
        _positive_int("observation_count", self.observation_count)
        for name in (
            "confirmation_count",
            "rejection_count",
            "inconclusive_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PerceptionCandidateValidationError(
                    f"{name} must be a non-negative integer"
                )
        if (
            self.confirmation_count + self.rejection_count + self.inconclusive_count
            != self.observation_count
        ):
            raise PerceptionCandidateValidationError(
                "candidate result counts disagree with observation_count"
            )
        _unit("confirmation_fraction", self.confirmation_fraction)
        _unit("rejection_fraction", self.rejection_fraction)
        if not _same_float(
            self.confirmation_fraction,
            self.confirmation_count / self.observation_count,
        ):
            raise PerceptionCandidateValidationError(
                "confirmation_fraction disagrees with counts"
            )
        if not _same_float(
            self.rejection_fraction,
            self.rejection_count / self.observation_count,
        ):
            raise PerceptionCandidateValidationError(
                "rejection_fraction disagrees with counts"
            )
        finding_ids = tuple(item.finding_id for item in self.findings)
        _ordered_unique("candidate validation finding identities", finding_ids)
        if len(self.findings) != self.observation_count:
            raise PerceptionCandidateValidationError(
                "candidate validation findings must cover every observation"
            )
        if any(
            item.expectation_id != self.expectation.expectation_id
            for item in self.findings
        ):
            raise PerceptionCandidateValidationError(
                "candidate validation finding expectation mismatch"
            )
        if self.version != CANDIDATE_VALIDATION_RESULT_VERSION:
            raise PerceptionCandidateValidationError(
                "unsupported candidate validation result version"
            )
        if self.result_id != _digest(_payload(self, "result_id")):
            raise PerceptionCandidateValidationError(
                "candidate validation result identity disagrees with canonical payload"
            )


@dataclass(frozen=True)
class CandidateValidationReportDTO:
    report_id: str
    status: str
    discovery_report_id: str
    discovery_cohort_id: str
    validation_cohort_id: str
    field_schema_id: str
    policy_id: str
    validation_observation_ids: tuple[str, ...]
    validation_interaction_ids: tuple[str, ...]
    validation_transition_evidence_ids: tuple[str, ...]
    expectation_ids: tuple[str, ...]
    results: tuple[CandidateValidationResultDTO, ...]
    semantics: str = CANDIDATE_VALIDATION_SEMANTICS
    version: str = CANDIDATE_VALIDATION_REPORT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.report_id,
                self.discovery_report_id,
                self.discovery_cohort_id,
                self.validation_cohort_id,
                self.field_schema_id,
                self.policy_id,
            )
        ):
            raise PerceptionCandidateValidationError(
                "candidate validation report identities must be non-empty"
            )
        if self.discovery_cohort_id == self.validation_cohort_id:
            raise PerceptionCandidateValidationError(
                "discovery and validation cohort identities must differ"
            )
        if self.status not in CANDIDATE_VALIDATION_REPORT_STATUSES:
            raise PerceptionCandidateValidationError(
                f"unsupported candidate validation report status: {self.status}"
            )
        _ordered_unique(
            "validation_observation_ids",
            self.validation_observation_ids,
            allow_empty=False,
        )
        _ordered_unique(
            "validation_interaction_ids",
            self.validation_interaction_ids,
            allow_empty=False,
        )
        _ordered_with_repetition(
            "validation_transition_evidence_ids",
            self.validation_transition_evidence_ids,
            allow_empty=False,
        )
        count = len(self.validation_observation_ids)
        if not (
            len(self.validation_interaction_ids)
            == len(self.validation_transition_evidence_ids)
            == count
        ):
            raise PerceptionCandidateValidationError(
                "validation report evidence counts must match observations"
            )
        _ordered_unique("candidate expectation identities", self.expectation_ids)
        result_ids = tuple(item.result_id for item in self.results)
        _ordered_unique("candidate validation result identities", result_ids)
        if (
            tuple(sorted(item.expectation.expectation_id for item in self.results))
            != self.expectation_ids
        ):
            raise PerceptionCandidateValidationError(
                "validation report expectation identities disagree with results"
            )
        if self.status != _report_status(self.results):
            raise PerceptionCandidateValidationError(
                "candidate validation report status disagrees with results"
            )
        if self.semantics != CANDIDATE_VALIDATION_SEMANTICS:
            raise PerceptionCandidateValidationError(
                "unsupported candidate validation semantics"
            )
        if self.version != CANDIDATE_VALIDATION_REPORT_VERSION:
            raise PerceptionCandidateValidationError(
                "unsupported candidate validation report version"
            )
        if self.report_id != _digest(_payload(self, "report_id")):
            raise PerceptionCandidateValidationError(
                "candidate validation report identity disagrees with canonical payload"
            )

    def results_for_status(
        self,
        status: str,
    ) -> tuple[CandidateValidationResultDTO, ...]:
        if status not in CANDIDATE_VALIDATION_RESULT_STATUSES:
            raise PerceptionCandidateValidationError(
                f"unsupported candidate validation result status: {status}"
            )
        return tuple(item for item in self.results if item.status == status)


def _source_statistic(
    discovery: TransitionDiscoveryReportDTO,
    candidate: MissingComponentCandidateDTO,
) -> RecurrentUnexplainedStatisticDTO:
    statistics = {item.statistic_id: item for item in discovery.statistics}
    try:
        statistic = statistics[candidate.source_statistic_id]
    except KeyError as exc:
        raise PerceptionCandidateValidationError(
            "candidate references an unknown discovery statistic"
        ) from exc
    if (
        statistic.statistic_kind != candidate.candidate_kind
        or statistic.field_ids != candidate.field_ids
    ):
        raise PerceptionCandidateValidationError(
            "candidate disagrees with its discovery statistic"
        )
    return statistic


def _expectation(
    *,
    discovery: TransitionDiscoveryReportDTO,
    candidate: MissingComponentCandidateDTO,
    policy: CandidateValidationPolicyDTO,
) -> CandidateValidationExpectationDTO:
    statistic = _source_statistic(discovery, candidate)
    retention = policy.minimum_magnitude_retention_fraction
    minimum_signed = (
        abs(statistic.mean_signed_change) * retention
        if candidate.proposed_expected_change in {"increase", "decrease"}
        else 0.0
    )
    values: dict[str, object] = {
        "discovery_report_id": discovery.report_id,
        "candidate_id": candidate.candidate_id,
        "source_statistic_id": statistic.statistic_id,
        "policy_id": policy.policy_id,
        "candidate_kind": candidate.candidate_kind,
        "field_schema_id": discovery.field_schema_id,
        "field_ids": candidate.field_ids,
        "expected_change": candidate.proposed_expected_change,
        "minimum_mean_absolute_change": (statistic.mean_absolute_change * retention),
        "minimum_changed_fraction": (statistic.mean_changed_fraction * retention),
        "minimum_signed_change_magnitude": minimum_signed,
        "version": CANDIDATE_VALIDATION_EXPECTATION_VERSION,
    }
    return CandidateValidationExpectationDTO(
        expectation_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )


def _aggregate_fields(
    fields: tuple[TransitionFieldEvidenceDTO, ...],
) -> tuple[float, float, float, int, int]:
    total_values = sum(item.total_value_count for item in fields)
    if total_values <= 0:
        raise PerceptionCandidateValidationError(
            "candidate validation target has no measurable values"
        )
    changed_values = sum(item.changed_value_count for item in fields)
    absolute = (
        sum(item.mean_absolute_change * item.total_value_count for item in fields)
        / total_values
    )
    signed = (
        sum(item.mean_signed_change * item.total_value_count for item in fields)
        / total_values
    )
    return absolute, signed, changed_values / total_values, changed_values, total_values


def _finding(
    *,
    expectation: CandidateValidationExpectationDTO,
    observation: HeldOutTransitionObservationDTO,
    policy: CandidateValidationPolicyDTO,
) -> CandidateValidationFindingDTO:
    field_map = {item.field_id: item for item in observation.fields}
    unknown = set(expectation.field_ids) - set(field_map)
    if unknown:
        raise PerceptionCandidateValidationError(
            f"validation observation is missing candidate fields: {sorted(unknown)}"
        )
    target_fields = tuple(field_map[field_id] for field_id in expectation.field_ids)
    absolute, signed, fraction, changed_count, total_count = _aggregate_fields(
        target_fields
    )
    if changed_count == 0:
        status = "missing_change"
        detail = "held-out evidence contained no threshold-crossing candidate change"
    elif (
        absolute < expectation.minimum_mean_absolute_change
        or fraction < expectation.minimum_changed_fraction
    ):
        status = "insufficient_change"
        detail = "held-out evidence did not retain the required discovery magnitude"
    elif expectation.expected_change == "change":
        status = "confirmed"
        detail = "held-out evidence confirmed the non-directional candidate change"
    else:
        required = max(
            expectation.minimum_signed_change_magnitude,
            policy.direction_epsilon,
        )
        if expectation.expected_change == "increase":
            wrong = signed < -required
            inconclusive = signed <= required
        else:
            wrong = signed > required
            inconclusive = signed >= -required
        if wrong:
            status = "wrong_change_direction"
            detail = "held-out signed change opposed the candidate hypothesis"
        elif inconclusive:
            status = "inconclusive_direction"
            detail = "held-out change was present but direction was not decisive"
        else:
            status = "confirmed"
            detail = "held-out evidence confirmed the candidate's directional change"
    values: dict[str, object] = {
        "expectation_id": expectation.expectation_id,
        "validation_observation_id": observation.observation_id,
        "interaction_id": observation.interaction_id,
        "transition_evidence_id": observation.transition_evidence_id,
        "field_ids": expectation.field_ids,
        "status": status,
        "observed_mean_absolute_change": absolute,
        "observed_mean_signed_change": signed,
        "observed_changed_fraction": fraction,
        "observed_changed_value_count": changed_count,
        "observed_total_value_count": total_count,
        "detail": detail,
        "version": CANDIDATE_VALIDATION_FINDING_VERSION,
    }
    return CandidateValidationFindingDTO(
        finding_id=_digest(values),
        **values,  # type: ignore[arg-type]
    )


def _result_status(
    *,
    observation_count: int,
    confirmation_fraction: float,
    rejection_fraction: float,
    policy: CandidateValidationPolicyDTO,
) -> str:
    if observation_count < policy.minimum_validation_observation_count:
        return "insufficient_validation_evidence"
    if confirmation_fraction >= policy.minimum_confirmation_fraction:
        return "validated"
    if rejection_fraction >= policy.minimum_rejection_fraction:
        return "rejected"
    return "inconclusive"


def _result(
    *,
    discovery: TransitionDiscoveryReportDTO,
    candidate: MissingComponentCandidateDTO,
    observations: tuple[HeldOutTransitionObservationDTO, ...],
    policy: CandidateValidationPolicyDTO,
) -> CandidateValidationResultDTO:
    expectation = _expectation(
        discovery=discovery,
        candidate=candidate,
        policy=policy,
    )
    findings = tuple(
        sorted(
            (
                _finding(
                    expectation=expectation,
                    observation=observation,
                    policy=policy,
                )
                for observation in observations
            ),
            key=lambda item: item.finding_id,
        )
    )
    confirmation_count = sum(item.status == "confirmed" for item in findings)
    rejection_count = sum(item.status in _REJECTION_FINDINGS for item in findings)
    inconclusive_count = len(findings) - confirmation_count - rejection_count
    confirmation_fraction = confirmation_count / len(findings)
    rejection_fraction = rejection_count / len(findings)
    status = _result_status(
        observation_count=len(findings),
        confirmation_fraction=confirmation_fraction,
        rejection_fraction=rejection_fraction,
        policy=policy,
    )
    values: dict[str, object] = {
        "candidate_id": candidate.candidate_id,
        "expectation": expectation,
        "status": status,
        "observation_count": len(findings),
        "confirmation_count": confirmation_count,
        "rejection_count": rejection_count,
        "inconclusive_count": inconclusive_count,
        "confirmation_fraction": confirmation_fraction,
        "rejection_fraction": rejection_fraction,
        "findings": findings,
        "version": CANDIDATE_VALIDATION_RESULT_VERSION,
    }
    identity_values = dict(values)
    identity_values["expectation"] = asdict(expectation)
    identity_values["findings"] = tuple(asdict(item) for item in findings)
    return CandidateValidationResultDTO(
        result_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )


def _report_status(results: tuple[CandidateValidationResultDTO, ...]) -> str:
    statuses = {item.status for item in results}
    if statuses == {"insufficient_validation_evidence"}:
        return "insufficient_evidence"
    if statuses == {"validated"}:
        return "all_validated"
    if (
        "validated" not in statuses
        and "insufficient_validation_evidence" not in statuses
    ):
        return "none_validated"
    return "mixed_outcomes"


def validate_discovered_transition_candidates(
    discovery: TransitionDiscoveryReportDTO,
    validation_observations: tuple[HeldOutTransitionObservationDTO, ...],
    policy: CandidateValidationPolicyDTO | None = None,
) -> CandidateValidationReportDTO:
    """Validate every P18C candidate on a disjoint held-out cohort."""

    if discovery.status != "candidates_found" or not discovery.candidates:
        raise PerceptionCandidateValidationError(
            "candidate validation requires a discovery report with candidates"
        )
    if not validation_observations:
        raise PerceptionCandidateValidationError(
            "candidate validation requires held-out observations"
        )
    resolved = policy or CandidateValidationPolicyDTO.create()
    if len({item.observation_id for item in validation_observations}) != len(
        validation_observations
    ):
        raise PerceptionCandidateValidationError(
            "held-out observations must have unique identities"
        )
    if len({item.interaction_id for item in validation_observations}) != len(
        validation_observations
    ):
        raise PerceptionCandidateValidationError(
            "held-out interactions must have unique identities"
        )
    cohort_ids = {item.cohort_id for item in validation_observations}
    schema_ids = {item.field_schema_id for item in validation_observations}
    if len(cohort_ids) != 1:
        raise PerceptionCandidateValidationError(
            "held-out observations must belong to one validation cohort"
        )
    validation_cohort_id = next(iter(cohort_ids))
    if validation_cohort_id == discovery.cohort_id:
        raise PerceptionCandidateValidationError(
            "validation cohort must differ from discovery cohort"
        )
    if schema_ids != {discovery.field_schema_id}:
        raise PerceptionCandidateValidationError(
            "held-out observations must use the discovery field schema"
        )
    validation_interactions = {item.interaction_id for item in validation_observations}
    overlap_interactions = validation_interactions & set(discovery.interaction_ids)
    if overlap_interactions:
        raise PerceptionCandidateValidationError(
            "discovery and validation interaction identities overlap: "
            f"{sorted(overlap_interactions)}"
        )
    validation_transitions = {
        item.transition_evidence_id for item in validation_observations
    }
    overlap_transitions = validation_transitions & set(
        discovery.transition_evidence_ids
    )
    if overlap_transitions:
        raise PerceptionCandidateValidationError(
            "discovery and validation transition evidence overlaps: "
            f"{sorted(overlap_transitions)}"
        )
    if set(item.observation_id for item in validation_observations) & set(
        discovery.observation_ids
    ):
        raise PerceptionCandidateValidationError(
            "discovery and validation observation identities overlap"
        )
    ordered_observations = tuple(
        sorted(validation_observations, key=lambda item: item.observation_id)
    )
    results = tuple(
        sorted(
            (
                _result(
                    discovery=discovery,
                    candidate=candidate,
                    observations=ordered_observations,
                    policy=resolved,
                )
                for candidate in discovery.candidates
            ),
            key=lambda item: item.result_id,
        )
    )
    values: dict[str, object] = {
        "status": _report_status(results),
        "discovery_report_id": discovery.report_id,
        "discovery_cohort_id": discovery.cohort_id,
        "validation_cohort_id": validation_cohort_id,
        "field_schema_id": discovery.field_schema_id,
        "policy_id": resolved.policy_id,
        "validation_observation_ids": tuple(
            sorted(item.observation_id for item in ordered_observations)
        ),
        "validation_interaction_ids": tuple(
            sorted(item.interaction_id for item in ordered_observations)
        ),
        "validation_transition_evidence_ids": tuple(
            sorted(item.transition_evidence_id for item in ordered_observations)
        ),
        "expectation_ids": tuple(
            sorted(item.expectation.expectation_id for item in results)
        ),
        "results": results,
        "semantics": CANDIDATE_VALIDATION_SEMANTICS,
        "version": CANDIDATE_VALIDATION_REPORT_VERSION,
    }
    identity_values = dict(values)
    identity_values["results"] = tuple(asdict(item) for item in results)
    return CandidateValidationReportDTO(
        report_id=_digest(identity_values),
        **values,  # type: ignore[arg-type]
    )
