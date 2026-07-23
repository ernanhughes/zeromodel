"""`ProviderEvaluationSummaryDTO` - a deterministic summary over evaluation cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    PROVIDER_EVALUATION_SUMMARY_VERSION,
)
from zeromodel.video.domains.video_action_set.dto import CanonicalJsonDTO
from zeromodel.video.domains.video_action_set.observation_common import (
    integer,
    json_mapping,
    require_keys,
    string,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_case_dto import (
    CASE_OUTCOME_ACTION_CHANGING,
    CASE_OUTCOME_ACTION_EQUIVALENT,
    CASE_OUTCOME_EXACT,
    ProviderEvaluationCaseDTO,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_common import (
    nonneg_int,
    optional_nonneg_int,
)


SUMMARY_KEYS = (
    "version",
    "attempted_count",
    "accepted_count",
    "rejected_count",
    "exact_count",
    "action_equivalent_count",
    "action_changing_count",
    "action_correct_count",
    "factor_correct_counts",
    "factor_denominators",
    "rejection_reason_counts",
    "latency_sample_count",
    "latency_min_us",
    "latency_max_us",
    "latency_total_us",
    "latency_median_us",
    "latency_p95_us",
    "summary_id",
)


def _nearest_rank(values: Sequence[int], percentile: float) -> int:
    ordered = sorted(values)
    count = len(ordered)
    rank = max(1, min(count, math.ceil(percentile * count)))
    return ordered[rank - 1]


@dataclass(frozen=True, slots=True)
class ProviderEvaluationSummaryDTO:
    """Deterministic, source-count-preserving summary over an ordered case list.

    Latency percentiles use the nearest-rank method on integer microseconds:
    ``rank(p) = ceil(p * n)`` clamped to ``[1, n]``; ``value = sorted(values)[rank - 1]``.
    Median uses ``p=0.50``, p95 uses ``p=0.95``. An empty case list has every
    count at zero and every latency field at ``None``.
    """

    version: str
    attempted_count: int
    accepted_count: int
    rejected_count: int
    exact_count: int
    action_equivalent_count: int
    action_changing_count: int
    action_correct_count: int
    factor_correct_counts: CanonicalJsonDTO
    factor_denominators: CanonicalJsonDTO
    rejection_reason_counts: CanonicalJsonDTO
    latency_sample_count: int
    latency_min_us: int | None
    latency_max_us: int | None
    latency_total_us: int | None
    latency_median_us: int | None
    latency_p95_us: int | None
    summary_id: str

    def __post_init__(self) -> None:
        if self.version != PROVIDER_EVALUATION_SUMMARY_VERSION:
            raise VPMValidationError("unsupported provider evaluation summary version")
        self._validate_counts()
        self._validate_factor_counts()
        self._validate_rejection_reason_counts()
        self._validate_latency_fields()
        expected_id = canonical_sha256(_summary_payload_without_id(self))
        if self.summary_id != expected_id:
            raise VPMValidationError("summary id mismatch")

    def _validate_counts(self) -> None:
        for value in (
            self.attempted_count,
            self.accepted_count,
            self.rejected_count,
            self.exact_count,
            self.action_equivalent_count,
            self.action_changing_count,
            self.action_correct_count,
            self.latency_sample_count,
        ):
            nonneg_int(value, "summary counts cannot be negative")
        if self.attempted_count != self.accepted_count + self.rejected_count:
            raise VPMValidationError("summary attempted count mismatch")
        if self.accepted_count != (
            self.exact_count + self.action_equivalent_count + self.action_changing_count
        ):
            raise VPMValidationError("summary accepted count mismatch")
        if self.action_correct_count != self.exact_count + self.action_equivalent_count:
            raise VPMValidationError("summary action correct count mismatch")

    def _validate_factor_counts(self) -> None:
        correct = json_mapping(
            self.factor_correct_counts, "summary factor correct counts mismatch"
        )
        denominators = json_mapping(
            self.factor_denominators, "summary factor denominators mismatch"
        )
        if set(correct) != set(denominators):
            raise VPMValidationError("summary factor keys mismatch")
        for key, denominator_value in denominators.items():
            denominator = nonneg_int(
                denominator_value, "summary factor denominators mismatch"
            )
            correct_value = nonneg_int(
                correct[key], "summary factor correct counts mismatch"
            )
            if correct_value > denominator:
                raise VPMValidationError(
                    "summary factor correct counts exceed denominator"
                )
            if denominator > self.accepted_count:
                raise VPMValidationError(
                    "summary factor denominator exceeds accepted count"
                )

    def _validate_rejection_reason_counts(self) -> None:
        reasons = json_mapping(
            self.rejection_reason_counts, "summary rejection reason counts mismatch"
        )
        reason_total = sum(
            nonneg_int(value, "summary rejection reason counts mismatch")
            for value in reasons.values()
        )
        if reason_total != self.rejected_count:
            raise VPMValidationError("summary rejection reason counts mismatch")

    def _validate_latency_fields(self) -> None:
        latency_fields = (
            self.latency_min_us,
            self.latency_max_us,
            self.latency_total_us,
            self.latency_median_us,
            self.latency_p95_us,
        )
        if self.latency_sample_count == 0:
            if any(value is not None for value in latency_fields):
                raise VPMValidationError("summary latency fields mismatch")
            return
        if any(value is None for value in latency_fields):
            raise VPMValidationError("summary latency fields mismatch")
        latency_min = nonneg_int(self.latency_min_us, "summary latency fields mismatch")
        latency_max = nonneg_int(self.latency_max_us, "summary latency fields mismatch")
        latency_total = nonneg_int(
            self.latency_total_us, "summary latency fields mismatch"
        )
        latency_median = nonneg_int(
            self.latency_median_us, "summary latency fields mismatch"
        )
        latency_p95 = nonneg_int(self.latency_p95_us, "summary latency fields mismatch")
        if not (
            latency_min <= latency_median <= latency_p95 <= latency_max <= latency_total
        ):
            raise VPMValidationError("summary latency fields mismatch")

    @classmethod
    def from_cases(
        cls, cases: Sequence[ProviderEvaluationCaseDTO]
    ) -> "ProviderEvaluationSummaryDTO":
        ordered = tuple(sorted(cases, key=lambda case: case.case_ordinal))
        accepted = tuple(case for case in ordered if case.accepted)
        rejected = tuple(case for case in ordered if not case.accepted)
        exact = sum(1 for case in accepted if case.outcome == CASE_OUTCOME_EXACT)
        equivalent = sum(
            1 for case in accepted if case.outcome == CASE_OUTCOME_ACTION_EQUIVALENT
        )
        changing = sum(
            1 for case in accepted if case.outcome == CASE_OUTCOME_ACTION_CHANGING
        )
        correct_counts, denominators = _factor_counts(accepted)
        reason_counts = _rejection_reason_counts(rejected)
        latency = _latency_stats(ordered)

        payload = {
            "version": PROVIDER_EVALUATION_SUMMARY_VERSION,
            "attempted_count": len(ordered),
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "exact_count": exact,
            "action_equivalent_count": equivalent,
            "action_changing_count": changing,
            "action_correct_count": exact + equivalent,
            "factor_correct_counts": correct_counts,
            "factor_denominators": denominators,
            "rejection_reason_counts": reason_counts,
            **latency,
        }
        summary_id = canonical_sha256(payload)
        return cls.from_dict(payload | {"summary_id": summary_id})

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProviderEvaluationSummaryDTO":
        require_keys(payload, SUMMARY_KEYS, "provider evaluation summary keys mismatch")
        return cls(
            version=string(
                payload, "version", "unsupported provider evaluation summary version"
            ),
            attempted_count=integer(
                payload, "attempted_count", "summary counts cannot be negative"
            ),
            accepted_count=integer(
                payload, "accepted_count", "summary counts cannot be negative"
            ),
            rejected_count=integer(
                payload, "rejected_count", "summary counts cannot be negative"
            ),
            exact_count=integer(
                payload, "exact_count", "summary counts cannot be negative"
            ),
            action_equivalent_count=integer(
                payload,
                "action_equivalent_count",
                "summary counts cannot be negative",
            ),
            action_changing_count=integer(
                payload,
                "action_changing_count",
                "summary counts cannot be negative",
            ),
            action_correct_count=integer(
                payload, "action_correct_count", "summary counts cannot be negative"
            ),
            factor_correct_counts=CanonicalJsonDTO.from_value(
                payload["factor_correct_counts"]
            ),
            factor_denominators=CanonicalJsonDTO.from_value(
                payload["factor_denominators"]
            ),
            rejection_reason_counts=CanonicalJsonDTO.from_value(
                payload["rejection_reason_counts"]
            ),
            latency_sample_count=integer(
                payload, "latency_sample_count", "summary counts cannot be negative"
            ),
            latency_min_us=optional_nonneg_int(
                payload.get("latency_min_us"), "summary latency fields mismatch"
            ),
            latency_max_us=optional_nonneg_int(
                payload.get("latency_max_us"), "summary latency fields mismatch"
            ),
            latency_total_us=optional_nonneg_int(
                payload.get("latency_total_us"), "summary latency fields mismatch"
            ),
            latency_median_us=optional_nonneg_int(
                payload.get("latency_median_us"), "summary latency fields mismatch"
            ),
            latency_p95_us=optional_nonneg_int(
                payload.get("latency_p95_us"), "summary latency fields mismatch"
            ),
            summary_id=string(payload, "summary_id", "summary id mismatch"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "attempted_count": self.attempted_count,
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "exact_count": self.exact_count,
            "action_equivalent_count": self.action_equivalent_count,
            "action_changing_count": self.action_changing_count,
            "action_correct_count": self.action_correct_count,
            "factor_correct_counts": self.factor_correct_counts.to_value(),
            "factor_denominators": self.factor_denominators.to_value(),
            "rejection_reason_counts": self.rejection_reason_counts.to_value(),
            "latency_sample_count": self.latency_sample_count,
            "latency_min_us": self.latency_min_us,
            "latency_max_us": self.latency_max_us,
            "latency_total_us": self.latency_total_us,
            "latency_median_us": self.latency_median_us,
            "latency_p95_us": self.latency_p95_us,
            "summary_id": self.summary_id,
        }


def _factor_counts(
    accepted: Sequence[ProviderEvaluationCaseDTO],
) -> tuple[dict[str, int], dict[str, int]]:
    factor_keys: set[str] = set()
    for case in accepted:
        factor_keys.update(
            json_mapping(case.expected_state, "case expected state mismatch")
        )
    correct_counts: dict[str, int] = {}
    denominators: dict[str, int] = {}
    for key in sorted(factor_keys):
        denominators[key] = sum(
            1
            for case in accepted
            if key in json_mapping(case.expected_state, "case expected state mismatch")
        )
        correct_counts[key] = sum(
            1
            for case in accepted
            if json_mapping(case.factor_matches, "case factor matches mismatch").get(
                key
            )
            is True
        )
    return correct_counts, denominators


def _rejection_reason_counts(
    rejected: Sequence[ProviderEvaluationCaseDTO],
) -> dict[str, int]:
    reason_counts: dict[str, int] = {}
    for case in rejected:
        reason = case.rejection_reason or "unknown"
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return reason_counts


def _latency_stats(
    ordered: Sequence[ProviderEvaluationCaseDTO],
) -> dict[str, int | None]:
    latencies = [
        case.provider_latency_us
        for case in ordered
        if case.provider_latency_us is not None
    ]
    if not latencies:
        return {
            "latency_sample_count": 0,
            "latency_min_us": None,
            "latency_max_us": None,
            "latency_total_us": None,
            "latency_median_us": None,
            "latency_p95_us": None,
        }
    return {
        "latency_sample_count": len(latencies),
        "latency_min_us": min(latencies),
        "latency_max_us": max(latencies),
        "latency_total_us": sum(latencies),
        "latency_median_us": _nearest_rank(latencies, 0.50),
        "latency_p95_us": _nearest_rank(latencies, 0.95),
    }


def _summary_payload_without_id(
    summary: ProviderEvaluationSummaryDTO,
) -> dict[str, object]:
    payload = summary.to_dict()
    payload.pop("summary_id")
    return payload


__all__ = ["SUMMARY_KEYS", "ProviderEvaluationSummaryDTO"]
