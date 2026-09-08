"""Future-memory authority: memory-about-memory regulating veto power.

Core principle: prediction accuracy is not decision authority. Remembering
something is not the same as allowing that memory to command enactment.

A ``FutureMemoryValidityDTO`` accumulates verification outcomes
(``record_verification_event``) into bounded, content-addressed evidence
about one transition model's reliability for one action: recent
verification rate, projection error, direction error, changed-field error,
and a recent-mismatch trend (staleness). ``assess_memory_authority`` then
derives the authority for the *next* decision from that evidence plus the
live projection:

- OOD: the projection itself is out of distribution (never vetoes);
- STALE: recent verification shows memory or world diverged (annotate only);
- MAY_VETO: supported projection, sufficient history, low mismatch,
  sufficient confidence (veto authorized);
- SUPPORT_ONLY: everything else (annotate only).

The coupling path vetoes a contradicted candidate only under MAY_VETO.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .expected_transition import ExpectedTransitionVPMDTO
from .transition_verification import FutureTransitionVerificationDTO

MEMORY_AUTHORITY_POLICY_VERSION: Final = "perception-memory-authority-policy/1"
FUTURE_MEMORY_VALIDITY_VERSION: Final = "perception-future-memory-validity/1"
MEMORY_AUTHORITY_ASSESSMENT_VERSION: Final = "perception-memory-authority-assessment/1"
MEMORY_AUTHORITY_CONTEXT_VERSION: Final = "perception-memory-authority-context/1"
MEMORY_AUTHORITIES: Final = frozenset({"SUPPORT_ONLY", "MAY_VETO", "STALE", "OOD"})
VALIDITY_OUTCOME_TOKENS: Final = frozenset(
    {"confirmed", "mismatch", "violation", "insufficient"}
)


class PerceptionMemoryAuthorityError(ValueError):
    """Raised when memory-authority evidence or assessment is ill-formed."""


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


def _round12(value: float) -> float:
    return round(float(value), 12)


@dataclass(frozen=True)
class MemoryAuthorityPolicyDTO:
    """Bounded policy for deriving veto authority from validity evidence."""

    minimum_events: int = 4
    stale_negative_rate: float = 0.5
    veto_min_confidence: float = 0.35
    version: str = MEMORY_AUTHORITY_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.minimum_events <= 0:
            raise PerceptionMemoryAuthorityError("minimum_events must be positive")
        if not 0.0 <= self.stale_negative_rate <= 1.0:
            raise PerceptionMemoryAuthorityError(
                "stale_negative_rate must be in [0, 1]"
            )
        if not 0.0 <= self.veto_min_confidence <= 1.0:
            raise PerceptionMemoryAuthorityError(
                "veto_min_confidence must be in [0, 1]"
            )
        if self.version != MEMORY_AUTHORITY_POLICY_VERSION:
            raise PerceptionMemoryAuthorityError(
                "unsupported memory authority policy version"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "minimum_events": self.minimum_events,
            "stale_negative_rate": self.stale_negative_rate,
            "version": self.version,
            "veto_min_confidence": self.veto_min_confidence,
        }

    @property
    def policy_id(self) -> str:
        return _digest(_canonical_json(self.canonical_payload()))


@dataclass(frozen=True)
class FutureMemoryValidityDTO:
    """Accumulated verification evidence about one model's memory for one action.

    The DTO stores evidence only (counts, bounded recent outcomes, summed
    error magnitudes). Authority itself is derived per decision by
    ``assess_memory_authority`` from this evidence plus the live projection.
    """

    validity_id: str
    transition_model_id: str
    action_label: str
    field_schema_id: str
    training_dataset_id: str
    window_size: int
    event_count: int
    confirmed_count: int
    mismatch_count: int
    violation_count: int
    insufficient_count: int
    recent_outcomes: tuple[str, ...]
    cumulative_abs_error: float
    error_events: int
    cumulative_direction_error: float
    direction_events: int
    cumulative_changed_error: float
    changed_events: int
    version: str = FUTURE_MEMORY_VALIDITY_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.validity_id,
                self.transition_model_id,
                self.action_label,
                self.field_schema_id,
                self.training_dataset_id,
            )
        ):
            raise PerceptionMemoryAuthorityError(
                "memory validity identities must be non-empty"
            )
        if not 1 <= self.window_size <= 64:
            raise PerceptionMemoryAuthorityError("window_size must be in [1, 64]")
        for name in (
            "event_count",
            "confirmed_count",
            "mismatch_count",
            "violation_count",
            "insufficient_count",
            "error_events",
            "direction_events",
            "changed_events",
        ):
            if getattr(self, name) < 0:
                raise PerceptionMemoryAuthorityError(f"{name} must be >= 0")
        if (
            self.confirmed_count
            + self.mismatch_count
            + self.violation_count
            + self.insufficient_count
        ) != self.event_count:
            raise PerceptionMemoryAuthorityError(
                "outcome counts must sum to event_count"
            )
        if len(self.recent_outcomes) > self.window_size:
            raise PerceptionMemoryAuthorityError("recent outcomes exceed window_size")
        if any(
            outcome not in VALIDITY_OUTCOME_TOKENS for outcome in self.recent_outcomes
        ):
            raise PerceptionMemoryAuthorityError("recent outcomes use unknown tokens")
        for name in (
            "cumulative_abs_error",
            "cumulative_direction_error",
            "cumulative_changed_error",
        ):
            value = getattr(self, name)
            if not (value >= 0.0) or value != value:
                raise PerceptionMemoryAuthorityError(
                    f"{name} must be a finite non-negative number"
                )
        if self.version != FUTURE_MEMORY_VALIDITY_VERSION:
            raise PerceptionMemoryAuthorityError(
                "unsupported future memory validity version"
            )
        if self.validity_id != _digest(_canonical_json(self.canonical_payload())):
            raise PerceptionMemoryAuthorityError(
                "validity identity disagrees with canonical payload"
            )

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "action_label": self.action_label,
            "changed_events": self.changed_events,
            "confirmed_count": self.confirmed_count,
            "cumulative_abs_error": _round12(self.cumulative_abs_error),
            "cumulative_changed_error": _round12(self.cumulative_changed_error),
            "cumulative_direction_error": _round12(self.cumulative_direction_error),
            "direction_events": self.direction_events,
            "error_events": self.error_events,
            "event_count": self.event_count,
            "field_schema_id": self.field_schema_id,
            "insufficient_count": self.insufficient_count,
            "mismatch_count": self.mismatch_count,
            "recent_outcomes": list(self.recent_outcomes),
            "training_dataset_id": self.training_dataset_id,
            "transition_model_id": self.transition_model_id,
            "version": self.version,
            "violation_count": self.violation_count,
            "window_size": self.window_size,
        }

    @property
    def recent_negative_rate(self) -> float:
        """Share of recent outcomes where memory or world diverged."""
        if not self.recent_outcomes:
            return 0.0
        negatives = sum(
            1
            for outcome in self.recent_outcomes
            if outcome in {"mismatch", "violation"}
        )
        return negatives / len(self.recent_outcomes)

    @property
    def staleness_score(self) -> float:
        return self.recent_negative_rate

    @property
    def mean_absolute_error(self) -> float:
        if self.error_events <= 0:
            return 0.0
        return self.cumulative_abs_error / self.error_events

    @property
    def mean_direction_error(self) -> float:
        if self.direction_events <= 0:
            return 0.0
        return self.cumulative_direction_error / self.direction_events

    @property
    def mean_changed_error(self) -> float:
        if self.changed_events <= 0:
            return 0.0
        return self.cumulative_changed_error / self.changed_events


def _outcome_for(status: str) -> tuple[str, str]:
    if status == "confirmed":
        return "confirmed", "confirmed_count"
    if status in {"future_projection_mismatch", "confirmed_with_unexpected_change"}:
        return "mismatch", "mismatch_count"
    if status == "declared_expectation_violation":
        return "violation", "violation_count"
    if status == "insufficient_evidence":
        return "insufficient", "insufficient_count"
    raise PerceptionMemoryAuthorityError(f"unknown verification status: {status!r}")


def record_verification_event(
    current: FutureMemoryValidityDTO | None,
    verification: FutureTransitionVerificationDTO,
    *,
    transition_model_id: str,
    action_label: str,
    field_schema_id: str,
    training_dataset_id: str,
    window_size: int = 16,
) -> FutureMemoryValidityDTO:
    """Fold one verification outcome into bounded validity evidence."""
    outcome, counter = _outcome_for(verification.status)
    if current is None:
        counts = {
            "confirmed_count": 0,
            "mismatch_count": 0,
            "violation_count": 0,
            "insufficient_count": 0,
        }
        recent: tuple[str, ...] = ()
        event_count = 0
        cumulative_abs_error = 0.0
        error_events = 0
        cumulative_direction_error = 0.0
        direction_events = 0
        cumulative_changed_error = 0.0
        changed_events = 0
    else:
        if current.transition_model_id != transition_model_id:
            raise PerceptionMemoryAuthorityError(
                "verification event targets a different transition model"
            )
        if current.action_label != action_label:
            raise PerceptionMemoryAuthorityError(
                "verification event targets a different action"
            )
        counts = {
            "confirmed_count": current.confirmed_count,
            "mismatch_count": current.mismatch_count,
            "violation_count": current.violation_count,
            "insufficient_count": current.insufficient_count,
        }
        recent = current.recent_outcomes
        event_count = current.event_count
        cumulative_abs_error = current.cumulative_abs_error
        error_events = current.error_events
        cumulative_direction_error = current.cumulative_direction_error
        direction_events = current.direction_events
        cumulative_changed_error = current.cumulative_changed_error
        changed_events = current.changed_events
        window_size = current.window_size
    counts[counter] += 1
    recent = (*recent, outcome)[-window_size:]
    values: dict[str, object] = {
        "action_label": action_label,
        "changed_events": changed_events + 1,
        "confirmed_count": counts["confirmed_count"],
        "cumulative_abs_error": _round12(
            cumulative_abs_error + verification.mean_absolute_error
        ),
        "cumulative_changed_error": _round12(
            cumulative_changed_error + verification.changed_field_error_rate
        ),
        "cumulative_direction_error": _round12(
            cumulative_direction_error + verification.direction_error_rate
        ),
        "direction_events": direction_events + 1,
        "error_events": error_events + 1,
        "event_count": event_count + 1,
        "field_schema_id": field_schema_id,
        "insufficient_count": counts["insufficient_count"],
        "mismatch_count": counts["mismatch_count"],
        "recent_outcomes": list(recent),
        "training_dataset_id": training_dataset_id,
        "transition_model_id": transition_model_id,
        "version": FUTURE_MEMORY_VALIDITY_VERSION,
        "violation_count": counts["violation_count"],
        "window_size": window_size,
    }
    return FutureMemoryValidityDTO(
        validity_id=_digest(_canonical_json(values)),
        transition_model_id=transition_model_id,
        action_label=action_label,
        field_schema_id=field_schema_id,
        training_dataset_id=training_dataset_id,
        window_size=window_size,
        event_count=event_count + 1,
        confirmed_count=counts["confirmed_count"],
        mismatch_count=counts["mismatch_count"],
        violation_count=counts["violation_count"],
        insufficient_count=counts["insufficient_count"],
        recent_outcomes=recent,
        cumulative_abs_error=cumulative_abs_error + verification.mean_absolute_error,
        error_events=error_events + 1,
        cumulative_direction_error=cumulative_direction_error
        + verification.direction_error_rate,
        direction_events=direction_events + 1,
        cumulative_changed_error=cumulative_changed_error
        + verification.changed_field_error_rate,
        changed_events=changed_events + 1,
    )


@dataclass(frozen=True)
class MemoryAuthorityAssessmentDTO:
    """Authority derived for one projection from validity evidence."""

    assessment_id: str
    validity_id: str | None
    expected_transition_id: str
    authority: str
    staleness_score: float
    recent_negative_rate: float
    confidence: float
    reasons: tuple[str, ...] = ()
    version: str = MEMORY_AUTHORITY_ASSESSMENT_VERSION

    def __post_init__(self) -> None:
        if not self.assessment_id or not self.expected_transition_id:
            raise PerceptionMemoryAuthorityError(
                "authority assessment identities must be non-empty"
            )
        if self.authority not in MEMORY_AUTHORITIES:
            raise PerceptionMemoryAuthorityError(
                f"unsupported memory authority: {self.authority!r}"
            )
        for name in ("staleness_score", "recent_negative_rate", "confidence"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise PerceptionMemoryAuthorityError(f"{name} must be in [0, 1]")
        if self.version != MEMORY_AUTHORITY_ASSESSMENT_VERSION:
            raise PerceptionMemoryAuthorityError(
                "unsupported memory authority assessment version"
            )


def assess_memory_authority(
    validity: FutureMemoryValidityDTO | None,
    projection: ExpectedTransitionVPMDTO,
    *,
    policy: MemoryAuthorityPolicyDTO | None = None,
) -> MemoryAuthorityAssessmentDTO:
    """Derive veto authority for one live projection.

    Cold memory (no history, or too few events) is permissive when the
    projection itself is supported and confident — preserving genuine
    corrections — while accumulated mismatch evidence demotes memory to
    STALE no matter how confident the projection claims to be.
    """
    resolved = policy or MemoryAuthorityPolicyDTO()
    reasons: list[str] = [f"projection status: {projection.status}"]
    staleness = validity.staleness_score if validity is not None else 0.0
    negative_rate = validity.recent_negative_rate if validity is not None else 0.0
    if projection.status == "out_of_distribution":
        authority = "OOD"
        reasons.append("projection is out of distribution: no veto authority")
    elif projection.status != "supported":
        authority = "SUPPORT_ONLY"
        reasons.append("projection lacks support: annotate only, never veto")
    elif validity is None or validity.event_count < resolved.minimum_events:
        if projection.confidence >= resolved.veto_min_confidence:
            authority = "MAY_VETO"
            reasons.append("cold memory with a supported confident projection")
        else:
            authority = "SUPPORT_ONLY"
            reasons.append("cold memory with low confidence: annotate only")
    elif negative_rate >= resolved.stale_negative_rate:
        authority = "STALE"
        reasons.append(
            f"recent negative verification rate {negative_rate:.3f} "
            f">= {resolved.stale_negative_rate:.3f}: veto authority withdrawn"
        )
    elif projection.confidence >= resolved.veto_min_confidence:
        authority = "MAY_VETO"
        reasons.append("supported projection with fresh verification support")
    else:
        authority = "SUPPORT_ONLY"
        reasons.append("supported projection below veto confidence")
    payload = {
        "authority": authority,
        "confidence": projection.confidence,
        "expected_transition_id": projection.expected_transition_id,
        "recent_negative_rate": negative_rate,
        "staleness_score": staleness,
        "validity_id": validity.validity_id if validity is not None else None,
        "version": MEMORY_AUTHORITY_ASSESSMENT_VERSION,
    }
    return MemoryAuthorityAssessmentDTO(
        assessment_id=_digest(_canonical_json(payload)),
        validity_id=validity.validity_id if validity is not None else None,
        expected_transition_id=projection.expected_transition_id,
        authority=authority,
        staleness_score=staleness,
        recent_negative_rate=negative_rate,
        confidence=projection.confidence,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class MemoryAuthorityContextDTO:
    """Validity evidence bundled for one coupled decision."""

    validity_by_action: tuple[tuple[str, FutureMemoryValidityDTO], ...]
    policy: MemoryAuthorityPolicyDTO
    version: str = MEMORY_AUTHORITY_CONTEXT_VERSION

    def __post_init__(self) -> None:
        actions = tuple(action for action, _ in self.validity_by_action)
        if actions != tuple(sorted(set(actions))):
            raise PerceptionMemoryAuthorityError(
                "validity actions must be unique and sorted"
            )
        if self.version != MEMORY_AUTHORITY_CONTEXT_VERSION:
            raise PerceptionMemoryAuthorityError(
                "unsupported memory authority context version"
            )

    @classmethod
    def create(
        cls,
        validity_by_action: Mapping[str, FutureMemoryValidityDTO],
        policy: MemoryAuthorityPolicyDTO | None = None,
    ) -> "MemoryAuthorityContextDTO":
        return cls(
            validity_by_action=tuple(
                sorted(validity_by_action.items(), key=lambda item: item[0])
            ),
            policy=policy or MemoryAuthorityPolicyDTO(),
        )

    def validity_for(self, action_label: str) -> FutureMemoryValidityDTO | None:
        for action, validity in self.validity_by_action:
            if action == action_label:
                return validity
        return None
