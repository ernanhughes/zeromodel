"""Expected-versus-observed transition verification (Strengthen / Return).

After the environment enacts an action, the caller builds the actual P18A
transition evidence (bound with the action declaration and conformance report
in a VisualTransitionAnalysis) and compares it against the remembered
expected transition. Prediction error and environment fault can both produce
mismatch; this report records the observed relation without diagnosing
its cause. Persistent learning is not performed: evidence is recorded only.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .expected_transition import ExpectedTransitionFieldDTO, ExpectedTransitionVPMDTO
from .transition_analysis import VisualTransitionAnalysisDTO
from .transition_evidence import TransitionFieldEvidenceDTO

FUTURE_TRANSITION_VERIFICATION_VERSION: Final = (
    "perception-future-transition-verification/1"
)
FUTURE_TRANSITION_VERIFICATION_STATUSES: Final = frozenset(
    {
        "confirmed",
        "confirmed_with_unexpected_change",
        "future_projection_mismatch",
        "declared_expectation_violation",
        "insufficient_evidence",
    }
)


class PerceptionTransitionVerificationError(ValueError):
    """Raised when expected-versus-observed comparison is ill-formed."""


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
class FutureTransitionVerificationDTO:
    """Explicit report comparing remembered future with observed evidence."""

    verification_id: str
    expected_transition_id: str
    observed_transition_evidence_id: str
    status: str
    field_notes: tuple[str, ...] = ()
    detail: str = ""
    version: str = FUTURE_TRANSITION_VERIFICATION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.verification_id,
                self.expected_transition_id,
                self.observed_transition_evidence_id,
            )
        ):
            raise PerceptionTransitionVerificationError(
                "verification identities must be non-empty"
            )
        if self.status not in FUTURE_TRANSITION_VERIFICATION_STATUSES:
            raise PerceptionTransitionVerificationError(
                f"unsupported verification status: {self.status!r}"
            )
        if self.field_notes != tuple(sorted(set(self.field_notes))):
            raise PerceptionTransitionVerificationError(
                "field notes must be unique and sorted"
            )
        if self.version != FUTURE_TRANSITION_VERIFICATION_VERSION:
            raise PerceptionTransitionVerificationError(
                "unsupported transition verification version"
            )


def _report(
    expected: ExpectedTransitionVPMDTO,
    observed_evidence_id: str,
    status: str,
    notes: list[str],
    detail: str,
) -> FutureTransitionVerificationDTO:
    ordered = tuple(sorted(set(notes)))
    payload = {
        "expected_transition_id": expected.expected_transition_id,
        "field_notes": list(ordered),
        "observed_transition_evidence_id": observed_evidence_id,
        "status": status,
        "version": FUTURE_TRANSITION_VERIFICATION_VERSION,
    }
    return FutureTransitionVerificationDTO(
        verification_id=_digest(_canonical_json(payload)),
        expected_transition_id=expected.expected_transition_id,
        observed_transition_evidence_id=observed_evidence_id,
        status=status,
        field_notes=ordered,
        detail=detail,
    )


def _compare_projected_field(
    projected: ExpectedTransitionFieldDTO,
    observed: TransitionFieldEvidenceDTO | None,
    *,
    tolerance: float,
    change_epsilon: float,
) -> tuple[str | None, bool, bool]:
    """Compare one projected field; returns (note, mismatch, unexpected)."""
    if observed is None:
        return f"{projected.field_id}: missing_observed_field", True, False
    error = abs(observed.after_mean - projected.expected_after_mean)
    allowed = tolerance + 2.0 * projected.signed_change_dispersion
    observed_signed = observed.mean_signed_change
    expected_signed = projected.expected_mean_signed_change
    expected_static = (
        projected.expected_changed_fraction <= 0.0
        or abs(expected_signed) <= change_epsilon
    )
    observed_changed = abs(observed_signed) > change_epsilon
    if error <= allowed:
        if expected_static and observed_changed:
            return f"{projected.field_id}: unexpected_change", False, True
        return None, False, False
    if expected_static and observed_changed:
        return f"{projected.field_id}: unexpected_change", True, True
    if expected_static:
        return f"{projected.field_id}: magnitude_mismatch", True, False
    if not observed_changed:
        return f"{projected.field_id}: missing_expected_change", True, False
    if (observed_signed > 0) != (expected_signed > 0):
        return f"{projected.field_id}: wrong_direction", True, False
    if abs(observed_signed) > abs(expected_signed) + allowed + change_epsilon:
        return f"{projected.field_id}: excessive_change", True, False
    return f"{projected.field_id}: magnitude_mismatch", True, False


def verify_expected_transition(
    expected: ExpectedTransitionVPMDTO,
    analysis: VisualTransitionAnalysisDTO,
    *,
    tolerance: float = 0.02,
    change_epsilon: float = 0.01,
) -> FutureTransitionVerificationDTO:
    """Compare a remembered expected transition with observed analysis."""
    observed_id = analysis.transition_evidence_id
    if expected.status != "supported":
        return _report(
            expected,
            observed_id,
            "insufficient_evidence",
            [],
            f"expected transition was {expected.status}; nothing to confirm",
        )
    enacted_label = analysis.action.payload.get("action_label")
    if enacted_label != expected.action_label:
        return _report(
            expected,
            observed_id,
            "insufficient_evidence",
            [],
            "action identity mismatch between expected and observed transition",
        )
    if analysis.before_source_vpm_id != expected.source_vpm_id:
        return _report(
            expected,
            observed_id,
            "insufficient_evidence",
            [],
            "before-source mismatch between expected and observed transition",
        )
    if analysis.status == "nonconformant":
        return _report(
            expected,
            observed_id,
            "declared_expectation_violation",
            [],
            "observed transition violates declared expectations",
        )
    observed_fields = {
        item.field_id: item for item in analysis.transition_evidence.fields
    }
    notes: list[str] = []
    mismatch = False
    unexpected_change = False
    for projected in expected.fields:
        note, field_mismatch, field_unexpected = _compare_projected_field(
            projected,
            observed_fields.get(projected.field_id),
            tolerance=tolerance,
            change_epsilon=change_epsilon,
        )
        if note is not None:
            notes.append(note)
        mismatch = mismatch or field_mismatch
        unexpected_change = unexpected_change or field_unexpected
    if mismatch:
        return _report(
            expected,
            observed_id,
            "future_projection_mismatch",
            notes,
            "observed transition differs from the projected future",
        )
    if unexpected_change:
        return _report(
            expected,
            observed_id,
            "confirmed_with_unexpected_change",
            notes,
            "projection confirmed with additional unexpected change",
        )
    return _report(
        expected,
        observed_id,
        "confirmed",
        notes,
        "observed transition matches the projected future",
    )
