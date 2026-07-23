"""`ProviderEvaluationCaseDTO` - one evaluated observation.

Distinguishes *observation correctness* (did the provider recover the exact
state) from *application behaviour correctness* (did the compiled policy
still choose the right action). A provider can predict the wrong exact state
while the compiled policy action still matches the expected action
(``action_equivalent``); a provider error that changes the compiled policy
action (``action_changing``) is a materially different failure. These are
never collapsed into one accuracy field.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    PROVIDER_EVALUATION_CASE_VERSION,
    PROVIDER_EVALUATION_RAW_RESPONSE_DIGEST_VERSION,
)
from zeromodel.video.domains.video_action_set.dto import CanonicalJsonDTO
from zeromodel.video.domains.video_action_set.observation_common import (
    boolean,
    integer,
    json_mapping,
    optional_sha256,
    optional_string,
    require_keys,
    sha256,
    string,
)
from zeromodel.video.domains.video_action_set.provider_evaluation_common import (
    basis_points,
    decision_payload,
    nonempty_str,
    nonneg_int,
    optional_basis_points,
    optional_canonical,
    optional_nonneg_int,
)


CASE_OUTCOME_EXACT = "exact"
CASE_OUTCOME_ACTION_EQUIVALENT = "action_equivalent"
CASE_OUTCOME_ACTION_CHANGING = "action_changing"
CASE_OUTCOME_REJECTED = "rejected"
CASE_OUTCOMES = frozenset(
    {
        CASE_OUTCOME_EXACT,
        CASE_OUTCOME_ACTION_EQUIVALENT,
        CASE_OUTCOME_ACTION_CHANGING,
        CASE_OUTCOME_REJECTED,
    }
)

DECISION_TRACE_KEYS = (
    "artifact_id",
    "row_id",
    "action",
    "metric_id",
    "value",
    "source_row_index",
    "source_metric_index",
    "view_row",
    "view_column",
    "candidates",
    "evidence",
)

CASE_KEYS = (
    "version",
    "case_ordinal",
    "frame_id",
    "policy_artifact_id",
    "provider_configuration_id",
    "expected_state",
    "expected_row_id",
    "expected_action",
    "expected_decision_trace",
    "accepted",
    "rejection_reason",
    "predicted_state",
    "predicted_row_id",
    "predicted_action",
    "predicted_decision_trace",
    "exact_state_match",
    "action_match",
    "factor_matches",
    "outcome",
    "provider_confidence_basis_points",
    "provider_latency_us",
    "provider_raw_response_digest",
    "provider_raw_response_text",
    "provider_response_metadata",
    "metadata",
    "case_id",
)


def _validate_decision_trace(
    trace: CanonicalJsonDTO,
    *,
    artifact_id: str,
    row_id: str,
    action: str,
    message: str,
) -> None:
    value = json_mapping(trace, message)
    if set(value) != set(DECISION_TRACE_KEYS):
        raise VPMValidationError(message)
    if (
        value.get("artifact_id") != artifact_id
        or value.get("row_id") != row_id
        or value.get("action") != action
    ):
        raise VPMValidationError(message)


def _raw_response_digest(text: str) -> str:
    return canonical_sha256(
        {
            "version": PROVIDER_EVALUATION_RAW_RESPONSE_DIGEST_VERSION,
            "text": text,
        }
    )


def _derive_case_outcome(
    *,
    accepted: bool,
    expected_state: object,
    predicted_state: object,
    expected_action: str,
    predicted_action: str | None,
) -> tuple[bool, bool, dict[str, bool], str]:
    if not accepted:
        return False, False, {}, CASE_OUTCOME_REJECTED
    exact = predicted_state == expected_state
    action_match = predicted_action == expected_action
    factor_matches: dict[str, bool] = {}
    if isinstance(expected_state, Mapping):
        predicted_mapping = (
            predicted_state if isinstance(predicted_state, Mapping) else {}
        )
        for key, value in expected_state.items():
            factor_matches[key] = (
                key in predicted_mapping and predicted_mapping[key] == value
            )
    if not action_match:
        outcome = CASE_OUTCOME_ACTION_CHANGING
    elif exact:
        outcome = CASE_OUTCOME_EXACT
    else:
        outcome = CASE_OUTCOME_ACTION_EQUIVALENT
    return exact, action_match, factor_matches, outcome


@dataclass(frozen=True, slots=True)
class ProviderEvaluationCaseContext:
    """The run-scoped identity every case in one run shares."""

    policy_artifact_id: str
    provider_configuration_id: str


@dataclass(frozen=True, slots=True)
class ProviderResponseEvidence:
    """Secondary provider-response evidence for one `ProviderEvaluationCaseDTO.build` call.

    ``provider_confidence_basis_points`` is the canonical scaled-integer
    confidence (``0..10000``, ten-thousandths). Convert a parsed model
    confidence float via the single ``confidence_to_basis_points`` helper
    below rather than scattering float-to-integer conversion across callers.
    """

    rejection_reason: str | None = None
    provider_confidence_basis_points: int | None = None
    provider_latency_us: int | None = None
    provider_raw_response_text: str | None = None
    provider_response_metadata: Mapping[str, object] | None = None
    metadata: Mapping[str, object] | None = None


def confidence_to_basis_points(value: float) -> int:
    """The one explicit float-to-basis-points conversion helper for provider
    confidence. ``value`` must be a finite fraction in ``[0.0, 1.0]``."""
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("confidence must be finite and in [0, 1]")
    return int(round(value * 10_000))


@dataclass(frozen=True, slots=True)
class ProviderEvaluationCaseDTO:
    """One evaluated observation: exact-state evidence plus policy-impact evidence.

    ``exact_state_match``, ``action_match``, ``factor_matches`` and ``outcome``
    are stored fields but are recomputed from ``expected_state``/
    ``predicted_state``/``expected_action``/``predicted_action`` in
    ``__post_init__`` and compared against the supplied value - exactly like
    ``ObservationDTO.action_known`` - so none of them can be supplied
    dishonestly.
    """

    version: str
    case_ordinal: int
    frame_id: str
    policy_artifact_id: str
    provider_configuration_id: str
    expected_state: CanonicalJsonDTO
    expected_row_id: str
    expected_action: str
    expected_decision_trace: CanonicalJsonDTO
    accepted: bool
    rejection_reason: str | None
    predicted_state: CanonicalJsonDTO | None
    predicted_row_id: str | None
    predicted_action: str | None
    predicted_decision_trace: CanonicalJsonDTO | None
    exact_state_match: bool
    action_match: bool
    factor_matches: CanonicalJsonDTO
    outcome: str
    provider_confidence_basis_points: int | None
    provider_latency_us: int | None
    provider_raw_response_digest: str | None
    provider_raw_response_text: str | None
    provider_response_metadata: CanonicalJsonDTO
    metadata: CanonicalJsonDTO
    case_id: str

    def __post_init__(self) -> None:
        if self.version != PROVIDER_EVALUATION_CASE_VERSION:
            raise VPMValidationError("unsupported provider evaluation case version")
        nonneg_int(self.case_ordinal, "case ordinal cannot be negative")
        nonempty_str(self.frame_id, "case frame id mismatch")
        sha256(self.policy_artifact_id, "case policy artifact id mismatch")
        sha256(
            self.provider_configuration_id,
            "case provider configuration id mismatch",
        )
        json_mapping(self.expected_state, "case expected state mismatch")
        nonempty_str(self.expected_row_id, "case expected row mismatch")
        nonempty_str(self.expected_action, "case expected action mismatch")
        _validate_decision_trace(
            self.expected_decision_trace,
            artifact_id=self.policy_artifact_id,
            row_id=self.expected_row_id,
            action=self.expected_action,
            message="case expected decision trace mismatch",
        )
        if not isinstance(self.accepted, bool):
            raise VPMValidationError("case accepted flag mismatch")
        if self.accepted:
            _validate_accepted_case(self)
        else:
            _validate_rejected_case(self)

        self._validate_derived_fields()

        if self.provider_confidence_basis_points is not None:
            basis_points(
                self.provider_confidence_basis_points,
                "case provider confidence mismatch",
            )
        if self.provider_latency_us is not None:
            nonneg_int(self.provider_latency_us, "case provider latency mismatch")
        if self.provider_raw_response_digest is not None:
            sha256(
                self.provider_raw_response_digest,
                "case provider raw response digest mismatch",
            )
        if self.provider_raw_response_text is not None:
            if self.provider_raw_response_digest != _raw_response_digest(
                self.provider_raw_response_text
            ):
                raise VPMValidationError("case provider raw response digest mismatch")
        json_mapping(
            self.provider_response_metadata,
            "case provider response metadata mismatch",
        )
        json_mapping(self.metadata, "case metadata mismatch")

        expected_id = canonical_sha256(_case_payload_without_id(self))
        if self.case_id != expected_id:
            raise VPMValidationError("case id mismatch")

    def _validate_derived_fields(self) -> None:
        expected_value = self.expected_state.to_value()
        predicted_value = (
            None if self.predicted_state is None else self.predicted_state.to_value()
        )
        exact, action_match, factor_matches, outcome = _derive_case_outcome(
            accepted=self.accepted,
            expected_state=expected_value,
            predicted_state=predicted_value,
            expected_action=self.expected_action,
            predicted_action=self.predicted_action,
        )
        if self.exact_state_match != exact:
            raise VPMValidationError("case exact_state_match mismatch")
        if self.action_match != action_match:
            raise VPMValidationError("case action_match mismatch")
        if (
            json_mapping(self.factor_matches, "case factor matches mismatch")
            != factor_matches
        ):
            raise VPMValidationError("case factor matches mismatch")
        if self.outcome not in CASE_OUTCOMES or self.outcome != outcome:
            raise VPMValidationError("case outcome mismatch")

    @property
    def provider_confidence(self) -> float | None:
        """Derived presentation convenience - not identity-bearing and not
        persisted as source truth. ``provider_confidence_basis_points``
        (an integer, ``0..10000``) remains authoritative; this float is
        recomputed from it on every access."""
        if self.provider_confidence_basis_points is None:
            return None
        return self.provider_confidence_basis_points / 10_000.0

    @classmethod
    def build(
        cls,
        *,
        case_ordinal: int,
        frame_id: str,
        context: ProviderEvaluationCaseContext,
        expected_state: Mapping[str, object],
        expected_decision: object,
        accepted: bool,
        predicted_state: Mapping[str, object] | None = None,
        predicted_decision: object | None = None,
        evidence: ProviderResponseEvidence | None = None,
    ) -> "ProviderEvaluationCaseDTO":
        evidence = evidence or ProviderResponseEvidence()
        expected_payload = dict(decision_payload(expected_decision))
        expected_row_id = nonempty_str(
            expected_payload.get("row_id"), "case expected row mismatch"
        )
        expected_action = nonempty_str(
            expected_payload.get("action"), "case expected action mismatch"
        )
        predicted_payload = (
            None
            if predicted_decision is None
            else dict(decision_payload(predicted_decision))
        )
        predicted_row_id = None
        predicted_action = None
        if predicted_payload is not None:
            predicted_row_id = nonempty_str(
                predicted_payload.get("row_id"), "case predicted row mismatch"
            )
            predicted_action = nonempty_str(
                predicted_payload.get("action"), "case predicted action mismatch"
            )
        expected_state_dto = CanonicalJsonDTO.from_value(dict(expected_state))
        predicted_state_dto = (
            None
            if predicted_state is None
            else CanonicalJsonDTO.from_value(dict(predicted_state))
        )
        exact, action_match, factor_matches, outcome = _derive_case_outcome(
            accepted=accepted,
            expected_state=expected_state_dto.to_value(),
            predicted_state=(
                None if predicted_state_dto is None else predicted_state_dto.to_value()
            ),
            expected_action=expected_action,
            predicted_action=predicted_action,
        )
        raw_response_digest = (
            None
            if evidence.provider_raw_response_text is None
            else _raw_response_digest(evidence.provider_raw_response_text)
        )
        payload = {
            "version": PROVIDER_EVALUATION_CASE_VERSION,
            "case_ordinal": case_ordinal,
            "frame_id": frame_id,
            "policy_artifact_id": context.policy_artifact_id,
            "provider_configuration_id": context.provider_configuration_id,
            "expected_state": expected_state_dto.to_value(),
            "expected_row_id": expected_row_id,
            "expected_action": expected_action,
            "expected_decision_trace": expected_payload,
            "accepted": accepted,
            "rejection_reason": evidence.rejection_reason,
            "predicted_state": (
                None if predicted_state_dto is None else predicted_state_dto.to_value()
            ),
            "predicted_row_id": predicted_row_id,
            "predicted_action": predicted_action,
            "predicted_decision_trace": predicted_payload,
            "exact_state_match": exact,
            "action_match": action_match,
            "factor_matches": factor_matches,
            "outcome": outcome,
            "provider_confidence_basis_points": evidence.provider_confidence_basis_points,
            "provider_latency_us": evidence.provider_latency_us,
            "provider_raw_response_digest": raw_response_digest,
            "provider_raw_response_text": evidence.provider_raw_response_text,
            "provider_response_metadata": dict(
                evidence.provider_response_metadata or {}
            ),
            "metadata": dict(evidence.metadata or {}),
        }
        case_id = canonical_sha256(payload)
        return cls.from_dict(payload | {"case_id": case_id})

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ProviderEvaluationCaseDTO":
        require_keys(payload, CASE_KEYS, "provider evaluation case keys mismatch")
        return cls(
            version=string(
                payload, "version", "unsupported provider evaluation case version"
            ),
            case_ordinal=integer(
                payload, "case_ordinal", "case ordinal cannot be negative"
            ),
            frame_id=string(payload, "frame_id", "case frame id mismatch"),
            policy_artifact_id=sha256(
                payload["policy_artifact_id"], "case policy artifact id mismatch"
            ),
            provider_configuration_id=sha256(
                payload["provider_configuration_id"],
                "case provider configuration id mismatch",
            ),
            expected_state=CanonicalJsonDTO.from_value(payload["expected_state"]),
            expected_row_id=string(
                payload, "expected_row_id", "case expected row mismatch"
            ),
            expected_action=string(
                payload, "expected_action", "case expected action mismatch"
            ),
            expected_decision_trace=CanonicalJsonDTO.from_value(
                payload["expected_decision_trace"]
            ),
            accepted=boolean(payload, "accepted", "case accepted flag mismatch"),
            rejection_reason=optional_string(
                payload, "rejection_reason", "case rejection reason mismatch"
            ),
            predicted_state=optional_canonical(payload["predicted_state"]),
            predicted_row_id=optional_string(
                payload, "predicted_row_id", "case predicted row mismatch"
            ),
            predicted_action=optional_string(
                payload, "predicted_action", "case predicted action mismatch"
            ),
            predicted_decision_trace=optional_canonical(
                payload["predicted_decision_trace"]
            ),
            exact_state_match=boolean(
                payload, "exact_state_match", "case exact_state_match mismatch"
            ),
            action_match=boolean(payload, "action_match", "case action_match mismatch"),
            factor_matches=CanonicalJsonDTO.from_value(payload["factor_matches"]),
            outcome=string(payload, "outcome", "case outcome mismatch"),
            provider_confidence_basis_points=optional_basis_points(
                payload.get("provider_confidence_basis_points"),
                "case provider confidence mismatch",
            ),
            provider_latency_us=optional_nonneg_int(
                payload.get("provider_latency_us"),
                "case provider latency mismatch",
            ),
            provider_raw_response_digest=optional_sha256(
                payload["provider_raw_response_digest"],
                "case provider raw response digest mismatch",
            ),
            provider_raw_response_text=optional_string(
                payload,
                "provider_raw_response_text",
                "case provider raw response text mismatch",
            ),
            provider_response_metadata=CanonicalJsonDTO.from_value(
                payload["provider_response_metadata"]
            ),
            metadata=CanonicalJsonDTO.from_value(payload["metadata"]),
            case_id=string(payload, "case_id", "case id mismatch"),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "case_ordinal": self.case_ordinal,
            "frame_id": self.frame_id,
            "policy_artifact_id": self.policy_artifact_id,
            "provider_configuration_id": self.provider_configuration_id,
            "expected_state": self.expected_state.to_value(),
            "expected_row_id": self.expected_row_id,
            "expected_action": self.expected_action,
            "expected_decision_trace": self.expected_decision_trace.to_value(),
            "accepted": self.accepted,
            "rejection_reason": self.rejection_reason,
            "predicted_state": (
                None
                if self.predicted_state is None
                else self.predicted_state.to_value()
            ),
            "predicted_row_id": self.predicted_row_id,
            "predicted_action": self.predicted_action,
            "predicted_decision_trace": (
                None
                if self.predicted_decision_trace is None
                else self.predicted_decision_trace.to_value()
            ),
            "exact_state_match": self.exact_state_match,
            "action_match": self.action_match,
            "factor_matches": self.factor_matches.to_value(),
            "outcome": self.outcome,
            "provider_confidence_basis_points": self.provider_confidence_basis_points,
            "provider_latency_us": self.provider_latency_us,
            "provider_raw_response_digest": self.provider_raw_response_digest,
            "provider_raw_response_text": self.provider_raw_response_text,
            "provider_response_metadata": self.provider_response_metadata.to_value(),
            "metadata": self.metadata.to_value(),
            "case_id": self.case_id,
        }


def _validate_accepted_case(case: ProviderEvaluationCaseDTO) -> None:
    if case.rejection_reason is not None:
        raise VPMValidationError("case rejection reason mismatch")
    if (
        case.predicted_state is None
        or case.predicted_row_id is None
        or case.predicted_action is None
        or case.predicted_decision_trace is None
    ):
        raise VPMValidationError("accepted case missing predicted result")
    json_mapping(case.predicted_state, "case predicted state mismatch")
    nonempty_str(case.predicted_row_id, "case predicted row mismatch")
    nonempty_str(case.predicted_action, "case predicted action mismatch")
    _validate_decision_trace(
        case.predicted_decision_trace,
        artifact_id=case.policy_artifact_id,
        row_id=case.predicted_row_id,
        action=case.predicted_action,
        message="case predicted decision trace mismatch",
    )


def _validate_rejected_case(case: ProviderEvaluationCaseDTO) -> None:
    if not case.rejection_reason:
        raise VPMValidationError("rejected case missing rejection reason")
    if (
        case.predicted_state is not None
        or case.predicted_row_id is not None
        or case.predicted_action is not None
        or case.predicted_decision_trace is not None
    ):
        raise VPMValidationError("rejected case must not carry a predicted result")


def _case_payload_without_id(case: ProviderEvaluationCaseDTO) -> dict[str, object]:
    payload = case.to_dict()
    payload.pop("case_id")
    return payload


__all__ = [
    "CASE_KEYS",
    "CASE_OUTCOMES",
    "CASE_OUTCOME_ACTION_CHANGING",
    "CASE_OUTCOME_ACTION_EQUIVALENT",
    "CASE_OUTCOME_EXACT",
    "CASE_OUTCOME_REJECTED",
    "ProviderEvaluationCaseContext",
    "ProviderEvaluationCaseDTO",
    "ProviderResponseEvidence",
    "confidence_to_basis_points",
]
