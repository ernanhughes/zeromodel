"""Typed observation-to-state claim compilation for bounded visual domains."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

STATE_CLAIMS_SEMANTICS: Final = (
    "typed_field_evidence_to_compatible_state_policy_unanimity"
)
OBSERVATION_ARTIFACT_VERSION: Final = "perception-observation-artifact/1"
EVIDENCE_REQUIREMENT_VERSION: Final = "perception-evidence-requirement/1"
FIELD_EVIDENCE_VERSION: Final = "perception-field-evidence/1"
OBSERVATION_VALIDITY_REPORT_VERSION: Final = "perception-observation-validity/1"
PANEL_REGISTRATION_RESULT_VERSION: Final = "perception-panel-registration/1"
FIELD_MEASUREMENT_VERSION: Final = "perception-field-measurement/1"
EVIDENCE_COMPILATION_REPORT_VERSION: Final = "perception-evidence-compilation/1"
STATE_DECISION_RECEIPT_VERSION: Final = "perception-state-decision-receipt/1"
STATE_SPECIFICATION_VERSION: Final = "perception-state-specification/1"
STATE_CLAIM_SET_VERSION: Final = "perception-state-claim-set/1"
POLICY_COMPATIBILITY_REPORT_VERSION: Final = "perception-policy-compatibility/1"

FIELD_EVIDENCE_STATUSES: Final = (
    "supported",
    "contradicted",
    "unresolved",
    "invalid_observation",
)
REJECT_AMBIGUOUS_DECISION: Final = "REJECT_AMBIGUOUS"
INVALID_OBSERVATION_DECISION: Final = "INVALID_OBSERVATION"


class PerceptionStateClaimError(ValueError):
    """Raised when state-claim compilation inputs or outputs are invalid."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object] | bytes) -> str:
    data = payload if isinstance(payload, bytes) else _canonical_json(payload)
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _unique_sorted(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if any(not value for value in result):
        raise PerceptionStateClaimError(f"{name} values must be non-empty")
    if len(result) != len(set(result)):
        raise PerceptionStateClaimError(f"{name} values must be unique")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class ObservationArtifact:
    observation_id: str
    image_digest: str
    capture_timestamp: str
    camera_id: str
    capture_profile_id: str
    width: int
    height: int
    version: str = OBSERVATION_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.observation_id,
                self.image_digest,
                self.capture_timestamp,
                self.camera_id,
                self.capture_profile_id,
            )
        ):
            raise PerceptionStateClaimError("observation identities must be non-empty")
        if not self.image_digest.startswith("sha256:"):
            raise PerceptionStateClaimError("image_digest must be a sha256 digest")
        if self.width <= 0 or self.height <= 0:
            raise PerceptionStateClaimError("observation dimensions must be positive")


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    field_id: str
    evidence_kind: str
    region_id: str
    allowed_values: tuple[str, ...]
    version: str = EVIDENCE_REQUIREMENT_VERSION

    def __post_init__(self) -> None:
        if not all((self.field_id, self.evidence_kind, self.region_id)):
            raise PerceptionStateClaimError(
                "evidence requirement ids must be non-empty"
            )
        object.__setattr__(
            self,
            "allowed_values",
            _unique_sorted(self.allowed_values, name="allowed_values"),
        )


@dataclass(frozen=True, slots=True)
class FieldEvidence:
    field_id: str
    status: str
    supported_values: tuple[str, ...]
    contradicted_values: tuple[str, ...]
    unresolved_values: tuple[str, ...]
    source_region: str
    observation_id: str
    compiler_id: str
    registered_panel_id: str = ""
    decoder_id: str = ""
    raw_measurement: Mapping[str, object] | None = None
    reason: str = ""
    evidence_id: str = ""
    version: str = FIELD_EVIDENCE_VERSION

    def __post_init__(self) -> None:
        if not all((self.field_id, self.status, self.observation_id, self.compiler_id)):
            raise PerceptionStateClaimError(
                "field evidence identities must be non-empty"
            )
        if self.status not in FIELD_EVIDENCE_STATUSES:
            raise PerceptionStateClaimError("unsupported field evidence status")
        for attr in ("supported_values", "contradicted_values", "unresolved_values"):
            object.__setattr__(
                self,
                attr,
                _unique_sorted(getattr(self, attr), name=attr),
            )
        overlap = (
            set(self.supported_values) & set(self.contradicted_values)
            | set(self.supported_values) & set(self.unresolved_values)
            | set(self.contradicted_values) & set(self.unresolved_values)
        )
        if overlap:
            raise PerceptionStateClaimError(
                "field evidence value sets must be disjoint"
            )
        if self.status == "supported" and not self.supported_values:
            raise PerceptionStateClaimError("supported evidence needs supported values")
        if self.status == "unresolved" and not self.unresolved_values:
            raise PerceptionStateClaimError(
                "unresolved evidence needs unresolved values"
            )
        if self.status == "invalid_observation" and not self.reason:
            raise PerceptionStateClaimError("invalid observations need a reason")
        if not self.evidence_id:
            payload: Mapping[str, object] = {
                "field_id": self.field_id,
                "status": self.status,
                "supported_values": self.supported_values,
                "contradicted_values": self.contradicted_values,
                "unresolved_values": self.unresolved_values,
                "source_region": self.source_region,
                "observation_id": self.observation_id,
                "compiler_id": self.compiler_id,
                "registered_panel_id": self.registered_panel_id,
                "decoder_id": self.decoder_id,
                "raw_measurement": self.raw_measurement or {},
                "reason": self.reason,
                "version": self.version,
            }
            object.__setattr__(self, "evidence_id", _digest(payload))


@dataclass(frozen=True, slots=True)
class ObservationValidityReport:
    observation_id: str
    valid: bool
    invalid_reasons: tuple[str, ...]
    measurement: Mapping[str, object]
    report_id: str = ""
    version: str = OBSERVATION_VALIDITY_REPORT_VERSION

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise PerceptionStateClaimError("observation_id must be non-empty")
        object.__setattr__(
            self,
            "invalid_reasons",
            _unique_sorted(self.invalid_reasons, name="invalid_reasons"),
        )
        if self.valid and self.invalid_reasons:
            raise PerceptionStateClaimError("valid observations cannot have reasons")
        if not self.valid and not self.invalid_reasons:
            raise PerceptionStateClaimError("invalid observations need reasons")
        if not self.report_id:
            object.__setattr__(
                self,
                "report_id",
                _digest(
                    {
                        "observation_id": self.observation_id,
                        "valid": self.valid,
                        "invalid_reasons": self.invalid_reasons,
                        "measurement": self.measurement,
                        "version": self.version,
                    }
                ),
            )


@dataclass(frozen=True, slots=True)
class PanelRegistrationResult:
    observation_id: str
    panel_layout_id: str
    registration_method_id: str
    registered_panel_id: str
    valid: bool
    anchor_count: int
    source_corners: tuple[tuple[float, float], ...]
    reason: str = ""
    version: str = PANEL_REGISTRATION_RESULT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.observation_id,
                self.panel_layout_id,
                self.registration_method_id,
                self.registered_panel_id,
            )
        ):
            raise PerceptionStateClaimError("registration identities must be non-empty")
        if self.anchor_count < 0:
            raise PerceptionStateClaimError("anchor_count must be non-negative")
        if self.valid and self.anchor_count < 4:
            raise PerceptionStateClaimError("valid registration requires four anchors")
        if not self.valid and not self.reason:
            raise PerceptionStateClaimError("invalid registration needs a reason")


@dataclass(frozen=True, slots=True)
class FieldMeasurement:
    field_id: str
    observation_id: str
    registered_panel_id: str
    source_region: str
    decoder_id: str
    raw_measurement: Mapping[str, object]
    measurement_id: str = ""
    version: str = FIELD_MEASUREMENT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.field_id,
                self.observation_id,
                self.registered_panel_id,
                self.source_region,
                self.decoder_id,
            )
        ):
            raise PerceptionStateClaimError("field measurement ids must be non-empty")
        if not self.raw_measurement:
            raise PerceptionStateClaimError("field measurements must preserve raw data")
        if not self.measurement_id:
            object.__setattr__(
                self,
                "measurement_id",
                _digest(
                    {
                        "field_id": self.field_id,
                        "observation_id": self.observation_id,
                        "registered_panel_id": self.registered_panel_id,
                        "source_region": self.source_region,
                        "decoder_id": self.decoder_id,
                        "raw_measurement": self.raw_measurement,
                        "version": self.version,
                    }
                ),
            )


@dataclass(frozen=True, slots=True)
class EvidenceCompilationReport:
    observation_id: str
    panel_layout_id: str
    calibration_id: str
    compiler_id: str
    validity_report: ObservationValidityReport
    registration_result: PanelRegistrationResult
    measurements: tuple[FieldMeasurement, ...]
    evidence: tuple[FieldEvidence, ...]
    report_id: str = ""
    version: str = EVIDENCE_COMPILATION_REPORT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.observation_id,
                self.panel_layout_id,
                self.calibration_id,
                self.compiler_id,
            )
        ):
            raise PerceptionStateClaimError("evidence report ids must be non-empty")
        if not self.report_id:
            object.__setattr__(
                self,
                "report_id",
                _digest(
                    {
                        "observation_id": self.observation_id,
                        "panel_layout_id": self.panel_layout_id,
                        "calibration_id": self.calibration_id,
                        "compiler_id": self.compiler_id,
                        "validity_report_id": self.validity_report.report_id,
                        "registered_panel_id": (
                            self.registration_result.registered_panel_id
                        ),
                        "measurement_ids": tuple(
                            item.measurement_id for item in self.measurements
                        ),
                        "evidence_ids": tuple(
                            item.evidence_id for item in self.evidence
                        ),
                        "version": self.version,
                    }
                ),
            )


@dataclass(frozen=True, slots=True)
class StateSpecification:
    state_id: str
    fields: Mapping[str, str]
    action_id: str
    criticality: int = 0
    version: str = STATE_SPECIFICATION_VERSION

    def __post_init__(self) -> None:
        if not self.state_id or not self.action_id:
            raise PerceptionStateClaimError("state_id and action_id must be non-empty")
        if not self.fields:
            raise PerceptionStateClaimError("state fields must be non-empty")
        if any(not key or not value for key, value in self.fields.items()):
            raise PerceptionStateClaimError(
                "state field ids and values must be non-empty"
            )
        if self.criticality < 0:
            raise PerceptionStateClaimError("criticality must be non-negative")


@dataclass(frozen=True, slots=True)
class EliminatedState:
    state_id: str
    contradictions: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.state_id or not self.contradictions:
            raise PerceptionStateClaimError(
                "eliminated states need an id and contradictions"
            )
        object.__setattr__(
            self,
            "contradictions",
            _unique_sorted(self.contradictions, name="contradictions"),
        )


@dataclass(frozen=True, slots=True)
class StateClaimSet:
    observation_id: str
    compatible_state_ids: tuple[str, ...]
    eliminated_states: tuple[EliminatedState, ...]
    unresolved_fields: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    invalid_observation: bool = False
    claim_set_id: str = ""
    version: str = STATE_CLAIM_SET_VERSION

    def __post_init__(self) -> None:
        if not self.observation_id:
            raise PerceptionStateClaimError("observation_id must be non-empty")
        for attr in ("compatible_state_ids", "unresolved_fields", "evidence_ids"):
            object.__setattr__(
                self, attr, _unique_sorted(getattr(self, attr), name=attr)
            )
        eliminated_ids = tuple(item.state_id for item in self.eliminated_states)
        if eliminated_ids != tuple(sorted(eliminated_ids)) or len(
            eliminated_ids
        ) != len(set(eliminated_ids)):
            raise PerceptionStateClaimError(
                "eliminated states must be unique and sorted"
            )
        if set(self.compatible_state_ids) & set(eliminated_ids):
            raise PerceptionStateClaimError(
                "a state cannot be compatible and eliminated"
            )
        if not self.claim_set_id:
            payload: Mapping[str, object] = {
                "observation_id": self.observation_id,
                "compatible_state_ids": self.compatible_state_ids,
                "eliminated_states": [
                    {
                        "state_id": item.state_id,
                        "contradictions": item.contradictions,
                    }
                    for item in self.eliminated_states
                ],
                "unresolved_fields": self.unresolved_fields,
                "evidence_ids": self.evidence_ids,
                "invalid_observation": self.invalid_observation,
                "version": self.version,
            }
            object.__setattr__(self, "claim_set_id", _digest(payload))


@dataclass(frozen=True, slots=True)
class PolicyCompatibilityReport:
    claim_set_id: str
    compatible_state_ids: tuple[str, ...]
    action_ids: tuple[str, ...]
    unanimous: bool
    decision: str
    unresolved_fields: tuple[str, ...]
    conflicting_actions: Mapping[str, tuple[str, ...]]
    report_id: str = ""
    version: str = POLICY_COMPATIBILITY_REPORT_VERSION

    def __post_init__(self) -> None:
        if not self.claim_set_id or not self.decision:
            raise PerceptionStateClaimError("policy report ids must be non-empty")
        for attr in ("compatible_state_ids", "action_ids", "unresolved_fields"):
            object.__setattr__(
                self, attr, _unique_sorted(getattr(self, attr), name=attr)
            )
        invalid = self.decision == INVALID_OBSERVATION_DECISION
        if self.unanimous != (len(self.action_ids) == 1):
            raise PerceptionStateClaimError("unanimous must match action cardinality")
        if self.unanimous and self.decision != self.action_ids[0]:
            raise PerceptionStateClaimError("unanimous decision must equal the action")
        if (
            not self.unanimous
            and not invalid
            and self.decision != REJECT_AMBIGUOUS_DECISION
        ):
            raise PerceptionStateClaimError("non-unanimous reports must reject")
        if not self.report_id:
            payload: Mapping[str, object] = {
                "claim_set_id": self.claim_set_id,
                "compatible_state_ids": self.compatible_state_ids,
                "action_ids": self.action_ids,
                "unanimous": self.unanimous,
                "decision": self.decision,
                "unresolved_fields": self.unresolved_fields,
                "conflicting_actions": {
                    key: tuple(value)
                    for key, value in sorted(self.conflicting_actions.items())
                },
                "version": self.version,
            }
            object.__setattr__(self, "report_id", _digest(payload))


@dataclass(frozen=True, slots=True)
class StateDecisionReceipt:
    observation_id: str
    evidence_report_id: str
    claim_set_id: str
    policy_report_id: str
    decision: str
    receipt_id: str = ""
    version: str = STATE_DECISION_RECEIPT_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.observation_id,
                self.evidence_report_id,
                self.claim_set_id,
                self.policy_report_id,
                self.decision,
            )
        ):
            raise PerceptionStateClaimError(
                "state decision receipt ids must be non-empty"
            )
        if not self.receipt_id:
            object.__setattr__(
                self,
                "receipt_id",
                _digest(
                    {
                        "observation_id": self.observation_id,
                        "evidence_report_id": self.evidence_report_id,
                        "claim_set_id": self.claim_set_id,
                        "policy_report_id": self.policy_report_id,
                        "decision": self.decision,
                        "version": self.version,
                    }
                ),
            )


def build_state_claim_set(
    observation_id: str,
    states: Sequence[StateSpecification],
    evidence: Sequence[FieldEvidence],
) -> StateClaimSet:
    """Return every declared state compatible with the supplied field evidence."""

    if not states:
        raise PerceptionStateClaimError("at least one state specification is required")
    state_ids = tuple(state.state_id for state in states)
    if len(state_ids) != len(set(state_ids)):
        raise PerceptionStateClaimError("state ids must be unique")
    evidence_by_field = {item.field_id: item for item in evidence}
    if len(evidence_by_field) != len(evidence):
        raise PerceptionStateClaimError("field evidence must be unique by field_id")
    invalid = any(item.status == "invalid_observation" for item in evidence)
    if invalid:
        return StateClaimSet(
            observation_id=observation_id,
            compatible_state_ids=(),
            eliminated_states=(),
            unresolved_fields=tuple(sorted(evidence_by_field)),
            evidence_ids=tuple(item.evidence_id for item in evidence),
            invalid_observation=True,
        )

    compatible: list[str] = []
    eliminated: list[EliminatedState] = []
    unresolved_fields: set[str] = set()
    for state in sorted(states, key=lambda item: item.state_id):
        contradictions: list[str] = []
        for field_id, value in state.fields.items():
            item = evidence_by_field.get(field_id)
            if item is None:
                unresolved_fields.add(field_id)
                continue
            if value in item.contradicted_values:
                contradictions.append(f"{field_id}={value}")
            elif item.supported_values and value not in item.supported_values:
                contradictions.append(f"{field_id}={value}")
            elif item.unresolved_values and value in item.unresolved_values:
                unresolved_fields.add(field_id)
        if contradictions:
            eliminated.append(
                EliminatedState(
                    state_id=state.state_id,
                    contradictions=tuple(contradictions),
                )
            )
        else:
            compatible.append(state.state_id)

    return StateClaimSet(
        observation_id=observation_id,
        compatible_state_ids=tuple(compatible),
        eliminated_states=tuple(sorted(eliminated, key=lambda item: item.state_id)),
        unresolved_fields=tuple(unresolved_fields),
        evidence_ids=tuple(item.evidence_id for item in evidence),
    )


def build_policy_compatibility_report(
    claim_set: StateClaimSet,
    states: Sequence[StateSpecification],
) -> PolicyCompatibilityReport:
    """Project compatible states through policy and act only on unanimous actions."""

    if claim_set.invalid_observation:
        return PolicyCompatibilityReport(
            claim_set_id=claim_set.claim_set_id,
            compatible_state_ids=(),
            action_ids=(),
            unanimous=False,
            decision=INVALID_OBSERVATION_DECISION,
            unresolved_fields=claim_set.unresolved_fields,
            conflicting_actions={},
        )
    by_id = {state.state_id: state for state in states}
    missing = set(claim_set.compatible_state_ids) - set(by_id)
    if missing:
        raise PerceptionStateClaimError("claim set references unknown states")
    action_to_states: dict[str, list[str]] = {}
    for state_id in claim_set.compatible_state_ids:
        action_to_states.setdefault(by_id[state_id].action_id, []).append(state_id)
    action_ids = tuple(sorted(action_to_states))
    unanimous = len(action_ids) == 1
    decision = action_ids[0] if unanimous else REJECT_AMBIGUOUS_DECISION
    return PolicyCompatibilityReport(
        claim_set_id=claim_set.claim_set_id,
        compatible_state_ids=claim_set.compatible_state_ids,
        action_ids=action_ids,
        unanimous=unanimous,
        decision=decision,
        unresolved_fields=claim_set.unresolved_fields,
        conflicting_actions={
            action: tuple(sorted(state_ids))
            for action, state_ids in sorted(action_to_states.items())
        },
    )
