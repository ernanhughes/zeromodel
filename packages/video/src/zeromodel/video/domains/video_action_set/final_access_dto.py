from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import cast

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import JsonValue, canonical_json_text, canonical_sha256


FINAL_EVALUATION_PROTOCOL_VERSION = "zeromodel-video-final-evaluation-protocol/v1"
FINAL_EXECUTION_AUTHORIZATION_VERSION = (
    "zeromodel-video-final-execution-authorization/v1"
)
FINAL_EXECUTION_REQUEST_VERSION = "zeromodel-video-final-execution-request/v1"
FINAL_ACCESS_RECORD_VERSION = "zeromodel-video-final-access-record/v1"
FINAL_ACCESS_EVENT_VERSION = "zeromodel-video-final-access-event/v1"
FINAL_EVIDENCE_BUNDLE_VERSION = "zeromodel-video-final-evidence-bundle/v1"
FINAL_EVALUATION_RESULT_VERSION = "zeromodel-video-final-evaluation-result/v1"
FINAL_EXECUTION_RECEIPT_VERSION = "zeromodel-video-final-execution-receipt/v1"
FINAL_EXECUTION_FAILURE_VERSION = "zeromodel-video-final-execution-failure/v1"

FINAL_ACCESS_STATES = frozenset(
    {"authorized", "reserved", "running", "completed", "failed", "interrupted"}
)
FINAL_ACCESS_TERMINAL_STATES = frozenset({"completed", "failed", "interrupted"})
FINAL_ACCESS_TRANSITIONS = {
    (None, "authorized"),
    ("authorized", "reserved"),
    ("reserved", "running"),
    ("reserved", "failed"),
    ("reserved", "interrupted"),
    ("running", "completed"),
    ("running", "failed"),
    ("running", "interrupted"),
}

SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
FINAL_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9._:-]+$")

PROTOCOL_KEYS = (
    "version",
    "protocol_id",
    "protocol_status",
    "created_utc",
    "approved_utc",
    "approved_by",
    "benchmark_seed_digest",
    "sealed_plan_digest",
    "policy_artifact_id",
    "candidate_set_id",
    "selected_provider_id",
    "decision_rule",
    "required_evidence",
    "claim_rule_set",
    "review_notes",
    "protocol_digest",
)
AUTHORIZATION_KEYS = (
    "version",
    "authorization_id",
    "authorization_status",
    "created_utc",
    "created_by",
    "protocol_digest",
    "expected_benchmark_seed_digest",
    "expected_sealed_plan_digest",
    "expected_policy_artifact_id",
    "output_dir",
    "database_path",
    "unattended_permitted",
    "operator_confirmation_text",
    "authorization_payload",
    "authorization_digest",
)
REQUEST_KEYS = (
    "version",
    "output_dir",
    "authorization_file",
    "expected_authorization_digest",
    "expected_sealed_plan_digest",
    "database_path",
    "preflight_only",
    "operator_identity",
    "unattended",
    "request_payload",
    "request_digest",
)
RECORD_KEYS = (
    "version",
    "access_id",
    "authorization_id",
    "state",
    "benchmark_seed_digest",
    "sealed_plan_digest",
    "protocol_digest",
    "authorization_digest",
    "created_utc",
    "updated_utc",
    "process_identity",
    "record_payload",
    "current_event_ordinal",
    "last_event_digest",
    "record_digest",
)
EVENT_KEYS = (
    "version",
    "access_id",
    "authorization_id",
    "ordinal",
    "previous_state",
    "new_state",
    "utc",
    "process_identity",
    "event_payload",
    "previous_event_digest",
    "event_digest",
)
RECEIPT_KEYS = (
    "version",
    "access_id",
    "authorization_id",
    "state",
    "completed_utc",
    "benchmark_seed_digest",
    "sealed_plan_digest",
    "protocol_digest",
    "authorization_digest",
    "evidence_digest",
    "evaluation_digest",
    "artifact_manifest_digest",
    "event_chain_digest",
    "decision",
    "execution_commit",
    "provider_order",
    "provider_versions",
    "expected_counts",
    "actual_counts",
    "historical_authority_id",
    "historical_database_sha256",
    "historical_evidence_manifest_digest",
    "historical_authority_digest",
    "receipt_digest",
)
EVIDENCE_BUNDLE_KEYS = (
    "version",
    "access_id",
    "authorization_digest",
    "protocol_digest",
    "benchmark_seed_digest",
    "sealed_plan_digest",
    "execution_commit",
    "provider_order",
    "provider_versions",
    "expected_counts",
    "actual_counts",
    "rows",
    "evidence_digest",
)
EVALUATION_RESULT_KEYS = (
    "version",
    "protocol_digest",
    "evidence_digest",
    "decision",
    "descriptive_measurements",
    "family_measurements",
    "rejections",
    "indeterminate_reasons",
    "actual_counts",
    "evaluation_digest",
)
FINAL_EVIDENCE_ROW_KEYS = (
    "family_id",
    "episode_id",
    "frame_ordinal",
    "frame_id",
    "provider_id",
    "split",
    "metrics",
)
FINAL_COUNT_KEYS = (
    "evidence_row_count",
    "episode_count",
    "frame_count",
    "provider_count",
)
FAILURE_KEYS = (
    "version",
    "access_id",
    "authorization_id",
    "state",
    "failed_utc",
    "benchmark_seed_digest",
    "sealed_plan_digest",
    "protocol_digest",
    "authorization_digest",
    "failure_kind",
    "error_code",
    "error_message",
    "event_chain_digest",
    "failure_digest",
)


@dataclass(frozen=True, slots=True)
class FinalJsonDTO:
    canonical_text: str

    def __post_init__(self) -> None:
        value = self.to_value()
        if canonical_json_text(value) != self.canonical_text:
            raise VPMValidationError("final access JSON is not canonical")

    @classmethod
    def from_value(cls, value: object) -> FinalJsonDTO:
        return cls(canonical_json_text(_json_value(value)))

    def to_value(self) -> JsonValue:
        import json

        try:
            return cast(JsonValue, json.loads(self.canonical_text))
        except json.JSONDecodeError as exc:
            raise VPMValidationError("final access JSON is not canonical") from exc


@dataclass(frozen=True, slots=True)
class FinalEvaluationProtocolDTO:
    version: str
    protocol_id: str
    protocol_status: str
    created_utc: str
    approved_utc: str | None
    approved_by: str | None
    benchmark_seed_digest: str
    sealed_plan_digest: str
    policy_artifact_id: str
    candidate_set_id: str
    selected_provider_id: str
    decision_rule: FinalJsonDTO
    required_evidence: FinalJsonDTO
    claim_rule_set: FinalJsonDTO
    review_notes: FinalJsonDTO
    protocol_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_EVALUATION_PROTOCOL_VERSION:
            raise VPMValidationError("final protocol version mismatch")
        validate_final_identifier(self.protocol_id, "final protocol id mismatch")
        if self.protocol_status not in {"draft", "review", "approved", "retired"}:
            raise VPMValidationError("final protocol status mismatch")
        if self.protocol_status == "approved" and (
            self.approved_utc is None or self.approved_by is None
        ):
            raise VPMValidationError("approved final protocol is missing approval")
        _require_digest(self.benchmark_seed_digest, "final protocol seed mismatch")
        _require_digest(self.sealed_plan_digest, "final protocol sealed plan mismatch")
        _require_digest(self.protocol_digest, "final protocol digest mismatch")
        _require_digest_match(
            self.to_dict(),
            "protocol_digest",
            "final protocol digest mismatch",
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> FinalEvaluationProtocolDTO:
        _require_exact_keys(
            payload, PROTOCOL_KEYS, "final protocol payload keys mismatch"
        )
        return cls(
            version=_str(payload, "version", "final protocol version mismatch"),
            protocol_id=_str(payload, "protocol_id", "final protocol id mismatch"),
            protocol_status=_str(
                payload, "protocol_status", "final protocol status mismatch"
            ),
            created_utc=_str(payload, "created_utc", "final protocol UTC mismatch"),
            approved_utc=_optional_str(
                payload, "approved_utc", "final protocol approval mismatch"
            ),
            approved_by=_optional_str(
                payload, "approved_by", "final protocol approval mismatch"
            ),
            benchmark_seed_digest=_str(
                payload, "benchmark_seed_digest", "final protocol seed mismatch"
            ),
            sealed_plan_digest=_str(
                payload, "sealed_plan_digest", "final protocol sealed plan mismatch"
            ),
            policy_artifact_id=_str(
                payload, "policy_artifact_id", "final protocol policy mismatch"
            ),
            candidate_set_id=_str(
                payload, "candidate_set_id", "final protocol candidate mismatch"
            ),
            selected_provider_id=_str(
                payload, "selected_provider_id", "final protocol provider mismatch"
            ),
            decision_rule=FinalJsonDTO.from_value(payload["decision_rule"]),
            required_evidence=FinalJsonDTO.from_value(payload["required_evidence"]),
            claim_rule_set=FinalJsonDTO.from_value(payload["claim_rule_set"]),
            review_notes=FinalJsonDTO.from_value(payload["review_notes"]),
            protocol_digest=_str(
                payload, "protocol_digest", "final protocol digest mismatch"
            ),
        )

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> FinalEvaluationProtocolDTO:
        _require_exact_keys(
            payload, PROTOCOL_KEYS[:-1], "final protocol payload keys mismatch"
        )
        return cls.from_dict(
            dict(payload)
            | {"protocol_digest": _digest_without_key(payload, "protocol_digest")}
        )

    @property
    def approved(self) -> bool:
        return self.protocol_status == "approved"

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "protocol_id": self.protocol_id,
            "protocol_status": self.protocol_status,
            "created_utc": self.created_utc,
            "approved_utc": self.approved_utc,
            "approved_by": self.approved_by,
            "benchmark_seed_digest": self.benchmark_seed_digest,
            "sealed_plan_digest": self.sealed_plan_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "candidate_set_id": self.candidate_set_id,
            "selected_provider_id": self.selected_provider_id,
            "decision_rule": self.decision_rule.to_value(),
            "required_evidence": self.required_evidence.to_value(),
            "claim_rule_set": self.claim_rule_set.to_value(),
            "review_notes": self.review_notes.to_value(),
            "protocol_digest": self.protocol_digest,
        }


@dataclass(frozen=True, slots=True)
class FinalExecutionAuthorizationDTO:
    version: str
    authorization_id: str
    authorization_status: str
    created_utc: str
    created_by: str
    protocol_digest: str
    expected_benchmark_seed_digest: str
    expected_sealed_plan_digest: str
    expected_policy_artifact_id: str
    output_dir: str
    database_path: str
    unattended_permitted: bool
    operator_confirmation_text: str
    authorization_payload: FinalJsonDTO
    authorization_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_EXECUTION_AUTHORIZATION_VERSION:
            raise VPMValidationError("final authorization version mismatch")
        validate_final_identifier(
            self.authorization_id,
            "final authorization id mismatch",
        )
        if self.authorization_status not in {"draft", "authorized", "retired"}:
            raise VPMValidationError("final authorization status mismatch")
        for value, message in (
            (self.protocol_digest, "final authorization protocol mismatch"),
            (
                self.expected_benchmark_seed_digest,
                "final authorization seed mismatch",
            ),
            (
                self.expected_sealed_plan_digest,
                "final authorization sealed plan mismatch",
            ),
            (self.authorization_digest, "final authorization digest mismatch"),
        ):
            _require_digest(value, message)
        _require_digest_match(
            self.to_dict(),
            "authorization_digest",
            "final authorization digest mismatch",
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, object],
    ) -> FinalExecutionAuthorizationDTO:
        _require_exact_keys(
            payload, AUTHORIZATION_KEYS, "final authorization payload keys mismatch"
        )
        return cls(
            version=_str(payload, "version", "final authorization version mismatch"),
            authorization_id=_str(
                payload, "authorization_id", "final authorization id mismatch"
            ),
            authorization_status=_str(
                payload, "authorization_status", "final authorization status mismatch"
            ),
            created_utc=_str(
                payload, "created_utc", "final authorization UTC mismatch"
            ),
            created_by=_str(
                payload, "created_by", "final authorization actor mismatch"
            ),
            protocol_digest=_str(
                payload, "protocol_digest", "final authorization protocol mismatch"
            ),
            expected_benchmark_seed_digest=_str(
                payload,
                "expected_benchmark_seed_digest",
                "final authorization seed mismatch",
            ),
            expected_sealed_plan_digest=_str(
                payload,
                "expected_sealed_plan_digest",
                "final authorization sealed plan mismatch",
            ),
            expected_policy_artifact_id=_str(
                payload,
                "expected_policy_artifact_id",
                "final authorization policy mismatch",
            ),
            output_dir=_str(payload, "output_dir", "final authorization path mismatch"),
            database_path=_str(
                payload, "database_path", "final authorization database mismatch"
            ),
            unattended_permitted=_bool(
                payload, "unattended_permitted", "final authorization mode mismatch"
            ),
            operator_confirmation_text=_str(
                payload,
                "operator_confirmation_text",
                "final authorization confirmation mismatch",
            ),
            authorization_payload=FinalJsonDTO.from_value(
                payload["authorization_payload"]
            ),
            authorization_digest=_str(
                payload, "authorization_digest", "final authorization digest mismatch"
            ),
        )

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> FinalExecutionAuthorizationDTO:
        _require_exact_keys(
            payload,
            AUTHORIZATION_KEYS[:-1],
            "final authorization payload keys mismatch",
        )
        return cls.from_dict(
            dict(payload)
            | {
                "authorization_digest": _digest_without_key(
                    payload, "authorization_digest"
                )
            }
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "authorization_id": self.authorization_id,
            "authorization_status": self.authorization_status,
            "created_utc": self.created_utc,
            "created_by": self.created_by,
            "protocol_digest": self.protocol_digest,
            "expected_benchmark_seed_digest": self.expected_benchmark_seed_digest,
            "expected_sealed_plan_digest": self.expected_sealed_plan_digest,
            "expected_policy_artifact_id": self.expected_policy_artifact_id,
            "output_dir": self.output_dir,
            "database_path": self.database_path,
            "unattended_permitted": self.unattended_permitted,
            "operator_confirmation_text": self.operator_confirmation_text,
            "authorization_payload": self.authorization_payload.to_value(),
            "authorization_digest": self.authorization_digest,
        }


@dataclass(frozen=True, slots=True)
class FinalExecutionRequestDTO:
    version: str
    output_dir: str
    authorization_file: str
    expected_authorization_digest: str
    expected_sealed_plan_digest: str
    database_path: str
    preflight_only: bool
    operator_identity: str
    unattended: bool
    request_payload: FinalJsonDTO
    request_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_EXECUTION_REQUEST_VERSION:
            raise VPMValidationError("final request version mismatch")
        _require_digest(
            self.expected_authorization_digest,
            "final request authorization digest mismatch",
        )
        _require_digest(
            self.expected_sealed_plan_digest,
            "final request sealed plan mismatch",
        )
        _require_digest(self.request_digest, "final request digest mismatch")
        _require_digest_match(
            self.to_dict(), "request_digest", "final request digest mismatch"
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FinalExecutionRequestDTO:
        _require_exact_keys(
            payload, REQUEST_KEYS, "final request payload keys mismatch"
        )
        return cls(
            version=_str(payload, "version", "final request version mismatch"),
            output_dir=_str(payload, "output_dir", "final request path mismatch"),
            authorization_file=_str(
                payload, "authorization_file", "final request authorization mismatch"
            ),
            expected_authorization_digest=_str(
                payload,
                "expected_authorization_digest",
                "final request authorization digest mismatch",
            ),
            expected_sealed_plan_digest=_str(
                payload,
                "expected_sealed_plan_digest",
                "final request sealed plan mismatch",
            ),
            database_path=_str(
                payload, "database_path", "final request database mismatch"
            ),
            preflight_only=_bool(
                payload, "preflight_only", "final request preflight mismatch"
            ),
            operator_identity=_str(
                payload, "operator_identity", "final request operator mismatch"
            ),
            unattended=_bool(payload, "unattended", "final request mode mismatch"),
            request_payload=FinalJsonDTO.from_value(payload["request_payload"]),
            request_digest=_str(
                payload, "request_digest", "final request digest mismatch"
            ),
        )

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> FinalExecutionRequestDTO:
        _require_exact_keys(
            payload, REQUEST_KEYS[:-1], "final request payload keys mismatch"
        )
        return cls.from_dict(
            dict(payload)
            | {"request_digest": _digest_without_key(payload, "request_digest")}
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "output_dir": self.output_dir,
            "authorization_file": self.authorization_file,
            "expected_authorization_digest": self.expected_authorization_digest,
            "expected_sealed_plan_digest": self.expected_sealed_plan_digest,
            "database_path": self.database_path,
            "preflight_only": self.preflight_only,
            "operator_identity": self.operator_identity,
            "unattended": self.unattended,
            "request_payload": self.request_payload.to_value(),
            "request_digest": self.request_digest,
        }


@dataclass(frozen=True, slots=True)
class FinalAccessRecordDTO:
    version: str
    access_id: str
    authorization_id: str
    state: str
    benchmark_seed_digest: str
    sealed_plan_digest: str
    protocol_digest: str
    authorization_digest: str
    created_utc: str
    updated_utc: str
    process_identity: str
    record_payload: FinalJsonDTO
    current_event_ordinal: int
    last_event_digest: str | None
    record_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_ACCESS_RECORD_VERSION:
            raise VPMValidationError("final access record version mismatch")
        if self.state not in FINAL_ACCESS_STATES:
            raise VPMValidationError("final access state mismatch")
        validate_final_identifier(self.access_id, "final access id mismatch")
        validate_final_identifier(
            self.authorization_id,
            "final access authorization mismatch",
        )
        if self.current_event_ordinal < -1:
            raise VPMValidationError("final access event ordinal mismatch")
        for value, message in (
            (self.benchmark_seed_digest, "final access seed mismatch"),
            (self.sealed_plan_digest, "final access sealed plan mismatch"),
            (self.protocol_digest, "final access protocol mismatch"),
            (self.authorization_digest, "final access authorization mismatch"),
            (self.record_digest, "final access record digest mismatch"),
        ):
            _require_digest(value, message)
        if self.last_event_digest is not None:
            _require_digest(
                self.last_event_digest, "final access event digest mismatch"
            )
        _require_digest_match(
            self.to_dict(), "record_digest", "final access record digest mismatch"
        )

    @classmethod
    def from_authorization(
        cls,
        authorization: FinalExecutionAuthorizationDTO,
        *,
        utc: str,
        process_identity: str,
        record_payload: Mapping[str, object] | None = None,
    ) -> FinalAccessRecordDTO:
        payload = {
            "version": FINAL_ACCESS_RECORD_VERSION,
            "access_id": access_id_for_authorization(authorization),
            "authorization_id": authorization.authorization_id,
            "state": "authorized",
            "benchmark_seed_digest": authorization.expected_benchmark_seed_digest,
            "sealed_plan_digest": authorization.expected_sealed_plan_digest,
            "protocol_digest": authorization.protocol_digest,
            "authorization_digest": authorization.authorization_digest,
            "created_utc": utc,
            "updated_utc": utc,
            "process_identity": process_identity,
            "record_payload": {} if record_payload is None else dict(record_payload),
            "current_event_ordinal": -1,
            "last_event_digest": None,
        }
        return cls.from_dict(
            payload | {"record_digest": _digest_without_key(payload, "record_digest")}
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FinalAccessRecordDTO:
        _require_exact_keys(
            payload, RECORD_KEYS, "final access record payload keys mismatch"
        )
        return cls(
            version=_str(payload, "version", "final access record version mismatch"),
            access_id=_str(payload, "access_id", "final access id mismatch"),
            authorization_id=_str(
                payload, "authorization_id", "final access authorization mismatch"
            ),
            state=_str(payload, "state", "final access state mismatch"),
            benchmark_seed_digest=_str(
                payload, "benchmark_seed_digest", "final access seed mismatch"
            ),
            sealed_plan_digest=_str(
                payload, "sealed_plan_digest", "final access sealed plan mismatch"
            ),
            protocol_digest=_str(
                payload, "protocol_digest", "final access protocol mismatch"
            ),
            authorization_digest=_str(
                payload, "authorization_digest", "final access authorization mismatch"
            ),
            created_utc=_str(payload, "created_utc", "final access UTC mismatch"),
            updated_utc=_str(payload, "updated_utc", "final access UTC mismatch"),
            process_identity=_str(
                payload, "process_identity", "final access process mismatch"
            ),
            record_payload=FinalJsonDTO.from_value(payload["record_payload"]),
            current_event_ordinal=_int(
                payload,
                "current_event_ordinal",
                "final access event ordinal mismatch",
            ),
            last_event_digest=_optional_str(
                payload, "last_event_digest", "final access event digest mismatch"
            ),
            record_digest=_str(
                payload, "record_digest", "final access record digest mismatch"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "access_id": self.access_id,
            "authorization_id": self.authorization_id,
            "state": self.state,
            "benchmark_seed_digest": self.benchmark_seed_digest,
            "sealed_plan_digest": self.sealed_plan_digest,
            "protocol_digest": self.protocol_digest,
            "authorization_digest": self.authorization_digest,
            "created_utc": self.created_utc,
            "updated_utc": self.updated_utc,
            "process_identity": self.process_identity,
            "record_payload": self.record_payload.to_value(),
            "current_event_ordinal": self.current_event_ordinal,
            "last_event_digest": self.last_event_digest,
            "record_digest": self.record_digest,
        }

    def with_state(
        self,
        *,
        state: str,
        utc: str,
        process_identity: str,
        current_event_ordinal: int,
        last_event_digest: str,
        record_payload: Mapping[str, object] | None = None,
    ) -> FinalAccessRecordDTO:
        payload = self.to_dict()
        payload.update(
            {
                "state": state,
                "updated_utc": utc,
                "process_identity": process_identity,
                "current_event_ordinal": current_event_ordinal,
                "last_event_digest": last_event_digest,
            }
        )
        if record_payload is not None:
            payload["record_payload"] = dict(record_payload)
        payload.pop("record_digest")
        return FinalAccessRecordDTO.from_dict(
            payload | {"record_digest": _digest_without_key(payload, "record_digest")}
        )


@dataclass(frozen=True, slots=True)
class FinalAccessEventDTO:
    version: str
    access_id: str
    authorization_id: str
    ordinal: int
    previous_state: str | None
    new_state: str
    utc: str
    process_identity: str
    event_payload: FinalJsonDTO
    previous_event_digest: str | None
    event_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_ACCESS_EVENT_VERSION:
            raise VPMValidationError("final access event version mismatch")
        validate_final_identifier(self.access_id, "final access id mismatch")
        validate_final_identifier(
            self.authorization_id,
            "final access authorization mismatch",
        )
        if self.ordinal < 0:
            raise VPMValidationError("final access event ordinal mismatch")
        payload = self.event_payload.to_value()
        kind = payload.get("kind") if isinstance(payload, Mapping) else None
        validate_final_access_event(self.previous_state, self.new_state, kind)
        if self.previous_event_digest is not None:
            _require_digest(
                self.previous_event_digest, "final access previous event mismatch"
            )
        _require_digest(self.event_digest, "final access event digest mismatch")
        _require_digest_match(
            self.to_dict(), "event_digest", "final access event digest mismatch"
        )

    @classmethod
    def build(
        cls,
        *,
        access_id: str,
        authorization_id: str,
        ordinal: int,
        previous_state: str | None,
        new_state: str,
        utc: str,
        process_identity: str,
        event_payload: Mapping[str, object] | None = None,
        previous_event_digest: str | None = None,
    ) -> FinalAccessEventDTO:
        payload = {
            "version": FINAL_ACCESS_EVENT_VERSION,
            "access_id": access_id,
            "authorization_id": authorization_id,
            "ordinal": ordinal,
            "previous_state": previous_state,
            "new_state": new_state,
            "utc": utc,
            "process_identity": process_identity,
            "event_payload": {} if event_payload is None else dict(event_payload),
            "previous_event_digest": previous_event_digest,
        }
        return cls.from_dict(
            payload | {"event_digest": _digest_without_key(payload, "event_digest")}
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FinalAccessEventDTO:
        _require_exact_keys(
            payload, EVENT_KEYS, "final access event payload keys mismatch"
        )
        return cls(
            version=_str(payload, "version", "final access event version mismatch"),
            access_id=_str(payload, "access_id", "final access id mismatch"),
            authorization_id=_str(
                payload, "authorization_id", "final access authorization mismatch"
            ),
            ordinal=_int(payload, "ordinal", "final access event ordinal mismatch"),
            previous_state=_optional_str(
                payload, "previous_state", "final access state mismatch"
            ),
            new_state=_str(payload, "new_state", "final access state mismatch"),
            utc=_str(payload, "utc", "final access UTC mismatch"),
            process_identity=_str(
                payload, "process_identity", "final access process mismatch"
            ),
            event_payload=FinalJsonDTO.from_value(payload["event_payload"]),
            previous_event_digest=_optional_str(
                payload,
                "previous_event_digest",
                "final access previous event mismatch",
            ),
            event_digest=_str(
                payload, "event_digest", "final access event digest mismatch"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "access_id": self.access_id,
            "authorization_id": self.authorization_id,
            "ordinal": self.ordinal,
            "previous_state": self.previous_state,
            "new_state": self.new_state,
            "utc": self.utc,
            "process_identity": self.process_identity,
            "event_payload": self.event_payload.to_value(),
            "previous_event_digest": self.previous_event_digest,
            "event_digest": self.event_digest,
        }


@dataclass(frozen=True, slots=True)
class FinalEvidenceBundleDTO:
    version: str
    access_id: str
    authorization_digest: str
    protocol_digest: str
    benchmark_seed_digest: str
    sealed_plan_digest: str
    execution_commit: str
    provider_order: tuple[str, ...]
    provider_versions: FinalJsonDTO
    expected_counts: FinalJsonDTO
    actual_counts: FinalJsonDTO
    rows: FinalJsonDTO
    evidence_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_EVIDENCE_BUNDLE_VERSION:
            raise VPMValidationError("final evidence version mismatch")
        validate_final_identifier(self.access_id, "final access id mismatch")
        for value, message in (
            (self.authorization_digest, "final evidence authorization mismatch"),
            (self.protocol_digest, "final evidence protocol mismatch"),
            (self.benchmark_seed_digest, "final evidence seed mismatch"),
            (self.sealed_plan_digest, "final evidence sealed plan mismatch"),
            (self.evidence_digest, "final evidence digest mismatch"),
        ):
            _require_digest(value, message)
        _validate_provider_contract(self.provider_order, self.provider_versions)
        expected = _count_mapping(
            self.expected_counts.to_value(),
            "final evidence expected counts mismatch",
        )
        actual = _count_mapping(
            self.actual_counts.to_value(),
            "final evidence actual counts mismatch",
        )
        rows = _evidence_rows(
            self.rows.to_value(),
            "final evidence rows mismatch",
        )
        if rows != _canonical_evidence_rows(rows):
            raise VPMValidationError("final evidence ordering mismatch")
        identities = {
            (
                row["family_id"],
                row["episode_id"],
                row["frame_ordinal"],
                row["frame_id"],
                row["provider_id"],
            )
            for row in rows
        }
        if len(identities) != len(rows):
            raise VPMValidationError("duplicate final evidence identity")
        if actual != _actual_counts(rows):
            raise VPMValidationError("final evidence actual counts mismatch")
        providers = {str(row["provider_id"]) for row in rows}
        if providers != set(self.provider_order):
            raise VPMValidationError("final evidence provider order mismatch")
        if any(actual[key] > expected[key] for key in FINAL_COUNT_KEYS):
            raise VPMValidationError("final evidence exceeds authorized counts")
        _require_digest_match(
            self.to_dict(),
            "evidence_digest",
            "final evidence digest mismatch",
        )

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> FinalEvidenceBundleDTO:
        expected_keys = tuple(
            key
            for key in EVIDENCE_BUNDLE_KEYS
            if key not in {"actual_counts", "evidence_digest"}
        )
        _require_exact_keys(payload, expected_keys, "final evidence keys mismatch")
        rows = _evidence_rows(payload["rows"], "final evidence rows mismatch")
        canonical_rows = _canonical_evidence_rows(rows)
        complete = dict(payload) | {
            "rows": canonical_rows,
            "actual_counts": _actual_counts(canonical_rows),
        }
        complete["evidence_digest"] = _digest_without_key(
            complete,
            "evidence_digest",
        )
        return cls.from_dict(complete)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FinalEvidenceBundleDTO:
        _require_exact_keys(
            payload, EVIDENCE_BUNDLE_KEYS, "final evidence keys mismatch"
        )
        return cls(
            version=_str(payload, "version", "final evidence version mismatch"),
            access_id=_str(payload, "access_id", "final access id mismatch"),
            authorization_digest=_str(
                payload,
                "authorization_digest",
                "final evidence authorization mismatch",
            ),
            protocol_digest=_str(
                payload,
                "protocol_digest",
                "final evidence protocol mismatch",
            ),
            benchmark_seed_digest=_str(
                payload,
                "benchmark_seed_digest",
                "final evidence seed mismatch",
            ),
            sealed_plan_digest=_str(
                payload,
                "sealed_plan_digest",
                "final evidence sealed plan mismatch",
            ),
            execution_commit=_str(
                payload,
                "execution_commit",
                "final evidence execution commit mismatch",
            ),
            provider_order=_string_tuple(
                payload["provider_order"],
                "final evidence provider order mismatch",
            ),
            provider_versions=FinalJsonDTO.from_value(payload["provider_versions"]),
            expected_counts=FinalJsonDTO.from_value(payload["expected_counts"]),
            actual_counts=FinalJsonDTO.from_value(payload["actual_counts"]),
            rows=FinalJsonDTO.from_value(payload["rows"]),
            evidence_digest=_str(
                payload,
                "evidence_digest",
                "final evidence digest mismatch",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "access_id": self.access_id,
            "authorization_digest": self.authorization_digest,
            "protocol_digest": self.protocol_digest,
            "benchmark_seed_digest": self.benchmark_seed_digest,
            "sealed_plan_digest": self.sealed_plan_digest,
            "execution_commit": self.execution_commit,
            "provider_order": list(self.provider_order),
            "provider_versions": self.provider_versions.to_value(),
            "expected_counts": self.expected_counts.to_value(),
            "actual_counts": self.actual_counts.to_value(),
            "rows": self.rows.to_value(),
            "evidence_digest": self.evidence_digest,
        }


@dataclass(frozen=True, slots=True)
class FinalEvaluationResultDTO:
    version: str
    protocol_digest: str
    evidence_digest: str
    decision: str
    descriptive_measurements: FinalJsonDTO
    family_measurements: FinalJsonDTO
    rejections: FinalJsonDTO
    indeterminate_reasons: FinalJsonDTO
    actual_counts: FinalJsonDTO
    evaluation_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_EVALUATION_RESULT_VERSION:
            raise VPMValidationError("final evaluation result version mismatch")
        if self.decision not in {"passed", "failed", "indeterminate"}:
            raise VPMValidationError("final evaluation decision mismatch")
        for value, message in (
            (self.protocol_digest, "final evaluation protocol mismatch"),
            (self.evidence_digest, "final evaluation evidence mismatch"),
            (self.evaluation_digest, "final evaluation digest mismatch"),
        ):
            _require_digest(value, message)
        _count_mapping(
            self.actual_counts.to_value(),
            "final evaluation actual counts mismatch",
        )
        if not isinstance(self.rejections.to_value(), list):
            raise VPMValidationError("final evaluation rejections mismatch")
        reasons = self.indeterminate_reasons.to_value()
        if not isinstance(reasons, list):
            raise VPMValidationError("final evaluation reasons mismatch")
        if self.decision == "indeterminate" and not reasons:
            raise VPMValidationError("final evaluation reasons mismatch")
        _require_digest_match(
            self.to_dict(),
            "evaluation_digest",
            "final evaluation digest mismatch",
        )

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> FinalEvaluationResultDTO:
        _require_exact_keys(
            payload,
            EVALUATION_RESULT_KEYS[:-1],
            "final evaluation result keys mismatch",
        )
        return cls.from_dict(
            dict(payload)
            | {
                "evaluation_digest": _digest_without_key(
                    payload,
                    "evaluation_digest",
                )
            }
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FinalEvaluationResultDTO:
        _require_exact_keys(
            payload,
            EVALUATION_RESULT_KEYS,
            "final evaluation result keys mismatch",
        )
        return cls(
            version=_str(
                payload,
                "version",
                "final evaluation result version mismatch",
            ),
            protocol_digest=_str(
                payload,
                "protocol_digest",
                "final evaluation protocol mismatch",
            ),
            evidence_digest=_str(
                payload,
                "evidence_digest",
                "final evaluation evidence mismatch",
            ),
            decision=_str(
                payload,
                "decision",
                "final evaluation decision mismatch",
            ),
            descriptive_measurements=FinalJsonDTO.from_value(
                payload["descriptive_measurements"]
            ),
            family_measurements=FinalJsonDTO.from_value(payload["family_measurements"]),
            rejections=FinalJsonDTO.from_value(payload["rejections"]),
            indeterminate_reasons=FinalJsonDTO.from_value(
                payload["indeterminate_reasons"]
            ),
            actual_counts=FinalJsonDTO.from_value(payload["actual_counts"]),
            evaluation_digest=_str(
                payload,
                "evaluation_digest",
                "final evaluation digest mismatch",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "protocol_digest": self.protocol_digest,
            "evidence_digest": self.evidence_digest,
            "decision": self.decision,
            "descriptive_measurements": self.descriptive_measurements.to_value(),
            "family_measurements": self.family_measurements.to_value(),
            "rejections": self.rejections.to_value(),
            "indeterminate_reasons": self.indeterminate_reasons.to_value(),
            "actual_counts": self.actual_counts.to_value(),
            "evaluation_digest": self.evaluation_digest,
        }


@dataclass(frozen=True, slots=True)
class FinalExecutionReceiptDTO:
    version: str
    access_id: str
    authorization_id: str
    state: str
    completed_utc: str
    benchmark_seed_digest: str
    sealed_plan_digest: str
    protocol_digest: str
    authorization_digest: str
    evidence_digest: str
    evaluation_digest: str
    artifact_manifest_digest: str
    event_chain_digest: str
    decision: str
    execution_commit: str
    provider_order: tuple[str, ...]
    provider_versions: FinalJsonDTO
    expected_counts: FinalJsonDTO
    actual_counts: FinalJsonDTO
    historical_authority_id: str
    historical_database_sha256: str
    historical_evidence_manifest_digest: str
    historical_authority_digest: str
    receipt_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_EXECUTION_RECEIPT_VERSION or self.state != "completed":
            raise VPMValidationError("final receipt state mismatch")
        if self.decision not in {"passed", "failed", "indeterminate"}:
            raise VPMValidationError("final receipt decision mismatch")
        for value, message in (
            (self.benchmark_seed_digest, "final receipt seed mismatch"),
            (self.sealed_plan_digest, "final receipt sealed plan mismatch"),
            (self.protocol_digest, "final receipt protocol mismatch"),
            (self.authorization_digest, "final receipt authorization mismatch"),
            (self.evidence_digest, "final receipt evidence mismatch"),
            (self.evaluation_digest, "final receipt evaluation mismatch"),
            (self.artifact_manifest_digest, "final receipt manifest mismatch"),
            (self.event_chain_digest, "final receipt event chain mismatch"),
            (
                self.historical_database_sha256,
                "final receipt historical database mismatch",
            ),
            (
                self.historical_evidence_manifest_digest,
                "final receipt historical evidence manifest mismatch",
            ),
            (
                self.historical_authority_digest,
                "final receipt historical authority mismatch",
            ),
            (self.receipt_digest, "final receipt digest mismatch"),
        ):
            _require_digest(value, message)
        validate_final_identifier(
            self.historical_authority_id,
            "final receipt historical authority id mismatch",
        )
        _validate_provider_contract(self.provider_order, self.provider_versions)
        _count_mapping(
            self.expected_counts.to_value(),
            "final receipt expected counts mismatch",
        )
        _count_mapping(
            self.actual_counts.to_value(),
            "final receipt actual counts mismatch",
        )
        _require_digest_match(
            self.to_dict(), "receipt_digest", "final receipt digest mismatch"
        )

    @classmethod
    def create(
        cls,
        payload: Mapping[str, object],
    ) -> FinalExecutionReceiptDTO:
        _require_exact_keys(payload, RECEIPT_KEYS[:-1], "final receipt keys mismatch")
        return cls.from_dict(
            dict(payload)
            | {"receipt_digest": _digest_without_key(payload, "receipt_digest")}
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FinalExecutionReceiptDTO:
        _require_exact_keys(payload, RECEIPT_KEYS, "final receipt keys mismatch")
        return cls(
            version=_str(payload, "version", "final receipt version mismatch"),
            access_id=_str(payload, "access_id", "final access id mismatch"),
            authorization_id=_str(
                payload, "authorization_id", "final access authorization mismatch"
            ),
            state=_str(payload, "state", "final receipt state mismatch"),
            completed_utc=_str(payload, "completed_utc", "final receipt UTC mismatch"),
            benchmark_seed_digest=_str(
                payload, "benchmark_seed_digest", "final receipt seed mismatch"
            ),
            sealed_plan_digest=_str(
                payload, "sealed_plan_digest", "final receipt sealed plan mismatch"
            ),
            protocol_digest=_str(
                payload, "protocol_digest", "final receipt protocol mismatch"
            ),
            authorization_digest=_str(
                payload, "authorization_digest", "final receipt authorization mismatch"
            ),
            evidence_digest=_str(
                payload, "evidence_digest", "final receipt evidence mismatch"
            ),
            evaluation_digest=_str(
                payload,
                "evaluation_digest",
                "final receipt evaluation mismatch",
            ),
            artifact_manifest_digest=_str(
                payload,
                "artifact_manifest_digest",
                "final receipt manifest mismatch",
            ),
            event_chain_digest=_str(
                payload, "event_chain_digest", "final receipt event chain mismatch"
            ),
            decision=_str(payload, "decision", "final receipt decision mismatch"),
            execution_commit=_str(
                payload,
                "execution_commit",
                "final receipt execution commit mismatch",
            ),
            provider_order=_string_tuple(
                payload["provider_order"],
                "final receipt provider order mismatch",
            ),
            provider_versions=FinalJsonDTO.from_value(payload["provider_versions"]),
            expected_counts=FinalJsonDTO.from_value(payload["expected_counts"]),
            actual_counts=FinalJsonDTO.from_value(payload["actual_counts"]),
            historical_authority_id=_str(
                payload,
                "historical_authority_id",
                "final receipt historical authority id mismatch",
            ),
            historical_database_sha256=_str(
                payload,
                "historical_database_sha256",
                "final receipt historical database mismatch",
            ),
            historical_evidence_manifest_digest=_str(
                payload,
                "historical_evidence_manifest_digest",
                "final receipt historical evidence manifest mismatch",
            ),
            historical_authority_digest=_str(
                payload,
                "historical_authority_digest",
                "final receipt historical authority mismatch",
            ),
            receipt_digest=_str(
                payload, "receipt_digest", "final receipt digest mismatch"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "access_id": self.access_id,
            "authorization_id": self.authorization_id,
            "state": self.state,
            "completed_utc": self.completed_utc,
            "benchmark_seed_digest": self.benchmark_seed_digest,
            "sealed_plan_digest": self.sealed_plan_digest,
            "protocol_digest": self.protocol_digest,
            "authorization_digest": self.authorization_digest,
            "evidence_digest": self.evidence_digest,
            "evaluation_digest": self.evaluation_digest,
            "artifact_manifest_digest": self.artifact_manifest_digest,
            "event_chain_digest": self.event_chain_digest,
            "decision": self.decision,
            "execution_commit": self.execution_commit,
            "provider_order": list(self.provider_order),
            "provider_versions": self.provider_versions.to_value(),
            "expected_counts": self.expected_counts.to_value(),
            "actual_counts": self.actual_counts.to_value(),
            "historical_authority_id": self.historical_authority_id,
            "historical_database_sha256": self.historical_database_sha256,
            "historical_evidence_manifest_digest": (
                self.historical_evidence_manifest_digest
            ),
            "historical_authority_digest": self.historical_authority_digest,
            "receipt_digest": self.receipt_digest,
        }


@dataclass(frozen=True, slots=True)
class FinalExecutionFailureDTO:
    version: str
    access_id: str
    authorization_id: str
    state: str
    failed_utc: str
    benchmark_seed_digest: str
    sealed_plan_digest: str
    protocol_digest: str
    authorization_digest: str
    failure_kind: str
    error_code: str
    error_message: str
    event_chain_digest: str
    failure_digest: str

    def __post_init__(self) -> None:
        if self.version != FINAL_EXECUTION_FAILURE_VERSION:
            raise VPMValidationError("final failure version mismatch")
        if self.state not in {"failed", "interrupted"}:
            raise VPMValidationError("final failure state mismatch")
        for value, message in (
            (self.benchmark_seed_digest, "final failure seed mismatch"),
            (self.sealed_plan_digest, "final failure sealed plan mismatch"),
            (self.protocol_digest, "final failure protocol mismatch"),
            (self.authorization_digest, "final failure authorization mismatch"),
            (self.event_chain_digest, "final failure event chain mismatch"),
            (self.failure_digest, "final failure digest mismatch"),
        ):
            _require_digest(value, message)
        _require_digest_match(
            self.to_dict(), "failure_digest", "final failure digest mismatch"
        )

    @classmethod
    def create(
        cls,
        payload: Mapping[str, object],
    ) -> FinalExecutionFailureDTO:
        _require_exact_keys(payload, FAILURE_KEYS[:-1], "final failure keys mismatch")
        return cls.from_dict(
            dict(payload)
            | {"failure_digest": _digest_without_key(payload, "failure_digest")}
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> FinalExecutionFailureDTO:
        _require_exact_keys(payload, FAILURE_KEYS, "final failure keys mismatch")
        return cls(
            version=_str(payload, "version", "final failure version mismatch"),
            access_id=_str(payload, "access_id", "final access id mismatch"),
            authorization_id=_str(
                payload, "authorization_id", "final access authorization mismatch"
            ),
            state=_str(payload, "state", "final failure state mismatch"),
            failed_utc=_str(payload, "failed_utc", "final failure UTC mismatch"),
            benchmark_seed_digest=_str(
                payload, "benchmark_seed_digest", "final failure seed mismatch"
            ),
            sealed_plan_digest=_str(
                payload, "sealed_plan_digest", "final failure sealed plan mismatch"
            ),
            protocol_digest=_str(
                payload, "protocol_digest", "final failure protocol mismatch"
            ),
            authorization_digest=_str(
                payload, "authorization_digest", "final failure authorization mismatch"
            ),
            failure_kind=_str(payload, "failure_kind", "final failure kind mismatch"),
            error_code=_str(payload, "error_code", "final failure code mismatch"),
            error_message=_str(
                payload, "error_message", "final failure message mismatch"
            ),
            event_chain_digest=_str(
                payload, "event_chain_digest", "final failure event chain mismatch"
            ),
            failure_digest=_str(
                payload, "failure_digest", "final failure digest mismatch"
            ),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "access_id": self.access_id,
            "authorization_id": self.authorization_id,
            "state": self.state,
            "failed_utc": self.failed_utc,
            "benchmark_seed_digest": self.benchmark_seed_digest,
            "sealed_plan_digest": self.sealed_plan_digest,
            "protocol_digest": self.protocol_digest,
            "authorization_digest": self.authorization_digest,
            "failure_kind": self.failure_kind,
            "error_code": self.error_code,
            "error_message": self.error_message,
            "event_chain_digest": self.event_chain_digest,
            "failure_digest": self.failure_digest,
        }


def access_id_for_authorization(
    authorization: FinalExecutionAuthorizationDTO,
) -> str:
    return f"final-access:{authorization.authorization_id}"


def validate_final_access_transition(
    previous_state: str | None,
    new_state: str,
) -> None:
    if previous_state is not None and previous_state not in FINAL_ACCESS_STATES:
        raise VPMValidationError("final access state mismatch")
    if new_state not in FINAL_ACCESS_STATES:
        raise VPMValidationError("final access state mismatch")
    if previous_state in FINAL_ACCESS_TERMINAL_STATES:
        raise VPMValidationError("final access terminal state is immutable")
    if (previous_state, new_state) not in FINAL_ACCESS_TRANSITIONS:
        raise VPMValidationError("final access state transition mismatch")


def validate_final_access_event(
    previous_state: str | None,
    new_state: str,
    kind: object,
) -> None:
    if previous_state == new_state:
        allowed = (
            previous_state == "running"
            and isinstance(kind, str)
            and kind
            in {
                "final_materialization_started",
                "final_materialization_completed",
                "provider_scoring_started",
                "provider_scoring_completed",
                "reachability_started",
                "reachability_completed",
                "evaluation_started",
                "evaluation_completed",
                "staged_artifact_generation_started",
                "staged_artifact_generation_completed",
                "artifact_validation_started",
                "artifact_validation_completed",
                "promotion_started",
                "promotion_completed",
            }
        ) or (
            previous_state == "completed"
            and isinstance(kind, str)
            and kind
            in {
                "receipt_publication_started",
                "receipt_publication_completed",
            }
        )
        if not allowed:
            raise VPMValidationError("final access event kind mismatch")
        return
    validate_final_access_transition(previous_state, new_state)


def event_chain_digest(events: tuple[FinalAccessEventDTO, ...]) -> str:
    return canonical_sha256([event.to_dict() for event in events])


def json_text(payload: Mapping[str, object]) -> str:
    return canonical_json_text(dict(payload))


def _require_exact_keys(
    payload: Mapping[str, object],
    keys: tuple[str, ...],
    message: str,
) -> None:
    if set(payload) != set(keys):
        raise VPMValidationError(message)


def _json_value(value: object) -> JsonValue:
    if value is None or isinstance(value, (str, int, float, bool)):
        return cast(JsonValue, value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    raise VPMValidationError("final access JSON value mismatch")


def _str(payload: Mapping[str, object], key: str, message: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise VPMValidationError(message)
    return value


def _optional_str(
    payload: Mapping[str, object],
    key: str,
    message: str,
) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise VPMValidationError(message)
    return value


def _bool(payload: Mapping[str, object], key: str, message: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise VPMValidationError(message)
    return value


def _int(payload: Mapping[str, object], key: str, message: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise VPMValidationError(message)
    return value


def _require_digest(value: str, message: str) -> None:
    if SHA256_RE.fullmatch(value) is None:
        raise VPMValidationError(message)


def _digest_without_key(payload: Mapping[str, object], digest_key: str) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != digest_key}
    )


def _require_digest_match(
    payload: Mapping[str, object],
    digest_key: str,
    message: str,
) -> None:
    if _digest_without_key(payload, digest_key) != payload[digest_key]:
        raise VPMValidationError(message)


def validate_final_identifier(value: str, message: str) -> None:
    if not value or FINAL_IDENTIFIER_RE.fullmatch(value) is None:
        raise VPMValidationError(message)


def _string_tuple(value: object, message: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise VPMValidationError(message)
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise VPMValidationError(message)
    if len(set(result)) != len(result):
        raise VPMValidationError(message)
    return cast(tuple[str, ...], result)


def _validate_provider_contract(
    provider_order: tuple[str, ...],
    provider_versions: FinalJsonDTO,
) -> None:
    if not provider_order or len(set(provider_order)) != len(provider_order):
        raise VPMValidationError("final evidence provider order mismatch")
    for provider_id in provider_order:
        validate_final_identifier(
            provider_id,
            "final evidence provider order mismatch",
        )
    versions = provider_versions.to_value()
    if not isinstance(versions, Mapping) or set(versions) != set(provider_order):
        raise VPMValidationError("final evidence provider versions mismatch")
    if any(not isinstance(value, str) or not value for value in versions.values()):
        raise VPMValidationError("final evidence provider versions mismatch")


def _count_mapping(value: object, message: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(FINAL_COUNT_KEYS):
        raise VPMValidationError(message)
    result: dict[str, int] = {}
    for key in FINAL_COUNT_KEYS:
        item = value[key]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise VPMValidationError(message)
        result[key] = item
    return result


def _evidence_rows(value: object, message: str) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise VPMValidationError(message)
    result: list[dict[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping) or set(item) != set(FINAL_EVIDENCE_ROW_KEYS):
            raise VPMValidationError(message)
        row = dict(item)
        for key in ("family_id", "episode_id", "frame_id", "provider_id"):
            field = row[key]
            if not isinstance(field, str):
                raise VPMValidationError(message)
            validate_final_identifier(field, message)
        ordinal = row["frame_ordinal"]
        if not isinstance(ordinal, int) or isinstance(ordinal, bool) or ordinal < 0:
            raise VPMValidationError(message)
        if row["split"] != "final":
            raise VPMValidationError("final evaluation evidence split mismatch")
        metrics = row["metrics"]
        if not isinstance(metrics, Mapping) or not metrics:
            raise VPMValidationError(message)
        if any(not isinstance(key, str) or not key for key in metrics):
            raise VPMValidationError(message)
        result.append(row)
    return result


def _canonical_evidence_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    return sorted(
        rows,
        key=lambda row: (
            cast(str, row["family_id"]),
            cast(str, row["episode_id"]),
            cast(int, row["frame_ordinal"]),
            cast(str, row["frame_id"]),
            cast(str, row["provider_id"]),
        ),
    )


def _actual_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        "evidence_row_count": len(rows),
        "episode_count": len({cast(str, row["episode_id"]) for row in rows}),
        "frame_count": len({cast(str, row["frame_id"]) for row in rows}),
        "provider_count": len({cast(str, row["provider_id"]) for row in rows}),
    }


__all__ = [
    "FINAL_ACCESS_EVENT_VERSION",
    "FINAL_ACCESS_RECORD_VERSION",
    "FINAL_ACCESS_STATES",
    "FINAL_ACCESS_TERMINAL_STATES",
    "FINAL_EVIDENCE_BUNDLE_VERSION",
    "FINAL_EVALUATION_PROTOCOL_VERSION",
    "FINAL_EVALUATION_RESULT_VERSION",
    "FINAL_EXECUTION_AUTHORIZATION_VERSION",
    "FINAL_EXECUTION_FAILURE_VERSION",
    "FINAL_EXECUTION_RECEIPT_VERSION",
    "FINAL_EXECUTION_REQUEST_VERSION",
    "FinalAccessEventDTO",
    "FinalAccessRecordDTO",
    "FinalEvidenceBundleDTO",
    "FinalEvaluationProtocolDTO",
    "FinalEvaluationResultDTO",
    "FinalExecutionAuthorizationDTO",
    "FinalExecutionFailureDTO",
    "FinalExecutionReceiptDTO",
    "FinalExecutionRequestDTO",
    "FinalJsonDTO",
    "access_id_for_authorization",
    "event_chain_digest",
    "json_text",
    "validate_final_access_transition",
    "validate_final_access_event",
    "validate_final_identifier",
]
