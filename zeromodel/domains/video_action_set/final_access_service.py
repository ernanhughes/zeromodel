from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import socket
from typing import Any

from ...artifact import VPMValidationError
from .final_access_dto import (
    FINAL_EXECUTION_FAILURE_VERSION,
    FINAL_EXECUTION_RECEIPT_VERSION,
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionFailureDTO,
    FinalExecutionReceiptDTO,
    FinalExecutionRequestDTO,
    event_chain_digest,
)
from .observation_dto import MaterializedObservationDTO, ObservationDTO
from .store import VideoActionSetStore


FinalExecutor = Callable[
    [FinalAccessRecordDTO, FinalExecutionRequestDTO], Mapping[str, object]
]


@dataclass(frozen=True, slots=True)
class FinalAccessService:
    store: VideoActionSetStore
    final_executor: FinalExecutor | None = None

    def create_authorization(
        self,
        authorization: FinalExecutionAuthorizationDTO,
        *,
        process_identity: str | None = None,
        utc: str | None = None,
    ) -> FinalAccessRecordDTO:
        process_identity = process_identity or default_process_identity()
        utc = utc or utc_now()
        record = FinalAccessRecordDTO.from_authorization(
            authorization,
            utc=utc,
            process_identity=process_identity,
            record_payload={"authorization_digest": authorization.authorization_digest},
        )
        event = FinalAccessEventDTO.build(
            access_id=record.access_id,
            authorization_id=record.authorization_id,
            ordinal=0,
            previous_state=None,
            new_state="authorized",
            utc=utc,
            process_identity=process_identity,
            event_payload={
                "kind": "authorization_created",
                "authorization_digest": authorization.authorization_digest,
            },
            previous_event_digest=None,
        )
        record = record.with_state(
            state="authorized",
            utc=utc,
            process_identity=process_identity,
            last_event_digest=event.event_digest,
            record_payload={"authorization_digest": authorization.authorization_digest},
        )
        return self.store.create_final_authorization(authorization, record, event)

    def load_authorization(
        self,
        authorization_id: str,
    ) -> FinalExecutionAuthorizationDTO | None:
        return self.store.load_final_authorization(authorization_id)

    def load_record(self, access_id: str) -> FinalAccessRecordDTO | None:
        return self.store.load_final_access_record(access_id)

    def list_events(self, access_id: str) -> tuple[FinalAccessEventDTO, ...]:
        return self.store.list_final_access_events(access_id)

    def reserve(
        self,
        access_id: str,
        *,
        process_identity: str | None = None,
        utc: str | None = None,
        event_payload: Mapping[str, object] | None = None,
    ) -> FinalAccessRecordDTO:
        record, event = self._next_transition(
            access_id,
            "reserved",
            process_identity=process_identity,
            utc=utc,
            event_payload=event_payload or {"kind": "reserved"},
        )
        return self.store.reserve_final_access(record, event)

    def mark_running(
        self,
        access_id: str,
        *,
        process_identity: str | None = None,
        utc: str | None = None,
        event_payload: Mapping[str, object] | None = None,
    ) -> FinalAccessRecordDTO:
        record, event = self._next_transition(
            access_id,
            "running",
            process_identity=process_identity,
            utc=utc,
            event_payload=event_payload or {"kind": "running"},
        )
        return self.store.mark_final_access_running(record, event)

    def complete(
        self,
        access_id: str,
        *,
        evidence_digest: str,
        measurements: Mapping[str, object],
        process_identity: str | None = None,
        utc: str | None = None,
    ) -> FinalAccessRecordDTO:
        process_identity = process_identity or default_process_identity()
        utc = utc or utc_now()
        current = self._require_record(access_id)
        events = self.store.list_final_access_events(access_id)
        event = self._build_event(
            current,
            "completed",
            process_identity=process_identity,
            utc=utc,
            event_payload={"kind": "completed", "evidence_digest": evidence_digest},
            previous_events=events,
        )
        receipt = FinalExecutionReceiptDTO.create(
            {
                "version": FINAL_EXECUTION_RECEIPT_VERSION,
                "access_id": current.access_id,
                "authorization_id": current.authorization_id,
                "state": "completed",
                "completed_utc": utc,
                "benchmark_seed_digest": current.benchmark_seed_digest,
                "sealed_plan_digest": current.sealed_plan_digest,
                "protocol_digest": current.protocol_digest,
                "authorization_digest": current.authorization_digest,
                "evidence_digest": evidence_digest,
                "event_chain_digest": event_chain_digest((*events, event)),
                "measurements": dict(measurements),
            }
        )
        record = current.with_state(
            state="completed",
            utc=utc,
            process_identity=process_identity,
            last_event_digest=event.event_digest,
            record_payload={"receipt": receipt.to_dict()},
        )
        return self.store.complete_final_access(record, event, receipt)

    def fail(
        self,
        access_id: str,
        *,
        failure_kind: str,
        error_code: str,
        error_message: str,
        process_identity: str | None = None,
        utc: str | None = None,
    ) -> FinalAccessRecordDTO:
        failure, record, event = self._failure_transition(
            access_id,
            "failed",
            failure_kind=failure_kind,
            error_code=error_code,
            error_message=error_message,
            process_identity=process_identity,
            utc=utc,
        )
        return self.store.fail_final_access(record, event, failure)

    def interrupt(
        self,
        access_id: str,
        *,
        failure_kind: str,
        error_code: str,
        error_message: str,
        process_identity: str | None = None,
        utc: str | None = None,
    ) -> FinalAccessRecordDTO:
        failure, record, event = self._failure_transition(
            access_id,
            "interrupted",
            failure_kind=failure_kind,
            error_code=error_code,
            error_message=error_message,
            process_identity=process_identity,
            utc=utc,
        )
        return self.store.interrupt_final_access(record, event, failure)

    def save_final_observation_record(
        self,
        access_id: str,
        record: Mapping[str, object],
    ) -> ObservationDTO:
        access = self._require_record(access_id)
        item = MaterializedObservationDTO.from_authorized_final_record(
            record,
            final_access_id=access.access_id,
        )
        return self.store.save_authorized_final_observations(access, (item,))[0]

    def execute_final_once(self, request: FinalExecutionRequestDTO) -> dict[str, Any]:
        preflight = self.preflight_final_execution(request)
        if request.preflight_only:
            return preflight
        if self.final_executor is None:
            raise VPMValidationError(
                "final execution requires an explicit registered final executor"
            )
        authorization = load_final_authorization_file(Path(request.authorization_file))
        existing = self.store.load_final_authorization(authorization.authorization_id)
        if existing is None:
            access = self.create_authorization(
                authorization,
                process_identity=request.operator_identity,
            )
        else:
            access = self._require_record(f"final-access:{authorization.authorization_id}")
        access = self.reserve(
            access.access_id,
            process_identity=request.operator_identity,
            event_payload={"kind": "reserved_by_execute_final_once"},
        )
        access = self.mark_running(
            access.access_id,
            process_identity=request.operator_identity,
            event_payload={"kind": "running_by_execute_final_once"},
        )
        try:
            result = self.final_executor(access, request)
            evidence_digest = _mapping_digest(result, "evidence_digest")
            return self.complete(
                access.access_id,
                evidence_digest=evidence_digest,
                measurements=dict(result),
                process_identity=request.operator_identity,
            ).to_dict()
        except Exception as exc:
            self.fail(
                access.access_id,
                failure_kind="executor_exception",
                error_code=exc.__class__.__name__,
                error_message=str(exc),
                process_identity=request.operator_identity,
            )
            raise

    def preflight_final_execution(
        self,
        request: FinalExecutionRequestDTO,
    ) -> dict[str, Any]:
        authorization = load_final_authorization_file(Path(request.authorization_file))
        if authorization.authorization_digest != request.expected_authorization_digest:
            raise VPMValidationError("final request authorization digest mismatch")
        if (
            authorization.expected_sealed_plan_digest
            != request.expected_sealed_plan_digest
        ):
            raise VPMValidationError("final request sealed plan mismatch")
        if authorization.output_dir != request.output_dir:
            raise VPMValidationError("final request output directory mismatch")
        if authorization.database_path != request.database_path:
            raise VPMValidationError("final request database mismatch")
        if request.unattended and not authorization.unattended_permitted:
            raise VPMValidationError("final authorization does not permit unattended use")
        if authorization.authorization_status != "authorized":
            raise VPMValidationError("final authorization status mismatch")
        return {
            "preflight": "passed",
            "read_only": True,
            "authorization_id": authorization.authorization_id,
            "authorization_digest": authorization.authorization_digest,
            "sealed_plan_digest": authorization.expected_sealed_plan_digest,
            "reservation_created": False,
        }

    def _next_transition(
        self,
        access_id: str,
        new_state: str,
        *,
        process_identity: str | None,
        utc: str | None,
        event_payload: Mapping[str, object],
    ) -> tuple[FinalAccessRecordDTO, FinalAccessEventDTO]:
        process_identity = process_identity or default_process_identity()
        utc = utc or utc_now()
        current = self._require_record(access_id)
        events = self.store.list_final_access_events(access_id)
        event = self._build_event(
            current,
            new_state,
            process_identity=process_identity,
            utc=utc,
            event_payload=event_payload,
            previous_events=events,
        )
        record = current.with_state(
            state=new_state,
            utc=utc,
            process_identity=process_identity,
            last_event_digest=event.event_digest,
        )
        return record, event

    def _failure_transition(
        self,
        access_id: str,
        state: str,
        *,
        failure_kind: str,
        error_code: str,
        error_message: str,
        process_identity: str | None,
        utc: str | None,
    ) -> tuple[
        FinalExecutionFailureDTO,
        FinalAccessRecordDTO,
        FinalAccessEventDTO,
    ]:
        process_identity = process_identity or default_process_identity()
        utc = utc or utc_now()
        current = self._require_record(access_id)
        events = self.store.list_final_access_events(access_id)
        event = self._build_event(
            current,
            state,
            process_identity=process_identity,
            utc=utc,
            event_payload={
                "kind": state,
                "failure_kind": failure_kind,
                "error_code": error_code,
            },
            previous_events=events,
        )
        failure = FinalExecutionFailureDTO.create(
            {
                "version": FINAL_EXECUTION_FAILURE_VERSION,
                "access_id": current.access_id,
                "authorization_id": current.authorization_id,
                "state": state,
                "failed_utc": utc,
                "benchmark_seed_digest": current.benchmark_seed_digest,
                "sealed_plan_digest": current.sealed_plan_digest,
                "protocol_digest": current.protocol_digest,
                "authorization_digest": current.authorization_digest,
                "failure_kind": failure_kind,
                "error_code": error_code,
                "error_message": error_message,
                "event_chain_digest": event_chain_digest((*events, event)),
            }
        )
        record = current.with_state(
            state=state,
            utc=utc,
            process_identity=process_identity,
            last_event_digest=event.event_digest,
            record_payload={"failure": failure.to_dict()},
        )
        return failure, record, event

    @staticmethod
    def _build_event(
        current: FinalAccessRecordDTO,
        new_state: str,
        *,
        process_identity: str,
        utc: str,
        event_payload: Mapping[str, object],
        previous_events: tuple[FinalAccessEventDTO, ...],
    ) -> FinalAccessEventDTO:
        previous_digest = None if not previous_events else previous_events[-1].event_digest
        return FinalAccessEventDTO.build(
            access_id=current.access_id,
            authorization_id=current.authorization_id,
            ordinal=len(previous_events),
            previous_state=current.state,
            new_state=new_state,
            utc=utc,
            process_identity=process_identity,
            event_payload=event_payload,
            previous_event_digest=previous_digest,
        )

    def _require_record(self, access_id: str) -> FinalAccessRecordDTO:
        record = self.store.load_final_access_record(access_id)
        if record is None:
            raise VPMValidationError("final access record is missing")
        return record


def load_final_authorization_file(path: Path) -> FinalExecutionAuthorizationDTO:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise VPMValidationError("final authorization payload keys mismatch")
    return FinalExecutionAuthorizationDTO.from_dict(payload)


def default_process_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _mapping_digest(payload: Mapping[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        raise VPMValidationError("final executor evidence digest mismatch")
    return value


__all__ = [
    "FinalAccessService",
    "FinalExecutor",
    "default_process_identity",
    "load_final_authorization_file",
    "utc_now",
]
