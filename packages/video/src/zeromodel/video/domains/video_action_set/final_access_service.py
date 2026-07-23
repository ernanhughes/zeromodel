from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
from typing import Any, Protocol, cast

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FINAL_EVIDENCE_BUNDLE_VERSION,
    FINAL_EXECUTION_FAILURE_VERSION,
    FINAL_EXECUTION_RECEIPT_VERSION,
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalEvaluationProtocolDTO,
    FinalEvaluationResultDTO,
    FinalEvidenceBundleDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionFailureDTO,
    FinalExecutionReceiptDTO,
    FinalExecutionRequestDTO,
    access_id_for_authorization,
)
from zeromodel.video.domains.video_action_set.final_evaluation import (
    evaluate_final_protocol,
)
from zeromodel.video.domains.video_action_set.final_historical_authority import (
    HISTORICAL_AUTHORITY_KEYS,
    VerifiedHistoricalAuthorityDTO,
    verify_historical_authority,
)
from zeromodel.video.domains.video_action_set.final_publication import (
    FinalArtifactManifestDTO,
    promote_staged_artifacts,
    publish_receipt_last,
    stage_final_artifacts,
    validate_canonical_artifacts,
    validate_staged_artifacts,
)
from zeromodel.video.domains.video_action_set.final_reconstruction import (
    reconstruct_final_access_ledger,
    validate_final_access_event_chain,
)
from zeromodel.video.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
)
from zeromodel.video.domains.video_action_set.store import VideoActionSetStore


class FinalExecutor(Protocol):
    """Authority-bearing internal executor registered only by final CLI bootstrap.

    Possession of writable access to the authoritative finalization database,
    an approved authorization and protocol, and an executor implementation is
    execution authority within the local trust scope. The service still derives
    and verifies every authoritative decision and digest.
    """

    def materialize(
        self,
        access: FinalAccessRecordDTO,
        request: FinalExecutionRequestDTO,
    ) -> object: ...

    def score_providers(
        self,
        access: FinalAccessRecordDTO,
        materialized: object,
    ) -> object: ...

    def assess_reachability(
        self,
        access: FinalAccessRecordDTO,
        scored: object,
    ) -> Sequence[Mapping[str, object]]: ...

    def build_artifacts(
        self,
        access: FinalAccessRecordDTO,
        evidence: FinalEvidenceBundleDTO,
        evaluation: FinalEvaluationResultDTO,
    ) -> Mapping[str, bytes]: ...


FailureInjector = Callable[[str], None]
AUTHORITY_CONTRACT_KEYS = frozenset(
    {
        "protocol_file",
        "execution_commit",
        "provider_order",
        "provider_versions",
        "expected_counts",
        "expected_episode_ids",
        "expected_artifacts",
        "historical_authority",
    }
)
COUNT_KEYS = frozenset(
    {"evidence_row_count", "episode_count", "frame_count", "provider_count"}
)


@dataclass(frozen=True, slots=True)
class FinalAuthorizationContract:
    protocol_file: Path
    execution_commit: str
    provider_order: tuple[str, ...]
    provider_versions: Mapping[str, str]
    expected_counts: Mapping[str, int]
    expected_episode_ids: tuple[str, ...]
    expected_artifacts: tuple[str, ...]
    historical_authority: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class ResolvedFinalExecutionPreflight:
    """Every authority-bearing preflight input, resolved and validated
    exactly once.

    Previously, `execute_final_once()` called `preflight_final_execution()`
    (which loaded and validated the authorization and protocol files) and
    then, for the non-preflight-only path, loaded the authorization file a
    *second* time to obtain the objects it actually executed with. A
    replacement of the file on disk between those two reads could make
    execution use a different authorization from the one the request and
    preflight approved (TOCTOU). This aggregate is built by
    `FinalAccessService._resolve_final_execution_preflight()` once per
    `execute_final_once()`/`preflight_final_execution()` call; both methods
    consume this same aggregate afterward and never read the authorization
    or protocol files again before reservation.
    """

    request: FinalExecutionRequestDTO
    authorization: FinalExecutionAuthorizationDTO
    protocol: FinalEvaluationProtocolDTO
    contract: FinalAuthorizationContract
    historical_authority: VerifiedHistoricalAuthorityDTO


@dataclass(frozen=True, slots=True)
class FinalAccessService:
    store: VideoActionSetStore
    final_executor: FinalExecutor | None = None
    failure_injector: FailureInjector | None = None

    def create_authorization(
        self,
        authorization: FinalExecutionAuthorizationDTO,
        protocol: FinalEvaluationProtocolDTO,
        *,
        process_identity: str | None = None,
        utc: str | None = None,
    ) -> FinalAccessRecordDTO:
        self.store.assert_finalization_authority()
        _validate_authorization_protocol(authorization, protocol)
        _authorization_contract(authorization)
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
                "protocol_digest": protocol.protocol_digest,
            },
            previous_event_digest=None,
        )
        record = record.with_state(
            state="authorized",
            utc=utc,
            process_identity=process_identity,
            current_event_ordinal=event.ordinal,
            last_event_digest=event.event_digest,
            record_payload={"authorization_digest": authorization.authorization_digest},
        )
        return self.store.create_final_authorization(
            authorization,
            protocol,
            record,
            event,
        )

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
        record, event = self._next_event(
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
        record, event = self._next_event(
            access_id,
            "running",
            process_identity=process_identity,
            utc=utc,
            event_payload=event_payload or {"kind": "running"},
        )
        return self.store.mark_final_access_running(record, event)

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
        resolved = self._resolve_final_execution_preflight(request)
        if request.preflight_only:
            return _preflight_response(resolved)
        if self.final_executor is None:
            raise VPMValidationError(
                "final execution requires an explicit registered final executor"
            )
        authorization = resolved.authorization
        protocol = resolved.protocol
        contract = resolved.contract
        verified_historical_authority = resolved.historical_authority
        existing = self.store.load_final_authorization(authorization.authorization_id)
        if existing is None:
            access = self.create_authorization(
                authorization,
                protocol,
                process_identity=request.operator_identity,
            )
        else:
            if existing != authorization:
                raise VPMValidationError("final execution authorization mismatch")
            access = self._require_record(access_id_for_authorization(authorization))
            stored_protocol = self.store.load_final_evaluation_protocol(
                authorization.protocol_digest
            )
            if stored_protocol != protocol:
                raise VPMValidationError("final evaluation protocol mismatch")
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
            return self._execute_running_access(
                access,
                request,
                authorization,
                contract,
                verified_historical_authority,
            )
        except Exception as exc:
            current = self.store.load_final_access_record(access.access_id)
            if current is not None and current.state in {"reserved", "running"}:
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
        resolved = self._resolve_final_execution_preflight(request)
        return _preflight_response(resolved)

    def _resolve_final_execution_preflight(
        self,
        request: FinalExecutionRequestDTO,
    ) -> ResolvedFinalExecutionPreflight:
        """Resolve and validate every authority-bearing preflight input
        exactly once. Read the authorization and protocol files here, and
        only here - `execute_final_once()` and `preflight_final_execution()`
        both consume the returned aggregate afterward and must never read
        either file again before reservation (see
        `ResolvedFinalExecutionPreflight`'s docstring).
        """
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
            raise VPMValidationError(
                "final authorization does not permit unattended use"
            )
        if authorization.authorization_status != "authorized":
            raise VPMValidationError("final authorization status mismatch")
        contract = _authorization_contract(authorization)
        protocol = load_final_protocol_file(contract.protocol_file)
        _validate_authorization_protocol(authorization, protocol)
        verified_historical_authority = verify_historical_authority(
            contract.historical_authority
        )
        return ResolvedFinalExecutionPreflight(
            request=request,
            authorization=authorization,
            protocol=protocol,
            contract=contract,
            historical_authority=verified_historical_authority,
        )

    def _execute_running_access(
        self,
        access: FinalAccessRecordDTO,
        request: FinalExecutionRequestDTO,
        authorization: FinalExecutionAuthorizationDTO,
        contract: FinalAuthorizationContract,
        verified_historical_authority: VerifiedHistoricalAuthorityDTO,
    ) -> dict[str, Any]:
        access = self._boundary(access, "final_materialization_started")
        materialized = cast(FinalExecutor, self.final_executor).materialize(
            access,
            request,
        )
        access = self._boundary(access, "final_materialization_completed")
        access = self._boundary(access, "provider_scoring_started")
        scored = cast(FinalExecutor, self.final_executor).score_providers(
            access,
            materialized,
        )
        access = self._boundary(access, "provider_scoring_completed")
        access = self._boundary(access, "reachability_started")
        rows = cast(FinalExecutor, self.final_executor).assess_reachability(
            access,
            scored,
        )
        access = self._boundary(access, "reachability_completed")
        raw_rows = [dict(row) for row in rows]
        evidence = FinalEvidenceBundleDTO.create(
            {
                "version": FINAL_EVIDENCE_BUNDLE_VERSION,
                "access_id": access.access_id,
                "authorization_digest": access.authorization_digest,
                "protocol_digest": access.protocol_digest,
                "benchmark_seed_digest": access.benchmark_seed_digest,
                "sealed_plan_digest": access.sealed_plan_digest,
                "execution_commit": contract.execution_commit,
                "provider_order": list(contract.provider_order),
                "provider_versions": dict(contract.provider_versions),
                "expected_counts": dict(contract.expected_counts),
                "rows": raw_rows,
            }
        )
        if evidence.rows.to_value() != raw_rows:
            raise VPMValidationError("final executor evidence ordering mismatch")
        access = self._boundary(access, "evaluation_started")
        evaluation = self._evaluate_authorized_evidence(
            access,
            authorization,
            evidence,
        )
        access = self._boundary(
            access,
            "evaluation_completed",
            {"evaluation_digest": evaluation.evaluation_digest},
        )
        access = self._boundary(access, "staged_artifact_generation_started")
        artifacts = cast(FinalExecutor, self.final_executor).build_artifacts(
            access,
            evidence,
            evaluation,
        )
        access = self._boundary(access, "staged_artifact_generation_completed")
        access = self._boundary(access, "artifact_validation_started")
        self._inject("before_staging_validation")
        staging_dir, manifest = stage_final_artifacts(
            output_dir=Path(authorization.output_dir),
            authorization_id=authorization.authorization_id,
            expected_executor_artifacts=contract.expected_artifacts,
            executor_artifacts=artifacts,
            evidence=evidence,
            evaluation=evaluation,
            before_validation=lambda: self._inject("during_staging_validation"),
        )
        access = self._boundary(
            access,
            "artifact_validation_completed",
            {"artifact_manifest_digest": manifest.artifact_manifest_digest},
        )
        self._inject("before_promotion")
        access = self._boundary(access, "promotion_started")
        self._inject("during_promotion")
        validate_staged_artifacts(staging_dir, manifest)
        promote_staged_artifacts(staging_dir, Path(authorization.output_dir))
        access = self._boundary(access, "promotion_completed")
        self._inject("after_promotion")
        return self._complete_and_publish_receipt(
            access=access,
            authorization=authorization,
            contract=contract,
            evidence=evidence,
            evaluation=evaluation,
            manifest=manifest,
            verified_historical_authority=verified_historical_authority,
        ).to_dict()

    def _evaluate_authorized_evidence(
        self,
        access: FinalAccessRecordDTO,
        authorization: FinalExecutionAuthorizationDTO,
        evidence: FinalEvidenceBundleDTO,
    ) -> FinalEvaluationResultDTO:
        current = self._require_record(access.access_id)
        if (
            current.state != "running"
            or current.authorization_digest != evidence.authorization_digest
        ):
            raise VPMValidationError("final access state transition mismatch")
        stored_authorization = self.store.load_final_authorization(
            current.authorization_id
        )
        if stored_authorization != authorization:
            raise VPMValidationError("final execution authorization mismatch")
        protocol = self.store.load_final_evaluation_protocol(current.protocol_digest)
        if protocol is None or not protocol.approved:
            raise VPMValidationError("final evaluation protocol is not approved")
        _validate_authorization_protocol(authorization, protocol)
        if (
            evidence.access_id != current.access_id
            or evidence.benchmark_seed_digest != current.benchmark_seed_digest
            or evidence.sealed_plan_digest != current.sealed_plan_digest
            or evidence.protocol_digest != current.protocol_digest
        ):
            raise VPMValidationError("final evidence authorization mismatch")
        contract = _authorization_contract(authorization)
        if evidence.actual_counts.to_value() != dict(contract.expected_counts):
            raise VPMValidationError("final evidence authorized counts mismatch")
        rows = evidence.rows.to_value()
        if not isinstance(rows, list):
            raise VPMValidationError("final evidence rows mismatch")
        episode_ids = {
            row.get("episode_id") for row in rows if isinstance(row, Mapping)
        }
        frame_provider_pairs = {
            (row.get("frame_id"), row.get("provider_id"))
            for row in rows
            if isinstance(row, Mapping)
        }
        if episode_ids != set(contract.expected_episode_ids) or len(
            frame_provider_pairs
        ) != len(rows):
            raise VPMValidationError("final evidence authorized identities mismatch")
        return evaluate_final_protocol(protocol, evidence)

    def _complete_and_publish_receipt(
        self,
        *,
        access: FinalAccessRecordDTO,
        authorization: FinalExecutionAuthorizationDTO,
        contract: FinalAuthorizationContract,
        evidence: FinalEvidenceBundleDTO,
        evaluation: FinalEvaluationResultDTO,
        manifest: FinalArtifactManifestDTO,
        verified_historical_authority: VerifiedHistoricalAuthorityDTO,
    ) -> FinalExecutionReceiptDTO:
        current = self._require_record(access.access_id)
        if current.state != "running":
            raise VPMValidationError("final access state transition mismatch")
        verified_evaluation = self._evaluate_authorized_evidence(
            current,
            authorization,
            evidence,
        )
        if verified_evaluation != evaluation:
            raise VPMValidationError("final evaluation digest mismatch")
        _validate_manifest_bindings(current, evidence, evaluation, manifest)
        validate_canonical_artifacts(Path(authorization.output_dir), manifest)
        self._inject("before_historical_authority_reverification")
        completion_historical_authority = verify_historical_authority(
            contract.historical_authority
        )
        if completion_historical_authority != verified_historical_authority:
            raise VPMValidationError("historical authority changed after reservation")
        self._inject("before_completed_transition")
        events = self.store.list_final_access_events(current.access_id)
        predecessor_chain_digest = validate_final_access_event_chain(current, events)
        completion_utc = utc_now()
        record, event = self._build_next_event(
            current,
            "completed",
            process_identity=current.process_identity,
            utc=completion_utc,
            event_payload={
                "kind": "completed",
                "verified_predecessor_chain_digest": predecessor_chain_digest,
                "evidence_digest": evidence.evidence_digest,
                "evaluation_digest": evaluation.evaluation_digest,
                "artifact_manifest_digest": manifest.artifact_manifest_digest,
                "historical_authority_digest": (
                    completion_historical_authority.historical_authority_digest
                ),
                "decision": evaluation.decision,
            },
            events=events,
            record_payload={
                "completion": {
                    "evidence_digest": evidence.evidence_digest,
                    "evaluation_digest": evaluation.evaluation_digest,
                    "artifact_manifest_digest": manifest.artifact_manifest_digest,
                    "historical_authority_digest": (
                        completion_historical_authority.historical_authority_digest
                    ),
                    "decision": evaluation.decision,
                }
            },
        )
        validate_final_access_event_chain(record, (*events, event))
        self._inject("during_completed_transition")
        completed = self.store.complete_final_access(record, event)
        self._inject("after_completed_transition")
        reconstruction = reconstruct_final_access_ledger(
            self.store, completed.access_id
        )
        if reconstruction["record"]["state"] != "completed":
            raise VPMValidationError("final completion reconstruction mismatch")
        self._inject("before_receipt")
        completed = self._boundary(completed, "receipt_publication_started")
        completed = self._boundary(
            completed,
            "receipt_publication_completed",
            {
                "evidence_digest": evidence.evidence_digest,
                "evaluation_digest": evaluation.evaluation_digest,
                "artifact_manifest_digest": manifest.artifact_manifest_digest,
            },
        )
        final_reconstruction = reconstruct_final_access_ledger(
            self.store,
            completed.access_id,
        )
        receipt = FinalExecutionReceiptDTO.create(
            {
                "version": FINAL_EXECUTION_RECEIPT_VERSION,
                "access_id": completed.access_id,
                "authorization_id": completed.authorization_id,
                "state": "completed",
                "completed_utc": completion_utc,
                "benchmark_seed_digest": completed.benchmark_seed_digest,
                "sealed_plan_digest": completed.sealed_plan_digest,
                "protocol_digest": completed.protocol_digest,
                "authorization_digest": completed.authorization_digest,
                "evidence_digest": evidence.evidence_digest,
                "evaluation_digest": evaluation.evaluation_digest,
                "artifact_manifest_digest": manifest.artifact_manifest_digest,
                "event_chain_digest": cast(
                    str,
                    final_reconstruction["event_chain_digest"],
                ),
                "decision": evaluation.decision,
                "execution_commit": contract.execution_commit,
                "provider_order": list(contract.provider_order),
                "provider_versions": dict(contract.provider_versions),
                "expected_counts": dict(contract.expected_counts),
                "actual_counts": evidence.actual_counts.to_value(),
                "historical_authority_id": (
                    completion_historical_authority.historical_authority_id
                ),
                "historical_database_sha256": (
                    completion_historical_authority.historical_database_sha256
                ),
                "historical_evidence_manifest_digest": (
                    completion_historical_authority.evidence_manifest_digest
                ),
                "historical_authority_digest": (
                    completion_historical_authority.historical_authority_digest
                ),
            }
        )
        _validate_receipt_bindings(
            receipt,
            completed,
            evidence,
            evaluation,
            manifest,
            completion_historical_authority,
        )
        publish_receipt_last(
            output_dir=Path(authorization.output_dir),
            authorization_id=authorization.authorization_id,
            receipt=receipt,
            before_write=lambda: self._inject("before_temporary_receipt_write"),
            after_write=lambda: self._inject("during_temporary_receipt_write"),
            before_publish=lambda: self._inject("during_receipt_write"),
            before_rename=lambda: self._inject("before_receipt_rename"),
            during_rename=lambda: self._inject("during_receipt_rename"),
            after_publish=lambda: self._inject("after_receipt_publication"),
        )
        return receipt

    def _boundary(
        self,
        current: FinalAccessRecordDTO,
        kind: str,
        payload: Mapping[str, object] | None = None,
    ) -> FinalAccessRecordDTO:
        record, event = self._next_event(
            current.access_id,
            current.state,
            process_identity=current.process_identity,
            utc=None,
            event_payload={"kind": kind} | ({} if payload is None else dict(payload)),
        )
        saved = self.store.append_final_access_event(record, event)
        self._inject(kind)
        return saved

    def _next_event(
        self,
        access_id: str,
        new_state: str,
        *,
        process_identity: str | None,
        utc: str | None,
        event_payload: Mapping[str, object],
    ) -> tuple[FinalAccessRecordDTO, FinalAccessEventDTO]:
        current = self._require_record(access_id)
        events = self.store.list_final_access_events(access_id)
        validate_final_access_event_chain(current, events)
        return self._build_next_event(
            current,
            new_state,
            process_identity=process_identity or default_process_identity(),
            utc=utc or utc_now(),
            event_payload=event_payload,
            events=events,
        )

    @staticmethod
    def _build_next_event(
        current: FinalAccessRecordDTO,
        new_state: str,
        *,
        process_identity: str,
        utc: str,
        event_payload: Mapping[str, object],
        events: Sequence[FinalAccessEventDTO],
        record_payload: Mapping[str, object] | None = None,
    ) -> tuple[FinalAccessRecordDTO, FinalAccessEventDTO]:
        previous_digest = events[-1].event_digest
        event = FinalAccessEventDTO.build(
            access_id=current.access_id,
            authorization_id=current.authorization_id,
            ordinal=current.current_event_ordinal + 1,
            previous_state=current.state,
            new_state=new_state,
            utc=utc,
            process_identity=process_identity,
            event_payload=event_payload,
            previous_event_digest=previous_digest,
        )
        record = current.with_state(
            state=new_state,
            utc=utc,
            process_identity=process_identity,
            current_event_ordinal=event.ordinal,
            last_event_digest=event.event_digest,
            record_payload=record_payload,
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
    ) -> tuple[FinalExecutionFailureDTO, FinalAccessRecordDTO, FinalAccessEventDTO]:
        process_identity = process_identity or default_process_identity()
        utc = utc or utc_now()
        current = self._require_record(access_id)
        events = self.store.list_final_access_events(access_id)
        predecessor_chain_digest = validate_final_access_event_chain(current, events)
        record, event = self._build_next_event(
            current,
            state,
            process_identity=process_identity,
            utc=utc,
            event_payload={
                "kind": state,
                "failure_kind": failure_kind,
                "error_code": error_code,
                "verified_predecessor_chain_digest": predecessor_chain_digest,
            },
            events=events,
        )
        verified_chain_digest = validate_final_access_event_chain(
            record,
            (*events, event),
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
                "event_chain_digest": verified_chain_digest,
            }
        )
        record = record.with_state(
            state=state,
            utc=utc,
            process_identity=process_identity,
            current_event_ordinal=event.ordinal,
            last_event_digest=event.event_digest,
            record_payload={"failure": failure.to_dict()},
        )
        validate_final_access_event_chain(record, (*events, event))
        return failure, record, event

    def _require_record(self, access_id: str) -> FinalAccessRecordDTO:
        record = self.store.load_final_access_record(access_id)
        if record is None:
            raise VPMValidationError("final access record is missing")
        return record

    def _inject(self, boundary: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(boundary)


def _preflight_response(resolved: ResolvedFinalExecutionPreflight) -> dict[str, Any]:
    """Project the read-only preflight summary from an already-resolved
    aggregate - never re-reads the authorization or protocol files."""
    authorization = resolved.authorization
    protocol = resolved.protocol
    historical_authority = resolved.historical_authority
    return {
        "preflight": "passed",
        "read_only": True,
        "validation_store": "in-memory-nonauthoritative",
        "authorization_id": authorization.authorization_id,
        "authorization_digest": authorization.authorization_digest,
        "protocol_digest": protocol.protocol_digest,
        "sealed_plan_digest": authorization.expected_sealed_plan_digest,
        "historical_authority_id": historical_authority.historical_authority_id,
        "historical_database_sha256": historical_authority.historical_database_sha256,
        "historical_evidence_manifest_digest": (
            historical_authority.evidence_manifest_digest
        ),
        "historical_authority_digest": historical_authority.historical_authority_digest,
        "reservation_created": False,
    }


def load_final_authorization_file(path: Path) -> FinalExecutionAuthorizationDTO:
    payload = _load_json_mapping(path, "final authorization payload keys mismatch")
    return FinalExecutionAuthorizationDTO.from_dict(payload)


def load_final_protocol_file(path: Path) -> FinalEvaluationProtocolDTO:
    payload = _load_json_mapping(path, "final protocol payload keys mismatch")
    return FinalEvaluationProtocolDTO.from_dict(payload)


def default_process_identity() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _authorization_contract(
    authorization: FinalExecutionAuthorizationDTO,
) -> FinalAuthorizationContract:
    value = authorization.authorization_payload.to_value()
    if not isinstance(value, Mapping) or set(value) != AUTHORITY_CONTRACT_KEYS:
        raise VPMValidationError("final authorization execution contract mismatch")
    protocol_file = value["protocol_file"]
    execution_commit = value["execution_commit"]
    provider_order = value["provider_order"]
    provider_versions = value["provider_versions"]
    expected_counts = value["expected_counts"]
    expected_episode_ids = value["expected_episode_ids"]
    expected_artifacts = value["expected_artifacts"]
    historical = value["historical_authority"]
    if not isinstance(protocol_file, str) or not protocol_file:
        raise VPMValidationError("final authorization protocol file mismatch")
    if not isinstance(execution_commit, str) or not execution_commit:
        raise VPMValidationError("final authorization execution commit mismatch")
    if not isinstance(provider_order, list) or not provider_order:
        raise VPMValidationError("final authorization provider order mismatch")
    if any(not isinstance(item, str) or not item for item in provider_order) or len(
        set(provider_order)
    ) != len(provider_order):
        raise VPMValidationError("final authorization provider order mismatch")
    if not isinstance(provider_versions, Mapping) or set(provider_versions) != set(
        provider_order
    ):
        raise VPMValidationError("final authorization provider versions mismatch")
    if any(
        not isinstance(item, str) or not item for item in provider_versions.values()
    ):
        raise VPMValidationError("final authorization provider versions mismatch")
    counts = _counts(expected_counts, "final authorization expected counts mismatch")
    if counts["provider_count"] != len(provider_order):
        raise VPMValidationError("final authorization expected counts mismatch")
    if (
        not isinstance(expected_episode_ids, list)
        or any(not isinstance(item, str) or not item for item in expected_episode_ids)
        or len(set(expected_episode_ids)) != len(expected_episode_ids)
        or len(expected_episode_ids) != counts["episode_count"]
    ):
        raise VPMValidationError("final authorization episode identities mismatch")
    if not isinstance(expected_artifacts, list):
        raise VPMValidationError("final authorization artifact list mismatch")
    if any(not isinstance(item, str) for item in expected_artifacts) or len(
        set(expected_artifacts)
    ) != len(expected_artifacts):
        raise VPMValidationError("final authorization artifact list mismatch")
    if (
        not isinstance(historical, Mapping)
        or set(historical) != HISTORICAL_AUTHORITY_KEYS
    ):
        raise VPMValidationError("final historical authority mismatch")
    for key in ("historical_database_sha256", "evidence_manifest_digest"):
        digest = historical[key]
        if (
            not isinstance(digest, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None
        ):
            raise VPMValidationError("final historical authority mismatch")
    for key in (
        "version",
        "historical_authority_id",
        "historical_database_path",
        "evidence_manifest_path",
        "stage8_commit",
    ):
        if not isinstance(historical[key], str) or not historical[key]:
            raise VPMValidationError("final historical authority mismatch")
    return FinalAuthorizationContract(
        protocol_file=Path(protocol_file),
        execution_commit=execution_commit,
        provider_order=tuple(cast(list[str], provider_order)),
        provider_versions=cast(Mapping[str, str], dict(provider_versions)),
        expected_counts=counts,
        expected_episode_ids=tuple(cast(list[str], expected_episode_ids)),
        expected_artifacts=tuple(cast(list[str], expected_artifacts)),
        historical_authority=dict(historical),
    )


def _validate_authorization_protocol(
    authorization: FinalExecutionAuthorizationDTO,
    protocol: FinalEvaluationProtocolDTO,
) -> None:
    if not protocol.approved:
        raise VPMValidationError("final evaluation protocol is not approved")
    if (
        authorization.authorization_status != "authorized"
        or authorization.protocol_digest != protocol.protocol_digest
        or authorization.expected_benchmark_seed_digest
        != protocol.benchmark_seed_digest
        or authorization.expected_sealed_plan_digest != protocol.sealed_plan_digest
        or authorization.expected_policy_artifact_id != protocol.policy_artifact_id
    ):
        raise VPMValidationError("final authorization protocol binding mismatch")
    contract = _authorization_contract(authorization)
    if protocol.selected_provider_id not in contract.provider_order:
        raise VPMValidationError("final authorization provider order mismatch")
    required = protocol.required_evidence.to_value()
    if (
        not isinstance(required, Mapping)
        or required.get("provider_id") != protocol.selected_provider_id
        or required.get("expected_row_count")
        != contract.expected_counts["evidence_row_count"]
    ):
        raise VPMValidationError("final authorization evidence count mismatch")


def _validate_manifest_bindings(
    access: FinalAccessRecordDTO,
    evidence: FinalEvidenceBundleDTO,
    evaluation: FinalEvaluationResultDTO,
    manifest: FinalArtifactManifestDTO,
) -> None:
    if (
        manifest.access_id != access.access_id
        or manifest.authorization_digest != access.authorization_digest
        or manifest.protocol_digest != access.protocol_digest
        or manifest.evidence_digest != evidence.evidence_digest
        or manifest.evaluation_digest != evaluation.evaluation_digest
        or manifest.execution_commit != evidence.execution_commit
        or manifest.provider_order != evidence.provider_order
        or manifest.provider_versions != evidence.provider_versions
        or manifest.expected_counts != evidence.expected_counts
        or manifest.actual_counts != evidence.actual_counts
    ):
        raise VPMValidationError("final artifact manifest binding mismatch")


def _validate_receipt_bindings(
    receipt: FinalExecutionReceiptDTO,
    access: FinalAccessRecordDTO,
    evidence: FinalEvidenceBundleDTO,
    evaluation: FinalEvaluationResultDTO,
    manifest: FinalArtifactManifestDTO,
    historical_authority: VerifiedHistoricalAuthorityDTO,
) -> None:
    if (
        receipt.access_id != access.access_id
        or receipt.authorization_digest != access.authorization_digest
        or receipt.protocol_digest != evaluation.protocol_digest
        or receipt.evidence_digest != evaluation.evidence_digest
        or receipt.evaluation_digest != evaluation.evaluation_digest
        or receipt.artifact_manifest_digest != manifest.artifact_manifest_digest
        or receipt.decision != evaluation.decision
        or receipt.execution_commit != evidence.execution_commit
        or receipt.provider_order != evidence.provider_order
        or receipt.provider_versions != evidence.provider_versions
        or receipt.expected_counts != evidence.expected_counts
        or receipt.actual_counts != evidence.actual_counts
        or receipt.historical_authority_id
        != historical_authority.historical_authority_id
        or receipt.historical_database_sha256
        != historical_authority.historical_database_sha256
        or receipt.historical_evidence_manifest_digest
        != historical_authority.evidence_manifest_digest
        or receipt.historical_authority_digest
        != historical_authority.historical_authority_digest
    ):
        raise VPMValidationError("final receipt binding mismatch")


def _counts(value: object, message: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != COUNT_KEYS:
        raise VPMValidationError(message)
    result: dict[str, int] = {}
    for key in COUNT_KEYS:
        item = value[key]
        if not isinstance(item, int) or isinstance(item, bool) or item < 0:
            raise VPMValidationError(message)
        result[key] = item
    return result


def _load_json_mapping(path: Path, message: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VPMValidationError(message) from exc
    if not isinstance(payload, Mapping):
        raise VPMValidationError(message)
    return payload


__all__ = [
    "FailureInjector",
    "FinalAccessService",
    "FinalAuthorizationContract",
    "FinalExecutor",
    "ResolvedFinalExecutionPreflight",
    "default_process_identity",
    "load_final_authorization_file",
    "load_final_protocol_file",
    "utc_now",
]
