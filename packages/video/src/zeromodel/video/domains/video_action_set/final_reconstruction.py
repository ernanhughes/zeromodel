from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalExecutionReceiptDTO,
    event_chain_digest,
)
from zeromodel.video.domains.video_action_set.final_publication import (
    FinalArtifactManifestDTO,
    load_canonical_artifact_manifest,
    load_published_receipt,
)
from zeromodel.video.domains.video_action_set.store import VideoActionSetStore


FINAL_RECONSTRUCTION_VERSION = "zeromodel-video-final-reconstruction/v1"
POST_FINAL_IMMUTABILITY_MANIFEST_VERSION = (
    "zeromodel-video-post-final-immutability-manifest/v1"
)
FINAL_PUBLICATION_STATUS_VERSION = "zeromodel-video-final-publication-status/v1"


def reconstruct_final_access_ledger(
    store: VideoActionSetStore,
    access_id: str,
) -> dict[str, Any]:
    record = store.load_final_access_record(access_id)
    if record is None:
        raise VPMValidationError("final access record is missing")
    events = store.list_final_access_events(access_id)
    chain_digest = validate_final_access_event_chain(record, events)
    publication = derive_final_publication_status(store, record, chain_digest)
    payload = {
        "version": FINAL_RECONSTRUCTION_VERSION,
        "record": record.to_dict(),
        "events": [event.to_dict() for event in events],
        "counters": final_access_counters_from_events(events),
        "event_chain_digest": chain_digest,
        "publication_status": publication["status"],
        "publishable_success": publication["publishable_success"],
        "publication": publication,
    }
    return payload | {"reconstruction_digest": canonical_sha256(payload)}


def final_access_counters_from_events(
    events: Sequence[FinalAccessEventDTO],
) -> dict[str, int]:
    """Return only counters supported by durable events actually present."""

    counts: Counter[str] = Counter()
    for event in events:
        counts[f"state_{event.new_state}_event_count"] += 1
        payload = event.event_payload.to_value()
        if isinstance(payload, Mapping):
            kind = payload.get("kind")
            if isinstance(kind, str):
                counts[f"{kind}_count"] += 1
    counts["durable_event_count"] = len(events)
    return dict(sorted(counts.items()))


def derive_final_publication_status(
    store: VideoActionSetStore,
    record: FinalAccessRecordDTO,
    chain_digest: str,
) -> dict[str, object]:
    authorization = store.load_final_authorization(record.authorization_id)
    if authorization is None:
        raise VPMValidationError("final authorization is missing")
    output_dir = Path(authorization.output_dir)
    manifest: FinalArtifactManifestDTO | None = None
    artifacts_valid = False
    receipt: FinalExecutionReceiptDTO | None = None
    receipt_status = "absent"
    try:
        manifest = load_canonical_artifact_manifest(output_dir)
        artifacts_valid = manifest is not None
    except (OSError, VPMValidationError):
        artifacts_valid = False
    try:
        receipt = load_published_receipt(output_dir)
        if receipt is not None:
            receipt_status = (
                "valid"
                if _receipt_matches_completion(
                    receipt,
                    record,
                    chain_digest,
                    manifest,
                )
                else "invalid"
            )
    except (OSError, VPMValidationError):
        receipt_status = "invalid"

    if record.state != "completed":
        status = record.state
    elif not artifacts_valid:
        status = "completed_artifacts_invalid"
    elif receipt_status == "absent":
        status = "completed_receipt_missing"
    elif receipt_status == "invalid":
        status = "completed_receipt_invalid"
    else:
        status = "completed_receipt_valid"
    publishable = status == "completed_receipt_valid"
    return {
        "version": FINAL_PUBLICATION_STATUS_VERSION,
        "status": status,
        "access_state": record.state,
        "canonical_artifacts_valid": artifacts_valid,
        "receipt_status": receipt_status,
        "publishable_success": publishable,
    }


def _receipt_matches_completion(
    receipt: FinalExecutionReceiptDTO,
    record: FinalAccessRecordDTO,
    chain_digest: str,
    manifest: FinalArtifactManifestDTO | None,
) -> bool:
    completion = record.record_payload.to_value()
    if not isinstance(completion, Mapping):
        return False
    completion = completion.get("completion")
    if not isinstance(completion, Mapping) or manifest is None:
        return False
    return (
        receipt.state == "completed"
        and receipt.access_id == record.access_id
        and receipt.authorization_id == record.authorization_id
        and receipt.authorization_digest == record.authorization_digest
        and receipt.protocol_digest == record.protocol_digest
        and receipt.benchmark_seed_digest == record.benchmark_seed_digest
        and receipt.sealed_plan_digest == record.sealed_plan_digest
        and receipt.event_chain_digest == chain_digest
        and receipt.artifact_manifest_digest == manifest.artifact_manifest_digest
        and receipt.evidence_digest == completion.get("evidence_digest")
        and receipt.evaluation_digest == completion.get("evaluation_digest")
        and receipt.artifact_manifest_digest
        == completion.get("artifact_manifest_digest")
        and receipt.historical_authority_digest
        == completion.get("historical_authority_digest")
        and receipt.decision == completion.get("decision")
    )


def build_post_final_immutability_manifest(
    *,
    access_record: FinalAccessRecordDTO,
    events: Sequence[FinalAccessEventDTO],
    artifact_digests: Mapping[str, str],
) -> dict[str, Any]:
    chain_digest = validate_final_access_event_chain(access_record, tuple(events))
    payload = {
        "version": POST_FINAL_IMMUTABILITY_MANIFEST_VERSION,
        "access_id": access_record.access_id,
        "authorization_id": access_record.authorization_id,
        "state": access_record.state,
        "sealed_plan_digest": access_record.sealed_plan_digest,
        "authorization_digest": access_record.authorization_digest,
        "event_chain_digest": chain_digest,
        "artifact_digests": dict(sorted(artifact_digests.items())),
    }
    return payload | {"manifest_digest": canonical_sha256(payload)}


def digest_files(paths: Sequence[Path]) -> dict[str, str]:
    return {
        path.as_posix(): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def validate_final_access_event_chain(
    record: FinalAccessRecordDTO,
    events: Sequence[FinalAccessEventDTO],
) -> str:
    if not events:
        raise VPMValidationError("final access event chain is empty")
    canonical_events: list[FinalAccessEventDTO] = []
    previous_digest: str | None = None
    previous_state: str | None = None
    for ordinal, event in enumerate(events):
        canonical = FinalAccessEventDTO.from_dict(event.to_dict())
        if (
            canonical.access_id != record.access_id
            or canonical.authorization_id != record.authorization_id
            or canonical.ordinal != ordinal
            or canonical.previous_event_digest != previous_digest
            or canonical.previous_state != previous_state
        ):
            raise VPMValidationError("final access event chain mismatch")
        if ordinal == 0:
            payload = canonical.event_payload.to_value()
            if (
                canonical.previous_state is not None
                or canonical.previous_event_digest is not None
                or canonical.new_state != "authorized"
                or not isinstance(payload, Mapping)
                or payload.get("kind") != "authorization_created"
                or payload.get("authorization_digest") != record.authorization_digest
            ):
                raise VPMValidationError("final access event chain genesis mismatch")
        canonical_events.append(canonical)
        previous_digest = canonical.event_digest
        previous_state = canonical.new_state
    if (
        record.state != previous_state
        or record.last_event_digest != previous_digest
        or record.current_event_ordinal != len(canonical_events) - 1
    ):
        raise VPMValidationError("final access event chain mismatch")
    return event_chain_digest(tuple(canonical_events))


__all__ = [
    "FINAL_RECONSTRUCTION_VERSION",
    "FINAL_PUBLICATION_STATUS_VERSION",
    "POST_FINAL_IMMUTABILITY_MANIFEST_VERSION",
    "build_post_final_immutability_manifest",
    "digest_files",
    "derive_final_publication_status",
    "final_access_counters_from_events",
    "reconstruct_final_access_ledger",
    "validate_final_access_event_chain",
]
