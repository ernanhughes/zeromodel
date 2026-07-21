from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
from pathlib import Path
from typing import Any

from ...artifact import VPMValidationError
from .canonical_json import canonical_sha256
from .final_access_dto import (
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    event_chain_digest,
)
from .store import VideoActionSetStore


FINAL_RECONSTRUCTION_VERSION = "zeromodel-video-final-reconstruction/v1"
POST_FINAL_IMMUTABILITY_MANIFEST_VERSION = (
    "zeromodel-video-post-final-immutability-manifest/v1"
)


def reconstruct_final_access_ledger(
    store: VideoActionSetStore,
    access_id: str,
) -> dict[str, Any]:
    record = store.load_final_access_record(access_id)
    if record is None:
        raise VPMValidationError("final access record is missing")
    events = store.list_final_access_events(access_id)
    _validate_event_chain(record, events)
    payload = {
        "version": FINAL_RECONSTRUCTION_VERSION,
        "record": record.to_dict(),
        "events": [event.to_dict() for event in events],
        "counters": final_access_counters_from_events(events),
        "event_chain_digest": event_chain_digest(events),
    }
    return payload | {"reconstruction_digest": canonical_sha256(payload)}


def final_access_counters_from_events(
    events: Sequence[FinalAccessEventDTO],
) -> dict[str, int]:
    counters = {
        "authorization_created_count": 0,
        "reservation_count": 0,
        "running_count": 0,
        "completion_count": 0,
        "failure_count": 0,
        "interruption_count": 0,
        "final_materialization_event_count": 0,
        "final_provider_score_event_count": 0,
        "final_reachability_event_count": 0,
        "final_evaluation_event_count": 0,
    }
    for event in events:
        if event.new_state == "authorized":
            counters["authorization_created_count"] += 1
        elif event.new_state == "reserved":
            counters["reservation_count"] += 1
        elif event.new_state == "running":
            counters["running_count"] += 1
        elif event.new_state == "completed":
            counters["completion_count"] += 1
        elif event.new_state == "failed":
            counters["failure_count"] += 1
        elif event.new_state == "interrupted":
            counters["interruption_count"] += 1
        payload = event.event_payload.to_value()
        if isinstance(payload, Mapping):
            kind = payload.get("kind")
            if kind == "final_materialization":
                counters["final_materialization_event_count"] += 1
            elif kind == "final_provider_score":
                counters["final_provider_score_event_count"] += 1
            elif kind == "final_reachability":
                counters["final_reachability_event_count"] += 1
            elif kind == "final_evaluation":
                counters["final_evaluation_event_count"] += 1
    return counters


def build_post_final_immutability_manifest(
    *,
    access_record: FinalAccessRecordDTO,
    events: Sequence[FinalAccessEventDTO],
    artifact_digests: Mapping[str, str],
) -> dict[str, Any]:
    _validate_event_chain(access_record, tuple(events))
    payload = {
        "version": POST_FINAL_IMMUTABILITY_MANIFEST_VERSION,
        "access_id": access_record.access_id,
        "authorization_id": access_record.authorization_id,
        "state": access_record.state,
        "sealed_plan_digest": access_record.sealed_plan_digest,
        "authorization_digest": access_record.authorization_digest,
        "event_chain_digest": event_chain_digest(tuple(events)),
        "artifact_digests": dict(sorted(artifact_digests.items())),
    }
    return payload | {"manifest_digest": canonical_sha256(payload)}


def digest_files(paths: Sequence[Path]) -> dict[str, str]:
    return {
        path.as_posix(): "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def _validate_event_chain(
    record: FinalAccessRecordDTO,
    events: Sequence[FinalAccessEventDTO],
) -> None:
    if not events:
        raise VPMValidationError("final access event chain is empty")
    previous_digest: str | None = None
    previous_state: str | None = None
    for ordinal, event in enumerate(events):
        if event.access_id != record.access_id:
            raise VPMValidationError("final access event chain mismatch")
        if event.ordinal != ordinal:
            raise VPMValidationError("final access event chain mismatch")
        if event.previous_event_digest != previous_digest:
            raise VPMValidationError("final access event chain mismatch")
        if event.previous_state != previous_state:
            raise VPMValidationError("final access event chain mismatch")
        previous_digest = event.event_digest
        previous_state = event.new_state
    if record.state != previous_state or record.last_event_digest != previous_digest:
        raise VPMValidationError("final access event chain mismatch")


__all__ = [
    "FINAL_RECONSTRUCTION_VERSION",
    "POST_FINAL_IMMUTABILITY_MANIFEST_VERSION",
    "build_post_final_immutability_manifest",
    "digest_files",
    "final_access_counters_from_events",
    "reconstruct_final_access_ledger",
]
