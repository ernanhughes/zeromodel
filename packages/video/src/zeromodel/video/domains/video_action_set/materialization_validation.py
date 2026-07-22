from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.arcade_observation import render_row_frame
from zeromodel.video.domains.video_action_set.contracts import (
    EPISODE_FAMILY_REGISTRY_VERSION,
    FAMILY_CLOSURE_VERSION,
)
from zeromodel.video.domains.video_action_set.control_histories import (
    reconstructed_control_causal_tuple_digest,
)
from zeromodel.video.domains.video_action_set.dto import BenchmarkIdentityDTO
from zeromodel.video.domains.video_action_set.episode_families import (
    episode_family_registry,
)
from zeromodel.video.domains.video_action_set.episode_materialization import (
    materialize_plan,
)
from zeromodel.video.domains.video_action_set.family_validation import (
    frame_invalid_closure_summary,
    validate_materialized_family_record,
)
from zeromodel.video.domains.video_action_set.pixel_digest import array_digest
from zeromodel.video.domains.video_action_set.provider_observation_boundary import (
    provider_observation_descriptor_for_record,
    provider_observation_digest,
)


def record_regeneration_view(record: Mapping[str, Any]) -> dict[str, Any]:
    pixels = record.get("pixels")
    return {
        "episode_id": record["episode_id"],
        "sequence_number": record["sequence_number"],
        "event_type": record.get("event_type", "frame"),
        "family": record["family"],
        "expected_disposition": record["expected_disposition"],
        "episode_family": record.get("episode_family"),
        "episode_disposition": record.get("episode_disposition"),
        "frame_disposition": record.get("frame_disposition"),
        "denominator_class": record.get("denominator_class"),
        "expected_row": record.get("expected_row"),
        "expected_action": record.get("expected_action"),
        "actual_executed_action": record.get("actual_executed_action"),
        "gap_declaration": record.get("gap_declaration"),
        "observation_pixel_digest": record.get("observation_pixel_digest"),
        "pixel_digest": None
        if pixels is None
        else array_digest(np.ascontiguousarray(pixels, dtype=np.uint8)),
        "sequence_digest": record.get("metadata", {}).get("sequence_digest"),
        "episode_plan_digest": record.get("metadata", {}).get("episode_plan_digest"),
    }


def _initial_family_closure_rows() -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for entry in episode_family_registry()["families"]:
        family_id = entry["family_id"]
        rows[family_id] = {
            "family_id": family_id,
            "family_version": entry["family_version"],
            "classification": entry["classification"],
            "planned_episode_count": 0,
            "regenerated_episode_count": 0,
            "validation_pass_count": 0,
            "no_op_count": 0,
            "malformed_count": 0,
            "distinguishable_invalid_count": 0,
            "information_theoretic_control_count": 0,
            "canonical_collision_count": 0,
            "valid_decode_count": 0,
            "denominator_eligibility": entry["denominator_treatment"],
            "reachability_applicable_count": 0,
            "reachability_trace_verification_count": 0,
            "closure_status": "not_planned",
        }
    return rows


def _reachability_trace_counts(
    provider_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    trace_counts: dict[str, int] = {}
    for row in provider_rows:
        trace = row.get("reachability_composition_trace")
        if trace:
            trace_counts[str(row["episode_id"])] = (
                trace_counts.get(str(row["episode_id"]), 0) + 1
            )
    return trace_counts


def _apply_frame_invalid_closure_counts(
    row: dict[str, Any], regenerated: Sequence[Mapping[str, Any]]
) -> None:
    changed = [
        item.get("metadata", {})
        .get("family_intervention_trace", {})
        .get("changed_pixel_count")
        for item in regenerated
    ]
    if any(value == 0 for value in changed if value is not None):
        row["no_op_count"] += 1
    row["distinguishable_invalid_count"] += 1
    validation_statuses = [
        validate_materialized_family_record(item) for item in regenerated
    ]
    canonical_collisions = sum(
        status == "invalid_family_valid_state_collision"
        for status in validation_statuses
    )
    if canonical_collisions:
        row["canonical_collision_count"] += int(canonical_collisions)
        row["valid_decode_count"] += int(canonical_collisions)
    if any(status != "ok" for status in validation_statuses):
        row["malformed_count"] += 1


def _update_family_closure_row(
    *,
    row: dict[str, Any],
    plan: Mapping[str, Any],
    actual: Sequence[Mapping[str, Any]],
    identity: BenchmarkIdentityDTO,
    reachability_tile: Mapping[str, Any],
    trace_counts: Mapping[str, int],
) -> None:
    family_id = str(plan["family_intervention"]["family_id"])
    row["planned_episode_count"] += 1
    try:
        regenerated = sorted(
            materialize_plan(plan, identity, reachability_tile),
            key=lambda item: int(item["sequence_number"]),
        )
        row["regenerated_episode_count"] += 1
        if [record_regeneration_view(item) for item in actual] == [
            record_regeneration_view(item) for item in regenerated
        ]:
            row["validation_pass_count"] += 1
        else:
            row["malformed_count"] += 1
        if family_id in {"conflicting_action_splice", "critical_evidence_corruption"}:
            _apply_frame_invalid_closure_counts(row, regenerated)
        if family_id == "information_control":
            row["information_theoretic_control_count"] += 1
        if family_id in {"valid", "impossible_transition"}:
            row["reachability_applicable_count"] += 1
        row["reachability_trace_verification_count"] += trace_counts.get(
            str(plan["episode_id"]), 0
        )
    except VPMValidationError:
        row["malformed_count"] += 1


def _finalize_family_closure_rows(rows: Mapping[str, dict[str, Any]]) -> None:
    for row in rows.values():
        if row["planned_episode_count"] == 0:
            continue
        row["closure_status"] = (
            "closed"
            if row["planned_episode_count"]
            == row["regenerated_episode_count"]
            == row["validation_pass_count"]
            and row["malformed_count"] == 0
            and row["no_op_count"] == 0
            and row["canonical_collision_count"] == 0
            and row["valid_decode_count"] == 0
            else "unresolved"
        )


def family_closure_report(
    *,
    split: str,
    records: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    identity: BenchmarkIdentityDTO,
    reachability_tile: Mapping[str, Any],
    provider_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_episode.setdefault(str(record["episode_id"]), []).append(record)
    rows = _initial_family_closure_rows()
    trace_counts = _reachability_trace_counts(provider_rows)
    for plan in plans:
        family_id = str(plan["family_intervention"]["family_id"])
        row = rows[family_id]
        actual = sorted(
            by_episode.get(str(plan["episode_id"]), ()),
            key=lambda item: int(item["sequence_number"]),
        )
        _update_family_closure_row(
            row=row,
            plan=plan,
            actual=actual,
            identity=identity,
            reachability_tile=reachability_tile,
            trace_counts=trace_counts,
        )
    _finalize_family_closure_rows(rows)
    return {
        "version": FAMILY_CLOSURE_VERSION,
        "split": split,
        "registry_version": EPISODE_FAMILY_REGISTRY_VERSION,
        "families": list(rows.values()),
        "negative_families_verified": False,
        "reachability_verified": False,
        "frame_invalid_closure": frame_invalid_closure_summary(records),
        "reference_instrument_correct": False,
        "materialization_ready": False,
        "repository_status": "reference_instrument_correctness_unresolved",
        "materialization_status": "prospective_materialization_prohibited",
    }


def _control_metadata_and_current_row(
    control_records: Sequence[Mapping[str, Any]],
) -> tuple[str | None, list[Mapping[str, Any]], str | None]:
    digests = {record.get("observation_pixel_digest") for record in control_records}
    if None in digests or len(digests) != 1:
        return "control_byte_identity_mismatch", [], None
    if any(
        record.get("metadata", {}).get("denominator_eligible")
        for record in control_records
    ):
        return "control_denominator_leak", [], None
    if any(
        record.get("episode_disposition") != "information_theoretic_control"
        for record in control_records
    ):
        return "control_disposition_mismatch", [], None
    if any(
        record.get("frame_disposition") != "information_theoretic_control"
        for record in control_records
    ):
        return "control_disposition_mismatch", [], None
    metadata = [record.get("metadata", {}) for record in control_records]
    hidden_history_ids = {item.get("hidden_source_history_id") for item in metadata}
    hidden_label_digests = {item.get("hidden_source_label_digest") for item in metadata}
    if None in hidden_history_ids or len(hidden_history_ids) < 2:
        return "control_hidden_history_not_ambiguous", metadata, None
    if None in hidden_label_digests or len(hidden_label_digests) < 2:
        return "control_hidden_label_not_ambiguous", metadata, None
    control_group_ids = {item.get("control_group_id") for item in metadata}
    if None in control_group_ids or len(control_group_ids) != 1:
        return "control_group_mismatch", metadata, None
    current_rows = {
        item.get("control_current_row_id") or item.get("source_row_id")
        for item in metadata
    }
    if None in current_rows or len(current_rows) != 1:
        return "control_current_state_mismatch", metadata, None
    current_row = str(next(iter(current_rows)))
    current_digest = array_digest(render_row_frame(current_row))
    if current_digest not in digests:
        return "control_current_state_mismatch", metadata, current_row
    return None, metadata, current_row


def _validate_control_provider_visibility(
    control_records: Sequence[Mapping[str, Any]],
    metadata: Sequence[Mapping[str, Any]],
) -> str | None:
    expected_visible_fields = [
        "pixels",
        "shape",
        "raw_digest",
        "timestamp",
        "source_id",
        "metadata",
        "version",
    ]
    if any(
        item.get("provider_visible_fields") != expected_visible_fields
        for item in metadata
    ):
        return "control_provider_visible_leak"

    provider_descriptors = []
    for record in control_records:
        meta = record.get("metadata", {})
        try:
            descriptor = provider_observation_descriptor_for_record(record)
        except (KeyError, TypeError, VPMValidationError, ValueError):
            return "control_provider_visible_leak"
        stored_descriptor = meta.get("provider_observation_descriptor")
        stored_digest = meta.get("provider_observation_digest")
        if (
            isinstance(stored_descriptor, Mapping)
            and dict(stored_descriptor) != descriptor
        ):
            return "control_provider_visible_leak"
        if stored_digest is not None and stored_digest != provider_observation_digest(
            descriptor
        ):
            return "control_provider_visible_leak"
        provider_descriptors.append(descriptor)
    if (
        len(
            {
                provider_observation_digest(descriptor)
                for descriptor in provider_descriptors
            }
        )
        != 1
    ):
        return "control_provider_visible_leak"
    return None


def _validate_control_history_ambiguity(
    metadata: Sequence[Mapping[str, Any]], current_row: str
) -> str:
    reconstructed_tuple_digests = set()
    resulting_rows = set()
    for meta in metadata:
        history = meta.get("grounded_causal_history") or meta.get(
            "hidden_source_history"
        )
        if not isinstance(history, Mapping):
            return "control_ambiguity_absent"
        try:
            tuple_digest = reconstructed_control_causal_tuple_digest(history)
        except VPMValidationError as exc:
            message = str(exc)
            if "result" in message or "choice" in message:
                return "control_history_result_mismatch"
            return "control_history_transition_invalid"
        if str(history.get("resulting_row_id")) != current_row:
            return "control_history_result_mismatch"
        reconstructed_tuple_digests.add(tuple_digest)
        resulting_rows.add(str(history.get("resulting_row_id")))
    if len(resulting_rows) != 1 or next(iter(resulting_rows)) != current_row:
        return "control_current_state_mismatch"
    if len(reconstructed_tuple_digests) < 2:
        return "control_ambiguity_absent"
    return "ok"


def validate_control_episode_records(records: Sequence[Mapping[str, Any]]) -> str:
    control_records = [
        record
        for record in records
        if record.get("expected_disposition") == "information_theoretic_control"
    ]
    if not control_records:
        return "no_control_records"
    status, metadata, current_row = _control_metadata_and_current_row(control_records)
    if status is not None:
        return status
    visible_status = _validate_control_provider_visibility(control_records, metadata)
    if visible_status is not None:
        return visible_status
    if current_row is None:
        return "control_current_state_mismatch"
    return _validate_control_history_ambiguity(metadata, current_row)


__all__ = [
    "family_closure_report",
    "record_regeneration_view",
    "validate_control_episode_records",
]
