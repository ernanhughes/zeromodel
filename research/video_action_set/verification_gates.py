"""Reference verification gate orchestration for video action-set artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Mapping,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.artifact_io import (
    _read_json,
    _read_jsonl,
    _sha256,
)
from research.video_action_set.build_orchestration import (
    _materialize_records,
    _measured_phase_access_counts,
)
from zeromodel.video.domains.video_action_set.contracts import (
    EPISODE_PLAN_VERSION,
    SEED_DERIVATION_VERSION,
    SPLICE_MASK_VERSION,
    TARGET_REGION_ID,
)
from zeromodel.video.domains.video_action_set.episode_families import (
    denominator_class as _denominator_class,
    episode_disposition as _episode_disposition,
    expected_frame_disposition,
)
from zeromodel.video.domains.video_action_set.episode_planning import (
    episode_ids_by_family as _episode_ids_by_family,
    final_observation_provenance as _final_observation_provenance,
    validate_episode_plan_collection as _validate_episode_plan_collection,
)
from research.evidence.evidence_audit import (
    access_prohibition_gate as _build_access_prohibition_gate,
)
from zeromodel.video.domains.video_action_set.frame_family_kernels import (
    apply_conflicting_splice as _apply_conflicting_splice,
    apply_critical_corruption as _apply_critical_corruption,
    critical_coordinates as _critical_coordinates,
    final_visible_target_action_evidence as _final_visible_target_action_evidence,
)
from zeromodel.video.domains.video_action_set.materialization_reachability import (
    tile_edge as _tile_edge,
)
from zeromodel.video.domains.video_action_set.materialization_validation import (
    validate_control_episode_records,
)
from zeromodel.video.domains.video_action_set.observation_replay import (
    replay_observation_operation_chain as _replay_observation_operation_chain_core,
    validate_observation_operation_chain as _validate_observation_operation_chain_core,
)
from zeromodel.video.domains.video_action_set.observation_universe import (
    _canonical_observation_digest_index,
    _valid_transformed_observation_digest_index,
)
from research.video_action_set.provider_measurement import (
    provider_version as _provider_version,
)
from zeromodel.video.domains.video_action_set.reachability_composition import (
    gap_reachability_state as _gap_reachability_state,
    state_from_trace as _state_from_trace,
    compose_reachability_trace,
    validate_reachability_trace,
)
from research.video_action_set.reference_verification import (
    _expected_semantic_for_row,
    _finding,
    _gate,
)
from zeromodel.video.domains.video_action_set.transformations import (
    _validate_transformation_parameters,
)
from zeromodel.video.domains.video_action_set.dto import BenchmarkIdentityDTO
from research.video.video_prospective_providers import PROSPECTIVE_PROVIDER_IDS

BenchmarkIdentity = BenchmarkIdentityDTO
_NON_FINAL_SPLITS = ("development", "calibration", "selection")
_ALL_SPLITS = ("development", "calibration", "selection", "final")


# fmt: off
def replay_observation_operation_chain(chain: Mapping[str, Any]) -> dict[str, Any]:
    return _replay_observation_operation_chain_core(
        chain,
        conflicting_splice_executor=_apply_conflicting_splice,
        critical_corruption_executor=_apply_critical_corruption,  # type: ignore[arg-type]
    )


def validate_observation_operation_chain(record: Mapping[str, Any]) -> str:
    return _validate_observation_operation_chain_core(
        record,
        conflicting_splice_executor=_apply_conflicting_splice,
        critical_corruption_executor=_apply_critical_corruption,  # type: ignore[arg-type]
    )


def _expected_split_counts() -> dict[str, int]:
    return {
        "development": 112,
        "calibration": 448,
        "selection": 1008,
    }


def _seed_and_plan_gate(output_dir: Path, context: Mapping[str, Any], *, max_findings: int | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    identity: BenchmarkIdentity = context["identity"]
    row_actions = context["row_actions"]
    plans_by_split = context["plans"]
    try:
        plan_payload = _read_json(output_dir / "episode-plan.json")
        final_payload = _read_json(output_dir / "final-split-sealed-plan.json")
    except FileNotFoundError:
        return _gate("seed_and_plan", [_finding("expected_file_missing", "episode plan file is missing")], unavailable=False)
    if plan_payload.get("version") != EPISODE_PLAN_VERSION or plan_payload.get("seed_derivation_version") != SEED_DERIVATION_VERSION:
        findings.append(_finding("episode_seed_derivation_mismatch", "episode plan schema or seed derivation version is unsupported"))
    seen: dict[str, str] = {}
    for split in _NON_FINAL_SPLITS:
        stored_split = plan_payload.get("splits", {}).get(split, {})
        stored_plans = list(stored_split.get("episodes", []))
        expected_plans = list(plans_by_split[split])
        if len(stored_plans) != len(expected_plans):
            findings.append(_finding("sealed_episode_identity_mismatch", "stored split plan count does not match deterministic regenerated plan count", split=split))
            continue
        for expected, stored in zip(expected_plans, stored_plans):
            episode_id = str(stored.get("episode_id"))
            if episode_id in seen:
                findings.append(_finding("duplicate_episode_id", "episode id is duplicated across sealed split plans", episode_id=episode_id, previous_split=seen[episode_id], split=split))
            seen[episode_id] = split
            if stored.get("split") != split or not episode_id.startswith(f"{split}:"):
                findings.append(_finding("episode_split_reassignment", "episode id or split field crosses its sealed split boundary", episode_id=episode_id, split=split))
            if stored.get("derived_seed_identity") != expected.get("derived_seed_identity") or stored.get("episode_seed") != expected.get("episode_seed"):
                findings.append(_finding("episode_seed_derivation_mismatch", "stored episode seed lineage does not match deterministic derivation", episode_id=episode_id))
                if max_findings is not None and len(findings) >= max_findings:
                    return _gate("seed_and_plan", findings, counts={"sealed_episode_count": sum(len(plans_by_split[item]) for item in _ALL_SPLITS)})
                continue
            if stored.get("source_row_id") != expected.get("source_row_id") or stored.get("secondary_row_id") != expected.get("secondary_row_id"):
                findings.append(_finding("episode_seed_derivation_mismatch", "stored source-row lineage does not match deterministic derivation", episode_id=episode_id))
                if max_findings is not None and len(findings) >= max_findings:
                    return _gate("seed_and_plan", findings, counts={"sealed_episode_count": sum(len(plans_by_split[item]) for item in _ALL_SPLITS)})
                continue
            if stored.get("episode_disposition") != expected.get("episode_disposition") or stored.get("denominator_class") != expected.get("denominator_class"):
                findings.append(_finding("family_disposition_mismatch", "stored episode disposition fields do not match deterministic derivation", episode_id=episode_id))
            if dict(stored) != expected:
                findings.append(_finding("sealed_episode_identity_mismatch", "stored sealed episode plan does not match deterministic regeneration", episode_id=episode_id))
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("seed_and_plan", findings, counts={"sealed_episode_count": sum(len(plans_by_split[item]) for item in _ALL_SPLITS)})
    expected_final = {
        "version": EPISODE_PLAN_VERSION,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "split": "final",
        "plan_only": True,
        "materialization_prohibited": True,
        "episode_counts": {
            "valid": 112,
            "frame_invalid": 56,
            "temporal_negative": 56,
            "information_control": 28,
        },
        "frame_count": 1008,
        "sealed_episode_ids": _episode_ids_by_family(plans_by_split["final"]),
        "episodes": plans_by_split["final"],
        "seed_commitment": identity.seed_digest,
    }
    expected_final = expected_final | {"sealed_plan_digest": _sha256(expected_final)}
    for plan in final_payload.get("episodes", []):
        if plan.get("final_observation_provenance") != _final_observation_provenance("final"):
            findings.append(
                _finding(
                    "final_observation_provenance_mismatch",
                    "final split episode claims materialized observation provenance",
                    episode_id=plan.get("episode_id"),
                )
            )
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("seed_and_plan", findings, counts={"sealed_episode_count": sum(len(plans_by_split[split]) for split in _ALL_SPLITS)})
    if final_payload != expected_final:
        findings.append(_finding("sealed_episode_identity_mismatch", "final sealed split identity does not match deterministic regeneration"))
        if max_findings is not None and len(findings) >= max_findings:
            return _gate("seed_and_plan", findings, counts={"sealed_episode_count": sum(len(plans_by_split[split]) for split in _ALL_SPLITS)})
    digest_payload = _read_json(output_dir / "final-split-sealed-digest.json")
    if digest_payload.get("digest") != expected_final["sealed_plan_digest"]:
        findings.append(_finding("sealed_episode_identity_mismatch", "final sealed split digest sidecar does not match the final sealed plan"))
        if max_findings is not None and len(findings) >= max_findings:
            return _gate("seed_and_plan", findings, counts={"sealed_episode_count": sum(len(plans_by_split[split]) for split in _ALL_SPLITS)})
    try:
        _validate_episode_plan_collection(context["identity"], plans_by_split, row_actions)
    except VPMValidationError as exc:
        findings.append(_finding("episode_seed_derivation_mismatch", "authoritative regenerated plan collection failed validation", error=str(exc)))
    return _gate("seed_and_plan", findings, counts={"sealed_episode_count": sum(len(plans_by_split[split]) for split in _ALL_SPLITS)})


def _episode_regeneration_gate(output_dir: Path, repo_root: Path, *, max_findings: int | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for split in _NON_FINAL_SPLITS:
        path = output_dir / split / "frame-metadata.jsonl"
        if not path.exists():
            findings.append(_finding("expected_record_missing", "split frame metadata is missing", split=split))
            continue
        stored_rows = _read_jsonl(path)
        expected_rows = _cached_materialized_metadata(repo_root, split)
        counts[f"{split}_stored_observations"] = len(stored_rows)
        counts[f"{split}_expected_observations"] = len(expected_rows)
        if len(stored_rows) != len(expected_rows):
            findings.append(_finding("expected_record_missing", "stored observation count does not match regenerated count", split=split))
        expected_by_key = {(row["episode_id"], int(row["sequence_number"])): row for row in expected_rows}
        stored_by_key = {(row.get("episode_id"), int(row.get("sequence_number", -1))): row for row in stored_rows}
        for key in sorted(set(expected_by_key) - set(stored_by_key)):
            findings.append(_finding("expected_record_missing", "regenerated observation is absent from stored frame metadata", split=split, episode_id=key[0], sequence_number=key[1]))
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("episode_regeneration", findings, counts=counts)
        for key in sorted(set(stored_by_key) - set(expected_by_key)):
            findings.append(_finding("orphan_observation_record", "stored frame metadata has no sealed regenerated observation", split=split, episode_id=key[0], sequence_number=key[1]))
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("episode_regeneration", findings, counts=counts)
        for key in sorted(set(expected_by_key) & set(stored_by_key)):
            expected = expected_by_key[key]
            stored = stored_by_key[key]
            if stored.get("frame_id") != expected.get("frame_id") or stored.get("clip_id") != expected.get("clip_id"):
                findings.append(_finding("frame_identity_mismatch", "stored frame identity does not match regenerated identity", split=split, episode_id=key[0], sequence_number=key[1]))
            if stored.get("event_type") == "gap_unknown" and stored.get("pixels") is not None:
                findings.append(_finding("gap_event_has_pixels", "declared gap event carries ordinary pixels", split=split, episode_id=key[0], sequence_number=key[1]))
            if stored.get("event_type") != expected.get("event_type") or stored.get("gap_declaration") != expected.get("gap_declaration"):
                findings.append(_finding("gap_event_structure_mismatch", "stored event type or gap declaration does not match regenerated observation", split=split, episode_id=key[0], sequence_number=key[1]))
            if stored.get("observation_pixel_digest") != expected.get("observation_pixel_digest"):
                code = "observation_digest_mismatch" if stored.get("observation_pixel_digest") and expected.get("observation_pixel_digest") else "observation_bytes_mismatch"
                findings.append(_finding(code, "stored observation identity does not match regenerated pixel bytes", split=split, episode_id=key[0], sequence_number=key[1]))
            comparable_fields = ("family", "expected_disposition", "expected_row", "expected_action", "actual_executed_action", "action_known")
            for field in comparable_fields:
                if stored.get(field) != expected.get(field):
                    findings.append(_finding("family_regeneration_mismatch", "stored frame classification does not match regenerated episode", split=split, episode_id=key[0], sequence_number=key[1], field=field))
                    break
            stored_meta = stored.get("metadata", {})
            expected_meta = expected.get("metadata", {})
            for field in ("seed_digest", "derived_seed_identity", "episode_plan_digest", "frame_seed_identity"):
                if stored_meta.get(field) != expected_meta.get(field):
                    findings.append(_finding("family_regeneration_mismatch", "stored frame seed or plan metadata does not match regenerated episode", split=split, episode_id=key[0], sequence_number=key[1], field=field))
                    break
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("episode_regeneration", findings, counts=counts)
    return _gate("episode_regeneration", findings, counts=counts)


def _semantic_outcome_gate(output_dir: Path, context: Mapping[str, Any], *, max_findings: int | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    row_ids = context["row_ids"]
    row_actions = context["row_actions"]
    counts = {"provider_score_records": 0}
    semantic_cache: dict[str, Any] = {}
    for split in _NON_FINAL_SPLITS:
        for row in _read_jsonl(output_dir / split / "provider-evidence.jsonl"):
            counts["provider_score_records"] += 1
            if row.get("policy_artifact_id") != context["policy"].artifact_id:
                findings.append(_finding("policy_action_mapping_mismatch", "provider row references a foreign policy artifact", split=split, frame_id=row.get("frame_id")))
                continue
            if row.get("provider_version") != _provider_version(str(row.get("provider_id"))):
                findings.append(_finding("provider_contract_mismatch", "provider row references a foreign provider contract", split=split, frame_id=row.get("frame_id"), provider_id=row.get("provider_id")))
                continue
            expected, evidence_findings = _expected_semantic_for_row(row, row_actions, row_ids, semantic_cache)
            findings.extend(evidence_findings)
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("semantic_outcome", findings, counts=counts)
            if expected is None:
                continue
            payload = row.get("semantic_top_set_outcome", {})
            payload_top_rows = {str(item) for item in payload.get("top_row_ids", [])}
            expected_top_rows = set(expected.top_row_ids)
            if payload_top_rows != expected_top_rows or set(row.get("top_row_ids", [])) != expected_top_rows:
                findings.append(_finding("semantic_status_mismatch", "stored semantic top-row set does not match reconstructed top set", split=split, frame_id=row.get("frame_id")))
            payload_actions = {str(item.get("row_id")): str(item.get("action_id")) for item in payload.get("top_row_actions", [])}
            if payload_actions != {row_id: row_actions[row_id] for row_id in expected_top_rows}:
                findings.append(_finding("policy_action_mapping_mismatch", "stored semantic row/action mapping does not match authoritative policy mapping", split=split, frame_id=row.get("frame_id")))
            if payload.get("status") != expected.status or row.get("semantic_status") != expected.status:
                findings.append(_finding("semantic_status_mismatch", "stored semantic status does not match independent reconstruction", split=split, frame_id=row.get("frame_id")))
            stored_resolved_row = payload.get("resolved_row_id", row.get("resolved_row"))
            stored_resolved_action = payload.get("resolved_action_id", row.get("resolved_action"))
            if expected.resolved_row_id is None and stored_resolved_row is not None:
                findings.append(_finding("resolved_row_not_permitted", "stored semantic outcome resolves a row where the frozen rules do not permit one", split=split, frame_id=row.get("frame_id")))
            elif stored_resolved_row != expected.resolved_row_id or row.get("resolved_row") != expected.resolved_row_id:
                findings.append(_finding("semantic_status_mismatch", "stored resolved row does not match independent reconstruction", split=split, frame_id=row.get("frame_id")))
            if expected.resolved_action_id is None and stored_resolved_action is not None:
                findings.append(_finding("resolved_action_not_permitted", "stored semantic outcome resolves an action where the frozen rules do not permit one", split=split, frame_id=row.get("frame_id")))
            elif stored_resolved_action != expected.resolved_action_id or row.get("resolved_action") != expected.resolved_action_id:
                findings.append(_finding("semantic_status_mismatch", "stored resolved action does not match independent reconstruction", split=split, frame_id=row.get("frame_id")))
            if payload.get("rejection_reason") != expected.rejection_reason:
                findings.append(_finding("semantic_status_mismatch", "stored semantic rejection reason does not match independent reconstruction", split=split, frame_id=row.get("frame_id")))
            if payload.get("semantic_outcome_digest") != expected.semantic_outcome_digest or row.get("semantic_outcome_digest") != expected.semantic_outcome_digest:
                findings.append(_finding("semantic_outcome_digest_mismatch", "stored semantic outcome digest does not match independent reconstruction", split=split, frame_id=row.get("frame_id")))
            if row.get("winner_row") != expected.resolved_row_id:
                code = "resolved_row_not_permitted" if expected.resolved_row_id is None and row.get("winner_row") is not None else "semantic_status_mismatch"
                findings.append(_finding(code, "stored winner row does not match semantic reconstruction", split=split, frame_id=row.get("frame_id")))
            if row.get("winner_action") != expected.resolved_action_id:
                code = "resolved_action_not_permitted" if expected.resolved_action_id is None and row.get("winner_action") is not None else "semantic_status_mismatch"
                findings.append(_finding(code, "stored winner action does not match semantic reconstruction", split=split, frame_id=row.get("frame_id")))
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("semantic_outcome", findings, counts=counts)
    return _gate("semantic_outcome", findings, counts=counts)


def _validate_family_frame_contracts(
    split: str,
    episode_id: str,
    ordered: list[dict[str, Any]],
    findings: list[dict[str, Any]],
) -> None:
    for row in ordered:
        episode_family = row.get("episode_family")
        if episode_family not in {"valid", "frame_invalid", "temporal_negative", "information_control"}:
            findings.append(_finding("family_disposition_mismatch", "frame omits a recognized episode family", split=split, episode_id=episode_id))
            continue
        if row.get("episode_disposition") != _episode_disposition(str(episode_family)):
            findings.append(_finding("family_disposition_mismatch", "frame episode disposition does not match its family", split=split, episode_id=episode_id))
        if row.get("denominator_class") != _denominator_class(str(episode_family)):
            findings.append(_finding("family_disposition_mismatch", "frame denominator class does not match its family", split=split, episode_id=episode_id))
        if not row.get("frame_disposition"):
            findings.append(_finding("family_disposition_mismatch", "frame disposition is missing", split=split, episode_id=episode_id))
        intervention = row.get("metadata", {}).get("family_intervention", {})
        mutation_kind = intervention.get("family_id") if episode_family != "valid" else None
        expected_frame = expected_frame_disposition(str(episode_family), None if mutation_kind is None else str(mutation_kind), int(row.get("sequence_number", -1)), intervention)
        if row.get("frame_disposition") != expected_frame:
            findings.append(_finding("frame_disposition_mismatch", "frame disposition does not match independent family derivation", split=split, episode_id=episode_id, frame_id=row.get("frame_id"), expected=expected_frame, actual=row.get("frame_disposition")))
        provenance_status = validate_observation_operation_chain(row)
        if provenance_status != "ok":
            findings.append(_finding(provenance_status, "stored observation operation chain does not independently replay to the emitted observation", split=split, episode_id=episode_id, frame_id=row.get("frame_id")))
    if any(row.get("event_type") == "gap_unknown" and (row.get("observation_pixel_digest") is not None or row.get("pixels") is not None) for row in ordered):
        findings.append(_finding("gap_event_has_pixels", "typed gap event carries ordinary observation identity", split=split, episode_id=episode_id))


def _validate_family_intervention_contracts(
    split: str,
    episode_id: str,
    family: Any,
    ordered: list[dict[str, Any]],
    row_actions: Mapping[str, str],
    findings: list[dict[str, Any]],
    has_transformed_valid_collision: Any,
) -> None:
    if family == "conflicting_action_splice":
        for row in ordered:
            metadata = row.get("metadata", {})
            trace = metadata.get("family_intervention_trace", {})
            if metadata.get("source_action_id") == metadata.get("competitor_action_id"):
                findings.append(_finding("family_contract_violation", "conflicting splice uses two rows governed by the same action", split=split, episode_id=episode_id))
            if int(trace.get("primary_contributing_pixel_count", 0)) <= 0 or int(trace.get("secondary_contributing_pixel_count", 0)) <= 0:
                findings.append(_finding("family_contract_violation", "conflicting splice does not include nonzero contributions from both sources", split=split, episode_id=episode_id))
            splice_counts = trace.get("action_relevant_region_contribution_counts", {})
            if int(splice_counts.get("primary_target_pixel_count", 0)) <= 0 or int(splice_counts.get("secondary_additive_target_pixel_count", 0)) <= 0:
                findings.append(_finding("family_contract_violation", "conflicting splice lacks two visible target evidence contributions", split=split, episode_id=episode_id))
            if trace.get("splice_mask_version") != SPLICE_MASK_VERSION or trace.get("target_region_id") != TARGET_REGION_ID:
                findings.append(_finding("family_contract_violation", "conflicting splice trace does not use the frozen target-evidence mask", split=split, episode_id=episode_id))
            if trace.get("output_observation_digest") != row.get("observation_pixel_digest"):
                findings.append(_finding("family_regeneration_mismatch", "conflicting splice output digest does not match stored observation", split=split, episode_id=episode_id))
            try:
                replay = replay_observation_operation_chain(metadata["observation_operation_chain"])
                visible_action_evidence = _final_visible_target_action_evidence(replay["pixels"], row_actions)
                if not visible_action_evidence["conflicting_action_evidence_present"]:
                    findings.append(_finding("conflicting_action_evidence_absent", "final visible splice target evidence does not imply at least two policy actions", split=split, episode_id=episode_id))
                if trace.get("visible_action_evidence_digest") != visible_action_evidence["visible_action_evidence_digest"]:
                    findings.append(_finding("conflicting_action_evidence_absent", "stored visible-action splice evidence does not match independent reconstruction", split=split, episode_id=episode_id))
            except (KeyError, TypeError, VPMValidationError, ValueError):
                findings.append(_finding("final_observation_provenance_mismatch", "conflicting splice operation chain could not be replayed for visible-action reconstruction", split=split, episode_id=episode_id))
            if _canonical_observation_digest_index().get(str(row.get("observation_pixel_digest"))):
                findings.append(_finding("invalid_family_valid_state_collision", "conflicting splice output decodes as a canonical valid observation", split=split, episode_id=episode_id))
            elif has_transformed_valid_collision(row.get("observation_pixel_digest")):
                findings.append(_finding("invalid_family_valid_transformation_collision", "conflicting splice output decodes as a bounded transformed valid observation", split=split, episode_id=episode_id))
    elif family == "critical_evidence_corruption":
        for row in ordered:
            metadata = row.get("metadata", {})
            trace = metadata.get("family_intervention_trace", {})
            changes = trace.get("changes", [])
            if not changes:
                findings.append(_finding("family_contract_violation", "critical corruption has an empty changed-coordinate set", split=split, episode_id=episode_id))
            if trace.get("changed_pixel_count") == 0:
                findings.append(_finding("family_no_op", "critical corruption is a no-op", split=split, episode_id=episode_id))
            if _canonical_observation_digest_index().get(str(row.get("observation_pixel_digest"))):
                findings.append(_finding("invalid_family_valid_state_collision", "critical corruption output decodes as a canonical valid observation", split=split, episode_id=episode_id))
            elif has_transformed_valid_collision(row.get("observation_pixel_digest")):
                findings.append(_finding("invalid_family_valid_transformation_collision", "critical corruption output decodes as a bounded transformed valid observation", split=split, episode_id=episode_id))
            for change in changes:
                coord = (int(change.get("y", -1)), int(change.get("x", -1)))
                if coord not in set(_critical_coordinates()) or change.get("original") == change.get("replacement"):
                    findings.append(_finding("family_contract_violation", "critical corruption changed a non-critical coordinate or preserved the original value", split=split, episode_id=episode_id))
    elif family == "bounded_translation" or family == "bounded_photometric" or family == "bounded_translation_photometric" or family == "bounded_translation_occlusion" or family == "compound_bounded" or family == "exact":
        for row in ordered:
            metadata = row.get("metadata", {})
            try:
                _validate_transformation_parameters(metadata.get("transformation_parameters", {}))
            except (KeyError, VPMValidationError) as exc:
                findings.append(_finding("family_contract_violation", "valid transformation parameters violate the frozen bounds", split=split, episode_id=episode_id, error=str(exc)))


def _validate_family_temporal_contracts(
    split: str,
    episode_id: str,
    ordered: list[dict[str, Any]],
    reachability_tile: Mapping[str, Any],
    findings: list[dict[str, Any]],
) -> None:
    if any(row.get("expected_disposition") == "information_theoretic_control" for row in ordered):
        code = validate_control_episode_records(ordered)
        if code != "ok":
            findings.append(_finding(code, "information control records violate byte-identity, hidden-history, or denominator rules", split=split, episode_id=episode_id))
    if any("stale_repeat" in row.get("metadata", {}) for row in ordered):
        repeat_rows = [row for row in ordered if "stale_repeat" in row.get("metadata", {})]
        for row in repeat_rows:
            repeat = row["metadata"]["stale_repeat"]
            if repeat.get("original_destination_digest") == repeat.get("replacement_digest"):
                findings.append(_finding("family_contract_violation", "stale repeat does not replace the destination with different bytes", split=split, episode_id=episode_id))
            if row.get("observation_pixel_digest") != repeat.get("replacement_digest"):
                findings.append(_finding("family_regeneration_mismatch", "stale repeat stored digest does not match replacement digest", split=split, episode_id=episode_id))
    if any("impossible_transition" in row.get("metadata", {}) for row in ordered):
        for row in ordered:
            transition = row.get("metadata", {}).get("impossible_transition")
            if not transition:
                continue
            edge = _tile_edge(reachability_tile, str(transition["source_row_id"]), str(transition["source_action_id"]))
            if str(transition["destination_row_id"]) in set(edge["reachable_row_ids"]):
                findings.append(_finding("transition_classification_mismatch", "impossible-transition episode contains at least one reachable pair", split=split, episode_id=episode_id))
    if any(row.get("metadata", {}).get("sequence_rule") == "non_identity_permutation" for row in ordered):
        materialized = [row.get("metadata", {}).get("original_frame_index") for row in ordered]
        if materialized == sorted(materialized) or sorted(materialized) != list(range(len(ordered))):
            findings.append(_finding("family_contract_violation", "reordered-frame metadata is not a non-identity complete permutation", split=split, episode_id=episode_id))


def _family_contract_gate(output_dir: Path, context: Mapping[str, Any], *, max_findings: int | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    counts = {"family_records_checked": 0}
    row_actions = context["row_actions"]
    reachability_tile = context["reachability_tile"]
    canonical_digest_set = set(_canonical_observation_digest_index())
    transformed_valid_digest_set: set[str] | None = None

    def has_transformed_valid_collision(digest: str | None) -> bool:
        nonlocal transformed_valid_digest_set
        if digest is None:
            return False
        if transformed_valid_digest_set is None:
            transformed_valid_digest_set = set(_valid_transformed_observation_digest_index()["digest_index"]) - canonical_digest_set
        return str(digest) in transformed_valid_digest_set

    for split in _NON_FINAL_SPLITS:
        by_episode: dict[str, list[dict[str, Any]]] = {}
        for row in _read_jsonl(output_dir / split / "frame-metadata.jsonl"):
            by_episode.setdefault(str(row.get("episode_id")), []).append(row)
        for episode_id, rows in by_episode.items():
            ordered = sorted(rows, key=lambda item: int(item.get("sequence_number", -1)))
            counts["family_records_checked"] += len(ordered)
            family = ordered[0].get("family")
            _validate_family_frame_contracts(split, episode_id, ordered, findings)
            _validate_family_intervention_contracts(
                split, episode_id, family, ordered, row_actions, findings, has_transformed_valid_collision
            )
            _validate_family_temporal_contracts(
                split, episode_id, ordered, reachability_tile, findings
            )
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("family_contract", findings, counts=counts)
    return _gate("family_contract", findings, counts=counts)


def _reachability_gate(output_dir: Path, repo_root: Path, context: Mapping[str, Any], *, max_findings: int | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    counts = {"reachability_traces_checked": 0}
    row_ids = context["row_ids"]
    row_actions = context["row_actions"]
    reachability_tile = context["reachability_tile"]
    semantic_cache: dict[str, Any] = {}
    for split in _NON_FINAL_SPLITS:
        frames = _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        evidence_rows = _read_jsonl(output_dir / split / "provider-evidence.jsonl")
        by_frame_provider = {(row.get("frame_id"), row.get("provider_id")): row for row in evidence_rows}
        reachability_state: dict[str, Mapping[str, Any] | None] = {
            provider_id: None for provider_id in PROSPECTIVE_PROVIDER_IDS
        }
        expected_frames = _cached_materialized_metadata(repo_root, split)
        # Use stored order only after sorting by the deterministic frame identity. This keeps JSONL row order non-semantic.
        if len(expected_frames) != len(frames):
            expected_frames = sorted(frames, key=lambda row: (str(row.get("episode_id")), int(row.get("sequence_number", -1))))
        for frame in expected_frames:
            if frame.get("event_type") == "gap_unknown" or frame.get("observation_pixel_digest") is None:
                for provider_id in reachability_state:
                    reachability_state[provider_id] = _gap_reachability_state(frame)
                continue
            for provider_id in PROSPECTIVE_PROVIDER_IDS:
                row = by_frame_provider.get((frame.get("frame_id"), provider_id))
                if row is None:
                    findings.append(_finding("reachability_trace_mismatch", "provider score row missing for scored frame", split=split, frame_id=frame.get("frame_id"), provider_id=provider_id))
                    if max_findings is not None and len(findings) >= max_findings:
                        return _gate("reachability", findings, counts=counts)
                    continue
                expected_outcome, evidence_findings = _expected_semantic_for_row(row, row_actions, row_ids, semantic_cache)
                if evidence_findings or expected_outcome is None:
                    continue
                trace = row.get("reachability_composition_trace")
                if not trace:
                    findings.append(_finding("reachability_trace_mismatch", "provider row is missing reachability composition trace", split=split, frame_id=frame.get("frame_id"), provider_id=provider_id))
                    if max_findings is not None and len(findings) >= max_findings:
                        return _gate("reachability", findings, counts=counts)
                    continue
                expected_trace = compose_reachability_trace(
                    frame_id=str(row["frame_id"]),
                    semantic_outcome=expected_outcome.to_dict(),
                    previous_state=reachability_state[provider_id],
                    reachability_tile=reachability_tile,
                    row_actions=row_actions,
                )
                counts["reachability_traces_checked"] += 1
                if trace.get("reachability_tile_identity") != reachability_tile["tile_digest"]:
                    findings.append(_finding("reachability_tile_mismatch", "reachability trace references a foreign tile identity", split=split, frame_id=row.get("frame_id"), provider_id=provider_id))
                elif trace.get("executed_action") != expected_trace.get("executed_action"):
                    findings.append(_finding("executed_action_mismatch", "reachability trace executed action does not match independent composition", split=split, frame_id=row.get("frame_id"), provider_id=provider_id))
                else:
                    code = validate_reachability_trace(
                        trace,
                        semantic_outcome=expected_outcome.to_dict(),
                        previous_state=reachability_state[provider_id],
                        reachability_tile=reachability_tile,
                        row_actions=row_actions,
                    )
                    if code != "ok":
                        if code in {"foreign_reachability_tile", "foreign_reachability_trace_digest"}:
                            code = "reachability_tile_mismatch" if code == "foreign_reachability_tile" else "reachability_trace_mismatch"
                        findings.append(_finding(code, "stored reachability trace does not match independent composition", split=split, frame_id=row.get("frame_id"), provider_id=provider_id))
                if max_findings is not None and len(findings) >= max_findings:
                    return _gate("reachability", findings, counts=counts)
                reachability_state[provider_id] = _state_from_trace(expected_trace)
    return _gate("reachability", findings, counts=counts)


def _completeness_orphan_gate(output_dir: Path, repo_root: Path, context: Mapping[str, Any], *, max_findings: int | None = None) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    final_ids = {episode_id for values in _episode_ids_by_family(context["plans"]["final"]).values() for episode_id in values}
    known_frame_ids: set[str] = set()
    known_episode_ids: set[str] = set()
    non_control_digest_owner: dict[str, str] = {}
    expected_digest_owners = _expected_digest_owners(repo_root)
    for split in _NON_FINAL_SPLITS:
        frame_rows = _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        evidence_rows = _read_jsonl(output_dir / split / "provider-evidence.jsonl")
        counts[f"{split}_frame_records"] = len(frame_rows)
        counts[f"{split}_provider_records"] = len(evidence_rows)
        for row in frame_rows:
            frame_id = str(row.get("frame_id"))
            episode_id = str(row.get("episode_id"))
            if frame_id in known_frame_ids:
                findings.append(_finding("duplicate_observation_record", "frame identity is duplicated", split=split, frame_id=frame_id))
                if max_findings is not None and len(findings) >= max_findings:
                    return _gate("completeness_orphan", findings, counts=counts)
            known_frame_ids.add(frame_id)
            known_episode_ids.add(episode_id)
            if episode_id in final_ids or row.get("split") == "final":
                findings.append(_finding("forbidden_final_materialization", "final sealed episode has a materialized frame descendant", split=split, episode_id=episode_id))
                if max_findings is not None and len(findings) >= max_findings:
                    return _gate("completeness_orphan", findings, counts=counts)
            digest = row.get("observation_pixel_digest")
            if digest is not None and row.get("expected_disposition") != "information_theoretic_control":
                previous = non_control_digest_owner.get(str(digest))
                expected_owners = expected_digest_owners.get(str(digest), set())
                if previous is not None and previous != episode_id and episode_id not in expected_owners:
                    findings.append(_finding("duplicate_observation_identity_unpermitted", "non-control observation identity is reused by multiple episodes", split=split, episode_id=episode_id, digest=digest))
                    if max_findings is not None and len(findings) >= max_findings:
                        return _gate("completeness_orphan", findings, counts=counts)
                non_control_digest_owner[str(digest)] = episode_id
        scored_frame_ids = {str(row.get("frame_id")) for row in evidence_rows}
        for row in evidence_rows:
            if row.get("frame_id") not in known_frame_ids:
                findings.append(_finding("orphan_score_vector_record", "score vector references an unknown observation", split=split, frame_id=row.get("frame_id")))
            if row.get("episode_id") not in known_episode_ids:
                findings.append(_finding("orphan_score_vector_record", "score vector references an unknown episode", split=split, episode_id=row.get("episode_id")))
            semantic = row.get("semantic_top_set_outcome", {})
            if semantic.get("quantized_score_vector_digest") != row.get("score_vector_digest"):
                findings.append(_finding("semantic_outcome_digest_mismatch", "semantic outcome does not reference the stored score vector identity", split=split, frame_id=row.get("frame_id")))
            trace = row.get("reachability_composition_trace")
            if trace and trace.get("semantic_outcome_digest") != row.get("semantic_outcome_digest"):
                findings.append(_finding("reachability_trace_mismatch", "reachability trace does not reference the stored semantic outcome identity", split=split, frame_id=row.get("frame_id")))
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("completeness_orphan", findings, counts=counts)
        gap_frame_ids = {str(row.get("frame_id")) for row in frame_rows if row.get("event_type") == "gap_unknown" or row.get("observation_pixel_digest") is None}
        if scored_frame_ids & gap_frame_ids:
            findings.append(_finding("gap_event_has_pixels", "typed gap frame has provider score evidence", split=split))
            if max_findings is not None and len(findings) >= max_findings:
                return _gate("completeness_orphan", findings, counts=counts)
    return _gate("completeness_orphan", findings, counts=counts)


def _access_prohibition_gate(output_dir: Path, *, max_findings: int | None = None) -> dict[str, Any]:
    measured = _measured_phase_access_counts(output_dir)
    stored_path = output_dir / "phase-access-audits.json"
    stored = _read_json(stored_path) if stored_path.exists() else {}
    return _build_access_prohibition_gate(measured, stored, max_findings=max_findings)


def _cached_materialized_metadata(repo_root: Path, split: str) -> list[dict[str, Any]]:
    cache_key = (str(repo_root.resolve()), str(split))
    if not hasattr(_cached_materialized_metadata, "_cache"):
        _cached_materialized_metadata._cache = {}  # type: ignore[attr-defined]
    cache: dict[tuple[str, str], list[dict[str, Any]]] = _cached_materialized_metadata._cache  # type: ignore[attr-defined]
    if cache_key not in cache:
        cache[cache_key] = [{key: value for key, value in row.items() if key != "pixels"} for row in _materialize_records(split, repo_root)]
    return cache[cache_key]


def _expected_digest_owners(repo_root: Path) -> dict[str, set[str]]:
    cache_key = str(repo_root.resolve())
    if not hasattr(_expected_digest_owners, "_cache"):
        _expected_digest_owners._cache = {}  # type: ignore[attr-defined]
    cache: dict[str, dict[str, set[str]]] = _expected_digest_owners._cache  # type: ignore[attr-defined]
    if cache_key not in cache:
        owners: dict[str, set[str]] = {}
        for split in _NON_FINAL_SPLITS:
            for row in _cached_materialized_metadata(repo_root, split):
                digest = row.get("observation_pixel_digest")
                if digest is not None and row.get("expected_disposition") != "information_theoretic_control":
                    owners.setdefault(str(digest), set()).add(str(row.get("episode_id")))
        cache[cache_key] = owners
    return cache[cache_key]
# fmt: on
