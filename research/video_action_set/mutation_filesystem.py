"""Video action-set mutation filesystem."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import (
    Any,
    Mapping,
    Sequence,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.observation_provenance import (
    valid_frame_operation_chain as _valid_frame_operation_chain,
)
from zeromodel.video.domains.video_action_set.observation_universe import (
    canonical_observation_universe,
)
from zeromodel.video.domains.video_action_set.reachability_composition import (
    trace_digest as _trace_digest,
)
from research.video_action_set.reference_verification import (
    REFERENCE_VERIFICATION_VERSION,
    _REQUIRED_VERIFICATION_GATES,
    _finding,
    _gate,
)
from zeromodel.video.domains.video_action_set.transformations import (
    _transformation_parameters,
)
from research.evidence.video_complete_row_evidence import QUANTIZATION_SCALE
from zeromodel.video.domains.video_action_set.artifact_io import (
    _read_json,
    _read_jsonl,
    _sha256,
    _write_json,
    _write_jsonl,
)


CLOSURE_REPORT_VERSION = "zeromodel-video-action-set-reference-closure/v1"


# fmt: off
def _file_digest(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _directory_snapshot(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    snapshot: dict[str, dict[str, Any]] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            snapshot[str(item.relative_to(path)).replace("\\", "/")] = {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "digest": _file_digest(item)}
    return snapshot


def _structural_payload(path: Path) -> Any:
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return path.read_text(encoding="utf-8")


def _mutation_structural_snapshot(output_dir: Path, *, only_files: Sequence[str] | None = None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    selected = None if only_files is None else {str(item).replace("\\", "/") for item in only_files}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(output_dir)).replace("\\", "/")
        if selected is not None and relative not in selected:
            continue
        try:
            snapshot[relative] = _structural_payload(path)
        except UnicodeDecodeError:
            snapshot[relative] = {"binary_digest": _file_digest(path)}
    return snapshot


def _rewrite_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _write_jsonl(path, [dict(row) for row in rows])


def _launder_split_manifest(output_dir: Path, split: str) -> None:
    manifest_path = output_dir / f"{split}-manifest.json"
    if not manifest_path.exists():
        return
    frame_rows = _read_jsonl(output_dir / split / "frame-metadata.jsonl")
    evidence_rows = _read_jsonl(output_dir / split / "provider-evidence.jsonl")
    manifest = _read_json(manifest_path)
    manifest["observation_count"] = len(frame_rows)
    manifest["provider_frame_record_count"] = len(evidence_rows)
    manifest["frame_digest"] = _sha256(frame_rows)
    manifest["provider_evidence_digest"] = _sha256(evidence_rows)
    _write_json(manifest_path, manifest)


def _mutate_first_provider_row(output_dir: Path, mutator: Any, *, predicate: Any | None = None, splits: Sequence[str] = ("development", "calibration", "selection")) -> None:
    for split in splits:
        path = output_dir / split / "provider-evidence.jsonl"
        rows = _read_jsonl(path)
        for row in rows:
            if predicate is None or predicate(row):
                mutator(row)
                _rewrite_jsonl(path, rows)
                _launder_split_manifest(output_dir, split)
                return
    raise VPMValidationError("mutation fixture lacks a matching provider row")


def _mutate_first_frame_row(output_dir: Path, mutator: Any, *, predicate: Any | None = None) -> None:
    path = output_dir / "selection" / "frame-metadata.jsonl"
    rows = _read_jsonl(path)
    for row in rows:
        if predicate is None or predicate(row):
            mutator(row)
            _rewrite_jsonl(path, rows)
            _launder_split_manifest(output_dir, "selection")
            return
    raise VPMValidationError("mutation fixture lacks a matching frame row")


def _apply_evidence_mutation(output_dir: Path, case_name: str) -> None:
    def mutate(row: dict[str, Any]) -> None:
        match case_name:
            case 'evidence_raw_score_preserve_quantized_bin':
                row["all_112_raw_scores"][0] = float(row["all_112_raw_scores"][0]) + 1e-12
            case 'evidence_raw_score_cross_quantization_boundary':
                row["all_112_raw_scores"][0] = 0.0 if int(row["all_112_quantized_scores"][0]) else 1.0
            case 'evidence_quantized_score_changed':
                row["all_112_quantized_scores"][0] = min(QUANTIZATION_SCALE, int(row["all_112_quantized_scores"][0]) + 1)
                row["score_vector_digest"] = _sha256({"laundered": row["all_112_quantized_scores"]})
                row["quantized_score_vector_digest"] = row["score_vector_digest"]
            case 'evidence_remove_row_score':
                for key in ("all_112_row_ids", "all_112_raw_scores", "all_112_quantized_scores"):
                    row[key].pop()
            case 'evidence_duplicate_row_score':
                row["all_112_row_ids"][1] = row["all_112_row_ids"][0]
            case 'evidence_introduce_foreign_row':
                row["all_112_row_ids"][0] = "foreign:row"
            case 'evidence_reorder_stored_rows':
                for key in ("all_112_row_ids", "all_112_raw_scores", "all_112_quantized_scores"):
                    row[key][0], row[key][1] = row[key][1], row[key][0]
            case 'evidence_alter_ranking_order':
                row["complete_ordered_ranking"][0], row["complete_ordered_ranking"][1] = row["complete_ordered_ranking"][1], row["complete_ordered_ranking"][0]
            case 'evidence_alter_tie_group_membership':
                row["tie_groups"][0]["row_ids"][0] = row["complete_ordered_ranking"][-1]
            case 'evidence_split_tie_group_incorrectly':
                top = row["tie_groups"][0]
                if len(top["row_ids"]) < 2:
                    top["row_ids"].append(row["complete_ordered_ranking"][1])
                moved = top["row_ids"].pop()
                row["tie_groups"].insert(1, {"tie_group_index": 1, "quantized_score": top["quantized_score"], "row_ids": [moved]})
            case 'evidence_merge_distinct_score_groups':
                if len(row["tie_groups"]) > 1:
                    row["tie_groups"][0]["row_ids"].extend(row["tie_groups"][1]["row_ids"])
                    row["tie_groups"].pop(1)
                else:
                    row["tie_groups"][0]["row_ids"].append(row["complete_ordered_ranking"][-1])
            case 'evidence_alter_quantized_score_vector_digest':
                row["score_vector_digest"] = "sha256:" + "0" * 64
                row["quantized_score_vector_digest"] = row["score_vector_digest"]
            case 'evidence_alter_raw_diagnostic_digest':
                row["raw_score_diagnostic_digest"] = "sha256:" + "0" * 64
    _mutate_first_provider_row(output_dir, mutate)
    return


def _apply_semantic_mutation(output_dir: Path, case_name: str) -> None:
    def semantic_mutate(row: dict[str, Any]) -> None:
        payload = row["semantic_top_set_outcome"]
        match case_name:
            case 'semantic_resolved_row_for_action_unanimous_tie':
                payload["resolved_row_id"] = payload["top_row_ids"][0]
                row["resolved_row"] = payload["resolved_row_id"]
            case 'semantic_resolved_action_for_conflicting_tie':
                payload["resolved_action_id"] = payload["top_action_ids"][0]
                row["resolved_action"] = payload["resolved_action_id"]
                row["winner_action"] = payload["resolved_action_id"]
            case 'semantic_convert_conflicting_tie_to_unique_row':
                payload["status"] = "unique_row"
                row["semantic_status"] = "unique_row"
            case 'semantic_change_top_row_policy_action':
                payload["top_row_actions"][0]["action_id"] = "FIRE" if payload["top_row_actions"][0]["action_id"] != "FIRE" else "LEFT"
            case 'semantic_change_rejection_reason':
                payload["rejection_reason"] = "mutated rejection reason"
            case 'semantic_alter_outcome_digest':
                payload["semantic_outcome_digest"] = "sha256:" + "1" * 64
                row["semantic_outcome_digest"] = payload["semantic_outcome_digest"]
            case 'semantic_lexically_reorder_tied_rows':
                payload["top_row_ids"] = list(reversed(payload["top_row_ids"]))
                row["top_row_ids"] = list(reversed(row["top_row_ids"]))
            case 'semantic_reorder_rows_preserving_action_equivalence':
                payload["top_row_actions"] = list(reversed(payload["top_row_actions"]))
    if case_name == "semantic_resolved_row_for_action_unanimous_tie":
        _mutate_first_provider_row(output_dir, semantic_mutate, predicate=lambda row: row.get("semantic_status") == "action_unanimous_tie")
    elif case_name in {"semantic_resolved_action_for_conflicting_tie", "semantic_convert_conflicting_tie_to_unique_row", "semantic_change_rejection_reason"}:
        _mutate_first_provider_row(output_dir, semantic_mutate, predicate=lambda row: row.get("semantic_status") == "conflicting_action_tie")
    elif case_name in {"semantic_lexically_reorder_tied_rows", "semantic_reorder_rows_preserving_action_equivalence"}:
        _mutate_first_provider_row(output_dir, semantic_mutate, predicate=lambda row: len(row.get("top_row_ids", [])) > 1)
    else:
        _mutate_first_provider_row(output_dir, semantic_mutate)
    return


def _apply_policy_mutation(output_dir: Path, case_name: str) -> None:
    payload = _read_json(output_dir / "policy-artifact.json")
    if case_name == "policy_remove_policy_row":
        payload["row_ids"].pop()
    elif case_name == "policy_add_undeclared_row":
        payload["row_ids"].append("foreign:row")
        payload["row_action"]["foreign:row"] = "LEFT"
    elif case_name == "policy_alter_artifact_identity":
        payload["policy_artifact_id"] = "sha256:foreign-policy"
    elif case_name == "policy_mapping_recomputed_superficial_metadata":
        payload["row_action_digest"] = _sha256({"laundered": payload["row_action"], "mutation": case_name})
    else:
        first = payload["row_ids"][0]
        payload["row_action"][first] = "FIRE" if payload["row_action"][first] != "FIRE" else "LEFT"
        payload["row_action_digest"] = _sha256({"laundered": payload["row_action"]})
    _write_json(output_dir / "policy-artifact.json", payload)
    return


def _apply_observation_mutation(output_dir: Path, case_name: str) -> None:
    if case_name == "observation_swap_two_frame_payloads":
        path = output_dir / "selection" / "frame-metadata.jsonl"
        rows = _read_jsonl(path)
        rows[0]["observation_pixel_digest"], rows[1]["observation_pixel_digest"] = rows[1]["observation_pixel_digest"], rows[0]["observation_pixel_digest"]
        _rewrite_jsonl(path, rows)
        _launder_split_manifest(output_dir, "selection")
    elif case_name == "observation_alter_frame_identity":
        _mutate_first_frame_row(output_dir, lambda row: row.__setitem__("frame_id", str(row["frame_id"]) + ":mutated"))
    elif case_name == "observation_reuse_under_two_episode_ids":
        path = output_dir / "selection" / "frame-metadata.jsonl"
        rows = _read_jsonl(path)
        first_row = next(row for row in rows if row.get("observation_pixel_digest") and row.get("expected_disposition") != "information_theoretic_control")
        first_digest = first_row["observation_pixel_digest"]
        first_episode = first_row["episode_id"]
        for row in rows:
            if row.get("observation_pixel_digest") and row.get("expected_disposition") != "information_theoretic_control" and row["episode_id"] != first_episode:
                row["observation_pixel_digest"] = first_digest
                break
        _rewrite_jsonl(path, rows)
        _launder_split_manifest(output_dir, "selection")
    elif case_name == "observation_substitute_frame_for_declared_gap":
        _mutate_first_frame_row(output_dir, lambda row: (row.__setitem__("event_type", "frame"), row.__setitem__("gap_declaration", None), row.__setitem__("observation_pixel_digest", "sha256:" + "2" * 64)), predicate=lambda row: row.get("event_type") == "gap_unknown")
    else:
        replacement = {
            "observation_flip_byte_digest": "sha256:" + "3" * 64,
            "observation_change_pixels_and_recompute_digest": "sha256:" + "b" * 64,
            "observation_change_digest_without_pixels": "sha256:" + "c" * 64,
        }[case_name]
        _mutate_first_frame_row(output_dir, lambda row: row.__setitem__("observation_pixel_digest", replacement), predicate=lambda row: row.get("observation_pixel_digest") is not None)
    return


def _apply_seed_mutation(output_dir: Path, case_name: str) -> None:
    if case_name in {"seed_alter_root_seed_material", "seed_alter_root_seed_digest"}:
        path = output_dir / ("generator-identity.json" if case_name == "seed_alter_root_seed_material" else "benchmark-contract-identity.json")
        payload = _read_json(path)
        if case_name == "seed_alter_root_seed_material":
            payload["seed_material"] = str(payload["seed_material"]) + "|mutated"
        else:
            payload["seed_digest"] = "sha256:" + "4" * 64
        _write_json(path, payload)
        return
    path = output_dir / "episode-plan.json"
    payload = _read_json(path)
    plan = payload["splits"]["selection"]["episodes"][0]
    match case_name:
        case 'seed_alter_derived_seed':
            plan["derived_seed_identity"] = "sha256:" + "5" * 64
        case 'seed_alter_derivation_namespace':
            plan["seed_lineage"]["concrete_episode_seed"]["namespace"] = "mutated_namespace"
        case 'seed_alter_episode_ordinal':
            plan["ordinal"] += 1
        case 'seed_alter_split_identity':
            plan["split"] = "development"
        case 'seed_move_episode_between_splits':
            moved = payload["splits"]["development"]["episodes"][0]
            moved["split"] = "selection"
            moved["episode_id"] = "selection:moved-from-development"
        case 'seed_duplicate_episode_id':
            payload["splits"]["selection"]["episodes"][1]["episode_id"] = plan["episode_id"]
        case 'seed_alter_source_row':
            plan["source_row_id"] = payload["policy_row_ids"][-1]
        case 'seed_alter_splice_partner':
            target = next(item for item in payload["splits"]["selection"]["episodes"] if item.get("mutation_kind") == "conflicting_action_splice")
            target["secondary_row_id"] = target["source_row_id"]
        case 'seed_alter_transformation_parameters':
            plan["frame_plans"][0]["transformation_parameters"]["dx"] = 99
        case 'seed_alter_planned_family':
            plan["family_label"] = "temporal_negative"
        case 'seed_alter_sealed_plan_digest':
            plan["plan_digest"] = "sha256:" + "6" * 64
        case 'seed_alter_final_sealed_identity':
            final_path = output_dir / "final-split-sealed-plan.json"
            final = _read_json(final_path)
            final["episodes"][0]["episode_id"] = "final:valid:mutated"
            final["sealed_episode_ids"]["valid"][0] = "final:valid:mutated"
            final["sealed_plan_digest"] = _sha256({key: value for key, value in final.items() if key != "sealed_plan_digest"})
            _write_json(final_path, final)
            _write_json(output_dir / "final-split-sealed-digest.json", {"digest": final["sealed_plan_digest"]})
            return
        case 'seed_final_observation_provenance_materialized':
            final_path = output_dir / "final-split-sealed-plan.json"
            final = _read_json(final_path)
            final["episodes"][0]["final_observation_provenance"] = {
                "materialization_status": "materialized",
                "observation_payload_included": True,
                "provenance": "mutated_final_payload_claim",
            }
            final["sealed_plan_digest"] = _sha256({key: value for key, value in final.items() if key != "sealed_plan_digest"})
            _write_json(final_path, final)
            _write_json(output_dir / "final-split-sealed-digest.json", {"digest": final["sealed_plan_digest"]})
            return
    _write_json(path, payload)
    return


def _apply_family_mutation(output_dir: Path, case_name: str) -> None:
    def family_mutate(row: dict[str, Any]) -> None:
        metadata = row.setdefault("metadata", {})
        trace = metadata.setdefault("family_intervention_trace", {})
        match case_name:
            case 'family_conflicting_splice_same_action_rows':
                metadata["competitor_action_id"] = metadata.get("source_action_id")
            case 'family_splice_zero_source_contribution':
                trace["secondary_contributing_pixel_count"] = 0
            case 'family_splice_output_equal_one_source':
                trace["output_observation_digest"] = trace.get("primary_source_digest")
            case 'family_splice_valid_state_collision':
                canonical = canonical_observation_universe()["rows"][0]
                row["observation_pixel_digest"] = canonical["observation_pixel_digest"]
                metadata["observation_operation_chain"] = _valid_frame_operation_chain(canonical["row_id"], _transformation_parameters("exact", 0))
                trace["output_observation_digest"] = canonical["observation_pixel_digest"]
                trace["canonical_collision_count"] = 1
                trace["canonical_collision_rows"] = [{"row_id": canonical["row_id"], "action_id": canonical["action_id"]}]
            case 'family_splice_target_evidence_count_removed':
                trace.setdefault("action_relevant_region_contribution_counts", {})["secondary_additive_target_pixel_count"] = 0
            case 'family_critical_empty_coordinate_set':
                trace["changes"] = []
            case 'family_corrupt_noncritical_coordinates':
                if trace.get("changes"):
                    trace["changes"][0]["y"] = 0
                    trace["changes"][0]["x"] = 0
            case 'family_replacement_value_identical':
                if trace.get("changes"):
                    trace["changes"][0]["replacement"] = trace["changes"][0]["original"]
            case 'family_clipping_quantization_noop':
                trace["changed_pixel_count"] = 0
            case 'family_reordered_metadata_original_payload_order' | 'family_identity_permutation_labelled_reordered':
                metadata["original_frame_index"] = int(row.get("sequence_number", 0)) if case_name == "family_reordered_metadata_original_payload_order" else -1
            case 'family_stale_label_without_repeated_bytes' | 'family_stale_repeat_naturally_identical':
                repeat = metadata.setdefault("stale_repeat", {})
                if case_name == "family_stale_label_without_repeated_bytes":
                    repeat["replacement_digest"] = repeat.get("original_destination_digest", row.get("observation_pixel_digest"))
                else:
                    repeat["original_destination_digest"] = repeat.get("replacement_digest", row.get("observation_pixel_digest"))
            case 'family_impossible_transition_reachable_pair':
                transition = metadata.setdefault("impossible_transition", {})
                edge = transition.get("consulted_edge", {})
                if edge.get("reachable_row_ids"):
                    transition["destination_row_id"] = edge["reachable_row_ids"][0]
            case 'family_gap_event_carrying_pixels':
                row["pixels"] = [[0]]
            case 'family_information_control_pixel_difference':
                row["observation_pixel_digest"] = "sha256:" + "7" * 64
            case 'family_control_denominator_leak':
                metadata["denominator_eligible"] = True
            case 'family_control_visible_source_leak':
                metadata["provider_visible_fields"] = ["pixels", "source_row_id"]
            case 'family_episode_disposition_mismatch':
                row["episode_disposition"] = "valid"
                row["denominator_class"] = "valid_denominator"
    if case_name == "family_control_hidden_history_collapse":
        path = output_dir / "selection" / "frame-metadata.jsonl"
        rows = _read_jsonl(path)
        control_episode = next(row["episode_id"] for row in rows if row.get("expected_disposition") == "information_theoretic_control")
        control_rows = [row for row in rows if row.get("episode_id") == control_episode]
        first_metadata = control_rows[0].setdefault("metadata", {})
        hidden_history_id = first_metadata.get("hidden_source_history_id")
        hidden_label = first_metadata.get("hidden_source_label_digest")
        for row in control_rows:
            metadata = row.setdefault("metadata", {})
            metadata["hidden_source_history_id"] = hidden_history_id
            metadata["hidden_source_label_digest"] = hidden_label
        _rewrite_jsonl(path, rows)
        _launder_split_manifest(output_dir, "selection")
        return
    if "splice" in case_name:
        _mutate_first_frame_row(output_dir, family_mutate, predicate=lambda row: row.get("family") == "conflicting_action_splice")
    elif "critical" in case_name or "corrupt" in case_name or "replacement" in case_name or "noop" in case_name:
        _mutate_first_frame_row(output_dir, family_mutate, predicate=lambda row: row.get("family") == "critical_evidence_corruption")
    elif "reordered" in case_name or "permutation" in case_name:
        _mutate_first_frame_row(output_dir, family_mutate, predicate=lambda row: row.get("metadata", {}).get("sequence_rule") == "non_identity_permutation")
    elif "stale" in case_name:
        _mutate_first_frame_row(output_dir, family_mutate, predicate=lambda row: "stale_repeat" in row.get("metadata", {}))
    elif "impossible" in case_name:
        _mutate_first_frame_row(output_dir, family_mutate, predicate=lambda row: "impossible_transition" in row.get("metadata", {}))
    elif "gap" in case_name:
        _mutate_first_frame_row(output_dir, family_mutate, predicate=lambda row: row.get("event_type") == "gap_unknown")
    else:
        _mutate_first_frame_row(output_dir, family_mutate, predicate=lambda row: row.get("expected_disposition") == "information_theoretic_control")
    return


def _apply_reachability_mutation(output_dir: Path, case_name: str) -> None:
    if case_name in {"reachability_add_impossible_edge", "reachability_change_tile_identity", "reachability_change_unrelated_edge"}:
        payload = _read_json(output_dir / "reachability-tile-reference.json")
        if case_name == "reachability_add_impossible_edge":
            payload["tile_digest"] = "sha256:" + "8" * 64
            payload["mutation_kind"] = "added-impossible-edge"
        elif case_name == "reachability_change_unrelated_edge":
            payload["tile_digest"] = "sha256:" + "7" * 64
            payload["mutation_kind"] = "changed-unrelated-edge"
        else:
            payload["tile_digest"] = "sha256:" + "6" * 64
        _write_json(output_dir / "reachability-tile-reference.json", payload)
        return
    def reach_mutate(row: dict[str, Any]) -> None:
        trace = row["reachability_composition_trace"]
        match case_name:
            case 'reachability_remove_applicable_edge':
                trace["consulted_edges"][0]["reachable_row_ids"] = []
            case 'reachability_redirect_applicable_edge':
                trace["reachable_candidate_pairs"][0]["destination_row_id"] = row["all_112_row_ids"][-1]
            case 'reachability_alter_destination_action':
                row["semantic_top_set_outcome"]["top_row_actions"][0]["action_id"] = "FIRE" if row["semantic_top_set_outcome"]["top_row_actions"][0]["action_id"] != "FIRE" else "LEFT"
                trace["mutation_kind"] = "altered_destination_action"
            case 'reachability_alter_consulted_edge_list' | 'reachability_omit_consulted_edge':
                if case_name == "reachability_alter_consulted_edge_list":
                    trace["consulted_edges"][0]["action_id"] = "FIRE" if trace["consulted_edges"][0]["action_id"] != "FIRE" else "LEFT"
                else:
                    trace["consulted_edges"] = []
            case 'reachability_add_unconsulted_edge_to_trace':
                trace["consulted_edges"].append({"source_row_id": "foreign", "action_id": "LEFT", "reachable_row_ids": []})
            case 'reachability_alter_reachable_pair_set':
                trace["reachable_candidate_pairs"].append({"source_row_id": "foreign", "action_id": "LEFT", "destination_row_id": "foreign"})
            case 'reachability_alter_retained_candidate_rows':
                trace["retained_rows"] = list(reversed(trace.get("retained_rows", []))) + ["foreign:retained"]
            case 'reachability_alter_removed_candidate_rows':
                trace["removed_rows"] = list(reversed(trace.get("removed_rows", []))) + ["foreign"]
            case 'reachability_replace_rejection_with_lexical_winner':
                trace["status"] = "resolved"
                trace["executed_action"] = "LEFT"
            case 'reachability_change_executed_action':
                trace["executed_action"] = "FIRE" if trace.get("executed_action") != "FIRE" else "LEFT"
            case 'reachability_use_foreign_trace_digest':
                trace["trace_digest"] = "sha256:" + "9" * 64
        if case_name != "reachability_use_foreign_trace_digest":
            trace["trace_digest"] = _trace_digest(trace)
    if case_name in {"reachability_remove_applicable_edge", "reachability_alter_consulted_edge_list", "reachability_omit_consulted_edge", "reachability_add_unconsulted_edge_to_trace"}:
        predicate = lambda row: bool(row.get("reachability_composition_trace", {}).get("consulted_edges"))  # noqa: E731
    elif case_name in {"reachability_redirect_applicable_edge", "reachability_alter_reachable_pair_set"}:
        predicate = lambda row: bool(row.get("reachability_composition_trace", {}).get("reachable_candidate_pairs"))  # noqa: E731
    elif case_name == "reachability_alter_removed_candidate_rows":
        predicate = lambda row: bool(row.get("reachability_composition_trace", {}).get("removed_rows"))  # noqa: E731
    elif case_name == "reachability_replace_rejection_with_lexical_winner":
        predicate = lambda row: row.get("reachability_composition_trace", {}).get("status") == "rejected"  # noqa: E731
    elif case_name == "reachability_change_executed_action":
        predicate = lambda row: row.get("reachability_composition_trace", {}).get("executed_action") is not None  # noqa: E731
    else:
        predicate = lambda row: row.get("reachability_composition_trace") is not None  # noqa: E731
    _mutate_first_provider_row(output_dir, reach_mutate, predicate=predicate)
    return


def _apply_access_mutation(output_dir: Path, case_name: str) -> None:
    if case_name in {"access_increment_final_materialization_count", "access_increment_forbidden_access_counter"}:
        payload = _read_json(output_dir / "phase-access-audits.json")
        payload["forbidden_final_access_counter"] = int(payload.get("forbidden_final_access_counter", 0)) + 1
        if case_name == "access_increment_final_materialization_count":
            payload["final_materialization_count"] = int(payload.get("final_materialization_count", 0)) + 1
        _write_json(output_dir / "phase-access-audits.json", payload)
    elif case_name in {"access_add_final_observation_artifact", "access_add_final_score_vector_record", "access_add_final_reachability_trace"}:
        final_plan = _read_json(output_dir / "final-split-sealed-plan.json")
        final_id = final_plan["sealed_episode_ids"]["valid"][0]
        if case_name == "access_add_final_observation_artifact":
            _write_jsonl(output_dir / "final" / "frame-metadata.jsonl", [{"split": "final", "episode_id": final_id, "frame_id": f"final:{final_id}:frame-00"}])
        else:
            row = _read_jsonl(output_dir / "selection" / "provider-evidence.jsonl")[0]
            row["split"] = "final"
            row["episode_id"] = final_id
            if case_name == "access_add_final_reachability_trace":
                row["reachability_composition_trace"] = {"trace_digest": "sha256:" + "a" * 64}
            _write_jsonl(output_dir / "final" / "provider-evidence.jsonl", [row])
    elif case_name == "access_record_calibration_execution":
        _write_json(output_dir / "selected-calibration.json", {"executed": True})
    elif case_name == "access_record_architecture_selection_execution":
        _write_json(output_dir / "selected-architecture.json", {"executed": True})
    else:
        report: dict[str, Any] = {
            "version": REFERENCE_VERIFICATION_VERSION,
            "verified": True,
            "gates": [_gate(name, []) for name in _REQUIRED_VERIFICATION_GATES],
            "primary_failure_code": None,
            "primary_failure_gate": None,
        }
        closure: dict[str, Any] = {"version": CLOSURE_REPORT_VERSION, "supported_status": "reference_instrument_correctness_unresolved", "verification": report}
        if case_name == "access_change_failed_gate_status_to_passed":
            closure["verification"]["gates"][0]["status"] = "passed"
            closure["verification"]["gates"][0]["findings"] = [_finding("status_claim_not_supported", "synthetic failed gate relabelled as passed")]
            closure["verification"]["gates"][0]["finding_count"] = 1
        if case_name == "access_remove_required_gate_from_closure_report":
            closure["verification"]["gates"] = closure["verification"]["gates"][1:]
        if case_name == "access_change_repository_status_to_correct":
            closure["supported_status"] = "reference_instrument_correct"
        closure["closure_report_digest"] = _sha256({key: value for key, value in closure.items() if key != "closure_report_digest"})
        _write_json(output_dir / "reference-closure-report.json", closure)
    return


def _apply_reference_mutation(output_dir: Path, case_name: str) -> None:
    if case_name.startswith('evidence_'):
        _apply_evidence_mutation(output_dir, case_name)
        return
    if case_name.startswith('semantic_'):
        _apply_semantic_mutation(output_dir, case_name)
        return
    if case_name.startswith('policy_'):
        _apply_policy_mutation(output_dir, case_name)
        return
    if case_name.startswith('observation_'):
        _apply_observation_mutation(output_dir, case_name)
        return
    if case_name.startswith('seed_'):
        _apply_seed_mutation(output_dir, case_name)
        return
    if case_name.startswith('family_'):
        _apply_family_mutation(output_dir, case_name)
        return
    if case_name.startswith('reachability_'):
        _apply_reachability_mutation(output_dir, case_name)
        return
    if case_name.startswith('access_'):
        _apply_access_mutation(output_dir, case_name)
        return
    raise VPMValidationError(f"unsupported mutation case: {case_name}")
# fmt: on
