from __future__ import annotations

from typing import Any, Mapping, Sequence

from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from research.video_action_set.reference_verification import (
    _finding,
    _report_failure_codes,
)

MUTATION_AUDIT_VERSION = "zeromodel-video-action-set-reference-mutation-audit/v1"

_MUTATION_NAMES = (
    "evidence_raw_score_preserve_quantized_bin",
    "evidence_raw_score_cross_quantization_boundary",
    "evidence_quantized_score_changed",
    "evidence_remove_row_score",
    "evidence_duplicate_row_score",
    "evidence_introduce_foreign_row",
    "evidence_reorder_stored_rows",
    "evidence_alter_ranking_order",
    "evidence_alter_tie_group_membership",
    "evidence_split_tie_group_incorrectly",
    "evidence_merge_distinct_score_groups",
    "evidence_alter_quantized_score_vector_digest",
    "evidence_alter_raw_diagnostic_digest",
    "semantic_resolved_row_for_action_unanimous_tie",
    "semantic_resolved_action_for_conflicting_tie",
    "semantic_convert_conflicting_tie_to_unique_row",
    "semantic_change_top_row_policy_action",
    "semantic_change_rejection_reason",
    "semantic_alter_outcome_digest",
    "semantic_lexically_reorder_tied_rows",
    "semantic_reorder_rows_preserving_action_equivalence",
    "policy_alter_row_to_action_mapping",
    "policy_remove_policy_row",
    "policy_add_undeclared_row",
    "policy_alter_artifact_identity",
    "policy_mapping_recomputed_superficial_metadata",
    "observation_flip_byte_digest",
    "observation_change_pixels_and_recompute_digest",
    "observation_change_digest_without_pixels",
    "observation_swap_two_frame_payloads",
    "observation_alter_frame_identity",
    "observation_reuse_under_two_episode_ids",
    "observation_substitute_frame_for_declared_gap",
    "seed_alter_root_seed_material",
    "seed_alter_root_seed_digest",
    "seed_alter_derived_seed",
    "seed_alter_derivation_namespace",
    "seed_alter_episode_ordinal",
    "seed_alter_split_identity",
    "seed_move_episode_between_splits",
    "seed_duplicate_episode_id",
    "seed_alter_source_row",
    "seed_alter_splice_partner",
    "seed_alter_transformation_parameters",
    "seed_alter_planned_family",
    "seed_alter_sealed_plan_digest",
    "seed_alter_final_sealed_identity",
    "seed_final_observation_provenance_materialized",
    "family_conflicting_splice_same_action_rows",
    "family_splice_zero_source_contribution",
    "family_splice_output_equal_one_source",
    "family_splice_valid_state_collision",
    "family_splice_target_evidence_count_removed",
    "family_critical_empty_coordinate_set",
    "family_corrupt_noncritical_coordinates",
    "family_replacement_value_identical",
    "family_clipping_quantization_noop",
    "family_reordered_metadata_original_payload_order",
    "family_identity_permutation_labelled_reordered",
    "family_stale_label_without_repeated_bytes",
    "family_stale_repeat_naturally_identical",
    "family_impossible_transition_reachable_pair",
    "family_gap_event_carrying_pixels",
    "family_information_control_pixel_difference",
    "family_control_denominator_leak",
    "family_control_hidden_history_collapse",
    "family_control_visible_source_leak",
    "family_episode_disposition_mismatch",
    "reachability_remove_applicable_edge",
    "reachability_redirect_applicable_edge",
    "reachability_alter_destination_action",
    "reachability_add_impossible_edge",
    "reachability_change_tile_identity",
    "reachability_change_unrelated_edge",
    "reachability_alter_consulted_edge_list",
    "reachability_omit_consulted_edge",
    "reachability_add_unconsulted_edge_to_trace",
    "reachability_alter_reachable_pair_set",
    "reachability_alter_retained_candidate_rows",
    "reachability_alter_removed_candidate_rows",
    "reachability_replace_rejection_with_lexical_winner",
    "reachability_change_executed_action",
    "reachability_use_foreign_trace_digest",
    "access_increment_final_materialization_count",
    "access_add_final_observation_artifact",
    "access_add_final_score_vector_record",
    "access_add_final_reachability_trace",
    "access_increment_forbidden_access_counter",
    "access_record_calibration_execution",
    "access_record_architecture_selection_execution",
    "access_change_failed_gate_status_to_passed",
    "access_change_repository_status_to_correct",
    "access_remove_required_gate_from_closure_report",
)

_MUTATION_FAILURE_CODES = (
    "raw_diagnostic_digest_mismatch",
    "quantized_score_vector_mismatch",
    "quantized_score_vector_mismatch",
    "score_row_universe_mismatch",
    "score_row_universe_mismatch",
    "score_row_universe_mismatch",
    "score_row_universe_mismatch",
    "ranking_reconstruction_mismatch",
    "tie_group_reconstruction_mismatch",
    "tie_group_reconstruction_mismatch",
    "tie_group_reconstruction_mismatch",
    "quantized_score_vector_mismatch",
    "raw_diagnostic_digest_mismatch",
    "resolved_row_not_permitted",
    "resolved_action_not_permitted",
    "semantic_status_mismatch",
    "policy_action_mapping_mismatch",
    "semantic_status_mismatch",
    "semantic_outcome_digest_mismatch",
    None,
    None,
    "policy_action_mapping_mismatch",
    "policy_action_mapping_mismatch",
    "policy_action_mapping_mismatch",
    "policy_action_mapping_mismatch",
    "policy_action_mapping_mismatch",
    "observation_digest_mismatch",
    "observation_digest_mismatch",
    "observation_digest_mismatch",
    "observation_digest_mismatch",
    "frame_identity_mismatch",
    "duplicate_observation_identity_unpermitted",
    "gap_event_structure_mismatch",
    "episode_seed_derivation_mismatch",
    "benchmark_contract_identity_mismatch",
    "episode_seed_derivation_mismatch",
    "sealed_episode_identity_mismatch",
    "sealed_episode_identity_mismatch",
    "episode_split_reassignment",
    "episode_split_reassignment",
    "duplicate_episode_id",
    "episode_seed_derivation_mismatch",
    "episode_seed_derivation_mismatch",
    "sealed_episode_identity_mismatch",
    "sealed_episode_identity_mismatch",
    "sealed_episode_identity_mismatch",
    "sealed_episode_identity_mismatch",
    "final_observation_provenance_mismatch",
    "family_contract_violation",
    "family_contract_violation",
    "family_regeneration_mismatch",
    "invalid_family_valid_state_collision",
    "family_contract_violation",
    "family_contract_violation",
    "family_contract_violation",
    "family_contract_violation",
    "family_no_op",
    "family_contract_violation",
    "family_contract_violation",
    "family_contract_violation",
    "family_contract_violation",
    "transition_classification_mismatch",
    "gap_event_has_pixels",
    "control_byte_identity_mismatch",
    "control_denominator_leak",
    "control_hidden_history_not_ambiguous",
    "control_provider_visible_leak",
    "family_disposition_mismatch",
    "consulted_edge_mismatch",
    "reachable_pair_mismatch",
    "policy_action_mapping_mismatch",
    "reachability_tile_mismatch",
    "reachability_tile_mismatch",
    "reachability_tile_mismatch",
    "consulted_edge_mismatch",
    "consulted_edge_mismatch",
    "consulted_edge_mismatch",
    "reachable_pair_mismatch",
    "reachability_trace_mismatch",
    "reachability_trace_mismatch",
    "executed_action_mismatch",
    "executed_action_mismatch",
    "reachability_trace_mismatch",
    "status_claim_not_supported",
    "forbidden_final_materialization",
    "forbidden_final_score_access",
    "forbidden_final_score_access",
    "status_claim_not_supported",
    "forbidden_calibration_execution",
    "forbidden_selection_execution",
    "status_claim_not_supported",
    "status_claim_not_supported",
    "closure_gate_missing",
)

_MUTATION_METADATA: dict[str, dict[str, Any]] = {
    "evidence_quantized_score_changed": {"digest_laundering": True},
    "semantic_alter_outcome_digest": {"digest_laundering": True},
    "semantic_lexically_reorder_tied_rows": {"invariant": True},
    "semantic_reorder_rows_preserving_action_equivalence": {"invariant": True},
    "observation_change_pixels_and_recompute_digest": {"digest_laundering": True},
    "observation_reuse_under_two_episode_ids": {
        "gate_scope": ("structural_identity", "completeness_orphan")
    },
    "seed_alter_final_sealed_identity": {"digest_laundering": True},
    "seed_final_observation_provenance_materialized": {"digest_laundering": True},
    "family_clipping_quantization_noop": {"digest_laundering": True},
    "reachability_change_tile_identity": {"digest_laundering": True},
    "access_increment_forbidden_access_counter": {"digest_laundering": True},
}

_MUTATION_ARTIFACT_CLASSES = {
    "evidence": "evidence",
    "semantic": "semantic",
    "policy": "policy",
    "observation": "observation",
    "seed": "episode_plan",
    "family": "family_output",
    "reachability": "reachability_trace",
    "access": "access_status",
}


def _build_mutation_case(name: str, failure_code: str | None) -> dict[str, Any]:
    prefix = name.split("_", 1)[0]
    case = {
        "name": name,
        "expected_primary_failure_code": failure_code,
        "artifact_class": _MUTATION_ARTIFACT_CLASSES[prefix],
    }
    case.update(_MUTATION_METADATA.get(name, {}))
    return case


_MUTATION_CASES: tuple[dict[str, Any], ...] = tuple(
    _build_mutation_case(name, failure_code)
    for name, failure_code in zip(
        _MUTATION_NAMES,
        _MUTATION_FAILURE_CODES,
        strict=True,
    )
)


_MUTATION_GATE_SCOPE = {
    "evidence": ("structural_identity", "semantic_outcome"),
    "semantic": ("structural_identity", "semantic_outcome"),
    "policy": ("structural_identity",),
    "observation": (
        "structural_identity",
        "episode_regeneration",
        "completeness_orphan",
    ),
    "episode_plan": ("structural_identity", "seed_and_plan"),
    "family_output": ("structural_identity", "family_contract"),
    "reachability_trace": ("structural_identity", "semantic_outcome", "reachability"),
    "access_status": ("structural_identity", "access_prohibition"),
}


def _mutation_case_by_name() -> dict[str, dict[str, Any]]:
    return {str(case["name"]): dict(case) for case in _MUTATION_CASES}


def _mutation_property(case: Mapping[str, Any]) -> str:
    return str(
        case.get("protected_scientific_property") or str(case["name"]).replace("_", " ")
    )


def _mutation_expected_files(case: Mapping[str, Any]) -> tuple[str, ...]:
    artifact_class = str(case["artifact_class"])
    name = str(case.get("name", case.get("mutation_id", "")))
    if artifact_class in {"evidence", "semantic", "reachability_trace"}:
        files = ["development/provider-evidence.jsonl", "development-manifest.json"]
        if name in {
            "reachability_add_impossible_edge",
            "reachability_change_tile_identity",
            "reachability_change_unrelated_edge",
        }:
            files = ["reachability-tile-reference.json"]
        return tuple(files)
    if artifact_class == "policy":
        return ("policy-artifact.json",)
    if artifact_class in {"observation", "family_output"}:
        return ("selection/frame-metadata.jsonl", "selection-manifest.json")
    if artifact_class == "episode_plan":
        if name in {
            "seed_alter_final_sealed_identity",
            "seed_final_observation_provenance_materialized",
        }:
            return ("final-split-sealed-plan.json", "final-split-sealed-digest.json")
        if name in {"seed_alter_root_seed_material", "seed_alter_root_seed_digest"}:
            return (
                ("generator-identity.json",)
                if name == "seed_alter_root_seed_material"
                else ("benchmark-contract-identity.json",)
            )
        return ("episode-plan.json",)
    if artifact_class == "access_status":
        if name in {
            "access_increment_final_materialization_count",
            "access_increment_forbidden_access_counter",
        }:
            return ("phase-access-audits.json",)
        if name == "access_add_final_observation_artifact":
            return ("final/frame-metadata.jsonl",)
        if name in {
            "access_add_final_score_vector_record",
            "access_add_final_reachability_trace",
        }:
            return ("final/provider-evidence.jsonl",)
        if name == "access_record_calibration_execution":
            return ("selected-calibration.json",)
        if name == "access_record_architecture_selection_execution":
            return ("selected-architecture.json",)
        return ("reference-closure-report.json",)
    return ()


def _flatten_payload(value: Any, prefix: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        rows: dict[str, Any] = {}
        for key in sorted(value):
            child = f"{prefix}.{key}" if prefix else str(key)
            rows.update(_flatten_payload(value[key], child))
        return rows
    if isinstance(value, list):
        rows = {}
        for index, item in enumerate(value):
            rows.update(_flatten_payload(item, f"{prefix}[{index}]"))
        if not value:
            rows[prefix] = []
        return rows
    return {prefix: value}


def _changed_snapshot_files(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    return [
        file_name
        for file_name in sorted(set(before) | set(after))
        if before.get(file_name, {}).get("digest")
        != after.get(file_name, {}).get("digest")
    ]


def _changed_fields(before: Mapping[str, Any], after: Mapping[str, Any]) -> list[str]:
    changed: set[str] = set()
    for file_name in sorted(set(before) | set(after)):
        if file_name not in before or file_name not in after:
            changed.add(file_name)
            continue
        if before[file_name] == after[file_name]:
            continue
        before_flat = _flatten_payload(before[file_name], file_name)
        after_flat = _flatten_payload(after[file_name], file_name)
        for field in sorted(set(before_flat) | set(after_flat)):
            if before_flat.get(field) != after_flat.get(field):
                changed.add(field)
    return sorted(changed)


def _mutation_isolation_report(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    changed = _changed_fields(before, after)
    expected_files = tuple(
        str(item)
        for item in case.get("expected_changed_files", _mutation_expected_files(case))
    )
    unexpected = [
        field
        for field in changed
        if not any(
            field == expected
            or field.startswith(f"{expected}.")
            or field.startswith(f"{expected}[")
            for expected in expected_files
        )
    ]
    before_flat = _flatten_all(before)
    after_flat = _flatten_all(after)
    effect_payload = [
        {
            "field": field,
            "before": before.get(field, before_flat.get(field)),
            "after": after.get(field, after_flat.get(field)),
        }
        for field in changed
    ]
    return {
        "changed_fields": changed,
        "expected_changed_files": list(expected_files),
        "unexpected_changed_fields": unexpected,
        "changed_field_count": len(changed),
        "isolation_passed": bool(changed) and not unexpected,
        "mutation_effect_digest": canonical_sha256(
            {"changed_fields": changed, "effect_payload": effect_payload}
        ),
    }


def _flatten_all(payloads: Mapping[str, Any]) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for file_name, payload in payloads.items():
        flattened.update(_flatten_payload(payload, file_name))
    return flattened


def evaluate_mutation_case(
    *,
    case: Mapping[str, Any],
    report: Mapping[str, Any],
    isolation: Mapping[str, Any],
    application_error: str | None = None,
) -> dict[str, Any]:
    primary = report.get("primary_failure_code")
    primary_gate = report.get("primary_failure_gate")
    if application_error is not None:
        primary = "mutation_application_error"
        primary_gate = "mutation_application"
    expected = case.get("expected_primary_failure_code")
    expected_detected = case["expected_result_type"] == "detected"
    secondary_codes = [
        row
        for row in _report_failure_codes(report)
        if row["code"] != primary or row["gate"] != primary_gate
    ]
    return {
        "mutation": case["mutation_id"],
        "artifact_class": case["artifact_class"],
        "protected_scientific_property": case["protected_scientific_property"],
        "fixture_selector": case["fixture_selector"],
        "mutator_id": case["mutator_id"],
        "expected_result_type": case["expected_result_type"],
        "expected_primary_failure_code": expected,
        "actual_primary_failure_code": primary,
        "actual_primary_gate": primary_gate,
        "secondary_failure_codes": secondary_codes,
        "detected": primary is not None,
        "expected_detected": expected_detected,
        "invariant": case["expected_result_type"] == "semantic_invariant",
        "semantic_invariant_passed": (
            case["expected_result_type"] == "semantic_invariant" and primary is None
        ),
        "digest_laundering": bool(case["validation_metadata"]["digest_laundering"]),
        "immediate_digest_recomputed": bool(case["immediate_digest_recomputed"]),
        "parent_digest_recomputed": bool(case["parent_digest_recomputed"]),
        "mutation_isolation": dict(isolation),
        "property_changed": isolation["changed_field_count"] > 0,
        "expected_code_matched": (
            primary == expected if expected_detected else primary is None
        ),
        "application_error": application_error,
    }


def build_mutation_audit_payload(
    *,
    matrix_version: str,
    catalogue: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
    catalogue_findings: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    base_verified: bool = True,
    base_primary_failure_code: str | None = None,
) -> dict[str, Any]:
    if not base_verified or catalogue_findings:
        return _unavailable_audit_payload(
            matrix_version=matrix_version,
            catalogue=catalogue,
            selected_cases=selected_cases,
            catalogue_findings=catalogue_findings,
            base_primary_failure_code=base_primary_failure_code,
        )
    normalized = [dict(row) for row in results]
    expected = [row for row in normalized if row["expected_detected"]]
    detected = [row for row in expected if row["detected"]]
    missed = [row for row in expected if not row["detected"]]
    unexpected = [row for row in normalized if not row["expected_code_matched"]]
    invariants = [
        row for row in normalized if row["expected_result_type"] == "semantic_invariant"
    ]
    invariant_passes = [row for row in invariants if row["semantic_invariant_passed"]]
    isolation_failures = [
        row for row in normalized if not row["mutation_isolation"]["isolation_passed"]
    ]
    property_failures = [row for row in expected if not row["property_changed"]]
    duplicate_findings = _duplicate_effect_findings(normalized)
    laundering = _laundering_closure(normalized)
    payload = {
        "version": MUTATION_AUDIT_VERSION,
        "matrix_version": matrix_version,
        "base_verified": True,
        "catalogue_findings": [dict(row) for row in catalogue_findings],
        "mutations": normalized,
        "declared_mutation_count": len(catalogue),
        "executable_mutation_count": len(selected_cases),
        "expected_detection_count": len(expected),
        "expected_mutation_count": len(expected),
        "detected_mutation_count": len(detected),
        "missed_mutation_count": len(missed),
        "undetected_mutation_count": len(missed),
        "unexpected_failure_code_count": len(unexpected),
        "invariant_count": len(invariants),
        "invariant_pass_count": len(invariant_passes),
        "invariant_failure_count": len(invariants) - len(invariant_passes),
        "digest_laundering_tests": [
            row["mutation"] for row in normalized if row["digest_laundering"]
        ],
        "digest_laundering_class_closure": laundering,
        "mutation_isolation_passed": not isolation_failures,
        "mutation_isolation_failure_count": len(isolation_failures),
        "property_change_failure_count": len(property_failures),
        "duplicate_effect_findings": duplicate_findings,
        "repeated_run_determinism": "not_measured_in_single_run",
        "status": (
            "passed"
            if not missed
            and not unexpected
            and len(invariant_passes) == len(invariants)
            and not isolation_failures
            and not property_failures
            and not duplicate_findings
            else "failed"
        ),
    }
    payload["mutation_audit_digest"] = canonical_sha256(payload)
    return payload


def _unavailable_audit_payload(
    *,
    matrix_version: str,
    catalogue: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
    catalogue_findings: Sequence[Mapping[str, Any]],
    base_primary_failure_code: str | None,
) -> dict[str, Any]:
    expected = [
        case for case in selected_cases if case["expected_result_type"] == "detected"
    ]
    invariants = [
        case
        for case in selected_cases
        if case["expected_result_type"] == "semantic_invariant"
    ]
    payload = {
        "version": MUTATION_AUDIT_VERSION,
        "matrix_version": matrix_version,
        "base_verified": False,
        "base_primary_failure_code": base_primary_failure_code,
        "catalogue_findings": [dict(row) for row in catalogue_findings],
        "mutations": [],
        "declared_mutation_count": len(catalogue),
        "executable_mutation_count": len(selected_cases),
        "expected_detection_count": len(expected),
        "expected_mutation_count": len(expected),
        "detected_mutation_count": 0,
        "missed_mutation_count": len(expected),
        "undetected_mutation_count": len(expected),
        "unexpected_failure_code_count": len(selected_cases),
        "invariant_count": len(invariants),
        "invariant_pass_count": 0,
        "invariant_failure_count": len(invariants),
        "digest_laundering_tests": [],
        "digest_laundering_class_closure": {},
        "mutation_isolation_passed": False,
        "duplicate_effect_findings": [],
        "status": "unavailable",
    }
    payload["mutation_audit_digest"] = canonical_sha256(payload)
    return payload


def _duplicate_effect_findings(
    results: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    owners: dict[str, str] = {}
    findings = []
    for row in results:
        digest = str(row["mutation_isolation"]["mutation_effect_digest"])
        previous = owners.get(digest)
        if previous is not None:
            findings.append(
                _finding(
                    "duplicate_mutation_effect",
                    "two mutation ids produced the same structural effect",
                    first_mutation=previous,
                    second_mutation=row["mutation"],
                )
            )
        owners[digest] = str(row["mutation"])
    return findings


def _laundering_closure(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    closure: dict[str, dict[str, Any]] = {}
    for row in results:
        if not row["digest_laundering"]:
            continue
        item = closure.setdefault(
            str(row["artifact_class"]),
            {
                "declared": 0,
                "executed": 0,
                "detected": 0,
                "expected_code_matches": 0,
                "laundering_depth_reached": "none",
            },
        )
        item["declared"] += 1
        item["executed"] += 1
        item["detected"] += int(bool(row["detected"]))
        item["expected_code_matches"] += int(bool(row["expected_code_matched"]))
        if row["immediate_digest_recomputed"] and row["parent_digest_recomputed"]:
            item["laundering_depth_reached"] = "immediate_and_parent"
        elif row["immediate_digest_recomputed"]:
            item["laundering_depth_reached"] = "immediate"
    return closure


def build_repeated_mutation_audit_payload(
    *, matrix_version: str, first: Mapping[str, Any], second: Mapping[str, Any]
) -> dict[str, Any]:
    deterministic = first == second and first.get(
        "mutation_audit_digest"
    ) == second.get("mutation_audit_digest")
    payload = {
        "version": "zeromodel-video-action-set-reference-mutation-audit-repeat/v1",
        "matrix_version": matrix_version,
        "deterministic": deterministic,
        "first_audit_digest": first.get("mutation_audit_digest"),
        "second_audit_digest": second.get("mutation_audit_digest"),
        "audit": dict(first),
    }
    payload["repeat_digest"] = canonical_sha256(payload)
    return payload


__all__ = [
    "MUTATION_AUDIT_VERSION",
    "_MUTATION_CASES",
    "_MUTATION_GATE_SCOPE",
    "_changed_fields",
    "_changed_snapshot_files",
    "_flatten_payload",
    "_mutation_case_by_name",
    "_mutation_expected_files",
    "_mutation_isolation_report",
    "_mutation_property",
    "build_mutation_audit_payload",
    "build_repeated_mutation_audit_payload",
    "evaluate_mutation_case",
]
