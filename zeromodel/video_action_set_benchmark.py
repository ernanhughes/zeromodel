"""Backward-compatible facade for the video action-set instrument."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zeromodel.domains.video_action_set import (
    build_orchestration as _build_orchestration,
)
from zeromodel.arcade_policy import (
    ACTIONS as ACTIONS,
    ShooterConfig as ShooterConfig,
    compile_policy_artifact as compile_policy_artifact,
    next_rows as next_rows,
    parse_state_row_id as parse_state_row_id,
    render_state_frame as render_state_frame,
)
from zeromodel.domains.video_action_set.arcade_observation import (
    render_row_frame as _render_row_frame,
    renderer_identity as _renderer_identity,
    shooter_config_payload as _transition_config_payload,
)
from zeromodel.domains.video_action_set.artifact_io import (
    _json_bytes as _json_bytes,
    _json_ready as _json_ready,
    _read_json as _read_json,
    _read_jsonl as _read_jsonl,
    _sha256 as _sha256,
    _write_csv as _write_csv,
    _write_json as _write_json,
    _write_jsonl as _write_jsonl,
)
from zeromodel.domains.video_action_set.build_orchestration import (
    _build_durable_runtime as _build_durable_runtime,
    _load_reachability_tile as _load_reachability_tile,
    _materialize_records as _materialize_records,
    _measured_phase_access_counts as _measured_phase_access_counts,
    _write_observation_identity_manifest as _write_observation_identity_manifest,
    _write_split_overlap_audit as _write_split_overlap_audit,
    freeze_benchmark as freeze_benchmark,
    load_identity as load_identity,
    profile_runtime as profile_runtime,
)
from zeromodel.domains.video_action_set.contracts import (
    BENCHMARK_VERSION as BENCHMARK_VERSION,
    CANONICAL_OBSERVATION_UNIVERSE_VERSION as CANONICAL_OBSERVATION_UNIVERSE_VERSION,
    CONFLICTING_ACTION_SPLICE_VERSION as CONFLICTING_ACTION_SPLICE_VERSION,
    CRITICAL_EVIDENCE_CORRUPTION_VERSION as CRITICAL_EVIDENCE_CORRUPTION_VERSION,
    DECLARED_GAP_OR_UNKNOWN_VERSION as DECLARED_GAP_OR_UNKNOWN_VERSION,
    EPISODE_FAMILY_REGISTRY_VERSION as EPISODE_FAMILY_REGISTRY_VERSION,
    EPISODE_PLAN_VERSION as EPISODE_PLAN_VERSION,
    FAMILY_CLOSURE_VERSION as FAMILY_CLOSURE_VERSION,
    FAMILY_INTERVENTION_VERSION as FAMILY_INTERVENTION_VERSION,
    FRAME_INVALID_CLOSURE_VERSION as FRAME_INVALID_CLOSURE_VERSION,
    FRAME_SHAPE as FRAME_SHAPE,
    GENERATOR_VERSION as GENERATOR_VERSION,
    IMPOSSIBLE_TRANSITION_VERSION as IMPOSSIBLE_TRANSITION_VERSION,
    INFORMATION_CONTROL_VERSION as INFORMATION_CONTROL_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION as OBSERVATION_OPERATION_CHAIN_VERSION,
    PROVIDER_OBSERVATION_BOUNDARY_VERSION as PROVIDER_OBSERVATION_BOUNDARY_VERSION,
    REACHABILITY_TILE_DIGEST as REACHABILITY_TILE_DIGEST,
    REACHABILITY_TILE_VERSION as REACHABILITY_TILE_VERSION,
    REORDERED_FRAMES_VERSION as REORDERED_FRAMES_VERSION,
    SEED_DERIVATION_VERSION as SEED_DERIVATION_VERSION,
    SPLICE_MASK_VERSION as SPLICE_MASK_VERSION,
    STALE_REPEATED_FRAME_VERSION as STALE_REPEATED_FRAME_VERSION,
    TRANSFORMATION_FAMILY_VERSION as TRANSFORMATION_FAMILY_VERSION,
    VALID_OBSERVATION_UNIVERSE_VERSION as VALID_OBSERVATION_UNIVERSE_VERSION,
)
from zeromodel.domains.video_action_set.control_histories import (
    control_source_rows as _control_source_rows,
    grounded_control_histories_for_current_row as _grounded_control_histories_for_current_row,
    grounded_control_history as _grounded_control_history,
    normalized_control_causal_tuple as _normalized_control_causal_tuple,
    reconstructed_control_causal_tuple_digest as _reconstructed_control_causal_tuple_digest,
    select_grounded_control_histories as _select_grounded_control_histories,
    transition_identity as _transition_identity,
    transition_input_digest as _transition_input_digest,
    transition_result_digest as _transition_result_digest,
)
from zeromodel.domains.video_action_set.dto import (
    BenchmarkIdentityDTO as BenchmarkIdentity,
)
from zeromodel.domains.video_action_set.episode_families import (
    denominator_class as _denominator_class,
    episode_disposition as _episode_disposition,
    episode_family_registry as _episode_family_registry,
    expected_frame_disposition as expected_frame_disposition,
    family_contract as _family_contract,
    family_schedule as _family_schedule,
    frame_disposition_for_episode as _frame_disposition_for_episode,
)
from zeromodel.domains.video_action_set.episode_materialization import (
    control_episode as _control_episode,
    invalid_episode as _invalid_episode,
    materialize_plan as _materialize_plan,
    temporal_negative_episode as _temporal_negative_episode,
    valid_episode as _valid_episode,
)
from zeromodel.domains.video_action_set.episode_planning import (
    derived_seed as _derived_seed,
    episode_ids_by_family as _episode_ids_by_family,
    episode_plans_for_split as _episode_plans_for_split,
    final_observation_provenance as _final_observation_provenance,
    frame_count_for_plan as _frame_count_for_plan,
    frame_plans as _frame_plans,
    make_episode_plan as _make_episode_plan,
    validate_episode_plan as _validate_episode_plan,
    validate_episode_plan_collection as _validate_episode_plan_collection,
)
from zeromodel.domains.video_action_set.evidence_audit import (
    PHASE_ACCESS_VERSION as PHASE_ACCESS_VERSION,
)
from zeromodel.domains.video_action_set.family_intervention_planning import (
    conflicting_splice_source_rows as _conflicting_splice_source_rows,
    family_intervention_plan as _family_intervention_plan,
    impossible_destination_row as _impossible_destination_row,
    secondary_row_for_splice as _secondary_row_for_splice,
    seed_int_from_digest as _seed_int_from_digest,
    state_row_values as _state_row_values,
)
from zeromodel.domains.video_action_set.family_provenance import (
    conflicting_splice_operation_chain as _conflicting_splice_operation_chain,
    critical_corruption_operation_chain as _critical_corruption_operation_chain,
    impossible_transition_operation_chain as _impossible_transition_operation_chain,
    information_control_operation_chain as _information_control_operation_chain,
    stale_repeat_operation_chain as _stale_repeat_operation_chain,
)
from zeromodel.domains.video_action_set.family_validation import (
    frame_invalid_closure_summary as _frame_invalid_closure_summary,
    validate_materialized_family_record as validate_materialized_family_record,
)
from zeromodel.domains.video_action_set.frame_family_kernels import (
    apply_conflicting_splice as _apply_conflicting_splice,
    apply_critical_corruption as _apply_critical_corruption,
    critical_coordinate_manifest as _critical_coordinate_manifest,
    critical_coordinates as _critical_coordinates,
    detect_cooldown_state as _detect_cooldown_state,
    detect_tank_slot as _detect_tank_slot,
    detect_visible_target_slots as _detect_visible_target_slots,
    final_visible_target_action_evidence as _final_visible_target_action_evidence,
    splice_evidence_counts as _splice_evidence_counts,
    splice_mask_manifest as _splice_mask_manifest,
    splice_pair_has_final_visible_action_conflict as _splice_pair_has_final_visible_action_conflict,
    target_signal_coordinates as _target_signal_coordinates,
    target_signal_mask as _target_signal_mask,
    target_slot_signal_coordinates as _target_slot_signal_coordinates,
)
from zeromodel.domains.video_action_set.materialization_kernels import (
    apply_family as _apply_family,
    apply_frame_plan as _apply_frame_plan,
    frame_descriptor as _frame_descriptor,
)
from zeromodel.domains.video_action_set.materialization_reachability import (
    next_materialized_row as _next_row,
    reachability_tile_digest as _reachability_tile_digest,
    tile_edge as _tile_edge,
    validate_reachability_tile_identity as _validate_reachability_tile_identity,
)
from zeromodel.domains.video_action_set.materialization_validation import (
    family_closure_report as _family_closure_report,
    record_regeneration_view as _record_regeneration_view,
    validate_control_episode_records as validate_control_episode_records,
)
from zeromodel.domains.video_action_set.mutation_audit import (
    MUTATION_AUDIT_VERSION as MUTATION_AUDIT_VERSION,
    _MUTATION_CASES as _MUTATION_CASES,
    _changed_fields as _changed_fields,
    _mutation_isolation_report as _mutation_isolation_report,
)
from zeromodel.domains.video_action_set.mutation_filesystem import (
    _apply_reference_mutation as _apply_reference_mutation,
    _directory_snapshot as _directory_snapshot,
    _file_digest as _file_digest,
    _launder_split_manifest as _launder_split_manifest,
    _mutate_first_frame_row as _mutate_first_frame_row,
    _mutate_first_provider_row as _mutate_first_provider_row,
    _mutation_structural_snapshot as _mutation_structural_snapshot,
    _rewrite_jsonl as _rewrite_jsonl,
    _structural_payload as _structural_payload,
)
from zeromodel.domains.video_action_set.mutation_matrix import (
    MUTATION_MATRIX_VERSION as MUTATION_MATRIX_VERSION,
    mutation_catalogue as mutation_catalogue,
    validate_mutation_catalogue as validate_mutation_catalogue,
)
from zeromodel.domains.video_action_set.mutation_orchestration import (
    _run_adversarial_mutation_checks as _run_adversarial_mutation_checks,
    build_reference_closure_report as build_reference_closure_report,
    run_reference_mutation_audit as run_reference_mutation_audit,
    run_repeated_reference_mutation_audit as run_repeated_reference_mutation_audit,
    verify_instrument as verify_instrument,
)
from zeromodel.domains.video_action_set.observation_legacy_adapters import (
    operation_record as _operation_record,
)
from zeromodel.domains.video_action_set.observation_provenance import (
    gap_event_operation_chain as _gap_event_operation_chain,
    valid_frame_operation_chain as _valid_frame_operation_chain,
)
from zeromodel.domains.video_action_set.observation_universe import (
    _canonical_collision_rows as _canonical_collision_rows,
    _canonical_observation_digest_index as _canonical_observation_digest_index,
    _valid_transformation_parameter_key as _valid_transformation_parameter_key,
    _valid_transformation_parameter_universe as _valid_transformation_parameter_universe,
    _valid_transformed_observation_digest_index as _valid_transformed_observation_digest_index,
    canonical_observation_universe as canonical_observation_universe,
    canonical_prototypes as canonical_prototypes,
    valid_observation_universe as valid_observation_universe,
)
from zeromodel.domains.video_action_set.pixel_digest import (
    array_digest as _array_digest,
    pixel_digest as _pixel_digest,
)
from zeromodel.domains.video_action_set.provider_measurement import (
    SOURCE_SCOPE as SOURCE_SCOPE,
    measure_record_collection as measure_record_collection,
    provider_version as _provider_version,
    score_record as _score_record,
    score_vector_to_payload as _score_vector_to_payload,
)
from zeromodel.domains.video_action_set.provider_observation_boundary import (
    control_provider_source_id as _control_provider_source_id,
    provider_observation_descriptor_for_record as provider_observation_descriptor_for_record,
    provider_observation_digest as _provider_observation_digest,
    provider_observation_for_record as provider_observation_for_record,
    refresh_provider_observation_metadata as _refresh_provider_observation_metadata,
)
from zeromodel.domains.video_action_set.reachability_composition import (
    REACHABILITY_COMPOSITION_VERSION as REACHABILITY_COMPOSITION_VERSION,
    REACHABILITY_TRACE_VERSION as REACHABILITY_TRACE_VERSION,
    compose_reachability_trace as compose_reachability_trace,
    gap_reachability_state as _gap_reachability_state,
    state_from_trace as _state_from_trace,
    top_row_action_map as _top_row_action_map,
    trace_digest as _trace_digest,
    validate_reachability_trace as validate_reachability_trace,
)
from zeromodel.domains.video_action_set.reference_verification import (
    REFERENCE_VERIFICATION_VERSION as REFERENCE_VERIFICATION_VERSION,
    _REQUIRED_VERIFICATION_GATES as _REQUIRED_VERIFICATION_GATES,
    _expected_semantic_for_row as _expected_semantic_for_row,
    _finding as _finding,
    _gate as _gate,
    _primary_failure as _primary_failure,
    _stored_quantized_evidence as _stored_quantized_evidence,
)
from zeromodel.domains.video_action_set.runtime_profiling import (
    profile_provider as _profile_provider,
)
from zeromodel.domains.video_action_set.transformations import (
    _apply_transformation as _apply_transformation,
    _transformation_contract as _transformation_contract,
    _transformation_parameters as _transformation_parameters,
    _translation_values_for_seed as _translation_values_for_seed,
)
from zeromodel.domains.video_action_set.verification import (
    CLOSURE_REPORT_VERSION as CLOSURE_REPORT_VERSION,
)
from zeromodel.domains.video_action_set.verification_gates import (
    _access_prohibition_gate as _access_prohibition_gate,
    _cached_materialized_metadata as _cached_materialized_metadata,
    _completeness_orphan_gate as _completeness_orphan_gate,
    _episode_regeneration_gate as _episode_regeneration_gate,
    _expected_digest_owners as _expected_digest_owners,
    _expected_split_counts as _expected_split_counts,
    _family_contract_gate as _family_contract_gate,
    _reachability_gate as _reachability_gate,
    _seed_and_plan_gate as _seed_and_plan_gate,
    _semantic_outcome_gate as _semantic_outcome_gate,
    replay_observation_operation_chain as replay_observation_operation_chain,
    validate_observation_operation_chain as validate_observation_operation_chain,
)
from zeromodel.domains.video_action_set.verification_orchestration import (
    _reference_context as _reference_context,
    _structural_identity_gate as _structural_identity_gate,
    audit_canonical_providers as audit_canonical_providers,
    audit_evidence_completeness as audit_evidence_completeness,
    verify_provider_runtime_equivalence as verify_provider_runtime_equivalence,
    verify_reference_instrument as verify_reference_instrument,
    verify_reference_read_only as verify_reference_read_only,
)
from zeromodel.policy_lookup import (
    VPMPolicyLookup as VPMPolicyLookup,
)
from zeromodel.video_prospective_providers import (
    PROSPECTIVE_P1_VERSION as PROSPECTIVE_P1_VERSION,
    score_b3_joint_fit as score_b3_joint_fit,
    score_normalized_pixel as score_normalized_pixel,
    score_registered_local_correlation as score_registered_local_correlation,
)

from zeromodel.video_action_set_cli import main as main


EPISODE_SCHEMA_VERSION = "zeromodel-video-policy-episode/v1"


def _profiling_records(repo_root: Path, frame_count: int) -> list[dict[str, Any]]:
    original = _build_orchestration._materialize_records
    _build_orchestration._materialize_records = _materialize_records
    try:
        return _build_orchestration._profiling_records(repo_root, frame_count)
    finally:
        _build_orchestration._materialize_records = original


def build_split(split: str, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    patch_names = (
        "_materialize_records",
        "canonical_prototypes",
        "measure_record_collection",
    )
    originals = {name: getattr(_build_orchestration, name) for name in patch_names}
    try:
        for name in patch_names:
            setattr(_build_orchestration, name, globals()[name])
        return _build_orchestration.build_split(split, output_dir, repo_root)
    finally:
        for name, value in originals.items():
            setattr(_build_orchestration, name, value)


# fmt: off
_STAGE7C_COMPATIBILITY_ALIASES = (ACTIONS, BENCHMARK_VERSION, BenchmarkIdentity, CANONICAL_OBSERVATION_UNIVERSE_VERSION,
    CLOSURE_REPORT_VERSION, CONFLICTING_ACTION_SPLICE_VERSION, CRITICAL_EVIDENCE_CORRUPTION_VERSION, DECLARED_GAP_OR_UNKNOWN_VERSION,
    EPISODE_FAMILY_REGISTRY_VERSION, EPISODE_PLAN_VERSION, FAMILY_CLOSURE_VERSION, FAMILY_INTERVENTION_VERSION,
    FRAME_INVALID_CLOSURE_VERSION, FRAME_SHAPE, GENERATOR_VERSION, IMPOSSIBLE_TRANSITION_VERSION, INFORMATION_CONTROL_VERSION,
    MUTATION_AUDIT_VERSION, MUTATION_MATRIX_VERSION, OBSERVATION_OPERATION_CHAIN_VERSION, PHASE_ACCESS_VERSION, PROSPECTIVE_P1_VERSION,
    PROVIDER_OBSERVATION_BOUNDARY_VERSION, REACHABILITY_COMPOSITION_VERSION, REACHABILITY_TILE_DIGEST, REACHABILITY_TILE_VERSION,
    REACHABILITY_TRACE_VERSION, REFERENCE_VERIFICATION_VERSION, REORDERED_FRAMES_VERSION, SEED_DERIVATION_VERSION, SOURCE_SCOPE,
    SPLICE_MASK_VERSION, STALE_REPEATED_FRAME_VERSION, ShooterConfig, TRANSFORMATION_FAMILY_VERSION, VALID_OBSERVATION_UNIVERSE_VERSION,
    VPMPolicyLookup, _MUTATION_CASES, _REQUIRED_VERIFICATION_GATES, _access_prohibition_gate, _apply_conflicting_splice,
    _apply_critical_corruption, _apply_family, _apply_frame_plan, _apply_reference_mutation, _apply_transformation, _array_digest,
    _build_durable_runtime, _cached_materialized_metadata, _canonical_collision_rows, _canonical_observation_digest_index, _changed_fields,
    _completeness_orphan_gate, _conflicting_splice_operation_chain, _conflicting_splice_source_rows, _control_episode,
    _control_provider_source_id, _control_source_rows, _critical_coordinate_manifest, _critical_coordinates,
    _critical_corruption_operation_chain, _denominator_class, _derived_seed, _detect_cooldown_state, _detect_tank_slot,
    _detect_visible_target_slots, _directory_snapshot, _episode_disposition, _episode_family_registry, _episode_ids_by_family,
    _episode_plans_for_split, _episode_regeneration_gate, _expected_digest_owners, _expected_semantic_for_row, _expected_split_counts,
    _family_closure_report, _family_contract, _family_contract_gate, _family_intervention_plan, _family_schedule, _file_digest,
    _final_observation_provenance, _final_visible_target_action_evidence, _finding, _frame_count_for_plan, _frame_descriptor,
    _frame_disposition_for_episode, _frame_invalid_closure_summary, _frame_plans, _gap_event_operation_chain, _gap_reachability_state,
    _gate, _grounded_control_histories_for_current_row, _grounded_control_history, _impossible_destination_row,
    _impossible_transition_operation_chain, _information_control_operation_chain, _invalid_episode, _json_bytes, _json_ready,
    _launder_split_manifest, _load_reachability_tile, _make_episode_plan, _materialize_plan, _materialize_records,
    _measured_phase_access_counts, _mutate_first_frame_row, _mutate_first_provider_row, _mutation_isolation_report,
    _mutation_structural_snapshot, _next_row, _normalized_control_causal_tuple, _operation_record, _pixel_digest, _primary_failure,
    _profile_provider, _profiling_records, _provider_observation_digest, _provider_version, _reachability_gate, _reachability_tile_digest,
    _read_json, _read_jsonl, _reconstructed_control_causal_tuple_digest, _record_regeneration_view, _reference_context,
    _refresh_provider_observation_metadata, _render_row_frame, _renderer_identity, _rewrite_jsonl, _run_adversarial_mutation_checks,
    _score_record, _score_vector_to_payload, _secondary_row_for_splice, _seed_and_plan_gate, _seed_int_from_digest,
    _select_grounded_control_histories, _semantic_outcome_gate, _sha256, _splice_evidence_counts, _splice_mask_manifest,
    _splice_pair_has_final_visible_action_conflict, _stale_repeat_operation_chain, _state_from_trace, _state_row_values,
    _stored_quantized_evidence, _structural_identity_gate, _structural_payload, _target_signal_coordinates, _target_signal_mask,
    _target_slot_signal_coordinates, _temporal_negative_episode, _tile_edge, _top_row_action_map, _trace_digest, _transformation_contract,
    _transformation_parameters, _transition_config_payload, _transition_identity, _transition_input_digest, _transition_result_digest,
    _translation_values_for_seed, _valid_episode, _valid_frame_operation_chain, _valid_transformation_parameter_key,
    _valid_transformation_parameter_universe, _valid_transformed_observation_digest_index, _validate_episode_plan,
    _validate_episode_plan_collection, _validate_reachability_tile_identity, _write_csv, _write_json, _write_jsonl,
    _write_observation_identity_manifest, _write_split_overlap_audit, audit_canonical_providers, audit_evidence_completeness,
    build_reference_closure_report, build_split, canonical_observation_universe, canonical_prototypes, compile_policy_artifact,
    compose_reachability_trace, expected_frame_disposition, freeze_benchmark, load_identity, main, mutation_catalogue, next_rows,
    parse_state_row_id, profile_runtime, provider_observation_descriptor_for_record, provider_observation_for_record, render_state_frame,
    score_b3_joint_fit, score_normalized_pixel, score_registered_local_correlation,
    replay_observation_operation_chain, run_reference_mutation_audit, run_repeated_reference_mutation_audit, valid_observation_universe,
    validate_control_episode_records, validate_materialized_family_record, validate_mutation_catalogue,
    validate_observation_operation_chain, validate_reachability_trace, verify_instrument, verify_provider_runtime_equivalence,
    verify_reference_instrument, verify_reference_read_only, )
# fmt: on


__all__ = [
    "BenchmarkIdentity",
    "BENCHMARK_VERSION",
    "EPISODE_PLAN_VERSION",
    "GENERATOR_VERSION",
    "REACHABILITY_TILE_DIGEST",
    "REACHABILITY_TILE_VERSION",
    "SEED_DERIVATION_VERSION",
    "REFERENCE_VERIFICATION_VERSION",
    "MUTATION_AUDIT_VERSION",
    "CLOSURE_REPORT_VERSION",
    "MUTATION_MATRIX_VERSION",
    "PHASE_ACCESS_VERSION",
    "SOURCE_SCOPE",
    "audit_canonical_providers",
    "audit_evidence_completeness",
    "build_reference_closure_report",
    "build_split",
    "canonical_prototypes",
    "freeze_benchmark",
    "load_identity",
    "mutation_catalogue",
    "run_reference_mutation_audit",
    "run_repeated_reference_mutation_audit",
    "validate_mutation_catalogue",
    "verify_reference_instrument",
    "verify_reference_read_only",
    "verify_instrument",
]
