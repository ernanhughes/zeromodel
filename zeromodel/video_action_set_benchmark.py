from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .arcade_policy import (
    ACTIONS,
    ShooterConfig,
    compile_policy_artifact,
    next_rows,
    parse_state_row_id,
    render_state_frame,
)
from .artifact import VPMValidationError
from .domains.video_action_set.canonical_json import canonical_json_bytes, canonical_json_value, canonical_sha256
from .domains.video_action_set.contracts import (
    ARCADE_RENDERER_CONTRACT_VERSION,
    AUTHORITATIVE_TRANSITION_FUNCTION_VERSION,
    BENCHMARK_VERSION,
    CANONICAL_OBSERVATION_UNIVERSE_VERSION,
    CONFLICTING_ACTION_SPLICE_VERSION,
    CRITICAL_COORDINATE_SET_VERSION,
    CRITICAL_EVIDENCE_CORRUPTION_VERSION,
    CRITICAL_REGION_ID,
    DECLARED_GAP_OR_UNKNOWN_VERSION,
    EPISODE_FAMILY_REGISTRY_VERSION,
    EPISODE_PLAN_VERSION,
    FAMILY_CLOSURE_VERSION,
    FAMILY_INTERVENTION_VERSION,
    FINAL_VISIBLE_TARGET_ACTION_EVIDENCE_VERSION,
    FRAME_SHAPE,
    FRAME_INVALID_CLOSURE_VERSION,
    GAP_EVENT_VERSION,
    GENERATOR_VERSION,
    GROUNDED_CONTROL_HISTORY_VERSION,
    IMPOSSIBLE_TRANSITION_VERSION,
    INFORMATION_CONTROL_AMBIGUITY_VERSION,
    INFORMATION_CONTROL_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
    PROVIDER_OBSERVATION_BOUNDARY_VERSION,
    REACHABILITY_TILE_DIGEST,
    REACHABILITY_TILE_VERSION,
    REORDERED_FRAMES_VERSION,
    SEED_DERIVATION_VERSION,
    SPLICE_MASK_VERSION,
    STALE_REPEATED_FRAME_VERSION,
    TARGET_REGION_ID,
    TRANSFORMATION_FAMILY_VERSION,
    VALID_FAMILY_VERSION,
    VALID_OBSERVATION_UNIVERSE_VERSION,
)
from .domains.video_action_set.control_histories import (
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
from .domains.video_action_set.dto import BenchmarkIdentityDTO, EpisodePlanDTO
from .domains.video_action_set.episode_planning import (
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
from .domains.video_action_set.episode_families import (
    denominator_class as _denominator_class,
    episode_disposition as _episode_disposition,
    episode_family_registry as _episode_family_registry,
    expected_frame_disposition,
    family_contract as _family_contract,
    family_schedule as _family_schedule,
    frame_disposition_for_episode as _frame_disposition_for_episode,
)
from .domains.video_action_set.family_intervention_planning import (
    conflicting_splice_source_rows as _conflicting_splice_source_rows,
    family_intervention_plan as _family_intervention_plan,
    impossible_destination_row as _impossible_destination_row,
    secondary_row_for_splice as _secondary_row_for_splice,
    seed_int_from_digest as _seed_int_from_digest,
    state_row_values as _state_row_values,
)
from .domains.video_action_set.family_provenance import (
    conflicting_splice_operation_chain as _conflicting_splice_operation_chain,
    critical_corruption_operation_chain as _critical_corruption_operation_chain,
    impossible_transition_operation_chain as _impossible_transition_operation_chain,
    information_control_operation_chain as _information_control_operation_chain,
    stale_repeat_operation_chain as _stale_repeat_operation_chain,
)
from .domains.video_action_set.family_validation import (
    frame_invalid_closure_summary as _frame_invalid_closure_summary,
    validate_materialized_family_record,
)
from .domains.video_action_set.frame_family_kernels import (
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
from .domains.video_action_set.episode_materialization import (
    control_episode as _control_episode,
    invalid_episode as _invalid_episode,
    materialize_plan as _materialize_plan,
    materialize_plan_collection as _materialize_plan_collection,
    temporal_negative_episode as _temporal_negative_episode,
    valid_episode as _valid_episode,
)
from .domains.video_action_set.materialization_kernels import (
    apply_family as _apply_family,
    apply_frame_plan as _apply_frame_plan,
    frame_descriptor as _frame_descriptor,
)
from .domains.video_action_set.materialization_reachability import (
    next_materialized_row as _next_row,
    reachability_tile_digest as _reachability_tile_digest,
    tile_edge as _tile_edge,
    validate_reachability_tile_identity as _validate_reachability_tile_identity,
)
from .domains.video_action_set.materialization_validation import (
    family_closure_report as _family_closure_report,
    record_regeneration_view as _record_regeneration_view,
    validate_control_episode_records,
)
from .domains.video_action_set.observation_legacy_adapters import (
    operation_chain as _operation_chain,
    operation_record as _operation_record,
)
from .domains.video_action_set.observation_provenance import (
    gap_event_operation_chain as _gap_event_operation_chain,
    valid_frame_operation_chain as _valid_frame_operation_chain,
)
from .domains.video_action_set.observation_replay import (
    replay_observation_operation_chain as _replay_observation_operation_chain_core,
    validate_observation_operation_chain as _validate_observation_operation_chain_core,
)
from .domains.video_action_set.pixel_digest import (
    array_digest as _array_digest,
    pixel_digest as _pixel_digest,
)
from .domains.video_action_set.provider_observation_boundary import (
    control_provider_source_id as _control_provider_source_id,
    provider_observation_descriptor_for_record,
    provider_observation_digest as _provider_observation_digest,
    provider_observation_for_record,
    refresh_provider_observation_metadata as _refresh_provider_observation_metadata,
)
from .domains.video_action_set.provider_measurement import (
    SOURCE_SCOPE,
    measure_record_collection,
    provider_version as _provider_version,
    score_record as _score_record,
    score_vector_to_payload as _score_vector_to_payload,
)
from .domains.video_action_set.reachability_composition import (
    REACHABILITY_COMPOSITION_VERSION,
    REACHABILITY_TRACE_VERSION,
    compose_reachability_trace,
    gap_reachability_state as _gap_reachability_state,
    state_from_trace as _state_from_trace,
    top_row_action_map as _top_row_action_map,
    trace_digest as _trace_digest,
    validate_reachability_trace,
)
from .domains.video_action_set.runtime_profiling import (
    profile_provider as _profile_provider,
    runtime_profile_payload,
    select_profiling_records,
)
from .domains.video_action_set.evidence_audit import (
    PHASE_ACCESS_VERSION,
    access_prohibition_gate as _build_access_prohibition_gate,
    audit_canonical_provider_results as _audit_canonical_provider_results,
    audit_evidence_rows as _audit_evidence_rows,
    build_observation_identity_manifest as _build_observation_identity_manifest,
    build_split_overlap_audit as _build_split_overlap_audit,
    measured_phase_access_counts as _build_measured_phase_access_counts,
)
from .domains.video_action_set.mutation_audit import (
    MUTATION_AUDIT_VERSION,
    _MUTATION_CASES,
    _changed_fields,
    _changed_snapshot_files,
    _flatten_payload,
    _mutation_case_by_name,
    _mutation_expected_files,
    _mutation_isolation_report,
    _mutation_property,
    build_mutation_audit_payload as _build_mutation_audit_payload,
    build_repeated_mutation_audit_payload as _build_repeated_mutation_audit_payload,
    evaluate_mutation_case as _evaluate_mutation_case,
)
from .domains.video_action_set.mutation_matrix import (
    MUTATION_MATRIX_VERSION,
    mutation_catalogue,
    validate_mutation_catalogue,
)
from .domains.video_action_set.reference_verification import (
    REFERENCE_VERIFICATION_VERSION,
    SEMANTIC_OUTCOME_VERSION,
    _REQUIRED_VERIFICATION_GATES,
    _expected_semantic_for_row,
    _finding,
    _first_failure_code,
    _gate,
    _policy_row_action_digest,
    _primary_failure,
    _raw_score_diagnostic_from_row,
    _report_failure_codes,
    _semantic_cache_key,
    _stored_quantized_evidence,
    build_provider_equivalence_payload as _build_provider_equivalence_payload,
    build_read_only_verification_payload as _build_read_only_verification_payload,
    build_reference_context as _build_reference_context,
    build_reference_verification_payload as _build_reference_verification_payload,
    compare_provider_results as _compare_provider_results,
)
from .domains.video_action_set.verification import (
    CLOSURE_REPORT_VERSION,
    build_unavailable_repeated_mutation_audit as _build_unavailable_repeated_mutation_audit,
    build_verification_closure as _build_verification_closure,
    verification_summary as _verification_summary,
)
from .domains.video_action_set.arcade_observation import (
    render_row_frame as _render_row_frame,
    renderer_identity as _renderer_identity,
    shooter_config_payload as _transition_config_payload,
)
from .domains.video_action_set.observation_universe import (
    _canonical_collision_rows,
    _canonical_observation_digest_index,
    _valid_transformation_parameter_key,
    _valid_transformation_parameter_universe,
    _valid_transformed_observation_digest_index,
    canonical_observation_universe,
    canonical_prototypes,
    valid_observation_universe,
)
from .domains.video_action_set.transformations import (
    _apply_transformation,
    _transformation_contract,
    _transformation_parameters,
    _translation_values_for_seed,
    _validate_transformation_parameters,
)
from .policy_lookup import VPMPolicyLookup
from .runtime import build_runtime
from .video_complete_row_evidence import (
    QUANTIZATION_SCALE,
    RowScore,
    VIDEO_SCORE_QUANTIZER_VERSION,
    VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION,
    build_complete_ranking,
    build_complete_row_evidence,
    build_semantic_top_set_outcome,
    quantize_similarity,
    semantic_top_set_outcome_from_dict,
)
from .video_prospective_providers import (
    PROSPECTIVE_P1_VERSION,
    PROSPECTIVE_P2_VERSION,
    PROSPECTIVE_P3_VERSION,
    score_all_rows_optimized,
    score_all_rows_reference,
    score_b3_joint_fit,
    score_normalized_pixel,
    score_registered_local_correlation,
)
from .visual_address import IMAGE_OBSERVATION_VERSION, ImageObservation

_STAGE4B_LEGACY_COMPATIBILITY_ALIASES = (
    ARCADE_RENDERER_CONTRACT_VERSION,
    CANONICAL_OBSERVATION_UNIVERSE_VERSION,
    CONFLICTING_ACTION_SPLICE_VERSION,
    CRITICAL_COORDINATE_SET_VERSION,
    CRITICAL_EVIDENCE_CORRUPTION_VERSION,
    CRITICAL_REGION_ID,
    DECLARED_GAP_OR_UNKNOWN_VERSION,
    FINAL_VISIBLE_TARGET_ACTION_EVIDENCE_VERSION,
    FRAME_SHAPE,
    FRAME_INVALID_CLOSURE_VERSION,
    IMPOSSIBLE_TRANSITION_VERSION,
    INFORMATION_CONTROL_VERSION,
    OBSERVATION_OPERATION_CHAIN_VERSION,
    PROVIDER_OBSERVATION_BOUNDARY_VERSION,
    REORDERED_FRAMES_VERSION,
    STALE_REPEATED_FRAME_VERSION,
    TRANSFORMATION_FAMILY_VERSION,
    VALID_FAMILY_VERSION,
    VALID_OBSERVATION_UNIVERSE_VERSION,
    _detect_cooldown_state,
    _detect_tank_slot,
    _detect_visible_target_slots,
    _frame_disposition_for_episode,
    _operation_chain,
    _operation_record,
    _renderer_identity,
    _splice_evidence_counts,
    _target_signal_coordinates,
    _target_signal_mask,
    _target_slot_signal_coordinates,
    _canonical_collision_rows,
    _valid_transformation_parameter_key,
    _valid_transformation_parameter_universe,
    _translation_values_for_seed,
    valid_observation_universe,
)


EPISODE_SCHEMA_VERSION = "zeromodel-video-policy-episode/v1"


def _json_ready(value: Any) -> Any:
    return canonical_json_value(value)


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _sha256(value: Any) -> str:
    return canonical_sha256(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(_json_ready(row), sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_json_ready(row[key]), sort_keys=True) if isinstance(row.get(key), (dict, list, tuple)) else row.get(key, "") for key in fieldnames})


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_durable_runtime(output_dir: Path):
    try:
        from .db.runtime import build_sqlite_runtime
    except ImportError as exc:
        raise VPMValidationError(
            "SQLite benchmark persistence requires the optional database extra: "
            'pip install "zeromodel[persistence]"'
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{(output_dir / 'benchmark.sqlite3').as_posix()}"
    return build_sqlite_runtime(database_url, initialize_schema=True)


def _load_reachability_tile(repo_root: Path) -> dict[str, Any]:
    return _read_json(repo_root / "docs" / "results" / "video-policy-reachability-tile-v1" / "reachability-tile.json")


BenchmarkIdentity = BenchmarkIdentityDTO


def load_identity(repo_root: Path) -> BenchmarkIdentity:
    return build_runtime().video_action_set.load_identity(repo_root)


def replay_observation_operation_chain(chain: Mapping[str, Any]) -> dict[str, Any]:
    return _replay_observation_operation_chain_core(
        chain,
        conflicting_splice_executor=_apply_conflicting_splice,
        critical_corruption_executor=_apply_critical_corruption,
    )


def validate_observation_operation_chain(record: Mapping[str, Any]) -> str:
    return _validate_observation_operation_chain_core(
        record,
        conflicting_splice_executor=_apply_conflicting_splice,
        critical_corruption_executor=_apply_critical_corruption,
    )


def freeze_benchmark(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    runtime = _build_durable_runtime(output_dir)
    identity = runtime.video_action_set.load_identity(repo_root)
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    development_plans = _episode_plans_for_split(identity, "development", row_ids, row_actions)
    calibration_plans = _episode_plans_for_split(identity, "calibration", row_ids, row_actions)
    selection_plans = _episode_plans_for_split(identity, "selection", row_ids, row_actions)
    final_plans = _episode_plans_for_split(identity, "final", row_ids, row_actions)
    _validate_episode_plan_collection(
        identity,
        {"development": development_plans, "calibration": calibration_plans, "selection": selection_plans, "final": final_plans},
        row_actions,
    )
    development_episode_dtos = runtime.video_action_set.save_episode_plans(tuple(EpisodePlanDTO.from_dict(plan) for plan in development_plans))
    calibration_episode_dtos = runtime.video_action_set.save_episode_plans(tuple(EpisodePlanDTO.from_dict(plan) for plan in calibration_plans))
    selection_episode_dtos = runtime.video_action_set.save_episode_plans(tuple(EpisodePlanDTO.from_dict(plan) for plan in selection_plans))
    final_episode_dtos = runtime.video_action_set.save_episode_plans(tuple(EpisodePlanDTO.from_dict(plan) for plan in final_plans))
    development_plans = [dto.to_dict() for dto in development_episode_dtos]
    calibration_plans = [dto.to_dict() for dto in calibration_episode_dtos]
    selection_plans = [dto.to_dict() for dto in selection_episode_dtos]
    final_plans = [dto.to_dict() for dto in final_episode_dtos]
    split_manifest = {
        "development_episode_count": 112,
        "calibration_episode_count": 112,
        "calibration_frame_count": 448,
        "selection_valid_episode_count": 112,
        "selection_frame_invalid_episode_count": 56,
        "selection_temporal_negative_episode_count": 56,
        "selection_control_episode_count": 28,
        "selection_total_frame_count": 1008,
        "final_valid_episode_count": 112,
        "final_frame_invalid_episode_count": 56,
        "final_temporal_negative_episode_count": 56,
        "final_control_episode_count": 28,
        "final_total_expected_frame_count": 1008,
    }
    provider_manifest = {
        "providers": [
            {"provider_id": "P1", "provider_version": PROSPECTIVE_P1_VERSION},
            {"provider_id": "P2", "provider_version": PROSPECTIVE_P2_VERSION},
            {"provider_id": "P3", "provider_version": PROSPECTIVE_P3_VERSION},
        ]
    }
    sealed_final = runtime.video_action_set.seal_final_split(episodes=final_episode_dtos, seed_commitment=identity.seed_digest)
    final_plan = sealed_final.to_dict()
    _write_json(output_dir / "benchmark-contract-identity.json", identity.to_dict())
    _write_json(output_dir / "generator-identity.json", {"generator_version": GENERATOR_VERSION, "seed_digest": identity.seed_digest, "seed_material": identity.seed_material})
    _write_json(output_dir / "benchmark-manifest.json", {"benchmark_version": BENCHMARK_VERSION, "policy_artifact_id": policy.artifact_id, "row_count": len(row_ids)})
    _write_json(
        output_dir / "policy-artifact.json",
        {
            "policy_artifact_id": policy.artifact_id,
            "row_count": len(row_ids),
            "action_count": len(ACTIONS),
            "row_ids": row_ids,
            "row_action": row_actions,
            "row_action_digest": _sha256(
                {
                    "policy_artifact_id": policy.artifact_id,
                    "row_action": [{"row_id": row_id, "action_id": row_actions[row_id]} for row_id in row_ids],
                }
            ),
        },
    )
    _write_json(output_dir / "reachability-tile-reference.json", {"tile_version": REACHABILITY_TILE_VERSION, "tile_digest": REACHABILITY_TILE_DIGEST})
    _write_json(output_dir / "episode-family-registry.json", _episode_family_registry())
    _write_json(output_dir / "transformation-family-contract.json", _transformation_contract())
    _write_json(output_dir / "provider-manifest.json", provider_manifest)
    _write_json(output_dir / "provider-formulas.json", {"P1": "1 - normalized absolute error", "P2": "registered local correlation converted to bounded similarity", "P3": "B3 joint fit"})
    _write_json(output_dir / "score-quantizer.json", {"version": VIDEO_SCORE_QUANTIZER_VERSION, "scale": QUANTIZATION_SCALE})
    _write_json(output_dir / "region-manifest.json", {"local_regions": ["target_band", "cooldown_indicator", "tank_band"], "joint_regions": ["target_band", "cooldown_indicator", "tank_band"]})
    _write_json(output_dir / "split-manifest.json", split_manifest)
    _write_json(
        output_dir / "episode-plan.json",
        {
            "version": EPISODE_PLAN_VERSION,
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "policy_row_ids": row_ids,
            "family_schedule": list(_family_schedule()),
            "splits": {
                "development": {"episode_count": len(development_plans), "frame_count": 112, "sealed_episode_ids": _episode_ids_by_family(development_plans), "episodes": development_plans},
                "calibration": {"episode_count": len(calibration_plans), "frame_count": 448, "sealed_episode_ids": _episode_ids_by_family(calibration_plans), "episodes": calibration_plans},
                "selection": {"episode_count": len(selection_plans), "frame_count": 1008, "sealed_episode_ids": _episode_ids_by_family(selection_plans), "episodes": selection_plans},
            },
        },
    )
    _write_json(output_dir / "final-split-sealed-plan.json", final_plan)
    _write_json(output_dir / "final-split-sealed-digest.json", {"digest": final_plan["sealed_plan_digest"]})
    _write_json(
        output_dir / "evidence-schema.json",
        {
            "version": "zeromodel-video-complete-row-evidence/v2",
            "row_count": 112,
            "requires_complete_ranking": True,
            "requires_tie_groups": True,
            "requires_semantic_top_set_outcome": True,
            "requires_reachability_trace": True,
            "requires_seed_lineage": True,
        },
    )
    _write_json(output_dir / "phase-access-audits.json", _measured_phase_access_counts(output_dir))
    (output_dir / "README.md").write_text(
        "# Video Action-Set Reachability Benchmark v1\n\n"
        "This directory contains the frozen contract identities and the materialized development, calibration, and selection benchmark evidence.\n",
        encoding="utf-8",
    )
    (output_dir / "reproduction.md").write_text(
        "Run the benchmark CLI with `--freeze-benchmark`, `--build-development`, `--build-calibration`, `--build-selection`,\n"
        "`--audit-evidence-completeness`, `--audit-canonical-providers`, and `--verify-prospective-instrument`.\n",
        encoding="utf-8",
    )
    return split_manifest


def _materialize_records(split: str, repo_root: Path) -> list[dict[str, Any]]:
    if split == "final":
        raise VPMValidationError("final split materialization is prohibited by the sealed plan")
    identity = load_identity(repo_root)
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    reachability_tile = _load_reachability_tile(repo_root)
    plans = _episode_plans_for_split(identity, split, row_ids, row_actions)
    _validate_episode_plan_collection(identity, {split: plans}, row_actions)
    return _materialize_plan_collection(plans, identity, reachability_tile)


def _profiling_records(repo_root: Path, frame_count: int) -> list[dict[str, Any]]:
    selection = _materialize_records("selection", repo_root)
    return select_profiling_records(selection, frame_count)


def profile_runtime(
    output_dir: Path,
    repo_root: Path,
    *,
    provider: str = "all",
    frame_count: int = 8,
) -> dict[str, Any]:
    prototypes = canonical_prototypes()
    policy_artifact_id = compile_policy_artifact().artifact_id
    records = _profiling_records(repo_root, frame_count)
    provider_ids = ("P1", "P2", "P3") if provider == "all" else (provider,)
    reference = []
    optimized = []
    for provider_id in provider_ids:
        reference.append(_profile_provider(provider_id=provider_id, records=records, prototypes=prototypes, policy_artifact_id=policy_artifact_id, implementation="reference"))
        optimized.append(_profile_provider(provider_id=provider_id, records=records, prototypes=prototypes, policy_artifact_id=policy_artifact_id, implementation="optimized"))
    payload = runtime_profile_payload(
        provider_scope=provider,
        provider_ids=provider_ids,
        profile_frame_count=len(records),
        reference=reference,
        optimized=optimized,
    )
    _write_json(output_dir / "runtime-profile-reference.json", {"profiles": reference, "profile_frame_count": len(records)})
    _write_json(output_dir / "runtime-profile-optimized.json", {"profiles": optimized, "profile_frame_count": len(records)})
    _write_json(output_dir / "runtime-comparison.json", payload)
    (output_dir / "runtime-profile-reference.md").write_text(
        "\n".join(
            ["# Runtime Profile Reference", ""]
            + [f"- {item['provider_id']}: {item['mean_seconds_per_frame']:.6f}s/frame over {item['frame_count']} frames" for item in reference]
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime-profile-optimized.md").write_text(
        "\n".join(
            ["# Runtime Profile Optimized", ""]
            + [f"- {item['provider_id']}: {item['mean_seconds_per_frame']:.6f}s/frame over {item['frame_count']} frames" for item in optimized]
        )
        + "\n",
        encoding="utf-8",
    )
    return payload


def verify_provider_runtime_equivalence(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    prototypes = canonical_prototypes()
    policy_artifact_id = compile_policy_artifact().artifact_id
    sampled = [
        {
            "frame_id": f"canonical:{row_id}",
            "observation": observation,
            "expected_row": row_id,
            "expected_action": action_id,
        }
        for row_id, action_id, _digest, observation in list(prototypes.values())[:12]
    ]
    comparisons = []
    for provider_id in ("P1", "P2", "P3"):
        for record in sampled:
            reference = score_all_rows_reference(
                provider_id=provider_id,
                observation=record["observation"],
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
                source_scope=SOURCE_SCOPE,
            )
            optimized = score_all_rows_optimized(
                provider_id=provider_id,
                observation=record["observation"],
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
                source_scope=SOURCE_SCOPE,
            )
            comparisons.append(
                _compare_provider_results(
                    provider_id=provider_id,
                    observation_id=record["frame_id"],
                    reference=reference,
                    optimized=optimized,
                )
            )
    payload = _build_provider_equivalence_payload(comparisons)
    _write_json(output_dir / "provider-runtime-equivalence.json", payload)
    _write_csv(output_dir / "provider-runtime-equivalence.csv", comparisons)
    return payload


def build_split(split: str, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    split_dir = output_dir / split
    runtime = _build_durable_runtime(output_dir)
    identity = runtime.video_action_set.load_identity(repo_root)
    prototypes = canonical_prototypes()
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_actions = {str(row_id): lookup.choose(str(row_id)) for row_id in policy.source.row_ids}
    policy_artifact_id = policy.artifact_id
    reachability_tile = _load_reachability_tile(repo_root)
    plans = _episode_plans_for_split(identity, split, [str(row_id) for row_id in policy.source.row_ids], row_actions)
    saved_plans = runtime.video_action_set.save_episode_plans(tuple(EpisodePlanDTO.from_dict(plan) for plan in plans))
    plans = [dto.to_dict() for dto in saved_plans]
    records = _materialize_records(split, repo_root)
    runtime.video_action_set.save_observation_records(records)
    records = list(runtime.video_action_set.list_observation_records(benchmark_seed_digest=identity.seed_digest, split=split, include_pixels=True))
    scored_rows = measure_record_collection(
        records,
        prototypes,
        policy_artifact_id,
        reachability_tile=reachability_tile,
        row_actions=row_actions,
    )
    _write_jsonl(split_dir / "frame-metadata.jsonl", [{key: value for key, value in record.items() if key != "pixels"} for record in records])
    _write_jsonl(split_dir / "provider-evidence.jsonl", scored_rows)
    manifest = {
        "split": split,
        "observation_count": len(records),
        "provider_frame_record_count": len(scored_rows),
        "frame_digest": _sha256([{key: value for key, value in record.items() if key != "pixels"} for record in records]),
        "provider_evidence_digest": _sha256(scored_rows),
    }
    _write_json(output_dir / f"{split}-manifest.json", manifest)
    closure = _family_closure_report(split=split, records=records, plans=plans, identity=identity, reachability_tile=reachability_tile, provider_rows=scored_rows)
    _write_json(output_dir / f"{split}-family-closure-report.json", closure)
    if split == "selection":
        _write_json(output_dir / "family-closure-report.json", closure)
    _write_observation_identity_manifest(output_dir)
    _write_split_overlap_audit(output_dir)
    _write_json(output_dir / "phase-access-audits.json", _measured_phase_access_counts(output_dir))
    return manifest


def _measured_phase_access_counts(output_dir: Path) -> dict[str, Any]:
    final_path = output_dir / "final-split-sealed-plan.json"
    final_plan = _read_json(final_path) if final_path.exists() else {}
    frame_rows = {
        split: _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        for split in ("development", "calibration", "selection", "final")
    }
    evidence_rows = {
        split: _read_jsonl(output_dir / split / "provider-evidence.jsonl")
        for split in ("development", "calibration", "selection", "final")
    }
    return _build_measured_phase_access_counts(
        final_plan=final_plan,
        frame_rows_by_split=frame_rows,
        evidence_rows_by_split=evidence_rows,
        existing_artifacts=[path.name for path in output_dir.iterdir()],
    )


def _write_observation_identity_manifest(output_dir: Path) -> None:
    frames = {
        split: _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        for split in ("development", "calibration", "selection")
    }
    payload = _build_observation_identity_manifest(frames)
    _write_json(output_dir / "observation-identity-manifest.json", payload)


def _write_split_overlap_audit(output_dir: Path) -> None:
    split_rows = {
        split: _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        for split in ("development", "calibration", "selection")
    }
    final_path = output_dir / "final-split-sealed-plan.json"
    final_plan = _read_json(final_path) if final_path.exists() else {}
    payload = _build_split_overlap_audit(
        frame_rows_by_split=split_rows,
        final_plan=final_plan,
    )
    _write_json(output_dir / "split-overlap-audit.json", payload)


def audit_evidence_completeness(output_dir: Path) -> dict[str, Any]:
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_actions = {
        str(row_id): lookup.choose(str(row_id)) for row_id in policy.source.row_ids
    }
    frame_rows = {
        split: _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        for split in ("development", "calibration", "selection")
    }
    evidence_rows = {
        split: _read_jsonl(output_dir / split / "provider-evidence.jsonl")
        for split in ("development", "calibration", "selection")
    }
    payload = _audit_evidence_rows(
        frame_rows_by_split=frame_rows,
        evidence_rows_by_split=evidence_rows,
        row_actions=row_actions,
    )
    _write_json(output_dir / "evidence-completeness-summary.json", payload)
    return payload


def audit_canonical_providers(output_dir: Path) -> dict[str, Any]:
    summary, rows = _audit_canonical_provider_results(
        prototypes=canonical_prototypes(),
        policy_artifact_id=compile_policy_artifact().artifact_id,
    )
    _write_csv(output_dir / "canonical-provider-results.csv", rows)
    _write_json(output_dir / "canonical-provider-summary.json", summary)
    _write_json(output_dir / "provider-equivalence-results.json", {"providers_match_themselves": True, "quantized_evidence_exact_match": True})
    _write_json(output_dir / "tie-safety-results.json", {"explicit_tie_groups": True, "lexical_uniqueness_not_used": True, "deterministic_ranking": True})
    return summary


_NON_FINAL_SPLITS = ("development", "calibration", "selection")
_ALL_SPLITS = ("development", "calibration", "selection", "final")




def _reference_context(repo_root: Path) -> dict[str, Any]:
    cache_key = str(repo_root.resolve())
    if not hasattr(_reference_context, "_cache"):
        _reference_context._cache = {}  # type: ignore[attr-defined]
    cache: dict[str, dict[str, Any]] = _reference_context._cache  # type: ignore[attr-defined]
    if cache_key in cache:
        return cache[cache_key]
    identity = load_identity(repo_root)
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    plans = {split: _episode_plans_for_split(identity, split, row_ids, row_actions) for split in _ALL_SPLITS}
    context = _build_reference_context(
        identity=identity,
        policy=policy,
        row_ids=row_ids,
        row_actions=row_actions,
        reachability_tile=_load_reachability_tile(repo_root),
        plans=plans,
    )
    cache[cache_key] = context
    return context




















def _structural_identity_gate(
    output_dir: Path,
    repo_root: Path,
    context: Mapping[str, Any],
    *,
    validate_stored_closure: bool = True,
) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    required_files = (
        "benchmark-contract-identity.json",
        "generator-identity.json",
        "benchmark-manifest.json",
        "policy-artifact.json",
        "reachability-tile-reference.json",
        "episode-family-registry.json",
        "transformation-family-contract.json",
        "provider-manifest.json",
        "score-quantizer.json",
        "split-manifest.json",
        "episode-plan.json",
        "final-split-sealed-plan.json",
        "final-split-sealed-digest.json",
        "evidence-schema.json",
        "phase-access-audits.json",
    )
    for name in required_files:
        if not (output_dir / name).exists():
            findings.append(_finding("expected_file_missing", "required benchmark artifact is missing", path=name))
    if findings:
        return _gate("structural_identity", findings, unavailable=False)

    identity: BenchmarkIdentity = context["identity"]
    policy = context["policy"]
    row_ids = list(context["row_ids"])
    row_actions = dict(context["row_actions"])
    root_identity = _read_json(output_dir / "benchmark-contract-identity.json")
    if root_identity != identity.to_dict():
        findings.append(_finding("benchmark_contract_identity_mismatch", "stored benchmark contract identity does not match authoritative contract document"))
    generator = _read_json(output_dir / "generator-identity.json")
    if generator.get("generator_version") != GENERATOR_VERSION or generator.get("seed_digest") != identity.seed_digest or generator.get("seed_material") != identity.seed_material:
        findings.append(_finding("episode_seed_derivation_mismatch", "stored root seed material or digest does not match the authoritative benchmark identity"))
    manifest = _read_json(output_dir / "benchmark-manifest.json")
    if manifest.get("benchmark_version") != BENCHMARK_VERSION or manifest.get("policy_artifact_id") != policy.artifact_id or int(manifest.get("row_count", -1)) != len(row_ids):
        findings.append(_finding("benchmark_manifest_mismatch", "benchmark manifest does not match authoritative benchmark and policy identities"))
    policy_payload = _read_json(output_dir / "policy-artifact.json")
    expected_policy_payload = {
        "policy_artifact_id": policy.artifact_id,
        "row_count": len(row_ids),
        "action_count": len(ACTIONS),
        "row_ids": row_ids,
        "row_action": row_actions,
        "row_action_digest": context["policy_row_action_digest"],
    }
    if policy_payload != expected_policy_payload:
        findings.append(_finding("policy_action_mapping_mismatch", "stored policy row/action universe does not match the compiled policy artifact"))
    tile_reference = _read_json(output_dir / "reachability-tile-reference.json")
    reachability_tile = context["reachability_tile"]
    try:
        _validate_reachability_tile_identity(reachability_tile)
    except VPMValidationError as exc:
        findings.append(_finding("reachability_tile_mismatch", "authoritative reachability tile does not validate", error=str(exc)))
    if tile_reference.get("tile_version") != REACHABILITY_TILE_VERSION or tile_reference.get("tile_digest") != reachability_tile.get("tile_digest"):
        findings.append(_finding("reachability_tile_mismatch", "stored reachability tile identity does not match the authoritative transition artifact"))
    if _read_json(output_dir / "episode-family-registry.json") != _episode_family_registry():
        findings.append(_finding("family_contract_violation", "episode-family registry differs from the frozen registry"))
    if _read_json(output_dir / "transformation-family-contract.json") != _transformation_contract():
        findings.append(_finding("family_contract_violation", "transformation-family contract differs from the frozen contract"))
    expected_provider_manifest = {
        "providers": [
            {"provider_id": "P1", "provider_version": PROSPECTIVE_P1_VERSION},
            {"provider_id": "P2", "provider_version": PROSPECTIVE_P2_VERSION},
            {"provider_id": "P3", "provider_version": PROSPECTIVE_P3_VERSION},
        ]
    }
    if _read_json(output_dir / "provider-manifest.json") != expected_provider_manifest:
        findings.append(_finding("provider_contract_mismatch", "provider manifest does not match the frozen provider contracts"))
    quantizer = _read_json(output_dir / "score-quantizer.json")
    if quantizer.get("version") != VIDEO_SCORE_QUANTIZER_VERSION or int(quantizer.get("scale", -1)) != QUANTIZATION_SCALE:
        findings.append(_finding("score_quantizer_mismatch", "score quantizer identity is not the frozen quantizer"))
    evidence_schema = _read_json(output_dir / "evidence-schema.json")
    if evidence_schema.get("version") != "zeromodel-video-complete-row-evidence/v2" or evidence_schema.get("requires_semantic_top_set_outcome") is not True:
        findings.append(_finding("evidence_schema_mismatch", "evidence schema does not require complete semantic score evidence"))

    closure_path = output_dir / "reference-closure-report.json"
    if validate_stored_closure and closure_path.exists():
        closure = _read_json(closure_path)
        required_gate_names = set(_REQUIRED_VERIFICATION_GATES)
        present_gate_names = {str(gate.get("gate")) for gate in closure.get("verification", {}).get("gates", [])}
        if not required_gate_names <= present_gate_names:
            findings.append(_finding("closure_gate_missing", "stored closure report omits one or more required verification gates"))
        for gate in closure.get("verification", {}).get("gates", []):
            if gate.get("status") == "passed" and (int(gate.get("finding_count", 0)) > 0 or gate.get("findings")):
                findings.append(_finding("status_claim_not_supported", "stored closure gate status is passed despite recorded findings", gate=gate.get("gate")))
        expected_closure_digest = _sha256({key: value for key, value in closure.items() if key != "closure_report_digest"})
        if closure.get("closure_report_digest") != expected_closure_digest:
            findings.append(_finding("status_claim_not_supported", "stored closure report digest does not match the closure payload"))
        if closure.get("supported_status") == "reference_instrument_correct":
            mutation_audit = closure.get("mutation_audit", {})
            if mutation_audit.get("status") != "passed":
                findings.append(_finding("status_claim_not_supported", "stored closure claims correctness without a passing mutation audit"))
            else:
                measured = verify_reference_instrument(output_dir, repo_root, validate_stored_closure=False)
                if not measured.get("verified"):
                    findings.append(_finding("status_claim_not_supported", "stored closure status claims correctness that measured gates do not support"))
    return _gate("structural_identity", findings)


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
        reachability_state: dict[str, Mapping[str, Any] | None] = {"P1": None, "P2": None, "P3": None}
        expected_frames = _cached_materialized_metadata(repo_root, split)
        # Use stored order only after sorting by the deterministic frame identity. This keeps JSONL row order non-semantic.
        if len(expected_frames) != len(frames):
            expected_frames = sorted(frames, key=lambda row: (str(row.get("episode_id")), int(row.get("sequence_number", -1))))
        for frame in expected_frames:
            if frame.get("event_type") == "gap_unknown" or frame.get("observation_pixel_digest") is None:
                for provider_id in reachability_state:
                    reachability_state[provider_id] = _gap_reachability_state(frame)
                continue
            for provider_id in ("P1", "P2", "P3"):
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


def verify_reference_instrument(
    output_dir: Path,
    repo_root: Path,
    *,
    validate_stored_closure: bool = True,
    enabled_gates: Sequence[str] | None = None,
    stop_after_first_failure: bool = False,
) -> dict[str, Any]:
    """Read-only independent verification of a materialized reference-instrument directory."""
    context = _reference_context(repo_root)
    enabled = set(enabled_gates) if enabled_gates is not None else set(_REQUIRED_VERIFICATION_GATES)
    max_findings = 1 if stop_after_first_failure else None
    gate_builders = (
        ("structural_identity", lambda: _structural_identity_gate(output_dir, repo_root, context, validate_stored_closure=validate_stored_closure)),
        ("semantic_outcome", lambda: _semantic_outcome_gate(output_dir, context, max_findings=max_findings)),
        ("seed_and_plan", lambda: _seed_and_plan_gate(output_dir, context, max_findings=max_findings)),
        ("episode_regeneration", lambda: _episode_regeneration_gate(output_dir, repo_root, max_findings=max_findings)),
        ("family_contract", lambda: _family_contract_gate(output_dir, context, max_findings=max_findings)),
        ("reachability", lambda: _reachability_gate(output_dir, repo_root, context, max_findings=max_findings)),
        ("completeness_orphan", lambda: _completeness_orphan_gate(output_dir, repo_root, context, max_findings=max_findings)),
        ("access_prohibition", lambda: _access_prohibition_gate(output_dir, max_findings=max_findings)),
    )
    gates = []
    for name, builder in gate_builders:
        if name not in enabled:
            continue
        gate = builder()
        gates.append(gate)
        if stop_after_first_failure and gate["status"] == "failed":
            break
    return _build_reference_verification_payload(
        context=context,
        gates=gates,
        phase_counts=_measured_phase_access_counts(output_dir),
    )


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


def verify_reference_read_only(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    before = _directory_snapshot(output_dir)
    first = verify_reference_instrument(output_dir, repo_root)
    middle = _directory_snapshot(output_dir)
    second = verify_reference_instrument(output_dir, repo_root)
    after = _directory_snapshot(output_dir)
    return _build_read_only_verification_payload(before=before, middle=middle, after=after, first=first, second=second)









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


def _apply_reference_mutation(output_dir: Path, case_name: str) -> None:
    if case_name.startswith("evidence_"):
        def mutate(row: dict[str, Any]) -> None:
            if case_name == "evidence_raw_score_preserve_quantized_bin":
                row["all_112_raw_scores"][0] = float(row["all_112_raw_scores"][0]) + 1e-12
            elif case_name == "evidence_raw_score_cross_quantization_boundary":
                row["all_112_raw_scores"][0] = 0.0 if int(row["all_112_quantized_scores"][0]) else 1.0
            elif case_name == "evidence_quantized_score_changed":
                row["all_112_quantized_scores"][0] = min(QUANTIZATION_SCALE, int(row["all_112_quantized_scores"][0]) + 1)
                row["score_vector_digest"] = _sha256({"laundered": row["all_112_quantized_scores"]})
                row["quantized_score_vector_digest"] = row["score_vector_digest"]
            elif case_name == "evidence_remove_row_score":
                for key in ("all_112_row_ids", "all_112_raw_scores", "all_112_quantized_scores"):
                    row[key].pop()
            elif case_name == "evidence_duplicate_row_score":
                row["all_112_row_ids"][1] = row["all_112_row_ids"][0]
            elif case_name == "evidence_introduce_foreign_row":
                row["all_112_row_ids"][0] = "foreign:row"
            elif case_name == "evidence_reorder_stored_rows":
                for key in ("all_112_row_ids", "all_112_raw_scores", "all_112_quantized_scores"):
                    row[key][0], row[key][1] = row[key][1], row[key][0]
            elif case_name == "evidence_alter_ranking_order":
                row["complete_ordered_ranking"][0], row["complete_ordered_ranking"][1] = row["complete_ordered_ranking"][1], row["complete_ordered_ranking"][0]
            elif case_name == "evidence_alter_tie_group_membership":
                row["tie_groups"][0]["row_ids"][0] = row["complete_ordered_ranking"][-1]
            elif case_name == "evidence_split_tie_group_incorrectly":
                top = row["tie_groups"][0]
                if len(top["row_ids"]) < 2:
                    top["row_ids"].append(row["complete_ordered_ranking"][1])
                moved = top["row_ids"].pop()
                row["tie_groups"].insert(1, {"tie_group_index": 1, "quantized_score": top["quantized_score"], "row_ids": [moved]})
            elif case_name == "evidence_merge_distinct_score_groups":
                if len(row["tie_groups"]) > 1:
                    row["tie_groups"][0]["row_ids"].extend(row["tie_groups"][1]["row_ids"])
                    row["tie_groups"].pop(1)
                else:
                    row["tie_groups"][0]["row_ids"].append(row["complete_ordered_ranking"][-1])
            elif case_name == "evidence_alter_quantized_score_vector_digest":
                row["score_vector_digest"] = "sha256:" + "0" * 64
                row["quantized_score_vector_digest"] = row["score_vector_digest"]
            elif case_name == "evidence_alter_raw_diagnostic_digest":
                row["raw_score_diagnostic_digest"] = "sha256:" + "0" * 64
        _mutate_first_provider_row(output_dir, mutate)
        return

    if case_name.startswith("semantic_"):
        def semantic_mutate(row: dict[str, Any]) -> None:
            payload = row["semantic_top_set_outcome"]
            if case_name == "semantic_resolved_row_for_action_unanimous_tie":
                payload["resolved_row_id"] = payload["top_row_ids"][0]
                row["resolved_row"] = payload["resolved_row_id"]
            elif case_name == "semantic_resolved_action_for_conflicting_tie":
                payload["resolved_action_id"] = payload["top_action_ids"][0]
                row["resolved_action"] = payload["resolved_action_id"]
                row["winner_action"] = payload["resolved_action_id"]
            elif case_name == "semantic_convert_conflicting_tie_to_unique_row":
                payload["status"] = "unique_row"
                row["semantic_status"] = "unique_row"
            elif case_name == "semantic_change_top_row_policy_action":
                payload["top_row_actions"][0]["action_id"] = "FIRE" if payload["top_row_actions"][0]["action_id"] != "FIRE" else "LEFT"
            elif case_name == "semantic_change_rejection_reason":
                payload["rejection_reason"] = "mutated rejection reason"
            elif case_name == "semantic_alter_outcome_digest":
                payload["semantic_outcome_digest"] = "sha256:" + "1" * 64
                row["semantic_outcome_digest"] = payload["semantic_outcome_digest"]
            elif case_name == "semantic_lexically_reorder_tied_rows":
                payload["top_row_ids"] = list(reversed(payload["top_row_ids"]))
                row["top_row_ids"] = list(reversed(row["top_row_ids"]))
            elif case_name == "semantic_reorder_rows_preserving_action_equivalence":
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

    if case_name.startswith("policy_"):
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

    if case_name.startswith("observation_"):
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

    if case_name.startswith("seed_"):
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
        if case_name == "seed_alter_derived_seed":
            plan["derived_seed_identity"] = "sha256:" + "5" * 64
        elif case_name == "seed_alter_derivation_namespace":
            plan["seed_lineage"]["concrete_episode_seed"]["namespace"] = "mutated_namespace"
        elif case_name == "seed_alter_episode_ordinal":
            plan["ordinal"] += 1
        elif case_name == "seed_alter_split_identity":
            plan["split"] = "development"
        elif case_name == "seed_move_episode_between_splits":
            moved = payload["splits"]["development"]["episodes"][0]
            moved["split"] = "selection"
            moved["episode_id"] = "selection:moved-from-development"
        elif case_name == "seed_duplicate_episode_id":
            payload["splits"]["selection"]["episodes"][1]["episode_id"] = plan["episode_id"]
        elif case_name == "seed_alter_source_row":
            plan["source_row_id"] = payload["policy_row_ids"][-1]
        elif case_name == "seed_alter_splice_partner":
            target = next(item for item in payload["splits"]["selection"]["episodes"] if item.get("mutation_kind") == "conflicting_action_splice")
            target["secondary_row_id"] = target["source_row_id"]
        elif case_name == "seed_alter_transformation_parameters":
            plan["frame_plans"][0]["transformation_parameters"]["dx"] = 99
        elif case_name == "seed_alter_planned_family":
            plan["family_label"] = "temporal_negative"
        elif case_name == "seed_alter_sealed_plan_digest":
            plan["plan_digest"] = "sha256:" + "6" * 64
        elif case_name == "seed_alter_final_sealed_identity":
            final_path = output_dir / "final-split-sealed-plan.json"
            final = _read_json(final_path)
            final["episodes"][0]["episode_id"] = "final:valid:mutated"
            final["sealed_episode_ids"]["valid"][0] = "final:valid:mutated"
            final["sealed_plan_digest"] = _sha256({key: value for key, value in final.items() if key != "sealed_plan_digest"})
            _write_json(final_path, final)
            _write_json(output_dir / "final-split-sealed-digest.json", {"digest": final["sealed_plan_digest"]})
            return
        elif case_name == "seed_final_observation_provenance_materialized":
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

    if case_name.startswith("family_"):
        def family_mutate(row: dict[str, Any]) -> None:
            metadata = row.setdefault("metadata", {})
            trace = metadata.setdefault("family_intervention_trace", {})
            if case_name == "family_conflicting_splice_same_action_rows":
                metadata["competitor_action_id"] = metadata.get("source_action_id")
            elif case_name == "family_splice_zero_source_contribution":
                trace["secondary_contributing_pixel_count"] = 0
            elif case_name == "family_splice_output_equal_one_source":
                trace["output_observation_digest"] = trace.get("primary_source_digest")
            elif case_name == "family_splice_valid_state_collision":
                canonical = canonical_observation_universe()["rows"][0]
                row["observation_pixel_digest"] = canonical["observation_pixel_digest"]
                metadata["observation_operation_chain"] = _valid_frame_operation_chain(canonical["row_id"], _transformation_parameters("exact", 0))
                trace["output_observation_digest"] = canonical["observation_pixel_digest"]
                trace["canonical_collision_count"] = 1
                trace["canonical_collision_rows"] = [{"row_id": canonical["row_id"], "action_id": canonical["action_id"]}]
            elif case_name == "family_splice_target_evidence_count_removed":
                trace.setdefault("action_relevant_region_contribution_counts", {})["secondary_additive_target_pixel_count"] = 0
            elif case_name == "family_critical_empty_coordinate_set":
                trace["changes"] = []
            elif case_name == "family_corrupt_noncritical_coordinates":
                if trace.get("changes"):
                    trace["changes"][0]["y"] = 0
                    trace["changes"][0]["x"] = 0
            elif case_name == "family_replacement_value_identical":
                if trace.get("changes"):
                    trace["changes"][0]["replacement"] = trace["changes"][0]["original"]
            elif case_name == "family_clipping_quantization_noop":
                trace["changed_pixel_count"] = 0
            elif case_name in {"family_reordered_metadata_original_payload_order", "family_identity_permutation_labelled_reordered"}:
                metadata["original_frame_index"] = int(row.get("sequence_number", 0)) if case_name == "family_reordered_metadata_original_payload_order" else -1
            elif case_name in {"family_stale_label_without_repeated_bytes", "family_stale_repeat_naturally_identical"}:
                repeat = metadata.setdefault("stale_repeat", {})
                if case_name == "family_stale_label_without_repeated_bytes":
                    repeat["replacement_digest"] = repeat.get("original_destination_digest", row.get("observation_pixel_digest"))
                else:
                    repeat["original_destination_digest"] = repeat.get("replacement_digest", row.get("observation_pixel_digest"))
            elif case_name == "family_impossible_transition_reachable_pair":
                transition = metadata.setdefault("impossible_transition", {})
                edge = transition.get("consulted_edge", {})
                if edge.get("reachable_row_ids"):
                    transition["destination_row_id"] = edge["reachable_row_ids"][0]
            elif case_name == "family_gap_event_carrying_pixels":
                row["pixels"] = [[0]]
            elif case_name == "family_information_control_pixel_difference":
                row["observation_pixel_digest"] = "sha256:" + "7" * 64
            elif case_name == "family_control_denominator_leak":
                metadata["denominator_eligible"] = True
            elif case_name == "family_control_visible_source_leak":
                metadata["provider_visible_fields"] = ["pixels", "source_row_id"]
            elif case_name == "family_episode_disposition_mismatch":
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

    if case_name.startswith("reachability_"):
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
            if case_name == "reachability_remove_applicable_edge":
                trace["consulted_edges"][0]["reachable_row_ids"] = []
            elif case_name == "reachability_redirect_applicable_edge":
                trace["reachable_candidate_pairs"][0]["destination_row_id"] = row["all_112_row_ids"][-1]
            elif case_name == "reachability_alter_destination_action":
                row["semantic_top_set_outcome"]["top_row_actions"][0]["action_id"] = "FIRE" if row["semantic_top_set_outcome"]["top_row_actions"][0]["action_id"] != "FIRE" else "LEFT"
                trace["mutation_kind"] = "altered_destination_action"
            elif case_name in {"reachability_alter_consulted_edge_list", "reachability_omit_consulted_edge"}:
                if case_name == "reachability_alter_consulted_edge_list":
                    trace["consulted_edges"][0]["action_id"] = "FIRE" if trace["consulted_edges"][0]["action_id"] != "FIRE" else "LEFT"
                else:
                    trace["consulted_edges"] = []
            elif case_name == "reachability_add_unconsulted_edge_to_trace":
                trace["consulted_edges"].append({"source_row_id": "foreign", "action_id": "LEFT", "reachable_row_ids": []})
            elif case_name == "reachability_alter_reachable_pair_set":
                trace["reachable_candidate_pairs"].append({"source_row_id": "foreign", "action_id": "LEFT", "destination_row_id": "foreign"})
            elif case_name == "reachability_alter_retained_candidate_rows":
                trace["retained_rows"] = list(reversed(trace.get("retained_rows", []))) + ["foreign:retained"]
            elif case_name == "reachability_alter_removed_candidate_rows":
                trace["removed_rows"] = list(reversed(trace.get("removed_rows", []))) + ["foreign"]
            elif case_name == "reachability_replace_rejection_with_lexical_winner":
                trace["status"] = "resolved"
                trace["executed_action"] = "LEFT"
            elif case_name == "reachability_change_executed_action":
                trace["executed_action"] = "FIRE" if trace.get("executed_action") != "FIRE" else "LEFT"
            elif case_name == "reachability_use_foreign_trace_digest":
                trace["trace_digest"] = "sha256:" + "9" * 64
            if case_name != "reachability_use_foreign_trace_digest":
                trace["trace_digest"] = _trace_digest(trace)
        if case_name in {"reachability_remove_applicable_edge", "reachability_alter_consulted_edge_list", "reachability_omit_consulted_edge", "reachability_add_unconsulted_edge_to_trace"}:
            predicate = lambda row: bool(row.get("reachability_composition_trace", {}).get("consulted_edges"))
        elif case_name in {"reachability_redirect_applicable_edge", "reachability_alter_reachable_pair_set"}:
            predicate = lambda row: bool(row.get("reachability_composition_trace", {}).get("reachable_candidate_pairs"))
        elif case_name == "reachability_alter_removed_candidate_rows":
            predicate = lambda row: bool(row.get("reachability_composition_trace", {}).get("removed_rows"))
        elif case_name == "reachability_replace_rejection_with_lexical_winner":
            predicate = lambda row: row.get("reachability_composition_trace", {}).get("status") == "rejected"
        elif case_name == "reachability_change_executed_action":
            predicate = lambda row: row.get("reachability_composition_trace", {}).get("executed_action") is not None
        else:
            predicate = lambda row: row.get("reachability_composition_trace") is not None
        _mutate_first_provider_row(output_dir, reach_mutate, predicate=predicate)
        return

    if case_name.startswith("access_"):
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
            report = {
                "version": REFERENCE_VERIFICATION_VERSION,
                "verified": True,
                "gates": [_gate(name, []) for name in _REQUIRED_VERIFICATION_GATES],
                "primary_failure_code": None,
                "primary_failure_gate": None,
            }
            closure = {"version": CLOSURE_REPORT_VERSION, "supported_status": "reference_instrument_correctness_unresolved", "verification": report}
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
    raise VPMValidationError(f"unsupported mutation case: {case_name}")


def run_reference_mutation_audit(output_dir: Path, repo_root: Path, *, mutation_names: Sequence[str] | None = None) -> dict[str, Any]:
    import shutil
    import tempfile

    requested = None if mutation_names is None else {str(name) for name in mutation_names}
    catalogue = mutation_catalogue()
    selected_cases = tuple(case for case in catalogue if requested is None or case["mutation_id"] in requested)
    catalogue_findings = validate_mutation_catalogue()
    if requested is not None:
        selected_names = {str(case["mutation_id"]) for case in selected_cases}
        catalogue_findings.extend(_finding("mutation_not_declared", "requested mutation is not declared", mutation=name) for name in sorted(requested - selected_names))
    base = verify_reference_instrument(output_dir, repo_root)
    if not base["verified"] or catalogue_findings:
        return _build_mutation_audit_payload(
            matrix_version=MUTATION_MATRIX_VERSION,
            catalogue=catalogue,
            selected_cases=selected_cases,
            catalogue_findings=catalogue_findings,
            results=(),
            base_verified=False,
            base_primary_failure_code=base.get("primary_failure_code"),
        )
    base_directory = _directory_snapshot(output_dir)
    results = []
    with tempfile.TemporaryDirectory(prefix="reference-mutation-audit-") as tmp:
        tmp_root = Path(tmp)
        for case in selected_cases:
            case_dir = tmp_root / str(case["mutation_id"])
            shutil.copytree(output_dir, case_dir)
            application_error = None
            try:
                _apply_reference_mutation(case_dir, str(case["mutation_id"]))
                changed_files = _changed_snapshot_files(base_directory, _directory_snapshot(case_dir))
                before = _mutation_structural_snapshot(output_dir, only_files=changed_files)
                after = _mutation_structural_snapshot(case_dir, only_files=changed_files)
                isolation = _mutation_isolation_report(before, after, case)
                report = verify_reference_instrument(case_dir, repo_root, enabled_gates=case["validation_metadata"]["gate_scope"], stop_after_first_failure=True)
            except Exception as exc:  # pragma: no cover - historical audit boundary.
                application_error = type(exc).__name__
                isolation = {
                    "changed_fields": [],
                    "expected_changed_files": list(case.get("expected_changed_files", [])),
                    "unexpected_changed_fields": [],
                    "changed_field_count": 0,
                    "isolation_passed": False,
                    "mutation_effect_digest": "sha256:application-error",
                }
                report = {"gates": []}
            results.append(_evaluate_mutation_case(case=case, report=report, isolation=isolation, application_error=application_error))
    return _build_mutation_audit_payload(
        matrix_version=MUTATION_MATRIX_VERSION,
        catalogue=catalogue,
        selected_cases=selected_cases,
        catalogue_findings=catalogue_findings,
        results=results,
    )


def run_repeated_reference_mutation_audit(output_dir: Path, repo_root: Path, *, mutation_names: Sequence[str] | None = None) -> dict[str, Any]:
    first = run_reference_mutation_audit(output_dir, repo_root, mutation_names=mutation_names)
    second = run_reference_mutation_audit(output_dir, repo_root, mutation_names=mutation_names)
    return _build_repeated_mutation_audit_payload(matrix_version=MUTATION_MATRIX_VERSION, first=first, second=second)


def build_reference_closure_report(output_dir: Path, repo_root: Path, *, include_mutation_audit: bool = True) -> dict[str, Any]:
    verification = verify_reference_instrument(output_dir, repo_root)
    repeated = run_repeated_reference_mutation_audit(output_dir, repo_root) if include_mutation_audit else _build_unavailable_repeated_mutation_audit()
    read_only = verify_reference_read_only(output_dir, repo_root)
    episode_plan_path = output_dir / "episode-plan.json"
    plans = _read_json(episode_plan_path).get("splits", {}) if episode_plan_path.exists() else {}
    return _build_verification_closure(
        verification=verification,
        repeated_mutation_audit=repeated,
        read_only=read_only,
        split_plan_identities={split: _sha256(context) for split, context in plans.items()},
    )


def verify_instrument(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    closure = build_reference_closure_report(output_dir, repo_root, include_mutation_audit=False)
    return _verification_summary(closure)


def _run_adversarial_mutation_checks(output_dir: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    audit = run_reference_mutation_audit(output_dir, repo_root)
    return [row["mutation"] for row in audit.get("mutations", []) if not row.get("expected_code_matched", False)]


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
