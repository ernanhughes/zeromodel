"""Video action-set build orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast
from zeromodel.arcade_policy import (
    ACTIONS,
    compile_policy_artifact,
)
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    EPISODE_PLAN_VERSION,
    GENERATOR_VERSION,
    REACHABILITY_TILE_DIGEST,
    REACHABILITY_TILE_VERSION,
    SEED_DERIVATION_VERSION,
)
from zeromodel.domains.video_action_set.dto import EpisodePlanDTO
from zeromodel.domains.video_action_set.episode_families import (
    episode_family_registry as _episode_family_registry,
    family_schedule as _family_schedule,
)
from zeromodel.domains.video_action_set.episode_materialization import (
    materialize_plan_collection as _materialize_plan_collection,
)
from zeromodel.domains.video_action_set.episode_planning import (
    episode_ids_by_family as _episode_ids_by_family,
    episode_plans_for_split as _episode_plans_for_split,
    validate_episode_plan_collection as _validate_episode_plan_collection,
)
from zeromodel.domains.video_action_set.evidence_audit import (
    measured_phase_access_counts as _build_measured_phase_access_counts,
    build_observation_identity_manifest as _build_observation_identity_manifest,
    build_split_overlap_audit as _build_split_overlap_audit,
)
from zeromodel.domains.video_action_set.materialization_validation import (
    family_closure_report as _family_closure_report,
)
from zeromodel.domains.video_action_set.observation_universe import canonical_prototypes
from zeromodel.domains.video_action_set.provider_measurement import (
    SplitBuildProgressObserver,
    measure_record_collection,
)
from zeromodel.domains.video_action_set.runtime_profiling import (
    profile_provider as _profile_provider,
    runtime_profile_payload,
    select_profiling_records,
)
from zeromodel.domains.video_action_set.transformations import _transformation_contract
from zeromodel.policy_lookup import VPMPolicyLookup
from zeromodel.runtime import build_runtime
from zeromodel.video_complete_row_evidence import (
    QUANTIZATION_SCALE,
    VIDEO_SCORE_QUANTIZER_VERSION,
)
from zeromodel.video_prospective_providers import (
    PROSPECTIVE_P1_VERSION,
    PROSPECTIVE_P2_VERSION,
    PROSPECTIVE_P3_VERSION,
)
from zeromodel.domains.video_action_set.artifact_io import (
    _read_json,
    _read_jsonl,
    _sha256,
    _write_json,
    _write_jsonl,
    _write_text,
)
from zeromodel.domains.video_action_set.artifact_layout import (
    benchmark_database_path,
    family_closure_path,
    reachability_tile_path,
    root_artifact_path,
    split_artifact_path,
    split_manifest_path,
)
from zeromodel.domains.video_action_set.report_rendering import (
    benchmark_readme,
    reproduction_instructions,
    runtime_profile_optimized,
    runtime_profile_reference,
)
from zeromodel.domains.video_action_set.dto import BenchmarkIdentityDTO


BenchmarkIdentity = BenchmarkIdentityDTO


def _build_durable_runtime(output_dir: Path):
    try:
        from zeromodel.db.runtime import build_sqlite_runtime
    except ImportError as exc:
        raise VPMValidationError(
            "SQLite benchmark persistence requires the optional database extra: "
            'pip install "zeromodel[persistence]"'
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    database_url = f"sqlite:///{benchmark_database_path(output_dir).as_posix()}"
    return build_sqlite_runtime(database_url, initialize_schema=True)


def _load_reachability_tile(repo_root: Path) -> dict[str, Any]:
    return _read_json(reachability_tile_path(repo_root))


def load_identity(repo_root: Path) -> BenchmarkIdentity:
    return build_runtime().video_action_set.load_identity(repo_root)


def _write_frozen_contract_artifacts(
    output_dir: Path,
    identity: BenchmarkIdentity,
    policy: Any,
    row_ids: list[str],
    row_actions: dict[str, str],
    provider_manifest: dict[str, Any],
    split_manifest: dict[str, Any],
) -> None:
    _write_json(
        root_artifact_path(output_dir, "benchmark-contract-identity.json"),
        identity.to_dict(),
    )
    _write_json(
        root_artifact_path(output_dir, "generator-identity.json"),
        {
            "generator_version": GENERATOR_VERSION,
            "seed_digest": identity.seed_digest,
            "seed_material": identity.seed_material,
        },
    )
    _write_json(
        root_artifact_path(output_dir, "benchmark-manifest.json"),
        {
            "benchmark_version": BENCHMARK_VERSION,
            "policy_artifact_id": policy.artifact_id,
            "row_count": len(row_ids),
        },
    )
    _write_json(
        root_artifact_path(output_dir, "policy-artifact.json"),
        {
            "policy_artifact_id": policy.artifact_id,
            "row_count": len(row_ids),
            "action_count": len(ACTIONS),
            "row_ids": row_ids,
            "row_action": row_actions,
            "row_action_digest": _sha256(
                {
                    "policy_artifact_id": policy.artifact_id,
                    "row_action": [
                        {"row_id": row_id, "action_id": row_actions[row_id]}
                        for row_id in row_ids
                    ],
                }
            ),
        },
    )
    _write_json(
        root_artifact_path(output_dir, "reachability-tile-reference.json"),
        {
            "tile_version": REACHABILITY_TILE_VERSION,
            "tile_digest": REACHABILITY_TILE_DIGEST,
        },
    )
    _write_json(
        root_artifact_path(output_dir, "episode-family-registry.json"),
        _episode_family_registry(),
    )
    _write_json(
        root_artifact_path(output_dir, "transformation-family-contract.json"),
        _transformation_contract(),
    )
    _write_json(
        root_artifact_path(output_dir, "provider-manifest.json"), provider_manifest
    )
    _write_json(
        root_artifact_path(output_dir, "provider-formulas.json"),
        {
            "P1": "1 - normalized absolute error",
            "P2": "registered local correlation converted to bounded similarity",
            "P3": "B3 joint fit",
        },
    )
    _write_json(
        root_artifact_path(output_dir, "score-quantizer.json"),
        {"version": VIDEO_SCORE_QUANTIZER_VERSION, "scale": QUANTIZATION_SCALE},
    )
    _write_json(
        root_artifact_path(output_dir, "region-manifest.json"),
        {
            "local_regions": ["target_band", "cooldown_indicator", "tank_band"],
            "joint_regions": ["target_band", "cooldown_indicator", "tank_band"],
        },
    )
    _write_json(root_artifact_path(output_dir, "split-manifest.json"), split_manifest)


def _write_frozen_plan_artifacts(
    output_dir: Path,
    row_ids: list[str],
    development_plans: list[dict[str, Any]],
    calibration_plans: list[dict[str, Any]],
    selection_plans: list[dict[str, Any]],
    final_plan: dict[str, Any],
) -> None:
    _write_json(
        root_artifact_path(output_dir, "episode-plan.json"),
        {
            "version": EPISODE_PLAN_VERSION,
            "seed_derivation_version": SEED_DERIVATION_VERSION,
            "policy_row_ids": row_ids,
            "family_schedule": list(_family_schedule()),
            "splits": {
                "development": {
                    "episode_count": len(development_plans),
                    "frame_count": 112,
                    "sealed_episode_ids": _episode_ids_by_family(development_plans),
                    "episodes": development_plans,
                },
                "calibration": {
                    "episode_count": len(calibration_plans),
                    "frame_count": 448,
                    "sealed_episode_ids": _episode_ids_by_family(calibration_plans),
                    "episodes": calibration_plans,
                },
                "selection": {
                    "episode_count": len(selection_plans),
                    "frame_count": 1008,
                    "sealed_episode_ids": _episode_ids_by_family(selection_plans),
                    "episodes": selection_plans,
                },
            },
        },
    )
    _write_json(
        root_artifact_path(output_dir, "final-split-sealed-plan.json"), final_plan
    )
    _write_json(
        root_artifact_path(output_dir, "final-split-sealed-digest.json"),
        {"digest": final_plan["sealed_plan_digest"]},
    )
    _write_json(
        root_artifact_path(output_dir, "evidence-schema.json"),
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
    _write_json(
        root_artifact_path(output_dir, "phase-access-audits.json"),
        _measured_phase_access_counts(output_dir),
    )
    _write_text(root_artifact_path(output_dir, "README.md"), benchmark_readme())
    _write_text(
        root_artifact_path(output_dir, "reproduction.md"), reproduction_instructions()
    )


def freeze_benchmark(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    runtime = _build_durable_runtime(output_dir)
    identity = runtime.video_action_set.load_identity(repo_root)
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    development_plans = _episode_plans_for_split(
        identity, "development", row_ids, row_actions
    )
    calibration_plans = _episode_plans_for_split(
        identity, "calibration", row_ids, row_actions
    )
    selection_plans = _episode_plans_for_split(
        identity, "selection", row_ids, row_actions
    )
    final_plans = _episode_plans_for_split(identity, "final", row_ids, row_actions)
    _validate_episode_plan_collection(
        identity,
        {
            "development": development_plans,
            "calibration": calibration_plans,
            "selection": selection_plans,
            "final": final_plans,
        },
        row_actions,
    )
    development_episode_dtos = runtime.video_action_set.save_episode_plans(
        tuple(EpisodePlanDTO.from_dict(plan) for plan in development_plans)
    )
    calibration_episode_dtos = runtime.video_action_set.save_episode_plans(
        tuple(EpisodePlanDTO.from_dict(plan) for plan in calibration_plans)
    )
    selection_episode_dtos = runtime.video_action_set.save_episode_plans(
        tuple(EpisodePlanDTO.from_dict(plan) for plan in selection_plans)
    )
    final_episode_dtos = runtime.video_action_set.save_episode_plans(
        tuple(EpisodePlanDTO.from_dict(plan) for plan in final_plans)
    )
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
    sealed_final = runtime.video_action_set.seal_final_split(
        episodes=final_episode_dtos, seed_commitment=identity.seed_digest
    )
    final_plan = sealed_final.to_dict()
    _write_frozen_contract_artifacts(
        output_dir,
        identity,
        policy,
        row_ids,
        row_actions,
        provider_manifest,
        split_manifest,
    )
    _write_frozen_plan_artifacts(
        output_dir,
        row_ids,
        development_plans,
        calibration_plans,
        selection_plans,
        final_plan,
    )
    return split_manifest


def _materialize_records(split: str, repo_root: Path) -> list[dict[str, Any]]:
    if split == "final":
        raise VPMValidationError(
            "final split materialization is prohibited by the sealed plan"
        )
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
        reference.append(
            _profile_provider(
                provider_id=provider_id,
                records=records,
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
                implementation="reference",
            )
        )
        optimized.append(
            _profile_provider(
                provider_id=provider_id,
                records=records,
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
                implementation="optimized",
            )
        )
    payload = runtime_profile_payload(
        provider_scope=provider,
        provider_ids=provider_ids,
        profile_frame_count=len(records),
        reference=reference,
        optimized=optimized,
    )
    _write_json(
        root_artifact_path(output_dir, "runtime-profile-reference.json"),
        {"profiles": reference, "profile_frame_count": len(records)},
    )
    _write_json(
        root_artifact_path(output_dir, "runtime-profile-optimized.json"),
        {"profiles": optimized, "profile_frame_count": len(records)},
    )
    _write_json(root_artifact_path(output_dir, "runtime-comparison.json"), payload)
    _write_text(
        root_artifact_path(output_dir, "runtime-profile-reference.md"),
        runtime_profile_reference(reference),
    )
    _write_text(
        root_artifact_path(output_dir, "runtime-profile-optimized.md"),
        runtime_profile_optimized(optimized),
    )
    return payload


def build_split(
    split: str,
    output_dir: Path,
    repo_root: Path,
    *,
    progress_observer: SplitBuildProgressObserver | None = None,
) -> dict[str, Any]:
    runtime = _build_durable_runtime(output_dir)
    identity = runtime.video_action_set.load_identity(repo_root)
    prototypes = canonical_prototypes()
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_actions = {
        str(row_id): lookup.choose(str(row_id)) for row_id in policy.source.row_ids
    }
    policy_artifact_id = policy.artifact_id
    reachability_tile = _load_reachability_tile(repo_root)
    plans = _episode_plans_for_split(
        identity, split, [str(row_id) for row_id in policy.source.row_ids], row_actions
    )
    saved_plans = runtime.video_action_set.save_episode_plans(
        tuple(EpisodePlanDTO.from_dict(plan) for plan in plans)
    )
    plans = [dto.to_dict() for dto in saved_plans]
    records = _materialize_records(split, repo_root)
    runtime.video_action_set.save_observation_records(records)
    records = list(
        runtime.video_action_set.list_observation_records(
            benchmark_seed_digest=identity.seed_digest, split=split, include_pixels=True
        )
    )
    scored_rows = measure_record_collection(
        records,
        prototypes,
        policy_artifact_id,
        reachability_tile=reachability_tile,
        row_actions=row_actions,
        split=split,
        progress_observer=progress_observer,
    )
    _write_jsonl(
        split_artifact_path(output_dir, split, "frame-metadata.jsonl"),
        [
            {key: value for key, value in record.items() if key != "pixels"}
            for record in records
        ],
    )
    _write_jsonl(
        split_artifact_path(output_dir, split, "provider-evidence.jsonl"),
        cast(list[Mapping[str, Any]], scored_rows),
    )
    manifest = {
        "split": split,
        "observation_count": len(records),
        "provider_frame_record_count": len(scored_rows),
        "frame_digest": _sha256(
            [
                {key: value for key, value in record.items() if key != "pixels"}
                for record in records
            ]
        ),
        "provider_evidence_digest": _sha256(scored_rows),
    }
    _write_json(split_manifest_path(output_dir, split), manifest)
    closure = _family_closure_report(
        split=split,
        records=records,
        plans=plans,
        identity=identity,
        reachability_tile=reachability_tile,
        provider_rows=scored_rows,
    )
    _write_json(family_closure_path(output_dir, split), closure)
    if split == "selection":
        _write_json(
            root_artifact_path(output_dir, "family-closure-report.json"), closure
        )
    _write_observation_identity_manifest(output_dir)
    _write_split_overlap_audit(output_dir)
    _write_json(
        root_artifact_path(output_dir, "phase-access-audits.json"),
        _measured_phase_access_counts(output_dir),
    )
    return manifest


def _measured_phase_access_counts(output_dir: Path) -> dict[str, Any]:
    final_path = root_artifact_path(output_dir, "final-split-sealed-plan.json")
    final_plan = _read_json(final_path) if final_path.exists() else {}
    frame_rows = {
        split: _read_jsonl(
            split_artifact_path(output_dir, split, "frame-metadata.jsonl")
        )
        for split in ("development", "calibration", "selection", "final")
    }
    evidence_rows = {
        split: _read_jsonl(
            split_artifact_path(output_dir, split, "provider-evidence.jsonl")
        )
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
        split: _read_jsonl(
            split_artifact_path(output_dir, split, "frame-metadata.jsonl")
        )
        for split in ("development", "calibration", "selection")
    }
    payload = _build_observation_identity_manifest(frames)
    _write_json(
        root_artifact_path(output_dir, "observation-identity-manifest.json"), payload
    )


def _write_split_overlap_audit(output_dir: Path) -> None:
    split_rows = {
        split: _read_jsonl(
            split_artifact_path(output_dir, split, "frame-metadata.jsonl")
        )
        for split in ("development", "calibration", "selection")
    }
    final_path = root_artifact_path(output_dir, "final-split-sealed-plan.json")
    final_plan = _read_json(final_path) if final_path.exists() else {}
    payload = _build_split_overlap_audit(
        frame_rows_by_split=split_rows,
        final_plan=final_plan,
    )
    _write_json(root_artifact_path(output_dir, "split-overlap-audit.json"), payload)
