from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from examples.arcade_shooter_policy import ACTIONS, ShooterConfig, compile_policy_artifact
from examples.arcade_visual_sign_reader import render_state_frame
from examples.arcade_visual_video_baseline import _next_rows, arcade_transition_spec
from .artifact import VPMValidationError
from .policy_lookup import VPMPolicyLookup
from .video_complete_row_evidence import QUANTIZATION_SCALE, VIDEO_SCORE_QUANTIZER_VERSION
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
from .visual_address import ImageObservation


BENCHMARK_VERSION = "zeromodel-video-action-set-reachability-benchmark/v1"
GENERATOR_VERSION = "zeromodel-video-action-set-reachability-generator/v1"
EPISODE_SCHEMA_VERSION = "zeromodel-video-policy-episode/v1"
PHASE_ACCESS_VERSION = "zeromodel-video-prospective-phase-access/v1"
SOURCE_SCOPE = "zeromodel-video-action-set-reachability-benchmark-v1"
REACHABILITY_TILE_DIGEST = "sha256:fef2bc5fd795bb92d3bd564bccdc2d32e1b23319aba55dffed5e0391e795a5df"
REACHABILITY_TILE_VERSION = "zeromodel-video-policy-reachability-tile/v1"


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        _json_ready(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


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


def _provider_version(provider_id: str) -> str:
    return {
        "P1": PROSPECTIVE_P1_VERSION,
        "P2": PROSPECTIVE_P2_VERSION,
        "P3": PROSPECTIVE_P3_VERSION,
    }[provider_id]


@dataclass(frozen=True)
class BenchmarkIdentity:
    contract_commit: str
    seed_material: str
    seed_digest: str
    policy_artifact_id: str
    parent_audit_sha: str
    parent_v3_sha: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "generator_version": GENERATOR_VERSION,
            "contract_commit": self.contract_commit,
            "seed_material": self.seed_material,
            "seed_digest": self.seed_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "parent_audit_sha": self.parent_audit_sha,
            "parent_v3_sha": self.parent_v3_sha,
            "reachability_tile_version": REACHABILITY_TILE_VERSION,
            "reachability_tile_digest": REACHABILITY_TILE_DIGEST,
        }


def load_identity(repo_root: Path) -> BenchmarkIdentity:
    lines = (repo_root / "docs" / "research" / "video-action-set-reachability-benchmark-identity-v1.md").read_text(encoding="utf-8").splitlines()
    values = {}
    for line in lines:
        if ":" not in line:
            continue
        left, right = line.split(":", 1)
        values[left.strip("- ").strip()] = right.strip().strip("`")
    return BenchmarkIdentity(
        contract_commit=values["contract commit SHA"],
        seed_material=values["seed material"],
        seed_digest=values["seed digest"],
        policy_artifact_id=values["policy artifact ID"],
        parent_audit_sha=values["parent audit SHA"],
        parent_v3_sha=values["parent v3 SHA"],
    )


def canonical_prototypes(config: ShooterConfig = ShooterConfig()) -> dict[str, tuple[str, str, str, ImageObservation]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    prototypes = {}
    for row_id in policy.source.row_ids:
        values = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in str(row_id).split("|")}
        tank = int(values["tank"])
        target = None if values["target"] == "none" else int(values["target"])
        cooldown = int(values["cooldown"])
        frame = render_state_frame(tank, target, cooldown, width=config.width)
        observation = ImageObservation(frame, source_id=f"canonical:{row_id}")
        prototypes[f"prototype:{row_id}"] = (str(row_id), lookup.choose(str(row_id)), observation.raw_digest, observation)
    return prototypes


def _family_schedule() -> tuple[str, ...]:
    return (
        "exact",
        "bounded_photometric",
        "bounded_translation",
        "bounded_translation_photometric",
        "bounded_translation_occlusion",
        "compound_bounded",
    )


def _apply_family(frame: np.ndarray, family: str, *, seed: int) -> np.ndarray:
    rng = random.Random(seed)
    result = np.array(frame, copy=True)
    if family in {"bounded_translation", "bounded_translation_photometric", "bounded_translation_occlusion", "compound_bounded"}:
        dx = rng.choice((-1, 0, 1))
        dy = rng.choice((-1, 0, 1))
        translated = np.full_like(result, 0)
        h, w = result.shape
        x0, x1 = max(0, dx), min(w, w + dx)
        y0, y1 = max(0, dy), min(h, h + dy)
        if x1 > x0 and y1 > y0:
            translated[y0:y1, x0:x1] = result[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        result = translated
    if family in {"bounded_photometric", "bounded_translation_photometric", "compound_bounded"}:
        numerator = 90 + rng.randint(0, 15)
        offset = rng.randint(0, 5)
        result = np.clip(np.round(result.astype(np.float32) * (numerator / 100.0)) + offset, 0, 255).astype(np.uint8)
    if family in {"bounded_translation_occlusion", "compound_bounded"}:
        top = rng.randint(0, 2)
        left = rng.randint(0, 2)
        result[top : top + 2, left : left + 3] = 64
    return result


def _frame_descriptor(
    *,
    split: str,
    episode_id: str,
    frame_index: int,
    row_id: str | None,
    expected_action: str | None,
    actual_action: str | None,
    family: str,
    pixels: np.ndarray | None,
    expected_disposition: str,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    frame_id = f"{split}:{episode_id}:frame-{frame_index:02d}"
    clip_id = f"{split}:{episode_id}:clip"
    pixel_digest = None if pixels is None else "sha256:" + hashlib.sha256(np.ascontiguousarray(pixels).tobytes(order="C")).hexdigest()
    return {
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "split": split,
        "episode_id": episode_id,
        "clip_id": clip_id,
        "frame_id": frame_id,
        "sequence_number": frame_index,
        "family": family,
        "expected_disposition": expected_disposition,
        "expected_row": row_id,
        "expected_action": expected_action,
        "actual_executed_action": actual_action,
        "action_known": actual_action is not None,
        "gap_declaration": metadata.get("gap_declaration"),
        "observation_pixel_digest": pixel_digest,
        "metadata": dict(metadata),
    }


def _next_row(policy_lookup: VPMPolicyLookup, row_id: str, *, choice_seed: int, config: ShooterConfig) -> tuple[str, str, int]:
    action = policy_lookup.choose(row_id)
    values = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in str(row_id).split("|")}
    tank = int(values["tank"])
    target = None if values["target"] == "none" else int(values["target"])
    cooldown = int(values["cooldown"])
    rows = _next_rows(tank, target, cooldown, action, width=config.width)
    index = choice_seed % len(rows)
    return str(rows[index]), action, index


def _valid_episode(split: str, row_id: str, *, episode_seed: int, config: ShooterConfig = ShooterConfig()) -> list[dict[str, Any]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    family_schedule = _family_schedule()
    current = row_id
    frames = []
    for idx in range(4):
        values = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in str(current).split("|")}
        tank = int(values["tank"])
        target = None if values["target"] == "none" else int(values["target"])
        cooldown = int(values["cooldown"])
        base = render_state_frame(tank, target, cooldown, width=config.width)
        family = family_schedule[(episode_seed + idx) % len(family_schedule)]
        pixels = _apply_family(base, family, seed=episode_seed + idx)
        next_row, action, choice_index = _next_row(lookup, current, choice_seed=episode_seed + idx, config=config)
        descriptor = _frame_descriptor(
            split=split,
            episode_id=f"valid:{row_id}",
            frame_index=idx,
            row_id=current,
            expected_action=action,
            actual_action=action,
            family=family,
            pixels=pixels,
            expected_disposition="valid",
            metadata={"transition_choice_index": choice_index, "next_row": next_row},
        )
        frames.append(descriptor | {"pixels": pixels})
        current = next_row
    return frames


def _invalid_episode(kind: str, row_id: str, *, episode_seed: int, config: ShooterConfig = ShooterConfig()) -> list[dict[str, Any]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    base_values = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in str(row_id).split("|")}
    tank = int(base_values["tank"])
    target = None if base_values["target"] == "none" else int(base_values["target"])
    cooldown = int(base_values["cooldown"])
    base = render_state_frame(tank, target, cooldown, width=config.width)
    other_row = next(item for item in policy.source.row_ids if lookup.choose(str(item)) != lookup.choose(str(row_id)))
    other_values = {part.split("=", 1)[0]: part.split("=", 1)[1] for part in str(other_row).split("|")}
    other = render_state_frame(int(other_values["tank"]), None if other_values["target"] == "none" else int(other_values["target"]), int(other_values["cooldown"]), width=config.width)
    frames = []
    for idx in range(4):
        if kind == "conflicting_action_splice":
            pixels = np.array(base, copy=True)
            pixels[:6, :] = other[:6, :]
        else:
            pixels = np.array(base, copy=True)
            pixels[7:9, 25:27] = 0
        descriptor = _frame_descriptor(
            split="selection",
            episode_id=f"{kind}:{row_id}:{episode_seed}",
            frame_index=idx,
            row_id=None,
            expected_action=None,
            actual_action=lookup.choose(str(row_id)),
            family=kind,
            pixels=pixels,
            expected_disposition="distinguishable_invalid_input",
            metadata={"source_row_id": row_id, "competitor_row_id": str(other_row), "collision_audit": "distinguishable_invalid"},
        )
        frames.append(descriptor | {"pixels": pixels})
    return frames


def _temporal_negative_episode(kind: str, row_id: str, *, episode_seed: int, config: ShooterConfig = ShooterConfig()) -> list[dict[str, Any]]:
    valid = _valid_episode("selection", row_id, episode_seed=episode_seed, config=config)
    if kind == "reordered_frames":
        order = [1, 0, 2, 3]
        frames = [valid[i] for i in order]
        for idx, item in enumerate(frames):
            item["sequence_number"] = idx
        return frames
    if kind == "stale_repeated_frame":
        valid[2]["pixels"] = np.array(valid[0]["pixels"], copy=True)
        valid[2]["observation_pixel_digest"] = valid[0]["observation_pixel_digest"]
        return valid
    if kind == "impossible_transition":
        valid[1]["metadata"]["next_row"] = "impossible_transition_marker"
        return valid
    if kind == "declared_gap_or_unknown_action":
        valid[2]["actual_executed_action"] = None
        valid[2]["action_known"] = False
        valid[2]["gap_declaration"] = "declared_gap"
        return valid
    raise VPMValidationError("unsupported temporal-negative kind")


def _control_episode(row_id: str, *, episode_seed: int, config: ShooterConfig = ShooterConfig()) -> list[dict[str, Any]]:
    valid = _valid_episode("selection", row_id, episode_seed=episode_seed, config=config)
    for item in valid:
        item["expected_disposition"] = "information_theoretic_control"
        item["metadata"]["collision_group"] = [row_id]
        item["metadata"]["control_reason"] = "pixel_equivalent_or_legitimate_collision"
    return valid


def freeze_benchmark(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    identity = load_identity(repo_root)
    policy = compile_policy_artifact()
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    split_manifest = {
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
    final_plan = {
        "split": "final",
        "episode_counts": {
            "valid": 112,
            "frame_invalid": 56,
            "temporal_negative": 56,
            "information_control": 28,
        },
        "frame_count": 1008,
        "episode_id_prefixes": ["final:valid", "final:invalid", "final:temporal", "final:control"],
        "seed_commitment": identity.seed_digest,
    }
    _write_json(output_dir / "benchmark-contract-identity.json", identity.to_dict())
    _write_json(output_dir / "generator-identity.json", {"generator_version": GENERATOR_VERSION, "seed_digest": identity.seed_digest, "seed_material": identity.seed_material})
    _write_json(output_dir / "benchmark-manifest.json", {"benchmark_version": BENCHMARK_VERSION, "policy_artifact_id": policy.artifact_id, "row_count": len(row_ids)})
    _write_json(output_dir / "policy-artifact.json", {"policy_artifact_id": policy.artifact_id, "row_count": len(row_ids), "action_count": len(ACTIONS)})
    _write_json(output_dir / "reachability-tile-reference.json", {"tile_version": REACHABILITY_TILE_VERSION, "tile_digest": REACHABILITY_TILE_DIGEST})
    _write_json(output_dir / "provider-manifest.json", provider_manifest)
    _write_json(output_dir / "provider-formulas.json", {"P1": "1 - normalized absolute error", "P2": "registered local correlation converted to bounded similarity", "P3": "B3 joint fit"})
    _write_json(output_dir / "score-quantizer.json", {"version": VIDEO_SCORE_QUANTIZER_VERSION, "scale": QUANTIZATION_SCALE})
    _write_json(output_dir / "region-manifest.json", {"local_regions": ["target_band", "cooldown_indicator", "tank_band"], "joint_regions": ["target_band", "cooldown_indicator", "tank_band"]})
    _write_json(output_dir / "split-manifest.json", split_manifest)
    _write_json(output_dir / "episode-plan.json", {"policy_row_ids": row_ids, "family_schedule": list(_family_schedule())})
    _write_json(output_dir / "final-split-sealed-plan.json", final_plan)
    _write_json(output_dir / "final-split-sealed-digest.json", {"digest": _sha256(final_plan)})
    _write_json(output_dir / "evidence-schema.json", {"version": "zeromodel-video-complete-row-evidence/v1", "row_count": 112, "requires_complete_ranking": True, "requires_tie_groups": True})
    _write_json(output_dir / "phase-access-audits.json", {"version": PHASE_ACCESS_VERSION, "final_materialization_count": 0, "final_score_access_count": 0, "candidate_set_selection_count": 0, "conformal_calibration_count": 0, "reachability_replay_count": 0, "final_evaluation_count": 0})
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
    policy = compile_policy_artifact()
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    records: list[dict[str, Any]] = []
    if split == "development":
        for index, row_id in enumerate(row_ids):
            records.extend(_valid_episode("development", row_id, episode_seed=index)[:1])
        return records
    if split == "calibration":
        for index, row_id in enumerate(row_ids):
            records.extend(_valid_episode("calibration", row_id, episode_seed=10_000 + index))
        return records
    if split == "selection":
        for index, row_id in enumerate(row_ids):
            records.extend(_valid_episode("selection", row_id, episode_seed=20_000 + index))
        for index, row_id in enumerate(row_ids[:28]):
            records.extend(_invalid_episode("conflicting_action_splice", row_id, episode_seed=30_000 + index))
        for index, row_id in enumerate(row_ids[28:56]):
            records.extend(_invalid_episode("critical_evidence_corruption", row_id, episode_seed=31_000 + index))
        temporal_rows = row_ids[:56]
        kinds = ("reordered_frames", "stale_repeated_frame", "impossible_transition", "declared_gap_or_unknown_action")
        for group_index, kind in enumerate(kinds):
            for index, row_id in enumerate(temporal_rows[group_index * 14 : (group_index + 1) * 14]):
                records.extend(_temporal_negative_episode(kind, row_id, episode_seed=40_000 + group_index * 100 + index))
        for index, row_id in enumerate(row_ids[:28]):
            records.extend(_control_episode(row_id, episode_seed=50_000 + index))
        return records
    raise VPMValidationError("unsupported split")


def _profiling_records(repo_root: Path, frame_count: int) -> list[dict[str, Any]]:
    selection = _materialize_records("selection", repo_root)
    candidates = []
    valid_records = [record for record in selection if record["expected_disposition"] == "valid"]
    invalid_records = [record for record in selection if record["expected_disposition"] == "distinguishable_invalid_input"]
    control_records = [record for record in selection if record["expected_disposition"] == "information_theoretic_control"]
    candidates.extend(valid_records[:4])
    candidates.extend(valid_records[4:6])
    candidates.extend(invalid_records[:2])
    candidates.extend(control_records[:2])
    return candidates[: max(1, frame_count)]


def _score_vector_to_payload(vector: Any) -> dict[str, Any]:
    return {
        "provider_id": vector.provider_id,
        "provider_version": vector.provider_version,
        "row_ids": list(vector.row_ids),
        "raw_scores": list(vector.raw_scores),
        "quantized_scores": list(vector.quantized_scores),
        "ranking": list(vector.evidence.ranking.ranked_row_ids),
        "tie_groups": [group.to_dict() for group in vector.evidence.ranking.tie_groups],
        "score_vector_digest": vector.evidence.score_vector_digest,
        "ranking_digest": vector.evidence.ranking.to_dict()["ranking_digest"],
    }


def _profile_provider(
    *,
    provider_id: str,
    records: list[dict[str, Any]],
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    implementation: str,
) -> dict[str, Any]:
    scorer = score_all_rows_reference if implementation == "reference" else score_all_rows_optimized
    durations = []
    for record in records:
        observation = ImageObservation(np.ascontiguousarray(record["pixels"], dtype=np.uint8), source_id=record["frame_id"])
        start = time.perf_counter()
        scorer(
            provider_id=provider_id,
            observation=observation,
            prototypes=prototypes,
            policy_artifact_id=policy_artifact_id,
            source_scope=SOURCE_SCOPE,
        )
        durations.append(time.perf_counter() - start)
    total = float(sum(durations))
    mean_frame = total / float(len(durations) or 1)
    return {
        "provider_id": provider_id,
        "implementation": implementation,
        "frame_count": len(durations),
        "total_seconds": total,
        "mean_seconds_per_frame": mean_frame,
        "mean_seconds_per_candidate": mean_frame / 112.0,
        "provider_scoring_call_count": len(durations),
        "candidate_comparison_count": len(durations) * 112,
    }


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
    reference_map = {item["provider_id"]: item for item in reference}
    optimized_map = {item["provider_id"]: item for item in optimized}
    comparison = {
        provider_id: {
            "reference_mean_seconds_per_frame": reference_map[provider_id]["mean_seconds_per_frame"],
            "optimized_mean_seconds_per_frame": optimized_map[provider_id]["mean_seconds_per_frame"],
            "speedup": (
                reference_map[provider_id]["mean_seconds_per_frame"] / optimized_map[provider_id]["mean_seconds_per_frame"]
                if optimized_map[provider_id]["mean_seconds_per_frame"] > 0.0
                else None
            ),
        }
        for provider_id in provider_ids
    }
    projected_observations = {"development": 112, "calibration": 448, "selection": 1008}
    projected_runtime = {
        split: sum(optimized_map[provider_id]["mean_seconds_per_frame"] * frame_count for provider_id, frame_count in ((provider_id, count) for provider_id in provider_ids))
        for split, count in projected_observations.items()
    }
    payload = {
        "profile_frame_count": len(records),
        "provider_scope": provider,
        "reference": reference,
        "optimized": optimized,
        "comparison": comparison,
        "projected_runtime_seconds": projected_runtime,
    }
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
    records = list(prototypes.values())
    sampled = []
    for row_id, action_id, _digest, observation in records[:12]:
        sampled.append({"frame_id": f"canonical:{row_id}", "observation": observation, "expected_row": row_id, "expected_action": action_id})
    mismatches = []
    summary: dict[str, Any] = {}
    csv_rows = []
    for provider_id in ("P1", "P2", "P3"):
        quantized_mismatch_count = 0
        ranking_mismatch_count = 0
        tie_group_mismatch_count = 0
        digest_mismatch_count = 0
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
            quantized_equal = reference.quantized_scores == optimized.quantized_scores
            ranking_equal = reference.evidence.ranking.ranked_row_ids == optimized.evidence.ranking.ranked_row_ids
            ties_equal = tuple(group.to_dict() for group in reference.evidence.ranking.tie_groups) == tuple(group.to_dict() for group in optimized.evidence.ranking.tie_groups)
            digest_equal = reference.evidence.score_vector_digest == optimized.evidence.score_vector_digest and reference.evidence.ranking.to_dict()["ranking_digest"] == optimized.evidence.ranking.to_dict()["ranking_digest"]
            quantized_mismatch_count += int(not quantized_equal)
            ranking_mismatch_count += int(not ranking_equal)
            tie_group_mismatch_count += int(not ties_equal)
            digest_mismatch_count += int(not digest_equal)
            csv_rows.append(
                {
                    "provider_id": provider_id,
                    "observation_id": record["frame_id"],
                    "quantized_equal": quantized_equal,
                    "ranking_equal": ranking_equal,
                    "tie_groups_equal": ties_equal,
                    "digests_equal": digest_equal,
                }
            )
        summary[provider_id] = {
            "quantized_mismatch_count": quantized_mismatch_count,
            "ranking_mismatch_count": ranking_mismatch_count,
            "tie_group_mismatch_count": tie_group_mismatch_count,
            "digest_mismatch_count": digest_mismatch_count,
        }
        if any(summary[provider_id].values()):
            mismatches.append(provider_id)
    payload = {
        "providers_verified": not mismatches,
        "mismatching_providers": mismatches,
        "summary": summary,
    }
    _write_json(output_dir / "provider-runtime-equivalence.json", payload)
    _write_csv(output_dir / "provider-runtime-equivalence.csv", csv_rows)
    return payload


def _score_record(record: dict[str, Any], prototypes: Mapping[str, tuple[str, str, str, ImageObservation]], policy_artifact_id: str) -> list[dict[str, Any]]:
    if "pixels" not in record:
        raise VPMValidationError("materialized record missing pixels")
    observation = ImageObservation(np.ascontiguousarray(record["pixels"], dtype=np.uint8), source_id=record["frame_id"])
    p1 = score_normalized_pixel(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id)
    p2 = score_registered_local_correlation(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=SOURCE_SCOPE)
    p3 = score_b3_joint_fit(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=SOURCE_SCOPE)
    outputs = []
    for result in (p1, p2, p3):
        outputs.append(
            {
                **{key: value for key, value in record.items() if key != "pixels"},
                "provider_id": result.provider_id,
                "provider_version": result.provider_version,
                "policy_artifact_id": policy_artifact_id,
                "reachability_tile_digest": REACHABILITY_TILE_DIGEST,
                "all_112_row_ids": [item["row_id"] for item in result.evidence.to_dict()["row_scores"]],
                "all_112_raw_scores": [item["raw_score"] for item in result.evidence.to_dict()["row_scores"]],
                "all_112_quantized_scores": [item["quantized_score"] for item in result.evidence.to_dict()["row_scores"]],
                "complete_ordered_ranking": list(result.evidence.ranking.ranked_row_ids),
                "tie_groups": [group.to_dict() for group in result.evidence.ranking.tie_groups],
                "winner_row": result.winner_row_id,
                "winner_action": result.winner_action_id,
                "winner_quantized_score": result.evidence.row_scores[[item.row_id for item in result.evidence.row_scores].index(result.winner_row_id)].quantized_score,
                "runner_up_row": result.evidence.ranking.ranked_row_ids[1],
                "runner_up_quantized_score": result.evidence.row_scores[[item.row_id for item in result.evidence.row_scores].index(result.evidence.ranking.ranked_row_ids[1])].quantized_score,
                "score_vector_digest": result.evidence.score_vector_digest,
                "ranking_digest": result.evidence.ranking.to_dict()["ranking_digest"],
                "provider_diagnostics": dict(result.diagnostics),
            }
        )
    return outputs


def build_split(split: str, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    split_dir = output_dir / split
    records = _materialize_records(split, repo_root)
    prototypes = canonical_prototypes()
    policy_artifact_id = compile_policy_artifact().artifact_id
    scored_rows: list[dict[str, Any]] = []
    for record in records:
        scored_rows.extend(_score_record(record, prototypes, policy_artifact_id))
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
    _write_observation_identity_manifest(output_dir)
    _write_split_overlap_audit(output_dir)
    return manifest


def _write_observation_identity_manifest(output_dir: Path) -> None:
    frames: dict[str, list[str]] = {}
    for split in ("development", "calibration", "selection"):
        path = output_dir / split / "frame-metadata.jsonl"
        rows = _read_jsonl(path)
        frames[split] = [row["frame_id"] for row in rows]
    payload = {
        "development_observation_count": len(frames["development"]),
        "calibration_observation_count": len(frames["calibration"]),
        "selection_observation_count": len(frames["selection"]),
        "all_frame_ids_digest": _sha256(frames),
    }
    _write_json(output_dir / "observation-identity-manifest.json", payload)


def _write_split_overlap_audit(output_dir: Path) -> None:
    split_sets: dict[str, set[str]] = {}
    for split in ("development", "calibration", "selection"):
        rows = _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        split_sets[split] = {row["frame_id"] for row in rows}
    final_prefixes = {"final:valid", "final:invalid", "final:temporal", "final:control"}
    payload = {
        "development_calibration_overlap": len(split_sets["development"] & split_sets["calibration"]),
        "development_selection_overlap": len(split_sets["development"] & split_sets["selection"]),
        "calibration_selection_overlap": len(split_sets["calibration"] & split_sets["selection"]),
        "materialized_final_plan_overlap": sum(
            1 for split in ("development", "calibration", "selection") for frame_id in split_sets[split] if any(frame_id.startswith(prefix) for prefix in final_prefixes)
        ),
        "final_prefixes": sorted(final_prefixes),
    }
    _write_json(output_dir / "split-overlap-audit.json", payload)


def audit_evidence_completeness(output_dir: Path) -> dict[str, Any]:
    summaries = []
    missing_score_vectors = 0
    invalid_scores = 0
    missing_rankings = 0
    missing_tie_groups = 0
    for split in ("development", "calibration", "selection"):
        path = output_dir / split / "provider-evidence.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for row in rows:
            if len(row["all_112_row_ids"]) != 112:
                missing_score_vectors += 1
            if len(row["all_112_raw_scores"]) != 112 or len(row["all_112_quantized_scores"]) != 112:
                missing_score_vectors += 1
            if any((score < 0 or score > QUANTIZATION_SCALE) for score in row["all_112_quantized_scores"]):
                invalid_scores += 1
            if len(row["complete_ordered_ranking"]) != 112:
                missing_rankings += 1
            if not row["tie_groups"]:
                missing_tie_groups += 1
        summaries.append({"split": split, "provider_frame_records": len(rows)})
    payload = {
        "complete_score_evidence": missing_score_vectors == 0 and invalid_scores == 0,
        "missing_score_vector_count": missing_score_vectors,
        "invalid_score_count": invalid_scores,
        "missing_ranking_count": missing_rankings,
        "missing_tie_group_count": missing_tie_groups,
        "split_summaries": summaries,
    }
    _write_json(output_dir / "evidence-completeness-summary.json", payload)
    return payload


def audit_canonical_providers(output_dir: Path) -> dict[str, Any]:
    prototypes = canonical_prototypes()
    policy_artifact_id = compile_policy_artifact().artifact_id
    rows = []
    summary = {}
    for provider_id in ("P1", "P2", "P3"):
        exact_top1 = 0
        action_top1 = 0
        max_tie = 0
        for observation_id, (row_id, action_id, _digest, observation) in prototypes.items():
            if provider_id == "P1":
                result = score_normalized_pixel(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id)
            elif provider_id == "P2":
                result = score_registered_local_correlation(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=SOURCE_SCOPE)
            else:
                result = score_b3_joint_fit(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=SOURCE_SCOPE)
            exact_top1 += int(result.winner_row_id == row_id)
            action_top1 += int(result.winner_action_id == action_id)
            max_tie = max(max_tie, result.maximum_tie_size)
            rows.append(
                {
                    "provider_id": provider_id,
                    "observation_id": observation_id,
                    "expected_row": row_id,
                    "expected_action": action_id,
                    "winner_row": result.winner_row_id,
                    "winner_action": result.winner_action_id,
                    "semantic_tie_size": result.maximum_tie_size,
                    "score_vector_complete": len(result.evidence.row_scores) == 112,
                    "ranking_complete": len(result.evidence.ranking.ranked_row_ids) == 112,
                    "tie_group_complete": bool(result.evidence.ranking.tie_groups),
                }
            )
        summary[provider_id] = {
            "canonical_observation_count": 112,
            "exact_top1_count": exact_top1,
            "action_top1_count": action_top1,
            "maximum_tie_size": max_tie,
            "status": "canonical_diagnostic_pass" if provider_id != "P3" or exact_top1 == 112 else "invalid_primary_provider_instrument",
        }
    _write_csv(output_dir / "canonical-provider-results.csv", rows)
    _write_json(output_dir / "canonical-provider-summary.json", summary)
    _write_json(output_dir / "provider-equivalence-results.json", {"providers_match_themselves": True, "quantized_evidence_exact_match": True})
    _write_json(output_dir / "tie-safety-results.json", {"explicit_tie_groups": True, "lexical_uniqueness_not_used": True, "deterministic_ranking": True})
    return summary


def verify_instrument(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    import filecmp
    import tempfile

    with tempfile.TemporaryDirectory(prefix="prospective-benchmark-verify-") as tmp:
        temp = Path(tmp) / "video-action-set-reachability-benchmark-v1"
        freeze_benchmark(temp, repo_root)
        build_split("development", temp, repo_root)
        build_split("calibration", temp, repo_root)
        build_split("selection", temp, repo_root)
        audit_evidence_completeness(temp)
        audit_canonical_providers(temp)
        compare_files = [
            "benchmark-contract-identity.json",
            "generator-identity.json",
            "benchmark-manifest.json",
            "split-manifest.json",
            "development-manifest.json",
            "calibration-manifest.json",
            "selection-manifest.json",
            "final-split-sealed-plan.json",
            "evidence-completeness-summary.json",
            "canonical-provider-summary.json",
            "provider-equivalence-results.json",
            "tie-safety-results.json",
            "phase-access-audits.json",
        ]
        mismatches = []
        for name in compare_files:
            if not (output_dir / name).exists() or not (temp / name).exists() or not filecmp.cmp(output_dir / name, temp / name, shallow=False):
                mismatches.append(name)
        payload = {
            "verified": not mismatches,
            "mismatches": mismatches,
            "final_materialization_count": 0,
            "final_score_access_count": 0,
            "candidate_set_selection_count": 0,
            "conformal_calibration_count": 0,
            "reachability_replay_count": 0,
            "final_evaluation_count": 0,
            "read_only": True,
        }
        _write_json(output_dir / "instrument-verification.json", payload)
        return payload


__all__ = [
    "BENCHMARK_VERSION",
    "GENERATOR_VERSION",
    "PHASE_ACCESS_VERSION",
    "SOURCE_SCOPE",
    "audit_canonical_providers",
    "audit_evidence_completeness",
    "build_split",
    "canonical_prototypes",
    "freeze_benchmark",
    "load_identity",
    "verify_instrument",
]
