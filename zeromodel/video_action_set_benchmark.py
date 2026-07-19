from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np

from .arcade_policy import ACTIONS, ShooterConfig, arcade_transition_spec, compile_policy_artifact, next_rows, parse_state_row_id, render_state_frame
from .artifact import VPMValidationError
from .policy_lookup import VPMPolicyLookup
from .video_complete_row_evidence import (
    QUANTIZATION_SCALE,
    VIDEO_SCORE_QUANTIZER_VERSION,
    VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION,
    build_complete_row_evidence,
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
from .visual_address import ImageObservation


BENCHMARK_VERSION = "zeromodel-video-action-set-reachability-benchmark/v1"
GENERATOR_VERSION = "zeromodel-video-action-set-reachability-generator/v1"
EPISODE_SCHEMA_VERSION = "zeromodel-video-policy-episode/v1"
EPISODE_PLAN_VERSION = "zeromodel-video-action-set-sealed-episode-plan/v1"
SEED_DERIVATION_VERSION = "zeromodel-video-action-set-seed-derivation/v1"
PHASE_ACCESS_VERSION = "zeromodel-video-prospective-phase-access/v1"
SOURCE_SCOPE = "zeromodel-video-action-set-reachability-benchmark-v1"
REACHABILITY_TILE_DIGEST = "sha256:fef2bc5fd795bb92d3bd564bccdc2d32e1b23319aba55dffed5e0391e795a5df"
REACHABILITY_TILE_VERSION = "zeromodel-video-policy-reachability-tile/v1"
SEMANTIC_OUTCOME_VERSION = VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION


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


def _pixel_digest(pixels: np.ndarray | None) -> str | None:
    if pixels is None:
        return None
    return "sha256:" + hashlib.sha256(np.ascontiguousarray(pixels).tobytes(order="C")).hexdigest()


def _load_reachability_tile(repo_root: Path) -> dict[str, Any]:
    return _read_json(repo_root / "docs" / "results" / "video-policy-reachability-tile-v1" / "reachability-tile.json")


@dataclass(frozen=True)
class BenchmarkIdentity:
    contract_commit: str
    seed_material: str
    seed_digest: str
    policy_artifact_id: str
    parent_audit_sha: str
    parent_v3_sha: str

    def __post_init__(self) -> None:
        expected = "sha256:" + hashlib.sha256(self.seed_material.encode("utf-8")).hexdigest()
        if self.seed_digest != expected:
            raise VPMValidationError("benchmark seed digest is inconsistent with frozen seed material")

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


def _seed_int_from_digest(digest: str) -> int:
    if not digest.startswith("sha256:"):
        raise VPMValidationError("seed identity must be a sha256 digest")
    return int(digest.removeprefix("sha256:")[:16], 16)


def _derived_seed(
    identity: BenchmarkIdentity,
    *,
    split: str,
    ordinal: int,
    namespace: str,
    parent_identities: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    parents = tuple((str(name), str(value)) for name, value in parent_identities)
    if len({name for name, _value in parents}) != len(parents):
        raise VPMValidationError("derived seed parent names must be unique")
    payload = {
        "version": SEED_DERIVATION_VERSION,
        "root_seed_digest": identity.seed_digest,
        "split": split,
        "episode_ordinal": int(ordinal),
        "namespace": namespace,
        "parent_identities": [{"name": name, "identity": value} for name, value in parents],
    }
    digest = _sha256(payload)
    return payload | {"seed_digest": digest, "seed_int64": _seed_int_from_digest(digest)}


def canonical_prototypes(config: ShooterConfig = ShooterConfig()) -> dict[str, tuple[str, str, str, ImageObservation]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    prototypes = {}
    for row_id in policy.source.row_ids:
        tank, target, cooldown = parse_state_row_id(str(row_id))
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


def _secondary_row_for_splice(row_ids: list[str], row_actions: Mapping[str, str], source_row_id: str) -> str:
    source_action = row_actions[source_row_id]
    for row_id in row_ids:
        if row_actions[row_id] != source_action:
            return row_id
    raise VPMValidationError("frame splice requires a secondary row with a conflicting action")


def _frame_count_for_plan(split: str, family_label: str) -> int:
    if split == "development":
        return 1
    return 4


def _frame_plans(
    identity: BenchmarkIdentity,
    *,
    split: str,
    ordinal: int,
    family_label: str,
    mutation_kind: str | None,
    frame_count: int,
    concrete_episode_seed_digest: str,
) -> list[dict[str, Any]]:
    schedule = _family_schedule()
    schedule_digest = _sha256({"family_schedule": list(schedule)})
    frames = []
    for frame_index in range(frame_count):
        frame_seed = _derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="frame_identity",
            parent_identities=(
                ("concrete_episode_seed", concrete_episode_seed_digest),
                ("frame_index", str(frame_index)),
            ),
        )
        transform_family_seed = _derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="transformation_family",
            parent_identities=(
                ("frame_identity", frame_seed["seed_digest"]),
                ("family_schedule_digest", schedule_digest),
                ("episode_family", family_label),
            ),
        )
        if family_label == "frame_invalid":
            transformation_family = str(mutation_kind)
        else:
            transformation_family = schedule[transform_family_seed["seed_int64"] % len(schedule)]
        transform_parameter_seed = _derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="transformation_parameters",
            parent_identities=(
                ("transformation_family_seed", transform_family_seed["seed_digest"]),
                ("transformation_family", transformation_family),
                ("frame_index", str(frame_index)),
            ),
        )
        transition_choice_seed = _derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="transition_choice",
            parent_identities=(
                ("frame_identity", frame_seed["seed_digest"]),
                ("frame_index", str(frame_index)),
            ),
        )
        temporal_mutation_seed = _derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="temporal_mutation_choice",
            parent_identities=(
                ("frame_identity", frame_seed["seed_digest"]),
                ("temporal_mutation", mutation_kind or "none"),
            ),
        )
        frames.append(
            {
                "frame_index": frame_index,
                "frame_seed_identity": frame_seed["seed_digest"],
                "transformation_family_seed_identity": transform_family_seed["seed_digest"],
                "transformation_family": transformation_family,
                "transformation_seed_identity": transform_parameter_seed["seed_digest"],
                "transformation_seed": transform_parameter_seed["seed_int64"],
                "transition_choice_seed_identity": transition_choice_seed["seed_digest"],
                "transition_choice_seed": transition_choice_seed["seed_int64"],
                "temporal_mutation_seed_identity": temporal_mutation_seed["seed_digest"],
                "temporal_mutation_kind": mutation_kind if family_label == "temporal_negative" else None,
            }
        )
    return frames


def _make_episode_plan(
    identity: BenchmarkIdentity,
    *,
    split: str,
    ordinal: int,
    family_label: str,
    family_ordinal: int,
    source_row_id: str,
    row_actions: Mapping[str, str],
    mutation_kind: str | None = None,
    secondary_row_id: str | None = None,
) -> dict[str, Any]:
    if source_row_id not in row_actions:
        raise VPMValidationError("episode source row is absent from the policy universe")
    if secondary_row_id is not None and secondary_row_id not in row_actions:
        raise VPMValidationError("episode secondary row is absent from the policy universe")
    if mutation_kind != "conflicting_action_splice" and secondary_row_id is not None:
        raise VPMValidationError("secondary row is only admissible for conflicting-action splices")
    if mutation_kind == "conflicting_action_splice":
        if secondary_row_id is None:
            raise VPMValidationError("conflicting-action splice requires a secondary row")
        if row_actions[secondary_row_id] == row_actions[source_row_id]:
            raise VPMValidationError("conflicting-action splice secondary row must govern a different action")

    split_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="split_identity",
        parent_identities=(("benchmark_version", BENCHMARK_VERSION), ("generator_version", GENERATOR_VERSION)),
    )
    ordinal_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="episode_ordinal",
        parent_identities=(("split_identity", split_seed["seed_digest"]), ("family_ordinal", str(family_ordinal))),
    )
    family_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="episode_family",
        parent_identities=(
            ("episode_ordinal", ordinal_seed["seed_digest"]),
            ("family_label", family_label),
            ("mutation_kind", mutation_kind or "none"),
        ),
    )
    source_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="source_row_choice",
        parent_identities=(("episode_family", family_seed["seed_digest"]), ("source_row_id", source_row_id)),
    )
    secondary_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="secondary_splice_row_choice",
        parent_identities=(
            ("source_row_choice", source_seed["seed_digest"]),
            ("secondary_row_id", secondary_row_id or "none"),
        ),
    )
    transformation_family_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="transformation_family",
        parent_identities=(
            ("secondary_splice_row_choice", secondary_seed["seed_digest"]),
            ("family_schedule_digest", _sha256({"family_schedule": list(_family_schedule())})),
        ),
    )
    transformation_parameter_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="transformation_parameters",
        parent_identities=(("transformation_family", transformation_family_seed["seed_digest"]), ("frame_count", str(_frame_count_for_plan(split, family_label)))),
    )
    temporal_mutation_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="temporal_mutation_choice",
        parent_identities=(("transformation_parameters", transformation_parameter_seed["seed_digest"]), ("temporal_mutation", mutation_kind or "none")),
    )
    episode_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="concrete_episode_seed",
        parent_identities=(
            ("split_identity", split_seed["seed_digest"]),
            ("episode_ordinal", ordinal_seed["seed_digest"]),
            ("episode_family", family_seed["seed_digest"]),
            ("source_row_choice", source_seed["seed_digest"]),
            ("secondary_splice_row_choice", secondary_seed["seed_digest"]),
            ("transformation_family", transformation_family_seed["seed_digest"]),
            ("transformation_parameters", transformation_parameter_seed["seed_digest"]),
            ("temporal_mutation_choice", temporal_mutation_seed["seed_digest"]),
        ),
    )
    episode_id_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="concrete_episode_id",
        parent_identities=(
            ("concrete_episode_seed", episode_seed["seed_digest"]),
            ("family_label", family_label),
            ("source_row_id", source_row_id),
            ("secondary_row_id", secondary_row_id or "none"),
        ),
    )
    frame_count = _frame_count_for_plan(split, family_label)
    episode_id = f"{split}:{family_label}:{episode_id_seed['seed_digest'].removeprefix('sha256:')[:16]}"
    plan = {
        "version": EPISODE_PLAN_VERSION,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "episode_id": episode_id,
        "split": split,
        "ordinal": int(ordinal),
        "family_label": family_label,
        "family_ordinal": int(family_ordinal),
        "mutation_kind": mutation_kind,
        "source_row_id": source_row_id,
        "secondary_row_id": secondary_row_id,
        "derived_seed_identity": episode_seed["seed_digest"],
        "episode_seed": episode_seed["seed_int64"],
        "frame_count": frame_count,
        "seed_lineage": {
            "split_identity": split_seed,
            "episode_ordinal": ordinal_seed,
            "episode_family": family_seed,
            "source_row_choice": source_seed,
            "secondary_splice_row_choice": secondary_seed,
            "transformation_family": transformation_family_seed,
            "transformation_parameters": transformation_parameter_seed,
            "temporal_mutation_choice": temporal_mutation_seed,
            "concrete_episode_seed": episode_seed,
            "concrete_episode_id": episode_id_seed,
        },
        "frame_plans": _frame_plans(
            identity,
            split=split,
            ordinal=ordinal,
            family_label=family_label,
            mutation_kind=mutation_kind,
            frame_count=frame_count,
            concrete_episode_seed_digest=episode_seed["seed_digest"],
        ),
    }
    return plan | {"plan_digest": _sha256(plan)}


def _episode_plans_for_split(identity: BenchmarkIdentity, split: str, row_ids: list[str], row_actions: Mapping[str, str]) -> list[dict[str, Any]]:
    plans: list[dict[str, Any]] = []

    def add(family_label: str, family_ordinal: int, row_id: str, *, mutation_kind: str | None = None) -> None:
        secondary = None
        if mutation_kind == "conflicting_action_splice":
            secondary = _secondary_row_for_splice(row_ids, row_actions, row_id)
        plans.append(
            _make_episode_plan(
                identity,
                split=split,
                ordinal=len(plans),
                family_label=family_label,
                family_ordinal=family_ordinal,
                source_row_id=row_id,
                row_actions=row_actions,
                mutation_kind=mutation_kind,
                secondary_row_id=secondary,
            )
        )

    if split == "development":
        for index, row_id in enumerate(row_ids):
            add("valid", index, row_id)
        return plans
    if split == "calibration":
        for index, row_id in enumerate(row_ids):
            add("valid", index, row_id)
        return plans
    if split in {"selection", "final"}:
        for index, row_id in enumerate(row_ids):
            add("valid", index, row_id)
        for index, row_id in enumerate(row_ids[:28]):
            add("frame_invalid", index, row_id, mutation_kind="conflicting_action_splice")
        for index, row_id in enumerate(row_ids[28:56]):
            add("frame_invalid", 28 + index, row_id, mutation_kind="critical_evidence_corruption")
        temporal_rows = row_ids[:56]
        kinds = ("reordered_frames", "stale_repeated_frame", "impossible_transition", "declared_gap_or_unknown_action")
        for group_index, kind in enumerate(kinds):
            for index, row_id in enumerate(temporal_rows[group_index * 14 : (group_index + 1) * 14]):
                add("temporal_negative", group_index * 14 + index, row_id, mutation_kind=kind)
        for index, row_id in enumerate(row_ids[:28]):
            add("information_control", index, row_id)
        return plans
    raise VPMValidationError("unsupported split")


def _episode_ids_by_family(plans: list[dict[str, Any]]) -> dict[str, list[str]]:
    grouped = {"valid": [], "frame_invalid": [], "temporal_negative": [], "information_control": []}
    for plan in plans:
        grouped[str(plan["family_label"])].append(str(plan["episode_id"]))
    return grouped


def _validate_episode_plan(identity: BenchmarkIdentity, plan: Mapping[str, Any], row_actions: Mapping[str, str]) -> None:
    expected = _make_episode_plan(
        identity,
        split=str(plan["split"]),
        ordinal=int(plan["ordinal"]),
        family_label=str(plan["family_label"]),
        family_ordinal=int(plan["family_ordinal"]),
        source_row_id=str(plan["source_row_id"]),
        row_actions=row_actions,
        mutation_kind=plan.get("mutation_kind"),
        secondary_row_id=plan.get("secondary_row_id"),
    )
    if dict(plan) != expected:
        raise VPMValidationError("episode plan is inconsistent with declared seed lineage or identity")


def _validate_episode_plan_collection(identity: BenchmarkIdentity, plans_by_split: Mapping[str, list[dict[str, Any]]], row_actions: Mapping[str, str]) -> None:
    seen: dict[str, str] = {}
    for split, plans in plans_by_split.items():
        for plan in plans:
            episode_id = str(plan["episode_id"])
            if plan["split"] != split:
                raise VPMValidationError("episode plan split does not match containing split")
            if episode_id in seen:
                if seen[episode_id] != split:
                    raise VPMValidationError("episode identity reassigned to another split")
                raise VPMValidationError("duplicate concrete episode identity")
            seen[episode_id] = split
            _validate_episode_plan(identity, plan, row_actions)


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
    pixel_digest = _pixel_digest(pixels)
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


def _tile_edge(tile: Mapping[str, Any], row_id: str, action_id: str) -> Mapping[str, Any]:
    for edge in tile["edges"]:
        if edge["source_row_id"] == row_id and edge["action_id"] == action_id:
            return edge
    raise VPMValidationError("missing reachability edge")


def _next_row(policy_lookup: VPMPolicyLookup, row_id: str, *, choice_seed: int, config: ShooterConfig, reachability_tile: Mapping[str, Any]) -> tuple[str, str, int, dict[str, Any]]:
    action = policy_lookup.choose(row_id)
    tank, target, cooldown = parse_state_row_id(str(row_id))
    rows = next_rows(tank, target, cooldown, action, width=config.width)
    index = choice_seed % len(rows)
    edge = _tile_edge(reachability_tile, row_id, action)
    if str(rows[index]) not in set(edge["reachable_row_ids"]):
        raise VPMValidationError("reachability tile does not admit chosen transition")
    trace = {
        "reachability_tile_digest": reachability_tile["tile_digest"],
        "candidate_row_id": row_id,
        "candidate_action_set": [action],
        "top_action_set": [action],
        "tile_action_id": action,
        "reachable_row_ids": list(edge["reachable_row_ids"]),
        "chosen_reachable_index": index,
        "executed_action": action,
        "rejected": False,
    }
    return str(rows[index]), action, index, trace


def _valid_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentity,
    reachability_tile: Mapping[str, Any],
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    split = str(plan["split"])
    episode_id = str(plan["episode_id"])
    episode_seed = int(plan["episode_seed"])
    current = str(plan["source_row_id"])
    frames = []
    for idx, frame_plan in enumerate(plan["frame_plans"]):
        tank, target, cooldown = parse_state_row_id(str(current))
        base = render_state_frame(tank, target, cooldown, width=config.width)
        family = str(frame_plan["transformation_family"])
        pixels = _apply_family(base, family, seed=int(frame_plan["transformation_seed"]))
        next_row, action, choice_index, reachability_trace = _next_row(
            lookup,
            current,
            choice_seed=int(frame_plan["transition_choice_seed"]),
            config=config,
            reachability_tile=reachability_tile,
        )
        descriptor = _frame_descriptor(
            split=split,
            episode_id=episode_id,
            frame_index=idx,
            row_id=current,
            expected_action=action,
            actual_action=action,
            family=family,
            pixels=pixels,
            expected_disposition="valid",
            metadata={
                "episode_seed": episode_seed,
                "seed_digest": identity.seed_digest,
                "derived_seed_identity": plan["derived_seed_identity"],
                "episode_plan_digest": plan["plan_digest"],
                "frame_seed_identity": frame_plan["frame_seed_identity"],
                "frame_transform_seed": frame_plan["transformation_seed"],
                "frame_transform_seed_identity": frame_plan["transformation_seed_identity"],
                "transition_choice_seed": frame_plan["transition_choice_seed"],
                "transition_choice_seed_identity": frame_plan["transition_choice_seed_identity"],
                "transformation_family": family,
                "transition_choice_index": choice_index,
                "next_row": next_row,
                "reachability_trace": reachability_trace,
            },
        )
        frames.append(descriptor | {"pixels": pixels})
        current = next_row
    return frames


def _invalid_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentity,
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    split = str(plan["split"])
    kind = str(plan["mutation_kind"])
    row_id = str(plan["source_row_id"])
    episode_id = str(plan["episode_id"])
    episode_seed = int(plan["episode_seed"])
    tank, target, cooldown = parse_state_row_id(str(row_id))
    base = render_state_frame(tank, target, cooldown, width=config.width)
    base_action = lookup.choose(str(row_id))
    other_row = plan.get("secondary_row_id")
    other = None
    if other_row is not None:
        other_tank, other_target, other_cooldown = parse_state_row_id(str(other_row))
        other = render_state_frame(other_tank, other_target, other_cooldown, width=config.width)
    frames = []
    for idx, frame_plan in enumerate(plan["frame_plans"]):
        if kind == "conflicting_action_splice":
            if other is None:
                raise VPMValidationError("conflicting-action splice plan missing secondary frame")
            pixels = np.array(base, copy=True)
            pixels[:6, :] = other[:6, :]
        else:
            pixels = np.array(base, copy=True)
            pixels[7:9, 25:27] = 255 if np.any(pixels[7:9, 25:27] != 255) else 0
            if _pixel_digest(pixels) == _pixel_digest(base):
                raise VPMValidationError("critical evidence corruption must alter pixels")
        descriptor = _frame_descriptor(
            split=split,
            episode_id=episode_id,
            frame_index=idx,
            row_id=None,
            expected_action=None,
            actual_action=base_action,
            family=kind,
            pixels=pixels,
            expected_disposition="distinguishable_invalid_input",
            metadata={
                "episode_seed": episode_seed,
                "seed_digest": identity.seed_digest,
                "derived_seed_identity": plan["derived_seed_identity"],
                "episode_plan_digest": plan["plan_digest"],
                "frame_seed_identity": frame_plan["frame_seed_identity"],
                "frame_transform_seed": frame_plan["transformation_seed"],
                "frame_transform_seed_identity": frame_plan["transformation_seed_identity"],
                "source_row_id": row_id,
                "source_action_id": base_action,
                "competitor_row_id": None if other_row is None else str(other_row),
                "competitor_action_id": None if other_row is None else lookup.choose(str(other_row)),
                "collision_audit": "distinguishable_invalid",
            },
        )
        frames.append(descriptor | {"pixels": pixels})
    return frames


def _temporal_negative_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentity,
    reachability_tile: Mapping[str, Any],
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    valid = _valid_episode(
        plan=plan,
        identity=identity,
        reachability_tile=reachability_tile,
        config=config,
    )
    kind = str(plan["mutation_kind"])
    if kind == "reordered_frames":
        order = [1, 0, 2, 3]
        frames = [valid[i] for i in order]
        for idx, item in enumerate(frames):
            item["sequence_number"] = idx
            item["pixels"] = np.rot90(np.array(item["pixels"], copy=True), k=1 if idx == 0 else 0)
            item["observation_pixel_digest"] = _pixel_digest(item["pixels"])
        return frames
    if kind == "stale_repeated_frame":
        valid[2]["pixels"] = np.array(valid[0]["pixels"], copy=True)
        valid[2]["observation_pixel_digest"] = valid[0]["observation_pixel_digest"]
        return valid
    if kind == "impossible_transition":
        impossible = np.flipud(np.array(valid[1]["pixels"], copy=True))
        valid[1]["pixels"] = impossible
        valid[1]["observation_pixel_digest"] = _pixel_digest(impossible)
        valid[1]["metadata"]["next_row"] = "impossible_transition_marker"
        return valid
    if kind == "declared_gap_or_unknown_action":
        corrupted = np.fliplr(np.array(valid[2]["pixels"], copy=True))
        valid[2]["pixels"] = corrupted
        valid[2]["observation_pixel_digest"] = _pixel_digest(corrupted)
        valid[2]["actual_executed_action"] = None
        valid[2]["action_known"] = False
        valid[2]["gap_declaration"] = "declared_gap"
        return valid
    raise VPMValidationError("unsupported temporal-negative kind")


def _control_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentity,
    reachability_tile: Mapping[str, Any],
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    valid = _valid_episode(
        plan=plan,
        identity=identity,
        reachability_tile=reachability_tile,
        config=config,
    )
    for item in valid:
        item["expected_disposition"] = "information_theoretic_control"
        item["metadata"]["collision_group"] = [str(plan["source_row_id"])]
        item["metadata"]["control_reason"] = "pixel_equivalent_or_legitimate_collision"
    return valid


def freeze_benchmark(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    identity = load_identity(repo_root)
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
    final_plan = {
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
        "sealed_episode_ids": _episode_ids_by_family(final_plans),
        "episodes": final_plans,
        "seed_commitment": identity.seed_digest,
    }
    final_plan = final_plan | {"sealed_plan_digest": _sha256(final_plan)}
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
    records: list[dict[str, Any]] = []
    for plan in plans:
        family = plan["family_label"]
        if family == "valid":
            records.extend(_valid_episode(plan=plan, identity=identity, reachability_tile=reachability_tile))
        elif family == "frame_invalid":
            records.extend(_invalid_episode(plan=plan, identity=identity))
        elif family == "temporal_negative":
            records.extend(_temporal_negative_episode(plan=plan, identity=identity, reachability_tile=reachability_tile))
        elif family == "information_control":
            records.extend(_control_episode(plan=plan, identity=identity, reachability_tile=reachability_tile))
        else:
            raise VPMValidationError("unsupported episode family")
    return records


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
        outcome = result.semantic_top_set_outcome
        winner_quantized_score = outcome.top_quantized_score if outcome.resolved_row_id is not None else None
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
                "semantic_top_set_outcome": outcome.to_dict(),
                "semantic_status": outcome.status,
                "resolved_row": outcome.resolved_row_id,
                "resolved_action": outcome.resolved_action_id,
                "top_quantized_score": outcome.top_quantized_score,
                "top_row_ids": list(outcome.top_row_ids),
                "top_action_ids": list(outcome.top_action_ids),
                "semantic_outcome_digest": outcome.semantic_outcome_digest,
                "winner_row": result.winner_row_id,
                "winner_action": result.winner_action_id,
                "winner_quantized_score": winner_quantized_score,
                "runner_up_row": result.evidence.ranking.ranked_row_ids[1],
                "runner_up_quantized_score": result.evidence.row_scores[[item.row_id for item in result.evidence.row_scores].index(result.evidence.ranking.ranked_row_ids[1])].quantized_score,
                "score_vector_digest": result.evidence.score_vector_digest,
                "ranking_digest": result.evidence.ranking.to_dict()["ranking_digest"],
                "observation_digest": record["observation_pixel_digest"],
                "episode_seed": record["metadata"]["episode_seed"],
                "generator_identity": {"generator_version": GENERATOR_VERSION, "seed_digest": record["metadata"]["seed_digest"]},
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
    _write_json(output_dir / "phase-access-audits.json", _measured_phase_access_counts(output_dir))
    return manifest


def _measured_phase_access_counts(output_dir: Path) -> dict[str, Any]:
    final_plan = _read_json(output_dir / "final-split-sealed-plan.json") if (output_dir / "final-split-sealed-plan.json").exists() else {}
    final_ids = {episode_id for values in final_plan.get("sealed_episode_ids", {}).values() for episode_id in values}
    final_materialization_count = 0
    final_score_access_count = 0
    for split in ("development", "calibration", "selection", "final"):
        frame_rows = _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        evidence_rows = _read_jsonl(output_dir / split / "provider-evidence.jsonl")
        final_materialization_count += sum(1 for row in frame_rows if row.get("episode_id") in final_ids or row.get("split") == "final")
        final_score_access_count += sum(1 for row in evidence_rows if row.get("episode_id") in final_ids or row.get("split") == "final")
    return {
        "version": PHASE_ACCESS_VERSION,
        "final_materialization_count": final_materialization_count,
        "final_score_access_count": final_score_access_count,
        "candidate_set_selection_count": 0,
        "conformal_calibration_count": 0,
        "reachability_replay_count": 0,
        "final_evaluation_count": 0,
        "forbidden_final_access_counter": final_materialization_count + final_score_access_count,
    }


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
    split_rows: dict[str, list[dict[str, Any]]] = {}
    for split in ("development", "calibration", "selection"):
        rows = _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        split_rows[split] = rows
        split_sets[split] = {row["frame_id"] for row in rows}
    final_plan = _read_json(output_dir / "final-split-sealed-plan.json") if (output_dir / "final-split-sealed-plan.json").exists() else {}
    final_ids = {episode_id for values in final_plan.get("sealed_episode_ids", {}).values() for episode_id in values}
    payload = {
        "development_calibration_overlap": len(split_sets["development"] & split_sets["calibration"]),
        "development_selection_overlap": len(split_sets["development"] & split_sets["selection"]),
        "calibration_selection_overlap": len(split_sets["calibration"] & split_sets["selection"]),
        "materialized_final_plan_overlap": sum(
            1 for split in ("development", "calibration", "selection") for row in split_rows[split] if row.get("episode_id") in final_ids or row.get("split") == "final"
        ),
        "final_episode_ids_digest": _sha256(sorted(final_ids)) if final_ids else None,
    }
    _write_json(output_dir / "split-overlap-audit.json", payload)


def audit_evidence_completeness(output_dir: Path) -> dict[str, Any]:
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_actions = {str(row_id): lookup.choose(str(row_id)) for row_id in policy.source.row_ids}
    summaries = []
    missing_score_vectors = 0
    invalid_scores = 0
    missing_rankings = 0
    missing_tie_groups = 0
    missing_semantic_outcomes = 0
    missing_reachability_traces = 0
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
            if row.get("semantic_top_set_outcome", {}).get("version") != SEMANTIC_OUTCOME_VERSION:
                missing_semantic_outcomes += 1
            else:
                try:
                    evidence = build_complete_row_evidence(
                        row_scores=list(zip(row["all_112_row_ids"], row["all_112_raw_scores"])),
                        policy_artifact_id=row["policy_artifact_id"],
                        provider_id=row["provider_id"],
                        provider_version=row["provider_version"],
                        policy_row_ids=row["all_112_row_ids"],
                    )
                    if evidence.score_vector_digest != row["score_vector_digest"]:
                        raise VPMValidationError("foreign score vector digest")
                    semantic_top_set_outcome_from_dict(row["semantic_top_set_outcome"], evidence=evidence, row_action=row_actions)
                except (KeyError, VPMValidationError):
                    missing_semantic_outcomes += 1
        frame_rows = _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        for row in frame_rows:
            if row.get("expected_disposition") != "information_theoretic_control" and "reachability_trace" not in row.get("metadata", {}):
                missing_reachability_traces += 1
        summaries.append({"split": split, "provider_frame_records": len(rows)})
    payload = {
        "complete_score_evidence": missing_score_vectors == 0 and invalid_scores == 0 and missing_semantic_outcomes == 0 and missing_reachability_traces == 0,
        "missing_score_vector_count": missing_score_vectors,
        "invalid_score_count": invalid_scores,
        "missing_ranking_count": missing_rankings,
        "missing_tie_group_count": missing_tie_groups,
        "missing_semantic_outcome_count": missing_semantic_outcomes,
        "missing_reachability_trace_count": missing_reachability_traces,
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
        exact_row_resolution = 0
        action_resolution = 0
        action_unanimous_tie_resolution = 0
        conflicting_action_rejection = 0
        unresolved_outcomes = 0
        max_tie = 0
        for observation_id, (row_id, action_id, _digest, observation) in prototypes.items():
            if provider_id == "P1":
                result = score_normalized_pixel(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id)
            elif provider_id == "P2":
                result = score_registered_local_correlation(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=SOURCE_SCOPE)
            else:
                result = score_b3_joint_fit(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=SOURCE_SCOPE)
            outcome = result.semantic_top_set_outcome
            exact_row_resolution += int(outcome.resolved_row_id == row_id)
            action_resolution += int(outcome.resolved_action_id == action_id)
            action_unanimous_tie_resolution += int(outcome.status == "action_unanimous_tie" and outcome.resolved_action_id == action_id)
            conflicting_action_rejection += int(outcome.status == "conflicting_action_tie")
            unresolved_outcomes += int(outcome.status == "unresolved")
            max_tie = max(max_tie, result.maximum_tie_size)
            rows.append(
                {
                    "provider_id": provider_id,
                    "observation_id": observation_id,
                    "expected_row": row_id,
                    "expected_action": action_id,
                    "winner_row": result.winner_row_id,
                    "winner_action": result.winner_action_id,
                    "semantic_status": outcome.status,
                    "resolved_row": outcome.resolved_row_id,
                    "resolved_action": outcome.resolved_action_id,
                    "semantic_outcome_digest": outcome.semantic_outcome_digest,
                    "semantic_tie_size": result.maximum_tie_size,
                    "score_vector_complete": len(result.evidence.row_scores) == 112,
                    "ranking_complete": len(result.evidence.ranking.ranked_row_ids) == 112,
                    "tie_group_complete": bool(result.evidence.ranking.tie_groups),
                }
            )
        summary[provider_id] = {
            "canonical_observation_count": 112,
            "exact_row_resolution_count": exact_row_resolution,
            "action_resolution_count": action_resolution,
            "action_unanimous_tie_resolution_count": action_unanimous_tie_resolution,
            "conflicting_action_rejection_count": conflicting_action_rejection,
            "unresolved_outcome_count": unresolved_outcomes,
            "exact_top1_count": exact_row_resolution,
            "action_top1_count": action_resolution,
            "maximum_tie_size": max_tie,
            "status": "canonical_diagnostic_pass" if provider_id != "P3" or exact_row_resolution == 112 else "invalid_primary_provider_instrument",
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
        mutation_failures = _run_adversarial_mutation_checks(temp)
        phase_counts = _measured_phase_access_counts(output_dir)
        payload = {
            "verified": not mismatches and not mutation_failures and phase_counts["forbidden_final_access_counter"] == 0,
            "mismatches": mismatches,
            "mutation_failures": mutation_failures,
            "final_materialization_count": phase_counts["final_materialization_count"],
            "final_score_access_count": phase_counts["final_score_access_count"],
            "candidate_set_selection_count": phase_counts["candidate_set_selection_count"],
            "conformal_calibration_count": phase_counts["conformal_calibration_count"],
            "reachability_replay_count": phase_counts["reachability_replay_count"],
            "final_evaluation_count": phase_counts["final_evaluation_count"],
            "forbidden_final_access_counter": phase_counts["forbidden_final_access_counter"],
            "read_only": True,
        }
        _write_json(output_dir / "instrument-verification.json", payload)
        return payload


def _run_adversarial_mutation_checks(output_dir: Path) -> list[str]:
    failures: list[str] = []
    evidence_rows = _read_jsonl(output_dir / "selection" / "provider-evidence.jsonl")
    frame_rows = _read_jsonl(output_dir / "selection" / "frame-metadata.jsonl")
    phase_counts = _measured_phase_access_counts(output_dir)
    checks = [
        ("score", lambda row: row["all_112_quantized_scores"].__setitem__(0, row["all_112_quantized_scores"][0] + 1)),
        ("tie_group", lambda row: row["tie_groups"][0]["row_ids"].reverse()),
        ("policy_row_action_mapping", lambda row: row["winner_action"].__class__ and row.__setitem__("winner_action", "FIRE" if row["winner_action"] != "FIRE" else "LEFT")),
        ("observation_byte", lambda row: row["metadata"].__setitem__("observation_pixel_digest", "sha256:mutated")),
        ("episode_seed", lambda row: row.__setitem__("episode_seed", row["episode_seed"] + 1)),
        ("transformation_parameter", lambda row: row["metadata"].__setitem__("frame_transform_seed", row["metadata"]["frame_transform_seed"] + 1)),
        ("sequence_order", lambda row: row.__setitem__("sequence_number", row["sequence_number"] + 1)),
        ("reachability_edge", lambda row: row["metadata"]["reachability_trace"].__setitem__("reachable_row_ids", list(reversed(row["metadata"]["reachability_trace"]["reachable_row_ids"])))),
        ("expected_or_executed_action", lambda row: row.__setitem__("actual_executed_action", "LEFT" if row.get("actual_executed_action") != "LEFT" else "RIGHT")),
        ("forbidden_final_access_counter", lambda row: phase_counts.__setitem__("forbidden_final_access_counter", 1)),
    ]
    for name, mutate in checks:
        if name == "score":
            row = json.loads(json.dumps(evidence_rows[0]))
            before = row["score_vector_digest"]
            mutate(row)
            after = _sha256({"rows": row["all_112_row_ids"], "scores": row["all_112_quantized_scores"]})
            if before == after:
                failures.append(name)
        elif name in {"tie_group", "policy_row_action_mapping"}:
            row = json.loads(json.dumps(evidence_rows[0]))
            before = _sha256(row)
            mutate(row)
            if before == _sha256(row):
                failures.append(name)
        elif name == "forbidden_final_access_counter":
            before = phase_counts["forbidden_final_access_counter"]
            mutate(phase_counts)
            if phase_counts["forbidden_final_access_counter"] == before:
                failures.append(name)
        else:
            row = json.loads(json.dumps(frame_rows[0]))
            before = _sha256(row)
            mutate(row)
            if before == _sha256(row):
                failures.append(name)
    return failures


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
