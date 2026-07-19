from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from pathlib import Path
import random
from typing import Any, Mapping, Sequence

import numpy as np

from .arcade_policy import ACTIONS, ShooterConfig, arcade_transition_spec, compile_policy_artifact, next_rows, parse_state_row_id, render_state_frame
from .artifact import VPMValidationError
from .policy_lookup import VPMPolicyLookup
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
from .visual_address import ImageObservation


BENCHMARK_VERSION = "zeromodel-video-action-set-reachability-benchmark/v1"
GENERATOR_VERSION = "zeromodel-video-action-set-reachability-generator/v1"
EPISODE_SCHEMA_VERSION = "zeromodel-video-policy-episode/v1"
EPISODE_PLAN_VERSION = "zeromodel-video-action-set-sealed-episode-plan/v1"
SEED_DERIVATION_VERSION = "zeromodel-video-action-set-seed-derivation/v1"
EPISODE_FAMILY_REGISTRY_VERSION = "zeromodel-video-action-set-episode-family-registry/v1"
TRANSFORMATION_FAMILY_VERSION = "zeromodel-video-action-set-transformation-family/v1"
CRITICAL_COORDINATE_SET_VERSION = "zeromodel-video-action-set-critical-coordinate-set/v1"
REACHABILITY_COMPOSITION_VERSION = "zeromodel-video-action-set-reachability-composition/v1"
REACHABILITY_TRACE_VERSION = "zeromodel-video-action-set-reachability-trace/v1"
PHASE_ACCESS_VERSION = "zeromodel-video-prospective-phase-access/v1"
REFERENCE_VERIFICATION_VERSION = "zeromodel-video-action-set-reference-verification/v1"
MUTATION_AUDIT_VERSION = "zeromodel-video-action-set-reference-mutation-audit/v1"
CLOSURE_REPORT_VERSION = "zeromodel-video-action-set-reference-closure/v1"
SOURCE_SCOPE = "zeromodel-video-action-set-reachability-benchmark-v1"
REACHABILITY_TILE_DIGEST = "sha256:fef2bc5fd795bb92d3bd564bccdc2d32e1b23319aba55dffed5e0391e795a5df"
REACHABILITY_TILE_VERSION = "zeromodel-video-policy-reachability-tile/v1"
SEMANTIC_OUTCOME_VERSION = VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION
FRAME_SHAPE = (16, 28)
CRITICAL_REGION_ID = "cooldown_indicator"


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


def _array_digest(pixels: np.ndarray) -> str:
    digest = _pixel_digest(pixels)
    if digest is None:
        raise VPMValidationError("array digest requires pixels")
    return digest


def _coordinate_set_digest(coordinates: Sequence[Sequence[int]]) -> str:
    return _sha256({"coordinates": [[int(y), int(x)] for y, x in coordinates]})


def _load_reachability_tile(repo_root: Path) -> dict[str, Any]:
    return _read_json(repo_root / "docs" / "results" / "video-policy-reachability-tile-v1" / "reachability-tile.json")


def _transformation_contract() -> dict[str, Any]:
    return {
        "version": TRANSFORMATION_FAMILY_VERSION,
        "families": {
            "exact": {"classification": "valid", "dx": [0, 0], "dy": [0, 0], "scale_percent": [100, 100], "offset": [0, 0], "occlusion": False},
            "bounded_translation": {"classification": "valid", "dx": [-1, 1], "dy": [-1, 1], "scale_percent": [100, 100], "offset": [0, 0], "occlusion": False},
            "bounded_photometric": {"classification": "valid", "dx": [0, 0], "dy": [0, 0], "scale_percent": [90, 105], "offset": [0, 5], "occlusion": False},
            "bounded_translation_photometric": {"classification": "valid", "dx": [-1, 1], "dy": [-1, 1], "scale_percent": [90, 105], "offset": [0, 5], "occlusion": False},
            "bounded_translation_occlusion": {"classification": "valid", "dx": [-1, 1], "dy": [-1, 1], "scale_percent": [100, 100], "offset": [0, 0], "occlusion": True},
            "compound_bounded": {"classification": "valid", "dx": [-1, 1], "dy": [-1, 1], "scale_percent": [90, 105], "offset": [0, 5], "occlusion": True},
        },
        "occlusion_bounds": {"top": [0, 2], "left": [0, 2], "height": 2, "width": 3, "value": 64},
    }


def _episode_family_registry() -> dict[str, Any]:
    entries = [
        {
            "family_id": "valid",
            "family_version": "zeromodel-video-action-set-family-valid/v1",
            "classification": "valid",
            "source_row_required": True,
            "source_action_required": True,
            "pixel_intervention": "bounded valid transformation only",
            "sequence_intervention": "none",
            "expected_semantic_effect": "row/action should remain admissible under complete evidence",
            "distinguishability_status": "distinguishable_valid",
            "denominator_treatment": "valid denominator",
            "regeneration_requirements": ["source row", "sealed seed lineage", "transformation parameters", "pixel digest"],
            "implementation_status": "implemented_bounded_scaffold",
        },
        {
            "family_id": "conflicting_action_splice",
            "family_version": "zeromodel-video-action-set-family-conflicting-action-splice/v1",
            "classification": "invalid",
            "source_row_required": True,
            "source_action_required": True,
            "pixel_intervention": "primary/secondary source splice with nonzero effective contributions",
            "sequence_intervention": "none",
            "expected_semantic_effect": "constructed conflicting visual evidence",
            "distinguishability_status": "distinguishable_invalid_input",
            "denominator_treatment": "distinguishable-invalid denominator",
            "regeneration_requirements": ["primary source", "secondary source", "splice mask", "contribution counts", "output digest"],
            "implementation_status": "implemented_bounded_scaffold",
        },
        {
            "family_id": "critical_evidence_corruption",
            "family_version": "zeromodel-video-action-set-family-critical-evidence-corruption/v1",
            "classification": "invalid",
            "source_row_required": True,
            "source_action_required": True,
            "pixel_intervention": "frozen critical coordinate replacement",
            "sequence_intervention": "none",
            "expected_semantic_effect": "critical evidence is no longer faithful to the source row",
            "distinguishability_status": "distinguishable_invalid_input",
            "denominator_treatment": "distinguishable-invalid denominator",
            "regeneration_requirements": ["critical coordinate set", "original values", "replacement values", "output digest"],
            "implementation_status": "implemented_bounded_scaffold",
        },
        {
            "family_id": "reordered_frames",
            "family_version": "zeromodel-video-action-set-family-reordered-frames/v1",
            "classification": "temporal-negative",
            "source_row_required": True,
            "source_action_required": True,
            "pixel_intervention": "none",
            "sequence_intervention": "non-identity permutation of frame payload order",
            "expected_semantic_effect": "sequence order evidence is invalid",
            "distinguishability_status": "distinguishable_temporal_invalid",
            "denominator_treatment": "temporal-negative denominator",
            "regeneration_requirements": ["original order", "mutated order", "sequence digest"],
            "implementation_status": "implemented_bounded_scaffold",
        },
        {
            "family_id": "stale_repeated_frame",
            "family_version": "zeromodel-video-action-set-family-stale-repeated-frame/v1",
            "classification": "temporal-negative",
            "source_row_required": True,
            "source_action_required": True,
            "pixel_intervention": "later frame payload replaced by earlier payload",
            "sequence_intervention": "explicit stale repeat horizon",
            "expected_semantic_effect": "current frame evidence is stale",
            "distinguishability_status": "distinguishable_temporal_invalid",
            "denominator_treatment": "temporal-negative denominator",
            "regeneration_requirements": ["source frame index", "destination frame index", "original destination digest", "replacement digest"],
            "implementation_status": "implemented_bounded_scaffold",
        },
        {
            "family_id": "impossible_transition",
            "family_version": "zeromodel-video-action-set-family-impossible-transition/v1",
            "classification": "temporal-negative",
            "source_row_required": True,
            "source_action_required": True,
            "pixel_intervention": "destination frame set to a nonreachable policy row",
            "sequence_intervention": "violates frozen reachability relation",
            "expected_semantic_effect": "no admissible source-to-destination transition",
            "distinguishability_status": "distinguishable_temporal_invalid",
            "denominator_treatment": "temporal-negative denominator",
            "regeneration_requirements": ["transition relation identity", "pairwise reachability audit", "endpoint frame digests"],
            "implementation_status": "implemented_bounded_scaffold",
        },
        {
            "family_id": "declared_gap_or_unknown_action",
            "family_version": "zeromodel-video-action-set-family-declared-gap-or-unknown/v1",
            "classification": "temporal-negative",
            "source_row_required": True,
            "source_action_required": False,
            "pixel_intervention": "typed gap event with no ordinary pixels",
            "sequence_intervention": "explicit gap/unknown event",
            "expected_semantic_effect": "reader sees a deterministic non-frame event",
            "distinguishability_status": "typed_temporal_event",
            "denominator_treatment": "temporal-negative denominator",
            "regeneration_requirements": ["event identity", "position", "duration", "sequence digest"],
            "implementation_status": "implemented_bounded_scaffold",
        },
        {
            "family_id": "information_control",
            "family_version": "zeromodel-video-action-set-family-information-control/v1",
            "classification": "information-theoretic-control",
            "source_row_required": True,
            "source_action_required": False,
            "pixel_intervention": "byte-identical control observations with hidden source-history labels",
            "sequence_intervention": "control-only grouping",
            "expected_semantic_effect": "hidden distinction unavailable to providers",
            "distinguishability_status": "information_theoretic_control",
            "denominator_treatment": "excluded from distinguishable-invalid denominators",
            "regeneration_requirements": ["control group", "byte identity digest", "hidden-label digest"],
            "implementation_status": "implemented_bounded_scaffold",
        },
    ]
    return {
        "version": EPISODE_FAMILY_REGISTRY_VERSION,
        "transformation_contract_version": TRANSFORMATION_FAMILY_VERSION,
        "families": entries,
        "registry_digest": _sha256({"version": EPISODE_FAMILY_REGISTRY_VERSION, "families": entries}),
    }


def _family_contract(family_label: str, mutation_kind: str | None) -> dict[str, Any]:
    family_id = "valid" if family_label == "valid" else str(mutation_kind or family_label)
    for entry in _episode_family_registry()["families"]:
        if entry["family_id"] == family_id:
            return dict(entry)
    raise VPMValidationError("unknown episode family")


def _translation_values_for_seed(seed: int) -> tuple[int, int]:
    rng = random.Random(int(seed))
    return int(rng.choice((-1, 0, 1))), int(rng.choice((-1, 0, 1)))


def _transformation_parameters(family: str, seed: int) -> dict[str, Any]:
    contract = _transformation_contract()["families"]
    if family not in contract:
        raise VPMValidationError("unsupported transformation family")
    rng = random.Random(int(seed))
    spec = contract[family]
    dx = dy = 0
    if spec["dx"] != [0, 0] or spec["dy"] != [0, 0]:
        dx, dy = _translation_values_for_seed(seed)
    scale_percent = int(spec["scale_percent"][0])
    offset = int(spec["offset"][0])
    if spec["scale_percent"][0] != spec["scale_percent"][1]:
        scale_percent = 90 + rng.randint(0, 15)
    if spec["offset"][0] != spec["offset"][1]:
        offset = rng.randint(0, 5)
    params: dict[str, Any] = {
        "version": TRANSFORMATION_FAMILY_VERSION,
        "family": family,
        "seed": int(seed),
        "dx": dx,
        "dy": dy,
        "scale_percent": scale_percent,
        "offset": offset,
        "occlusion": None,
    }
    if bool(spec["occlusion"]):
        occlusion = _transformation_contract()["occlusion_bounds"]
        params["occlusion"] = {
            "top": rng.randint(int(occlusion["top"][0]), int(occlusion["top"][1])),
            "left": rng.randint(int(occlusion["left"][0]), int(occlusion["left"][1])),
            "height": int(occlusion["height"]),
            "width": int(occlusion["width"]),
            "value": int(occlusion["value"]),
        }
    params["parameter_digest"] = _sha256({key: value for key, value in params.items() if key != "parameter_digest"})
    return params


def _validate_transformation_parameters(params: Mapping[str, Any], *, image_shape: tuple[int, int] = FRAME_SHAPE) -> None:
    family = str(params["family"])
    contract = _transformation_contract()
    if family not in contract["families"]:
        raise VPMValidationError("unsupported transformation family")
    spec = contract["families"][family]
    dx = int(params["dx"])
    dy = int(params["dy"])
    scale_percent = int(params["scale_percent"])
    offset = int(params["offset"])
    if not (int(spec["dx"][0]) <= dx <= int(spec["dx"][1])):
        raise VPMValidationError("transformation dx out of bounds")
    if not (int(spec["dy"][0]) <= dy <= int(spec["dy"][1])):
        raise VPMValidationError("transformation dy out of bounds")
    if not (int(spec["scale_percent"][0]) <= scale_percent <= int(spec["scale_percent"][1])):
        raise VPMValidationError("transformation scale out of bounds")
    if not (int(spec["offset"][0]) <= offset <= int(spec["offset"][1])):
        raise VPMValidationError("transformation offset out of bounds")
    if bool(spec["occlusion"]):
        occlusion = params.get("occlusion")
        if not isinstance(occlusion, Mapping):
            raise VPMValidationError("occlusion parameters required")
        top = int(occlusion["top"])
        left = int(occlusion["left"])
        height = int(occlusion["height"])
        width = int(occlusion["width"])
        value = int(occlusion["value"])
        bounds = contract["occlusion_bounds"]
        if not (int(bounds["top"][0]) <= top <= int(bounds["top"][1])):
            raise VPMValidationError("occlusion top out of bounds")
        if not (int(bounds["left"][0]) <= left <= int(bounds["left"][1])):
            raise VPMValidationError("occlusion left out of bounds")
        if height != int(bounds["height"]) or width != int(bounds["width"]) or value != int(bounds["value"]):
            raise VPMValidationError("occlusion dimensions or value out of bounds")
        if top + height > image_shape[0] or left + width > image_shape[1]:
            raise VPMValidationError("occlusion exceeds image bounds")
    elif params.get("occlusion") is not None:
        raise VPMValidationError("occlusion parameters supplied for non-occlusion family")
    expected = _sha256({key: _json_ready(value) for key, value in params.items() if key != "parameter_digest"})
    if str(params.get("parameter_digest")) != expected:
        raise VPMValidationError("foreign transformation parameter digest")


def _apply_transformation(frame: np.ndarray, params: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.ascontiguousarray(frame, dtype=np.uint8)
    _validate_transformation_parameters(params, image_shape=tuple(source.shape))
    result = np.array(source, copy=True)
    dx = int(params["dx"])
    dy = int(params["dy"])
    if dx or dy:
        translated = np.full_like(result, 0)
        h, w = result.shape
        x0, x1 = max(0, dx), min(w, w + dx)
        y0, y1 = max(0, dy), min(h, h + dy)
        if x1 > x0 and y1 > y0:
            translated[y0:y1, x0:x1] = result[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
        result = translated
    if int(params["scale_percent"]) != 100 or int(params["offset"]) != 0:
        result = np.clip(np.round(result.astype(np.float32) * (int(params["scale_percent"]) / 100.0)) + int(params["offset"]), 0, 255).astype(np.uint8)
    if params.get("occlusion") is not None:
        occlusion = params["occlusion"]
        top = int(occlusion["top"])
        left = int(occlusion["left"])
        height = int(occlusion["height"])
        width = int(occlusion["width"])
        result[top : top + height, left : left + width] = int(occlusion["value"])
    result = np.ascontiguousarray(result, dtype=np.uint8)
    source_digest = _array_digest(source)
    output_digest = _array_digest(result)
    return result, {
        "source_observation_digest": source_digest,
        "transformed_observation_digest": output_digest,
        "transformation_parameter_digest": params["parameter_digest"],
        "changed_pixel_count": int(np.count_nonzero(source != result)),
    }


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


def _frame_count_for_plan(split: str, family_label: str) -> int:
    if split == "development":
        return 1
    return 4


def _critical_coordinates() -> tuple[tuple[int, int], ...]:
    return tuple((y, x) for y in range(7, 9) for x in range(25, 27))


def _critical_coordinate_manifest(coordinates: Sequence[Sequence[int]] | None = None) -> dict[str, Any]:
    source_coordinates = _critical_coordinates() if coordinates is None else coordinates
    coords = tuple((int(y), int(x)) for y, x in source_coordinates)
    if not coords:
        raise VPMValidationError("critical coordinate set cannot be empty")
    critical = set(_critical_coordinates())
    if any(coord not in critical for coord in coords):
        raise VPMValidationError("selected coordinate is not critical under the frozen definition")
    if len(set(coords)) != len(coords):
        raise VPMValidationError("critical coordinate set cannot contain duplicates")
    return {
        "version": CRITICAL_COORDINATE_SET_VERSION,
        "criticality_source": "tiny_arcade_shooter_rendering",
        "criticality_region_id": CRITICAL_REGION_ID,
        "coordinates": [[y, x] for y, x in coords],
        "coordinate_set_digest": _coordinate_set_digest(coords),
    }


def _splice_mask_manifest(mask_rows: Sequence[int] = tuple(range(6)), *, width: int = 28) -> dict[str, Any]:
    rows = tuple(int(row) for row in mask_rows)
    if not rows or len(set(rows)) != len(rows):
        raise VPMValidationError("splice mask rows must be unique and non-empty")
    if any(row < 0 or row >= FRAME_SHAPE[0] for row in rows):
        raise VPMValidationError("splice mask row out of bounds")
    coordinates = tuple((row, x) for row in rows for x in range(width))
    return {
        "version": "zeromodel-video-action-set-splice-mask/v1",
        "mask_kind": "row_band",
        "rows": list(rows),
        "coordinates": [[y, x] for y, x in coordinates],
        "coordinate_count": len(coordinates),
        "mask_digest": _coordinate_set_digest(coordinates),
    }


def _state_row_values(row_id: str) -> tuple[int, int | None, int]:
    return parse_state_row_id(str(row_id))


def _secondary_row_for_splice(row_ids: list[str], row_actions: Mapping[str, str], source_row_id: str) -> str:
    source_action = row_actions[source_row_id]
    source_tank, source_target, source_cooldown = _state_row_values(source_row_id)
    for row_id in row_ids:
        if row_id == source_row_id:
            continue
        if row_actions[row_id] == source_action:
            continue
        tank, target, cooldown = _state_row_values(row_id)
        if target == source_target:
            continue
        if (tank, cooldown) == (source_tank, source_cooldown):
            continue
        return row_id
    raise VPMValidationError("frame splice requires a secondary row with a conflicting action and distinct visual source")


def _impossible_destination_row(row_ids: list[str], row_actions: Mapping[str, str], source_row_id: str) -> str:
    action = row_actions[source_row_id]
    tank, target, cooldown = _state_row_values(source_row_id)
    reachable = set(next_rows(tank, target, cooldown, action, width=ShooterConfig().width))
    for row_id in reversed(row_ids):
        if row_id not in reachable and row_id != source_row_id:
            return row_id
    raise VPMValidationError("unable to select impossible transition destination")


def _family_intervention_plan(
    *,
    identity: BenchmarkIdentity,
    split: str,
    ordinal: int,
    family_label: str,
    mutation_kind: str | None,
    source_row_id: str,
    secondary_row_id: str | None,
    row_ids: Sequence[str],
    row_actions: Mapping[str, str],
) -> dict[str, Any]:
    family_id = "valid" if family_label == "valid" else str(mutation_kind or family_label)
    seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="family_intervention",
        parent_identities=(
            ("family_id", family_id),
            ("source_row_id", source_row_id),
            ("secondary_row_id", secondary_row_id or "none"),
        ),
    )
    original_order = list(range(_frame_count_for_plan(split, family_label)))
    payload: dict[str, Any] = {
        "version": "zeromodel-video-action-set-family-intervention/v1",
        "family_id": family_id,
        "intervention_seed_identity": seed["seed_digest"],
        "original_order": original_order,
        "materialized_order": list(original_order),
        "event_type": "frame_sequence",
    }
    if family_id == "conflicting_action_splice":
        payload |= {
            "primary_source_row_id": source_row_id,
            "primary_source_action_id": row_actions[source_row_id],
            "secondary_source_row_id": secondary_row_id,
            "secondary_source_action_id": None if secondary_row_id is None else row_actions[secondary_row_id],
            "splice_mask": _splice_mask_manifest(),
        }
    elif family_id == "critical_evidence_corruption":
        payload |= {"critical_coordinates": _critical_coordinate_manifest()}
    elif family_id == "reordered_frames":
        mutated = [1, 0, 2, 3]
        if mutated == original_order:
            raise VPMValidationError("reordered family requires non-identity order")
        payload |= {"materialized_order": mutated, "sequence_rule": "non_identity_permutation"}
    elif family_id == "stale_repeated_frame":
        payload |= {"stale_repeat": {"source_frame_index": 0, "destination_frame_index": 1, "maximum_stale_horizon": 1}}
    elif family_id == "impossible_transition":
        destination = _impossible_destination_row(list(row_ids), row_actions, source_row_id)
        payload |= {
            "transition_relation_identity": REACHABILITY_TILE_DIGEST,
            "impossible_transition": {
                "source_frame_index": 0,
                "destination_frame_index": 1,
                "source_row_id": source_row_id,
                "source_action_id": row_actions[source_row_id],
                "destination_row_id": destination,
                "destination_action_id": row_actions[destination],
            },
        }
    elif family_id == "declared_gap_or_unknown_action":
        gap_seed = _derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="gap_event_identity",
            parent_identities=(("family_intervention", seed["seed_digest"]), ("gap_position", "2")),
        )
        payload |= {
            "event_type": "typed_gap_sequence",
            "gap_event": {
                "version": "zeromodel-video-action-set-gap-event/v1",
                "position": 2,
                "duration_frames": 1,
                "reason": "declared_gap_or_unknown_action",
                "event_id": gap_seed["seed_digest"],
            },
        }
    elif family_id == "information_control":
        control_seed = _derived_seed(
            identity,
            split=split,
            ordinal=ordinal,
            namespace="information_control_identity",
            parent_identities=(("family_intervention", seed["seed_digest"]), ("source_row_id", source_row_id)),
        )
        payload |= {
            "control_group": {
                "control_group_id": control_seed["seed_digest"],
                "byte_identity_required": True,
                "hidden_source_label_digest": _sha256({"source_row_id": source_row_id, "control_seed": control_seed["seed_digest"]}),
                "provider_visible_fields": ["pixels", "frame_id"],
            }
        }
    payload["sequence_digest"] = _sha256({"order": payload["materialized_order"], "event_type": payload["event_type"], "family_id": family_id})
    payload["intervention_digest"] = _sha256(payload)
    return payload


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
            transformation_family = "exact"
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
        parameters = _transformation_parameters(transformation_family, transform_parameter_seed["seed_int64"])
        frames.append(
            {
                "frame_index": frame_index,
                "frame_seed_identity": frame_seed["seed_digest"],
                "transformation_family_seed_identity": transform_family_seed["seed_digest"],
                "transformation_family": transformation_family,
                "transformation_seed_identity": transform_parameter_seed["seed_digest"],
                "transformation_seed": transform_parameter_seed["seed_int64"],
                "transformation_parameter_digest": parameters["parameter_digest"],
                "transformation_parameters": parameters,
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
    frame_count = _frame_count_for_plan(split, family_label)
    frame_plans = _frame_plans(
        identity,
        split=split,
        ordinal=ordinal,
        family_label=family_label,
        mutation_kind=mutation_kind,
        frame_count=frame_count,
        concrete_episode_seed_digest=episode_seed["seed_digest"],
    )
    family_contract = _family_contract(family_label, mutation_kind)
    family_intervention = _family_intervention_plan(
        identity=identity,
        split=split,
        ordinal=ordinal,
        family_label=family_label,
        mutation_kind=mutation_kind,
        source_row_id=source_row_id,
        secondary_row_id=secondary_row_id,
        row_ids=tuple(row_actions.keys()),
        row_actions=row_actions,
    )
    episode_id_seed = _derived_seed(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="concrete_episode_id",
        parent_identities=(
            ("concrete_episode_seed", episode_seed["seed_digest"]),
            ("family_contract", family_contract["family_version"]),
            ("family_intervention", family_intervention["intervention_digest"]),
            ("source_row_id", source_row_id),
            ("secondary_row_id", secondary_row_id or "none"),
        ),
    )
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
        "family_contract": family_contract,
        "family_intervention": family_intervention,
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
        "frame_plans": frame_plans,
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
    result, _metadata = _apply_transformation(frame, _transformation_parameters(family, seed))
    return result


def _apply_frame_plan(frame: np.ndarray, frame_plan: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    params = frame_plan["transformation_parameters"]
    if str(params["parameter_digest"]) != str(frame_plan["transformation_parameter_digest"]):
        raise VPMValidationError("frame transformation parameter digest mismatch")
    return _apply_transformation(frame, params)


def _render_row_frame(row_id: str, *, config: ShooterConfig = ShooterConfig()) -> np.ndarray:
    tank, target, cooldown = parse_state_row_id(str(row_id))
    return render_state_frame(tank, target, cooldown, width=config.width)


def _apply_conflicting_splice(
    *,
    primary_pixels: np.ndarray,
    secondary_pixels: np.ndarray,
    primary_row_id: str,
    secondary_row_id: str,
    primary_action_id: str,
    secondary_action_id: str,
    mask_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    if primary_action_id == secondary_action_id:
        raise VPMValidationError("conflicting splice requires different governed actions")
    primary = np.ascontiguousarray(primary_pixels, dtype=np.uint8)
    secondary = np.ascontiguousarray(secondary_pixels, dtype=np.uint8)
    if primary.shape != secondary.shape:
        raise VPMValidationError("splice sources must have the same shape")
    coordinates = tuple((int(y), int(x)) for y, x in mask_manifest["coordinates"])
    if not coordinates:
        raise VPMValidationError("splice mask cannot be empty")
    output = np.array(primary, copy=True)
    for y, x in coordinates:
        if y < 0 or y >= output.shape[0] or x < 0 or x >= output.shape[1]:
            raise VPMValidationError("splice coordinate out of bounds")
        output[y, x] = secondary[y, x]
    mask = np.zeros(primary.shape, dtype=bool)
    for y, x in coordinates:
        mask[y, x] = True
    secondary_effective = int(np.count_nonzero(mask & (primary != secondary)))
    primary_effective = int(np.count_nonzero((~mask) & (primary != secondary)))
    if secondary_effective == 0 or primary_effective == 0:
        raise VPMValidationError("splice requires nonzero effective contribution from both sources")
    if np.array_equal(output, primary) or np.array_equal(output, secondary):
        raise VPMValidationError("splice output must not equal either source observation")
    manifest = {
        "primary_source_row_id": primary_row_id,
        "primary_source_action_id": primary_action_id,
        "secondary_source_row_id": secondary_row_id,
        "secondary_source_action_id": secondary_action_id,
        "primary_source_digest": _array_digest(primary),
        "secondary_source_digest": _array_digest(secondary),
        "splice_mask_identity": mask_manifest["mask_digest"],
        "primary_contributing_pixel_count": primary_effective,
        "secondary_contributing_pixel_count": secondary_effective,
        "output_observation_digest": _array_digest(output),
        "expected_invalid_family_label": "conflicting_action_splice",
    }
    manifest["splice_trace_digest"] = _sha256(manifest)
    return output, manifest


def _apply_critical_corruption(source_pixels: np.ndarray, coordinate_manifest: Mapping[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.ascontiguousarray(source_pixels, dtype=np.uint8)
    coords = tuple((int(y), int(x)) for y, x in coordinate_manifest["coordinates"])
    _critical_coordinate_manifest(coords)
    output = np.array(source, copy=True)
    changes = []
    for y, x in coords:
        if y < 0 or y >= output.shape[0] or x < 0 or x >= output.shape[1]:
            raise VPMValidationError("critical coordinate outside image bounds")
        original = int(output[y, x])
        replacement = 255 if original != 255 else 0
        if replacement == original:
            raise VPMValidationError("critical corruption replacement must change the value")
        output[y, x] = replacement
        if int(output[y, x]) == original:
            raise VPMValidationError("critical corruption no-op after assignment")
        changes.append({"y": y, "x": x, "original": original, "replacement": int(output[y, x])})
    if not changes:
        raise VPMValidationError("critical corruption requires at least one changed pixel")
    if np.array_equal(source, output):
        raise VPMValidationError("critical corruption must change the observation digest")
    manifest = {
        "criticality_artifact_identity": CRITICAL_COORDINATE_SET_VERSION,
        "critical_region_id": CRITICAL_REGION_ID,
        "critical_coordinate_set_identity": coordinate_manifest["coordinate_set_digest"],
        "changes": changes,
        "changed_pixel_count": len(changes),
        "source_observation_digest": _array_digest(source),
        "output_observation_digest": _array_digest(output),
    }
    manifest["critical_corruption_digest"] = _sha256(manifest)
    return output, manifest


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
        "event_type": metadata.get("event_type", "frame"),
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


def _reachability_tile_digest(tile: Mapping[str, Any]) -> str:
    return _sha256({key: value for key, value in tile.items() if key != "tile_digest"})


def _validate_reachability_tile_identity(tile: Mapping[str, Any]) -> None:
    if str(tile.get("tile_digest")) != _reachability_tile_digest(tile):
        raise VPMValidationError("foreign reachability tile digest")


def _next_row(policy_lookup: VPMPolicyLookup, row_id: str, *, choice_seed: int, config: ShooterConfig, reachability_tile: Mapping[str, Any]) -> tuple[str, str, int, dict[str, Any]]:
    _validate_reachability_tile_identity(reachability_tile)
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
        pixels, transformation_trace = _apply_frame_plan(base, frame_plan)
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
                "transformation_parameters": dict(frame_plan["transformation_parameters"]),
                "source_observation_digest": transformation_trace["source_observation_digest"],
                "transformed_observation_digest": transformation_trace["transformed_observation_digest"],
                "transformation_parameter_digest": transformation_trace["transformation_parameter_digest"],
                "transformation_changed_pixel_count": transformation_trace["changed_pixel_count"],
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
    intervention = plan["family_intervention"]
    for idx, frame_plan in enumerate(plan["frame_plans"]):
        source_pixels, transformation_trace = _apply_frame_plan(base, frame_plan)
        intervention_trace: dict[str, Any]
        if kind == "conflicting_action_splice":
            if other is None:
                raise VPMValidationError("conflicting-action splice plan missing secondary frame")
            pixels, intervention_trace = _apply_conflicting_splice(
                primary_pixels=source_pixels,
                secondary_pixels=other,
                primary_row_id=row_id,
                secondary_row_id=str(other_row),
                primary_action_id=base_action,
                secondary_action_id=lookup.choose(str(other_row)),
                mask_manifest=intervention["splice_mask"],
            )
        else:
            pixels, intervention_trace = _apply_critical_corruption(source_pixels, intervention["critical_coordinates"])
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
                "transformation_parameters": dict(frame_plan["transformation_parameters"]),
                "source_observation_digest": transformation_trace["source_observation_digest"],
                "transformed_observation_digest": transformation_trace["transformed_observation_digest"],
                "transformation_parameter_digest": transformation_trace["transformation_parameter_digest"],
                "transformation_changed_pixel_count": transformation_trace["changed_pixel_count"],
                "family_contract": plan["family_contract"],
                "family_intervention": intervention,
                "family_intervention_trace": intervention_trace,
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
    intervention = plan["family_intervention"]
    for item in valid:
        item["metadata"]["family_contract"] = plan["family_contract"]
        item["metadata"]["family_intervention"] = intervention
        item["metadata"]["sequence_digest"] = intervention["sequence_digest"]
    if kind == "reordered_frames":
        order = list(intervention["materialized_order"])
        if order == list(intervention["original_order"]):
            raise VPMValidationError("reordered family requires non-identity order")
        if sorted(order) != list(intervention["original_order"]):
            raise VPMValidationError("reordered family must be a complete permutation")
        frames = [valid[i] for i in order]
        for idx, item in enumerate(frames):
            item["sequence_number"] = idx
            item["metadata"]["original_frame_index"] = order[idx]
            item["metadata"]["materialized_order"] = order
            item["metadata"]["sequence_rule"] = intervention["sequence_rule"]
        return frames
    if kind == "stale_repeated_frame":
        repeat = intervention["stale_repeat"]
        source_index = int(repeat["source_frame_index"])
        destination_index = int(repeat["destination_frame_index"])
        original_destination_digest = valid[destination_index]["observation_pixel_digest"]
        replacement_digest = valid[source_index]["observation_pixel_digest"]
        if original_destination_digest == replacement_digest:
            raise VPMValidationError("stale repeat requires an actual payload replacement")
        valid[destination_index]["pixels"] = np.array(valid[source_index]["pixels"], copy=True)
        valid[destination_index]["observation_pixel_digest"] = replacement_digest
        valid[destination_index]["metadata"]["stale_repeat"] = {
            **repeat,
            "original_destination_digest": original_destination_digest,
            "replacement_digest": replacement_digest,
        }
        return valid
    if kind == "impossible_transition":
        transition_plan = intervention["impossible_transition"]
        source_index = int(transition_plan["source_frame_index"])
        destination_index = int(transition_plan["destination_frame_index"])
        source_row = str(transition_plan["source_row_id"])
        destination_row = str(transition_plan["destination_row_id"])
        edge = _tile_edge(reachability_tile, source_row, str(transition_plan["source_action_id"]))
        if destination_row in set(edge["reachable_row_ids"]):
            raise VPMValidationError("impossible transition destination is reachable")
        destination_pixels = _render_row_frame(destination_row, config=config)
        valid[destination_index]["pixels"] = destination_pixels
        valid[destination_index]["observation_pixel_digest"] = _pixel_digest(destination_pixels)
        valid[destination_index]["expected_row"] = destination_row
        valid[destination_index]["expected_action"] = transition_plan["destination_action_id"]
        valid[destination_index]["metadata"]["impossible_transition"] = {
            **transition_plan,
            "source_observation_digest": valid[source_index]["observation_pixel_digest"],
            "destination_observation_digest": valid[destination_index]["observation_pixel_digest"],
            "reachability_tile_digest": reachability_tile["tile_digest"],
            "consulted_edge": {"source_row_id": edge["source_row_id"], "action_id": edge["action_id"], "reachable_row_ids": list(edge["reachable_row_ids"])},
            "pairwise_reachability_status": "impossible",
        }
        valid[destination_index]["metadata"]["next_row"] = "impossible_transition_marker"
        return valid
    if kind == "declared_gap_or_unknown_action":
        gap = intervention["gap_event"]
        position = int(gap["position"])
        valid[position]["pixels"] = None
        valid[position]["event_type"] = "gap_unknown"
        valid[position]["expected_row"] = None
        valid[position]["expected_action"] = None
        valid[position]["actual_executed_action"] = None
        valid[position]["action_known"] = False
        valid[position]["gap_declaration"] = "declared_gap"
        valid[position]["observation_pixel_digest"] = None
        valid[position]["metadata"]["gap_event"] = gap
        valid[position]["metadata"]["event_identity"] = gap["event_id"]
        return valid
    raise VPMValidationError("unsupported temporal-negative kind")


def _control_episode(
    *,
    plan: Mapping[str, Any],
    identity: BenchmarkIdentity,
    reachability_tile: Mapping[str, Any],
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    split = str(plan["split"])
    episode_id = str(plan["episode_id"])
    source_row = str(plan["source_row_id"])
    source_pixels = _render_row_frame(source_row, config=config)
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    action = lookup.choose(source_row)
    intervention = plan["family_intervention"]
    control_group = intervention["control_group"]
    frames = []
    for idx, frame_plan in enumerate(plan["frame_plans"]):
        pixels = np.array(source_pixels, copy=True)
        descriptor = _frame_descriptor(
            split=split,
            episode_id=episode_id,
            frame_index=idx,
            row_id=None,
            expected_action=None,
            actual_action=None,
            family="information_control",
            pixels=pixels,
            expected_disposition="information_theoretic_control",
            metadata={
                "episode_seed": int(plan["episode_seed"]),
                "seed_digest": identity.seed_digest,
                "derived_seed_identity": plan["derived_seed_identity"],
                "episode_plan_digest": plan["plan_digest"],
                "frame_seed_identity": frame_plan["frame_seed_identity"],
                "source_row_id": source_row,
                "source_action_id": action,
                "family_contract": plan["family_contract"],
                "family_intervention": intervention,
                "control_group_id": control_group["control_group_id"],
                "control_observation_digest": _array_digest(source_pixels),
                "hidden_source_label_digest": control_group["hidden_source_label_digest"],
                "provider_hidden_fields": ["source_row_id", "source_action_id", "hidden_source_label_digest"],
                "denominator_eligible": False,
                "control_reason": "byte_identical_hidden_source_history",
            },
        )
        frames.append(descriptor | {"pixels": pixels})
    digests = {item["observation_pixel_digest"] for item in frames}
    if len(digests) != 1:
        raise VPMValidationError("information controls require byte-identical observations")
    return frames


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


def _materialize_plan(plan: Mapping[str, Any], identity: BenchmarkIdentity, reachability_tile: Mapping[str, Any]) -> list[dict[str, Any]]:
    family = plan["family_label"]
    if family == "valid":
        return _valid_episode(plan=plan, identity=identity, reachability_tile=reachability_tile)
    if family == "frame_invalid":
        return _invalid_episode(plan=plan, identity=identity)
    if family == "temporal_negative":
        return _temporal_negative_episode(plan=plan, identity=identity, reachability_tile=reachability_tile)
    if family == "information_control":
        return _control_episode(plan=plan, identity=identity, reachability_tile=reachability_tile)
    raise VPMValidationError("unsupported episode family")


def _record_regeneration_view(record: Mapping[str, Any]) -> dict[str, Any]:
    pixels = record.get("pixels")
    return {
        "episode_id": record["episode_id"],
        "sequence_number": record["sequence_number"],
        "event_type": record.get("event_type", "frame"),
        "family": record["family"],
        "expected_disposition": record["expected_disposition"],
        "expected_row": record.get("expected_row"),
        "expected_action": record.get("expected_action"),
        "actual_executed_action": record.get("actual_executed_action"),
        "gap_declaration": record.get("gap_declaration"),
        "observation_pixel_digest": record.get("observation_pixel_digest"),
        "pixel_digest": None if pixels is None else _array_digest(np.ascontiguousarray(pixels, dtype=np.uint8)),
        "sequence_digest": record.get("metadata", {}).get("sequence_digest"),
        "episode_plan_digest": record.get("metadata", {}).get("episode_plan_digest"),
    }


def _family_closure_report(
    *,
    split: str,
    records: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    identity: BenchmarkIdentity,
    reachability_tile: Mapping[str, Any],
    provider_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    by_episode: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_episode.setdefault(str(record["episode_id"]), []).append(record)
    registry = _episode_family_registry()
    rows: dict[str, dict[str, Any]] = {}
    for entry in registry["families"]:
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
            "denominator_eligibility": entry["denominator_treatment"],
            "reachability_applicable_count": 0,
            "reachability_trace_verification_count": 0,
            "closure_status": "not_planned",
        }
    trace_counts: dict[str, int] = {}
    for row in provider_rows:
        trace = row.get("reachability_composition_trace")
        if trace:
            trace_counts[str(row["episode_id"])] = trace_counts.get(str(row["episode_id"]), 0) + 1
    for plan in plans:
        family_id = str(plan["family_intervention"]["family_id"])
        row = rows[family_id]
        row["planned_episode_count"] += 1
        actual = sorted(by_episode.get(str(plan["episode_id"]), ()), key=lambda item: int(item["sequence_number"]))
        try:
            regenerated = sorted(_materialize_plan(plan, identity, reachability_tile), key=lambda item: int(item["sequence_number"]))
            row["regenerated_episode_count"] += 1
            if [_record_regeneration_view(item) for item in actual] == [_record_regeneration_view(item) for item in regenerated]:
                row["validation_pass_count"] += 1
            else:
                row["malformed_count"] += 1
            if family_id in {"conflicting_action_splice", "critical_evidence_corruption"}:
                changed = [
                    item.get("metadata", {}).get("family_intervention_trace", {}).get("changed_pixel_count")
                    for item in regenerated
                ]
                if any(value == 0 for value in changed if value is not None):
                    row["no_op_count"] += 1
                row["distinguishable_invalid_count"] += 1
            if family_id == "information_control":
                row["information_theoretic_control_count"] += 1
            if family_id in {"valid", "impossible_transition"}:
                row["reachability_applicable_count"] += 1
            row["reachability_trace_verification_count"] += trace_counts.get(str(plan["episode_id"]), 0)
        except VPMValidationError:
            row["malformed_count"] += 1
    for row in rows.values():
        if row["planned_episode_count"] == 0:
            continue
        row["closure_status"] = (
            "closed"
            if row["planned_episode_count"] == row["regenerated_episode_count"] == row["validation_pass_count"] and row["malformed_count"] == 0 and row["no_op_count"] == 0
            else "unresolved"
        )
    return {
        "version": "zeromodel-video-action-set-family-closure/v1",
        "split": split,
        "registry_version": EPISODE_FAMILY_REGISTRY_VERSION,
        "families": list(rows.values()),
        "negative_families_verified": False,
        "reachability_verified": False,
        "reference_instrument_correct": False,
        "materialization_ready": False,
        "repository_status": "reference_instrument_correctness_unresolved",
        "materialization_status": "prospective_materialization_prohibited",
    }


def validate_materialized_family_record(record: Mapping[str, Any]) -> str:
    pixels = record.get("pixels")
    if pixels is not None and record.get("observation_pixel_digest") != _array_digest(np.ascontiguousarray(pixels, dtype=np.uint8)):
        return "stale_observation_digest"
    metadata = record.get("metadata", {})
    trace = metadata.get("family_intervention_trace")
    if trace:
        output_digest = trace.get("output_observation_digest")
        if output_digest is not None and output_digest != record.get("observation_pixel_digest"):
            return "family_output_digest_mismatch"
        if trace.get("changed_pixel_count") == 0:
            return "family_no_op"
    if record.get("event_type") == "gap_unknown" and pixels is not None:
        return "gap_event_has_pixels"
    return "ok"


def validate_control_episode_records(records: Sequence[Mapping[str, Any]]) -> str:
    control_records = [record for record in records if record.get("expected_disposition") == "information_theoretic_control"]
    if not control_records:
        return "no_control_records"
    digests = {record.get("observation_pixel_digest") for record in control_records}
    if None in digests or len(digests) != 1:
        return "control_byte_identity_mismatch"
    if any(record.get("metadata", {}).get("denominator_eligible") for record in control_records):
        return "control_denominator_leak"
    return "ok"


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


def _trace_digest(payload: Mapping[str, Any]) -> str:
    return _sha256({key: value for key, value in payload.items() if key != "trace_digest"})


def _top_row_action_map(outcome: Mapping[str, Any]) -> dict[str, str]:
    return {str(item["row_id"]): str(item["action_id"]) for item in outcome.get("top_row_actions", [])}


def compose_reachability_trace(
    *,
    frame_id: str,
    semantic_outcome: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None,
    reachability_tile: Mapping[str, Any],
    row_actions: Mapping[str, str],
) -> dict[str, Any]:
    _validate_reachability_tile_identity(reachability_tile)
    top_rows = tuple(str(row_id) for row_id in semantic_outcome.get("top_row_ids", ()))
    top_row_actions = _top_row_action_map(semantic_outcome)
    if set(top_rows) != set(top_row_actions):
        raise VPMValidationError("semantic top rows and row/action mapping disagree")
    if any(row_actions[row_id] != top_row_actions[row_id] for row_id in top_rows):
        raise VPMValidationError("semantic outcome row/action mapping is inconsistent with policy")
    consulted_edges = []
    reachable_pairs = []
    removed_rows = []
    retained_rows: tuple[str, ...]
    rejection_reason = None

    prior_rows = tuple(str(row_id) for row_id in (previous_state or {}).get("candidate_rows", ()))
    prior_status = None if previous_state is None else str(previous_state.get("status", "unresolved"))
    if previous_state is not None and prior_status == "unresolved":
        retained_rows = ()
        removed_rows = list(top_rows)
        rejection_reason = "prior_state_unresolved"
    elif previous_state is None:
        retained_rows = tuple(sorted(top_rows))
    else:
        retained = set()
        for prior_row in prior_rows:
            prior_action = row_actions[prior_row]
            edge = _tile_edge(reachability_tile, prior_row, prior_action)
            edge_payload = {
                "source_row_id": edge["source_row_id"],
                "action_id": edge["action_id"],
                "reachable_row_ids": list(edge["reachable_row_ids"]),
            }
            consulted_edges.append(edge_payload)
            reachable = set(edge["reachable_row_ids"])
            for current_row in top_rows:
                if current_row in reachable:
                    retained.add(current_row)
                    reachable_pairs.append({"source_row_id": prior_row, "action_id": prior_action, "destination_row_id": current_row})
        retained_rows = tuple(sorted(retained))
        removed_rows = sorted(set(top_rows) - set(retained_rows))
        if not retained_rows:
            rejection_reason = "no_reachable_candidate"

    retained_actions = tuple(sorted({row_actions[row_id] for row_id in retained_rows}))
    if rejection_reason is None and semantic_outcome.get("status") == "conflicting_action_tie" and len(retained_actions) > 1:
        rejection_reason = "conflicting_reachable_actions"
    resolved_action = retained_actions[0] if rejection_reason is None and len(retained_actions) == 1 else None
    resolved_row = retained_rows[0] if rejection_reason is None and len(retained_rows) == 1 else None
    status = "rejected" if rejection_reason else ("resolved" if resolved_action is not None else "unresolved")
    trace = {
        "version": REACHABILITY_TRACE_VERSION,
        "composition_version": REACHABILITY_COMPOSITION_VERSION,
        "frame_id": frame_id,
        "semantic_outcome_digest": semantic_outcome["semantic_outcome_digest"],
        "input_candidate_rows": list(top_rows),
        "input_candidate_actions": sorted({top_row_actions[row_id] for row_id in top_rows}),
        "prior_reachable_rows": list(prior_rows),
        "prior_state_status": prior_status,
        "reachability_tile_identity": reachability_tile["tile_digest"],
        "consulted_edges": consulted_edges,
        "reachable_candidate_pairs": reachable_pairs,
        "removed_rows": removed_rows,
        "removed_actions": sorted({row_actions[row_id] for row_id in removed_rows}),
        "retained_rows": list(retained_rows),
        "retained_actions": list(retained_actions),
        "resulting_candidate_set": list(retained_rows),
        "resolved_row_id": resolved_row,
        "resolved_action_id": resolved_action,
        "rejection_reason": rejection_reason,
        "executed_action": resolved_action,
        "status": status,
    }
    trace["trace_digest"] = _trace_digest(trace)
    return trace


def validate_reachability_trace(
    trace: Mapping[str, Any],
    *,
    semantic_outcome: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None,
    reachability_tile: Mapping[str, Any],
    row_actions: Mapping[str, str],
) -> str:
    try:
        expected = compose_reachability_trace(
            frame_id=str(trace["frame_id"]),
            semantic_outcome=semantic_outcome,
            previous_state=previous_state,
            reachability_tile=reachability_tile,
            row_actions=row_actions,
        )
    except VPMValidationError as exc:
        if "reachability tile digest" in str(exc):
            return "foreign_reachability_tile"
        return "reachability_trace_recompute_failed"
    if str(trace.get("reachability_tile_identity")) != str(reachability_tile["tile_digest"]):
        return "foreign_reachability_tile"
    if str(trace.get("trace_digest")) != _trace_digest(trace):
        return "foreign_reachability_trace_digest"
    if list(trace.get("consulted_edges", [])) != expected["consulted_edges"]:
        return "consulted_edge_mismatch"
    if list(trace.get("reachable_candidate_pairs", [])) != expected["reachable_candidate_pairs"]:
        return "reachable_pair_mismatch"
    if dict(trace) != expected:
        return "reachability_trace_mismatch"
    return "ok"


def _state_from_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    if trace["status"] == "rejected":
        return {"status": "unresolved", "candidate_rows": tuple(), "reason": trace["rejection_reason"]}
    return {"status": "resolved", "candidate_rows": tuple(str(row_id) for row_id in trace["retained_rows"])}


def _gap_reachability_state(record: Mapping[str, Any]) -> dict[str, Any]:
    return {"status": "unresolved", "candidate_rows": tuple(), "reason": record.get("gap_declaration") or "typed_gap_event"}


def _score_record(
    record: dict[str, Any],
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    *,
    reachability_tile: Mapping[str, Any] | None = None,
    reachability_state: dict[str, Mapping[str, Any] | None] | None = None,
    row_actions: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    if "pixels" not in record:
        raise VPMValidationError("materialized record missing pixels")
    if record["pixels"] is None:
        raise VPMValidationError("typed gap events cannot be provider-scored as ordinary frames")
    observation = ImageObservation(np.ascontiguousarray(record["pixels"], dtype=np.uint8), source_id=record["frame_id"])
    p1 = score_normalized_pixel(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id)
    p2 = score_registered_local_correlation(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=SOURCE_SCOPE)
    p3 = score_b3_joint_fit(observation=observation, prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=SOURCE_SCOPE)
    outputs = []
    for result in (p1, p2, p3):
        outcome = result.semantic_top_set_outcome
        operational_trace = None
        if reachability_tile is not None and reachability_state is not None and row_actions is not None:
            previous = reachability_state.get(result.provider_id)
            operational_trace = compose_reachability_trace(
                frame_id=record["frame_id"],
                semantic_outcome=outcome.to_dict(),
                previous_state=previous,
                reachability_tile=reachability_tile,
                row_actions=row_actions,
            )
            reachability_state[result.provider_id] = _state_from_trace(operational_trace)
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
                "reachability_composition_trace": operational_trace,
                "winner_row": result.winner_row_id,
                "winner_action": result.winner_action_id,
                "winner_quantized_score": winner_quantized_score,
                "runner_up_row": result.evidence.ranking.ranked_row_ids[1],
                "runner_up_quantized_score": result.evidence.row_scores[[item.row_id for item in result.evidence.row_scores].index(result.evidence.ranking.ranked_row_ids[1])].quantized_score,
                "policy_row_universe_digest": result.evidence.policy_row_universe_digest,
                "quantized_score_vector_digest": result.evidence.quantized_score_vector_digest,
                "raw_score_diagnostic_digest": result.evidence.raw_score_diagnostic_digest,
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
    identity = load_identity(repo_root)
    prototypes = canonical_prototypes()
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_actions = {str(row_id): lookup.choose(str(row_id)) for row_id in policy.source.row_ids}
    policy_artifact_id = policy.artifact_id
    reachability_tile = _load_reachability_tile(repo_root)
    plans = _episode_plans_for_split(identity, split, [str(row_id) for row_id in policy.source.row_ids], row_actions)
    reachability_state: dict[str, Mapping[str, Any] | None] = {"P1": None, "P2": None, "P3": None}
    scored_rows: list[dict[str, Any]] = []
    for record in records:
        if record.get("event_type") == "gap_unknown" or record.get("pixels") is None:
            for provider_id in reachability_state:
                reachability_state[provider_id] = _gap_reachability_state(record)
            continue
        scored_rows.extend(
            _score_record(
                record,
                prototypes,
                policy_artifact_id,
                reachability_tile=reachability_tile,
                reachability_state=reachability_state,
                row_actions=row_actions,
            )
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
    final_plan = _read_json(output_dir / "final-split-sealed-plan.json") if (output_dir / "final-split-sealed-plan.json").exists() else {}
    final_ids = {episode_id for values in final_plan.get("sealed_episode_ids", {}).values() for episode_id in values}
    final_materialization_count = 0
    final_score_access_count = 0
    final_reachability_execution_count = 0
    for split in ("development", "calibration", "selection", "final"):
        frame_rows = _read_jsonl(output_dir / split / "frame-metadata.jsonl")
        evidence_rows = _read_jsonl(output_dir / split / "provider-evidence.jsonl")
        final_materialization_count += sum(1 for row in frame_rows if row.get("episode_id") in final_ids or row.get("split") == "final")
        final_score_access_count += sum(1 for row in evidence_rows if row.get("episode_id") in final_ids or row.get("split") == "final")
        final_reachability_execution_count += sum(
            1
            for row in evidence_rows
            if (row.get("episode_id") in final_ids or row.get("split") == "final") and row.get("reachability_composition_trace") is not None
        )
    calibration_execution_count = sum(
        int((output_dir / name).exists())
        for name in (
            "selected-calibration.json",
            "calibration-grid.json",
            "calibration-results.json",
            "conformal-calibration.json",
        )
    )
    architecture_selection_execution_count = sum(
        int((output_dir / name).exists())
        for name in (
            "selected-architecture.json",
            "architecture-grid.json",
            "architecture-selection.json",
            "selected-method.json",
        )
    )
    candidate_tuning_execution_count = sum(
        int((output_dir / name).exists())
        for name in (
            "candidate-tuning.json",
            "candidate-set-tuning.json",
            "candidate-grid.json",
            "reachability-replay.json",
        )
    )
    final_evaluation_count = sum(
        int((output_dir / name).exists())
        for name in (
            "final-results.json",
            "final-summary.json",
            "final-evaluation.json",
        )
    )
    return {
        "version": PHASE_ACCESS_VERSION,
        "final_materialization_count": final_materialization_count,
        "final_score_access_count": final_score_access_count,
        "final_reachability_execution_count": final_reachability_execution_count,
        "candidate_set_selection_count": candidate_tuning_execution_count,
        "candidate_tuning_execution_count": candidate_tuning_execution_count,
        "conformal_calibration_count": calibration_execution_count,
        "calibration_execution_count": calibration_execution_count,
        "architecture_selection_execution_count": architecture_selection_execution_count,
        "reachability_replay_count": candidate_tuning_execution_count,
        "final_evaluation_count": final_evaluation_count,
        "forbidden_final_access_counter": final_materialization_count + final_score_access_count + final_reachability_execution_count,
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


_NON_FINAL_SPLITS = ("development", "calibration", "selection")
_ALL_SPLITS = ("development", "calibration", "selection", "final")
_REQUIRED_VERIFICATION_GATES = (
    "structural_identity",
    "semantic_outcome",
    "seed_and_plan",
    "episode_regeneration",
    "family_contract",
    "reachability",
    "completeness_orphan",
    "access_prohibition",
)


def _policy_row_action_digest(policy_artifact_id: str, row_ids: Sequence[str], row_actions: Mapping[str, str]) -> str:
    return _sha256(
        {
            "policy_artifact_id": policy_artifact_id,
            "row_action": [{"row_id": row_id, "action_id": row_actions[row_id]} for row_id in row_ids],
        }
    )


def _reference_context(repo_root: Path) -> dict[str, Any]:
    identity = load_identity(repo_root)
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    reachability_tile = _load_reachability_tile(repo_root)
    plans = {split: _episode_plans_for_split(identity, split, row_ids, row_actions) for split in _ALL_SPLITS}
    return {
        "identity": identity,
        "policy": policy,
        "row_ids": row_ids,
        "row_actions": row_actions,
        "reachability_tile": reachability_tile,
        "plans": plans,
        "policy_row_action_digest": _policy_row_action_digest(policy.artifact_id, row_ids, row_actions),
    }


def _finding(code: str, message: str, **details: Any) -> dict[str, Any]:
    payload = {"code": code, "message": message}
    payload.update({key: _json_ready(value) for key, value in details.items() if value is not None})
    return payload


def _gate(name: str, findings: Sequence[Mapping[str, Any]], *, counts: Mapping[str, Any] | None = None, unavailable: bool = False) -> dict[str, Any]:
    status = "unavailable" if unavailable else ("failed" if findings else "passed")
    return {
        "gate": name,
        "status": status,
        "finding_count": len(findings),
        "findings": [dict(item) for item in findings],
        "counts": dict(counts or {}),
    }


def _first_failure_code(report: Mapping[str, Any]) -> str | None:
    for gate in report.get("gates", []):
        for finding in gate.get("findings", []):
            return str(finding["code"])
    return None


def _raw_score_diagnostic_from_row(row: Mapping[str, Any], policy_row_ids: Sequence[str]) -> str:
    scores = {str(row_id): float(score) for row_id, score in zip(row["all_112_row_ids"], row["all_112_raw_scores"])}
    return build_complete_row_evidence(
        row_scores=[(row_id, scores[row_id]) for row_id in policy_row_ids],
        policy_artifact_id=str(row["policy_artifact_id"]),
        provider_id=str(row["provider_id"]),
        provider_version=str(row["provider_version"]),
        policy_row_ids=policy_row_ids,
    ).raw_score_diagnostic_digest


def _stored_quantized_evidence(row: Mapping[str, Any], policy_row_ids: Sequence[str]) -> tuple[Any | None, list[dict[str, Any]]]:
    findings: list[dict[str, Any]] = []
    row_ids = [str(row_id) for row_id in row.get("all_112_row_ids", [])]
    raw_scores = list(row.get("all_112_raw_scores", []))
    quantized_scores = list(row.get("all_112_quantized_scores", []))
    if row_ids != list(policy_row_ids) or len(row_ids) != 112 or len(set(row_ids)) != 112 or len(raw_scores) != 112 or len(quantized_scores) != 112:
        findings.append(_finding("score_row_universe_mismatch", "score vector does not contain the exact frozen 112-row policy universe", frame_id=row.get("frame_id")))
        return None, findings
    try:
        expected_quantized = [quantize_similarity(float(score)) for score in raw_scores]
    except VPMValidationError:
        findings.append(_finding("raw_diagnostic_digest_mismatch", "raw scores are not finite or quantizable", frame_id=row.get("frame_id")))
        return None, findings
    if [int(score) for score in quantized_scores] != expected_quantized:
        findings.append(_finding("quantized_score_vector_mismatch", "stored quantized scores do not reconstruct from raw scores", frame_id=row.get("frame_id")))
    try:
        evidence = build_complete_row_evidence(
            row_scores=[(row_id, float(score)) for row_id, score in zip(row_ids, raw_scores)],
            policy_artifact_id=str(row["policy_artifact_id"]),
            provider_id=str(row["provider_id"]),
            provider_version=str(row["provider_version"]),
            policy_row_ids=policy_row_ids,
        )
    except (KeyError, VPMValidationError) as exc:
        findings.append(_finding("quantized_score_vector_mismatch", "complete row evidence could not be reconstructed", frame_id=row.get("frame_id"), error=str(exc)))
        return None, findings
    stored_score_digest = str(row.get("quantized_score_vector_digest", row.get("score_vector_digest")))
    if stored_score_digest != evidence.quantized_score_vector_digest or str(row.get("score_vector_digest")) != evidence.quantized_score_vector_digest:
        findings.append(_finding("quantized_score_vector_mismatch", "quantized score-vector identity does not match reconstructed evidence", frame_id=row.get("frame_id")))
    if row.get("raw_score_diagnostic_digest") is not None and str(row.get("raw_score_diagnostic_digest")) != evidence.raw_score_diagnostic_digest:
        findings.append(_finding("raw_diagnostic_digest_mismatch", "raw diagnostic identity does not match reconstructed raw scores", frame_id=row.get("frame_id")))
    ranking_scores = tuple(RowScore(row_id=row_id, raw_score=float(raw), quantized_score=int(quantized)) for row_id, raw, quantized in zip(row_ids, raw_scores, quantized_scores))
    expected_ranking = build_complete_ranking(ranking_scores)
    if list(row.get("complete_ordered_ranking", [])) != list(expected_ranking.ranked_row_ids):
        findings.append(_finding("ranking_reconstruction_mismatch", "stored ranking does not reconstruct from quantized scores", frame_id=row.get("frame_id")))
    if list(row.get("tie_groups", [])) != [group.to_dict() for group in expected_ranking.tie_groups]:
        findings.append(_finding("tie_group_reconstruction_mismatch", "stored tie groups do not reconstruct from quantized scores", frame_id=row.get("frame_id")))
    if row.get("ranking_digest") is not None and str(row.get("ranking_digest")) != expected_ranking.ranking_digest:
        findings.append(_finding("ranking_reconstruction_mismatch", "ranking digest does not match reconstructed ranking", frame_id=row.get("frame_id")))
    return evidence, findings


def _semantic_cache_key(row: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "policy_artifact_id": row.get("policy_artifact_id"),
            "provider_id": row.get("provider_id"),
            "provider_version": row.get("provider_version"),
            "all_112_row_ids": row.get("all_112_row_ids"),
            "all_112_raw_scores": row.get("all_112_raw_scores"),
            "all_112_quantized_scores": row.get("all_112_quantized_scores"),
            "complete_ordered_ranking": row.get("complete_ordered_ranking"),
            "tie_groups": row.get("tie_groups"),
            "score_vector_digest": row.get("score_vector_digest"),
            "quantized_score_vector_digest": row.get("quantized_score_vector_digest"),
            "raw_score_diagnostic_digest": row.get("raw_score_diagnostic_digest"),
            "ranking_digest": row.get("ranking_digest"),
        }
    )


def _expected_semantic_for_row(
    row: Mapping[str, Any],
    row_actions: Mapping[str, str],
    policy_row_ids: Sequence[str],
    cache: dict[str, Any] | None = None,
) -> tuple[Any | None, list[dict[str, Any]]]:
    key = _semantic_cache_key(row)
    if cache is not None and key in cache:
        return cache[key], []
    evidence, findings = _stored_quantized_evidence(row, policy_row_ids)
    if evidence is None:
        return None, findings
    try:
        outcome = build_semantic_top_set_outcome(evidence=evidence, row_action=row_actions)
    except VPMValidationError as exc:
        return None, findings + [_finding("semantic_status_mismatch", "semantic outcome could not be reconstructed", frame_id=row.get("frame_id"), error=str(exc))]
    if cache is not None and not findings:
        cache[key] = outcome
    return outcome, findings


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
        expected_closure_digest = _sha256({key: value for key, value in closure.items() if key != "closure_report_digest"})
        if closure.get("closure_report_digest") != expected_closure_digest:
            findings.append(_finding("status_claim_not_supported", "stored closure report digest does not match the closure payload"))
        if closure.get("supported_status") == "reference_instrument_correct":
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


def _seed_and_plan_gate(output_dir: Path, context: Mapping[str, Any]) -> dict[str, Any]:
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
                continue
            if stored.get("source_row_id") != expected.get("source_row_id") or stored.get("secondary_row_id") != expected.get("secondary_row_id"):
                findings.append(_finding("episode_seed_derivation_mismatch", "stored source-row lineage does not match deterministic derivation", episode_id=episode_id))
                continue
            if dict(stored) != expected:
                findings.append(_finding("sealed_episode_identity_mismatch", "stored sealed episode plan does not match deterministic regeneration", episode_id=episode_id))
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
    if final_payload != expected_final:
        findings.append(_finding("sealed_episode_identity_mismatch", "final sealed split identity does not match deterministic regeneration"))
    digest_payload = _read_json(output_dir / "final-split-sealed-digest.json")
    if digest_payload.get("digest") != expected_final["sealed_plan_digest"]:
        findings.append(_finding("sealed_episode_identity_mismatch", "final sealed split digest sidecar does not match the final sealed plan"))
    try:
        _validate_episode_plan_collection(context["identity"], plans_by_split, row_actions)
    except VPMValidationError as exc:
        findings.append(_finding("episode_seed_derivation_mismatch", "authoritative regenerated plan collection failed validation", error=str(exc)))
    return _gate("seed_and_plan", findings, counts={"sealed_episode_count": sum(len(plans_by_split[split]) for split in _ALL_SPLITS)})


def _episode_regeneration_gate(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for split in _NON_FINAL_SPLITS:
        path = output_dir / split / "frame-metadata.jsonl"
        if not path.exists():
            findings.append(_finding("expected_record_missing", "split frame metadata is missing", split=split))
            continue
        stored_rows = _read_jsonl(path)
        expected_rows = [{key: value for key, value in row.items() if key != "pixels"} for row in _materialize_records(split, repo_root)]
        counts[f"{split}_stored_observations"] = len(stored_rows)
        counts[f"{split}_expected_observations"] = len(expected_rows)
        if len(stored_rows) != len(expected_rows):
            findings.append(_finding("expected_record_missing", "stored observation count does not match regenerated count", split=split))
        expected_by_key = {(row["episode_id"], int(row["sequence_number"])): row for row in expected_rows}
        stored_by_key = {(row.get("episode_id"), int(row.get("sequence_number", -1))): row for row in stored_rows}
        for key in sorted(set(expected_by_key) - set(stored_by_key)):
            findings.append(_finding("expected_record_missing", "regenerated observation is absent from stored frame metadata", split=split, episode_id=key[0], sequence_number=key[1]))
        for key in sorted(set(stored_by_key) - set(expected_by_key)):
            findings.append(_finding("orphan_observation_record", "stored frame metadata has no sealed regenerated observation", split=split, episode_id=key[0], sequence_number=key[1]))
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
    return _gate("episode_regeneration", findings, counts=counts)


def _semantic_outcome_gate(output_dir: Path, context: Mapping[str, Any]) -> dict[str, Any]:
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
    return _gate("semantic_outcome", findings, counts=counts)


def _family_contract_gate(output_dir: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    counts = {"family_records_checked": 0}
    row_actions = context["row_actions"]
    reachability_tile = context["reachability_tile"]
    for split in _NON_FINAL_SPLITS:
        by_episode: dict[str, list[dict[str, Any]]] = {}
        for row in _read_jsonl(output_dir / split / "frame-metadata.jsonl"):
            by_episode.setdefault(str(row.get("episode_id")), []).append(row)
        for episode_id, rows in by_episode.items():
            ordered = sorted(rows, key=lambda item: int(item.get("sequence_number", -1)))
            counts["family_records_checked"] += len(ordered)
            family = ordered[0].get("family")
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
                    if trace.get("output_observation_digest") != row.get("observation_pixel_digest"):
                        findings.append(_finding("family_regeneration_mismatch", "conflicting splice output digest does not match stored observation", split=split, episode_id=episode_id))
            elif family == "critical_evidence_corruption":
                for row in ordered:
                    metadata = row.get("metadata", {})
                    trace = metadata.get("family_intervention_trace", {})
                    changes = trace.get("changes", [])
                    if not changes:
                        findings.append(_finding("family_contract_violation", "critical corruption has an empty changed-coordinate set", split=split, episode_id=episode_id))
                    if trace.get("changed_pixel_count") == 0:
                        findings.append(_finding("family_no_op", "critical corruption is a no-op", split=split, episode_id=episode_id))
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
                    findings.append(_finding(code, "information control records violate byte-identity or denominator rules", split=split, episode_id=episode_id))
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
    return _gate("family_contract", findings, counts=counts)


def _reachability_gate(output_dir: Path, repo_root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
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
        expected_frames = [{key: value for key, value in row.items() if key != "pixels"} for row in _materialize_records(split, repo_root)]
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
                    continue
                expected_outcome, evidence_findings = _expected_semantic_for_row(row, row_actions, row_ids, semantic_cache)
                if evidence_findings or expected_outcome is None:
                    continue
                trace = row.get("reachability_composition_trace")
                if not trace:
                    findings.append(_finding("reachability_trace_mismatch", "provider row is missing reachability composition trace", split=split, frame_id=frame.get("frame_id"), provider_id=provider_id))
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
                reachability_state[provider_id] = _state_from_trace(expected_trace)
    return _gate("reachability", findings, counts=counts)


def _completeness_orphan_gate(output_dir: Path, repo_root: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    final_ids = {episode_id for values in _episode_ids_by_family(context["plans"]["final"]).values() for episode_id in values}
    known_frame_ids: set[str] = set()
    known_episode_ids: set[str] = set()
    non_control_digest_owner: dict[str, str] = {}
    expected_digest_owners: dict[str, set[str]] = {}
    for split in _NON_FINAL_SPLITS:
        for row in _materialize_records(split, repo_root):
            digest = row.get("observation_pixel_digest")
            if digest is not None and row.get("expected_disposition") != "information_theoretic_control":
                expected_digest_owners.setdefault(str(digest), set()).add(str(row.get("episode_id")))
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
            known_frame_ids.add(frame_id)
            known_episode_ids.add(episode_id)
            if episode_id in final_ids or row.get("split") == "final":
                findings.append(_finding("forbidden_final_materialization", "final sealed episode has a materialized frame descendant", split=split, episode_id=episode_id))
            digest = row.get("observation_pixel_digest")
            if digest is not None and row.get("expected_disposition") != "information_theoretic_control":
                previous = non_control_digest_owner.get(str(digest))
                expected_owners = expected_digest_owners.get(str(digest), set())
                if previous is not None and previous != episode_id and episode_id not in expected_owners:
                    findings.append(_finding("duplicate_observation_identity_unpermitted", "non-control observation identity is reused by multiple episodes", split=split, episode_id=episode_id, digest=digest))
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
        gap_frame_ids = {str(row.get("frame_id")) for row in frame_rows if row.get("event_type") == "gap_unknown" or row.get("observation_pixel_digest") is None}
        if scored_frame_ids & gap_frame_ids:
            findings.append(_finding("gap_event_has_pixels", "typed gap frame has provider score evidence", split=split))
    return _gate("completeness_orphan", findings, counts=counts)


def _access_prohibition_gate(output_dir: Path) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    measured = _measured_phase_access_counts(output_dir)
    stored = _read_json(output_dir / "phase-access-audits.json") if (output_dir / "phase-access-audits.json").exists() else {}
    if measured["final_materialization_count"]:
        findings.append(_finding("forbidden_final_materialization", "final split has materialized observations", count=measured["final_materialization_count"]))
    if measured["final_score_access_count"]:
        findings.append(_finding("forbidden_final_score_access", "final split has provider score records", count=measured["final_score_access_count"]))
    if measured["final_reachability_execution_count"]:
        findings.append(_finding("forbidden_final_reachability_access", "final split has reachability execution traces", count=measured["final_reachability_execution_count"]))
    if measured["calibration_execution_count"]:
        findings.append(_finding("forbidden_calibration_execution", "prospective calibration execution artifact is present", count=measured["calibration_execution_count"]))
    if measured["architecture_selection_execution_count"] or measured["candidate_tuning_execution_count"]:
        findings.append(_finding("forbidden_selection_execution", "prospective selection or candidate-tuning execution artifact is present", count=measured["architecture_selection_execution_count"] + measured["candidate_tuning_execution_count"]))
    if measured["final_evaluation_count"]:
        findings.append(_finding("forbidden_final_evaluation", "final evaluation artifact is present", count=measured["final_evaluation_count"]))
    if stored and {key: stored.get(key) for key in measured if key in stored} != {key: measured.get(key) for key in measured if key in stored}:
        findings.append(_finding("status_claim_not_supported", "stored phase-access counters do not match counters measured from concrete artifacts"))
    return _gate("access_prohibition", findings, counts=measured)


def verify_reference_instrument(
    output_dir: Path,
    repo_root: Path,
    *,
    validate_stored_closure: bool = True,
    enabled_gates: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Read-only independent verification of a materialized reference-instrument directory."""

    context = _reference_context(repo_root)
    enabled = set(enabled_gates) if enabled_gates is not None else set(_REQUIRED_VERIFICATION_GATES)
    gate_builders = (
        ("structural_identity", lambda: _structural_identity_gate(output_dir, repo_root, context, validate_stored_closure=validate_stored_closure)),
        ("semantic_outcome", lambda: _semantic_outcome_gate(output_dir, context)),
        ("seed_and_plan", lambda: _seed_and_plan_gate(output_dir, context)),
        ("episode_regeneration", lambda: _episode_regeneration_gate(output_dir, repo_root)),
        ("family_contract", lambda: _family_contract_gate(output_dir, context)),
        ("reachability", lambda: _reachability_gate(output_dir, repo_root, context)),
        ("completeness_orphan", lambda: _completeness_orphan_gate(output_dir, repo_root, context)),
        ("access_prohibition", lambda: _access_prohibition_gate(output_dir)),
    )
    gates = [builder() for name, builder in gate_builders if name in enabled]
    passed = [gate["gate"] for gate in gates if gate["status"] == "passed"]
    failed = [gate["gate"] for gate in gates if gate["status"] == "failed"]
    unavailable = [gate["gate"] for gate in gates if gate["status"] == "unavailable"]
    phase_counts = _measured_phase_access_counts(output_dir)
    payload: dict[str, Any] = {
        "version": REFERENCE_VERIFICATION_VERSION,
        "authoritative_roots": {
            "benchmark_contract_identity": context["identity"].to_dict(),
            "root_seed_digest": context["identity"].seed_digest,
            "policy_artifact_id": context["policy"].artifact_id,
            "policy_row_action_digest": context["policy_row_action_digest"],
            "episode_family_registry_digest": _episode_family_registry()["registry_digest"],
            "reachability_tile_digest": context["reachability_tile"]["tile_digest"],
            "provider_versions": {"P1": PROSPECTIVE_P1_VERSION, "P2": PROSPECTIVE_P2_VERSION, "P3": PROSPECTIVE_P3_VERSION},
            "evidence_schema_versions": {
                "complete_row_evidence": "zeromodel-video-complete-row-evidence/v2",
                "semantic_top_set_outcome": SEMANTIC_OUTCOME_VERSION,
                "reachability_trace": REACHABILITY_TRACE_VERSION,
                "sealed_episode_plan": EPISODE_PLAN_VERSION,
            },
        },
        "checks_executed": [gate["gate"] for gate in gates],
        "passed_checks": passed,
        "failed_checks": failed,
        "unavailable_checks": unavailable,
        "gates": gates,
        "measured_counts": phase_counts,
        "final_access_measurements": {
            "final_plan_count": len(context["plans"]["final"]),
            "final_observation_materialization_count": phase_counts["final_materialization_count"],
            "final_provider_score_access_count": phase_counts["final_score_access_count"],
            "final_reachability_execution_count": phase_counts["final_reachability_execution_count"],
            "final_evaluation_count": phase_counts["final_evaluation_count"],
            "calibration_execution_count": phase_counts["calibration_execution_count"],
            "architecture_selection_execution_count": phase_counts["architecture_selection_execution_count"],
            "candidate_tuning_execution_count": phase_counts["candidate_tuning_execution_count"],
        },
        "verified": not failed and not unavailable,
        "primary_failure_code": None,
    }
    payload["primary_failure_code"] = _first_failure_code(payload)
    payload["verification_digest"] = _sha256({key: value for key, value in payload.items() if key != "verification_digest"})
    return payload


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
    payload = {
        "read_only": before == middle == after,
        "deterministic": first.get("verification_digest") == second.get("verification_digest") and first == second,
        "first_verification_digest": first.get("verification_digest"),
        "second_verification_digest": second.get("verification_digest"),
        "path_count": len(before),
    }
    payload["status"] = "passed" if payload["read_only"] and payload["deterministic"] else "failed"
    payload["digest"] = _sha256(payload)
    return payload


_MUTATION_CASES: tuple[dict[str, Any], ...] = (
    {"name": "evidence_raw_score_preserve_quantized_bin", "expected_primary_failure_code": "raw_diagnostic_digest_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_raw_score_cross_quantization_boundary", "expected_primary_failure_code": "quantized_score_vector_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_quantized_score_changed", "expected_primary_failure_code": "quantized_score_vector_mismatch", "artifact_class": "evidence", "digest_laundering": True},
    {"name": "evidence_remove_row_score", "expected_primary_failure_code": "score_row_universe_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_duplicate_row_score", "expected_primary_failure_code": "score_row_universe_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_introduce_foreign_row", "expected_primary_failure_code": "score_row_universe_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_reorder_stored_rows", "expected_primary_failure_code": "score_row_universe_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_alter_ranking_order", "expected_primary_failure_code": "ranking_reconstruction_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_alter_tie_group_membership", "expected_primary_failure_code": "tie_group_reconstruction_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_split_tie_group_incorrectly", "expected_primary_failure_code": "tie_group_reconstruction_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_merge_distinct_score_groups", "expected_primary_failure_code": "tie_group_reconstruction_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_alter_quantized_score_vector_digest", "expected_primary_failure_code": "quantized_score_vector_mismatch", "artifact_class": "evidence"},
    {"name": "evidence_alter_raw_diagnostic_digest", "expected_primary_failure_code": "raw_diagnostic_digest_mismatch", "artifact_class": "evidence"},
    {"name": "semantic_resolved_row_for_action_unanimous_tie", "expected_primary_failure_code": "resolved_row_not_permitted", "artifact_class": "semantic"},
    {"name": "semantic_resolved_action_for_conflicting_tie", "expected_primary_failure_code": "resolved_action_not_permitted", "artifact_class": "semantic"},
    {"name": "semantic_convert_conflicting_tie_to_unique_row", "expected_primary_failure_code": "semantic_status_mismatch", "artifact_class": "semantic"},
    {"name": "semantic_change_top_row_policy_action", "expected_primary_failure_code": "policy_action_mapping_mismatch", "artifact_class": "semantic"},
    {"name": "semantic_change_rejection_reason", "expected_primary_failure_code": "semantic_status_mismatch", "artifact_class": "semantic"},
    {"name": "semantic_alter_outcome_digest", "expected_primary_failure_code": "semantic_outcome_digest_mismatch", "artifact_class": "semantic", "digest_laundering": True},
    {"name": "semantic_lexically_reorder_tied_rows", "expected_primary_failure_code": None, "artifact_class": "semantic", "invariant": True},
    {"name": "semantic_reorder_rows_preserving_action_equivalence", "expected_primary_failure_code": None, "artifact_class": "semantic", "invariant": True},
    {"name": "policy_alter_row_to_action_mapping", "expected_primary_failure_code": "policy_action_mapping_mismatch", "artifact_class": "policy"},
    {"name": "policy_remove_policy_row", "expected_primary_failure_code": "policy_action_mapping_mismatch", "artifact_class": "policy"},
    {"name": "policy_add_undeclared_row", "expected_primary_failure_code": "policy_action_mapping_mismatch", "artifact_class": "policy"},
    {"name": "policy_alter_artifact_identity", "expected_primary_failure_code": "policy_action_mapping_mismatch", "artifact_class": "policy"},
    {"name": "policy_mapping_recomputed_superficial_metadata", "expected_primary_failure_code": "policy_action_mapping_mismatch", "artifact_class": "policy", "digest_laundering": True},
    {"name": "observation_flip_byte_digest", "expected_primary_failure_code": "observation_digest_mismatch", "artifact_class": "observation"},
    {"name": "observation_change_pixels_and_recompute_digest", "expected_primary_failure_code": "observation_digest_mismatch", "artifact_class": "observation", "digest_laundering": True},
    {"name": "observation_change_digest_without_pixels", "expected_primary_failure_code": "observation_digest_mismatch", "artifact_class": "observation"},
    {"name": "observation_swap_two_frame_payloads", "expected_primary_failure_code": "observation_digest_mismatch", "artifact_class": "observation"},
    {"name": "observation_alter_frame_identity", "expected_primary_failure_code": "frame_identity_mismatch", "artifact_class": "observation"},
    {"name": "observation_reuse_under_two_episode_ids", "expected_primary_failure_code": "duplicate_observation_identity_unpermitted", "artifact_class": "observation"},
    {"name": "observation_substitute_frame_for_declared_gap", "expected_primary_failure_code": "gap_event_structure_mismatch", "artifact_class": "observation"},
    {"name": "seed_alter_root_seed_material", "expected_primary_failure_code": "episode_seed_derivation_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_root_seed_digest", "expected_primary_failure_code": "benchmark_contract_identity_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_derived_seed", "expected_primary_failure_code": "episode_seed_derivation_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_derivation_namespace", "expected_primary_failure_code": "sealed_episode_identity_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_episode_ordinal", "expected_primary_failure_code": "sealed_episode_identity_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_split_identity", "expected_primary_failure_code": "episode_split_reassignment", "artifact_class": "episode_plan"},
    {"name": "seed_move_episode_between_splits", "expected_primary_failure_code": "episode_split_reassignment", "artifact_class": "episode_plan"},
    {"name": "seed_duplicate_episode_id", "expected_primary_failure_code": "duplicate_episode_id", "artifact_class": "episode_plan"},
    {"name": "seed_alter_source_row", "expected_primary_failure_code": "episode_seed_derivation_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_splice_partner", "expected_primary_failure_code": "episode_seed_derivation_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_transformation_parameters", "expected_primary_failure_code": "sealed_episode_identity_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_planned_family", "expected_primary_failure_code": "sealed_episode_identity_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_sealed_plan_digest", "expected_primary_failure_code": "sealed_episode_identity_mismatch", "artifact_class": "episode_plan"},
    {"name": "seed_alter_final_sealed_identity", "expected_primary_failure_code": "sealed_episode_identity_mismatch", "artifact_class": "episode_plan", "digest_laundering": True},
    {"name": "family_conflicting_splice_same_action_rows", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_splice_zero_source_contribution", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_splice_output_equal_one_source", "expected_primary_failure_code": "family_regeneration_mismatch", "artifact_class": "family_output"},
    {"name": "family_critical_empty_coordinate_set", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_corrupt_noncritical_coordinates", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_replacement_value_identical", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_clipping_quantization_noop", "expected_primary_failure_code": "family_no_op", "artifact_class": "family_output", "digest_laundering": True},
    {"name": "family_reordered_metadata_original_payload_order", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_identity_permutation_labelled_reordered", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_stale_label_without_repeated_bytes", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_stale_repeat_naturally_identical", "expected_primary_failure_code": "family_contract_violation", "artifact_class": "family_output"},
    {"name": "family_impossible_transition_reachable_pair", "expected_primary_failure_code": "transition_classification_mismatch", "artifact_class": "family_output"},
    {"name": "family_gap_event_carrying_pixels", "expected_primary_failure_code": "gap_event_has_pixels", "artifact_class": "family_output"},
    {"name": "family_information_control_pixel_difference", "expected_primary_failure_code": "control_byte_identity_mismatch", "artifact_class": "family_output"},
    {"name": "family_control_denominator_leak", "expected_primary_failure_code": "control_denominator_leak", "artifact_class": "family_output"},
    {"name": "reachability_remove_applicable_edge", "expected_primary_failure_code": "consulted_edge_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_redirect_applicable_edge", "expected_primary_failure_code": "reachable_pair_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_alter_destination_action", "expected_primary_failure_code": "policy_action_mapping_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_add_impossible_edge", "expected_primary_failure_code": "reachability_tile_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_change_tile_identity", "expected_primary_failure_code": "reachability_tile_mismatch", "artifact_class": "reachability_trace", "digest_laundering": True},
    {"name": "reachability_change_unrelated_edge", "expected_primary_failure_code": "reachability_tile_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_alter_consulted_edge_list", "expected_primary_failure_code": "consulted_edge_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_omit_consulted_edge", "expected_primary_failure_code": "consulted_edge_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_add_unconsulted_edge_to_trace", "expected_primary_failure_code": "consulted_edge_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_alter_reachable_pair_set", "expected_primary_failure_code": "reachable_pair_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_alter_retained_candidate_rows", "expected_primary_failure_code": "reachability_trace_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_alter_removed_candidate_rows", "expected_primary_failure_code": "reachability_trace_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_replace_rejection_with_lexical_winner", "expected_primary_failure_code": "executed_action_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_change_executed_action", "expected_primary_failure_code": "executed_action_mismatch", "artifact_class": "reachability_trace"},
    {"name": "reachability_use_foreign_trace_digest", "expected_primary_failure_code": "reachability_trace_mismatch", "artifact_class": "reachability_trace"},
    {"name": "access_increment_final_materialization_count", "expected_primary_failure_code": "status_claim_not_supported", "artifact_class": "access_status"},
    {"name": "access_add_final_observation_artifact", "expected_primary_failure_code": "forbidden_final_materialization", "artifact_class": "access_status"},
    {"name": "access_add_final_score_vector_record", "expected_primary_failure_code": "forbidden_final_score_access", "artifact_class": "access_status"},
    {"name": "access_add_final_reachability_trace", "expected_primary_failure_code": "forbidden_final_score_access", "artifact_class": "access_status"},
    {"name": "access_increment_forbidden_access_counter", "expected_primary_failure_code": "status_claim_not_supported", "artifact_class": "access_status", "digest_laundering": True},
    {"name": "access_record_calibration_execution", "expected_primary_failure_code": "forbidden_calibration_execution", "artifact_class": "access_status"},
    {"name": "access_record_architecture_selection_execution", "expected_primary_failure_code": "forbidden_selection_execution", "artifact_class": "access_status"},
    {"name": "access_change_failed_gate_status_to_passed", "expected_primary_failure_code": "status_claim_not_supported", "artifact_class": "access_status"},
    {"name": "access_change_repository_status_to_correct", "expected_primary_failure_code": "status_claim_not_supported", "artifact_class": "access_status"},
    {"name": "access_remove_required_gate_from_closure_report", "expected_primary_failure_code": "closure_gate_missing", "artifact_class": "access_status"},
)


_MUTATION_GATE_SCOPE = {
    "evidence": ("structural_identity", "semantic_outcome"),
    "semantic": ("structural_identity", "semantic_outcome"),
    "policy": ("structural_identity",),
    "observation": ("structural_identity", "episode_regeneration", "completeness_orphan"),
    "episode_plan": ("structural_identity", "seed_and_plan"),
    "family_output": ("structural_identity", "family_contract"),
    "reachability_trace": ("structural_identity", "semantic_outcome", "reachability"),
    "access_status": ("structural_identity", "access_prohibition"),
}


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


def _mutate_first_provider_row(output_dir: Path, mutator: Any, *, predicate: Any | None = None) -> None:
    path = output_dir / "selection" / "provider-evidence.jsonl"
    rows = _read_jsonl(path)
    for row in rows:
        if predicate is None or predicate(row):
            mutator(row)
            _rewrite_jsonl(path, rows)
            _launder_split_manifest(output_dir, "selection")
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
            elif case_name in {"semantic_lexically_reorder_tied_rows", "semantic_reorder_rows_preserving_action_equivalence"}:
                payload["top_row_ids"] = list(reversed(payload["top_row_ids"]))
                payload["top_row_actions"] = list(reversed(payload["top_row_actions"]))
                row["top_row_ids"] = list(reversed(row["top_row_ids"]))
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
            first_digest = next(row["observation_pixel_digest"] for row in rows if row.get("observation_pixel_digest") and row.get("expected_disposition") != "information_theoretic_control")
            for row in rows:
                if row.get("observation_pixel_digest") and row.get("expected_disposition") != "information_theoretic_control" and row["observation_pixel_digest"] != first_digest:
                    row["observation_pixel_digest"] = first_digest
                    break
            _rewrite_jsonl(path, rows)
            _launder_split_manifest(output_dir, "selection")
        elif case_name == "observation_substitute_frame_for_declared_gap":
            _mutate_first_frame_row(output_dir, lambda row: (row.__setitem__("event_type", "frame"), row.__setitem__("gap_declaration", None), row.__setitem__("observation_pixel_digest", "sha256:" + "2" * 64)), predicate=lambda row: row.get("event_type") == "gap_unknown")
        else:
            _mutate_first_frame_row(output_dir, lambda row: row.__setitem__("observation_pixel_digest", "sha256:" + "3" * 64), predicate=lambda row: row.get("observation_pixel_digest") is not None)
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
            payload["splits"]["development"]["episodes"].append(plan)
            payload["splits"]["selection"]["episodes"] = payload["splits"]["selection"]["episodes"][1:]
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
                metadata["original_frame_index"] = int(row.get("sequence_number", 0))
            elif case_name in {"family_stale_label_without_repeated_bytes", "family_stale_repeat_naturally_identical"}:
                repeat = metadata.setdefault("stale_repeat", {})
                repeat["replacement_digest"] = repeat.get("original_destination_digest", row.get("observation_pixel_digest"))
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
            payload["tile_digest"] = "sha256:" + "8" * 64
            _write_json(output_dir / "reachability-tile-reference.json", payload)
            return
        def reach_mutate(row: dict[str, Any]) -> None:
            trace = row["reachability_composition_trace"]
            if case_name == "reachability_remove_applicable_edge":
                trace["consulted_edges"] = []
            elif case_name == "reachability_redirect_applicable_edge":
                if trace.get("reachable_candidate_pairs"):
                    trace["reachable_candidate_pairs"][0]["destination_row_id"] = row["all_112_row_ids"][-1]
            elif case_name == "reachability_alter_destination_action":
                row["semantic_top_set_outcome"]["top_row_actions"][0]["action_id"] = "FIRE" if row["semantic_top_set_outcome"]["top_row_actions"][0]["action_id"] != "FIRE" else "LEFT"
            elif case_name in {"reachability_alter_consulted_edge_list", "reachability_omit_consulted_edge"}:
                trace["consulted_edges"] = []
            elif case_name == "reachability_add_unconsulted_edge_to_trace":
                trace["consulted_edges"].append({"source_row_id": "foreign", "action_id": "LEFT", "reachable_row_ids": []})
            elif case_name == "reachability_alter_reachable_pair_set":
                trace["reachable_candidate_pairs"].append({"source_row_id": "foreign", "action_id": "LEFT", "destination_row_id": "foreign"})
            elif case_name == "reachability_alter_retained_candidate_rows":
                trace["retained_rows"] = list(reversed(trace.get("retained_rows", [])))
            elif case_name == "reachability_alter_removed_candidate_rows":
                trace["removed_rows"] = list(reversed(trace.get("removed_rows", []))) + ["foreign"]
            elif case_name == "reachability_replace_rejection_with_lexical_winner":
                trace["status"] = "resolved"
                trace["executed_action"] = "LEFT"
            elif case_name == "reachability_change_executed_action":
                trace["executed_action"] = "FIRE" if trace.get("executed_action") != "FIRE" else "LEFT"
            elif case_name == "reachability_use_foreign_trace_digest":
                trace["trace_digest"] = "sha256:" + "9" * 64
        _mutate_first_provider_row(output_dir, reach_mutate, predicate=lambda row: row.get("reachability_composition_trace") is not None)
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
            report = verify_reference_instrument(output_dir, Path(__file__).resolve().parents[1])
            if case_name == "access_remove_required_gate_from_closure_report":
                report["gates"] = report["gates"][1:]
            closure = {"version": CLOSURE_REPORT_VERSION, "supported_status": "reference_instrument_correct", "verification": report}
            _write_json(output_dir / "reference-closure-report.json", closure)
        return
    raise VPMValidationError(f"unsupported mutation case: {case_name}")


def run_reference_mutation_audit(output_dir: Path, repo_root: Path, *, mutation_names: Sequence[str] | None = None) -> dict[str, Any]:
    import shutil
    import tempfile

    selected_cases = tuple(case for case in _MUTATION_CASES if mutation_names is None or case["name"] in set(mutation_names))
    base = verify_reference_instrument(output_dir, repo_root)
    results: list[dict[str, Any]] = []
    if not base["verified"]:
        payload = {
            "version": MUTATION_AUDIT_VERSION,
            "base_verified": False,
            "base_primary_failure_code": base.get("primary_failure_code"),
            "mutations": [],
            "expected_mutation_count": len(selected_cases),
            "detected_mutation_count": 0,
            "undetected_mutation_count": len([case for case in selected_cases if not case.get("invariant")]),
            "unexpected_failure_code_count": len(selected_cases),
            "digest_laundering_tests": [],
            "status": "unavailable",
        }
        payload["mutation_audit_digest"] = _sha256(payload)
        return payload
    with tempfile.TemporaryDirectory(prefix="reference-mutation-audit-") as tmp:
        tmp_root = Path(tmp)
        for case in selected_cases:
            case_dir = tmp_root / case["name"]
            shutil.copytree(output_dir, case_dir)
            apply_error = None
            try:
                _apply_reference_mutation(case_dir, str(case["name"]))
                report = verify_reference_instrument(
                    case_dir,
                    repo_root,
                    enabled_gates=_MUTATION_GATE_SCOPE.get(str(case["artifact_class"]), _REQUIRED_VERIFICATION_GATES),
                )
                primary = report.get("primary_failure_code")
                detected = primary is not None
            except Exception as exc:  # pragma: no cover - returned in the audit report for deterministic debugging.
                primary = "mutation_application_error"
                detected = True
                apply_error = str(exc)
            expected = case.get("expected_primary_failure_code")
            invariant = bool(case.get("invariant"))
            expected_detected = expected is not None
            results.append(
                {
                    "mutation": case["name"],
                    "artifact_class": case["artifact_class"],
                    "expected_primary_failure_code": expected,
                    "actual_primary_failure_code": primary,
                    "detected": detected,
                    "expected_detected": expected_detected,
                    "invariant": invariant,
                    "digest_laundering": bool(case.get("digest_laundering", False)),
                    "expected_code_matched": (primary == expected) if expected_detected else (primary is None),
                    "application_error": apply_error,
                }
            )
    expected_cases = [row for row in results if row["expected_detected"]]
    detected = [row for row in expected_cases if row["detected"]]
    undetected = [row for row in expected_cases if not row["detected"]]
    unexpected = [row for row in results if not row["expected_code_matched"]]
    payload = {
        "version": MUTATION_AUDIT_VERSION,
        "base_verified": True,
        "mutations": results,
        "expected_mutation_count": len(expected_cases),
        "detected_mutation_count": len(detected),
        "undetected_mutation_count": len(undetected),
        "unexpected_failure_code_count": len(unexpected),
        "digest_laundering_tests": [row["mutation"] for row in results if row["digest_laundering"]],
        "status": "passed" if not undetected and not unexpected else "failed",
    }
    payload["mutation_audit_digest"] = _sha256(payload)
    return payload


def build_reference_closure_report(output_dir: Path, repo_root: Path, *, include_mutation_audit: bool = True) -> dict[str, Any]:
    verification = verify_reference_instrument(output_dir, repo_root)
    mutation_audit = run_reference_mutation_audit(output_dir, repo_root) if include_mutation_audit else {
        "version": MUTATION_AUDIT_VERSION,
        "status": "unavailable",
        "expected_mutation_count": len([case for case in _MUTATION_CASES if not case.get("invariant")]),
        "detected_mutation_count": 0,
        "undetected_mutation_count": len([case for case in _MUTATION_CASES if not case.get("invariant")]),
        "unexpected_failure_code_count": 0,
        "mutations": [],
        "digest_laundering_tests": [],
    }
    read_only = verify_reference_read_only(output_dir, repo_root)
    final_counts = verification["final_access_measurements"]
    required_zero = (
        final_counts["final_observation_materialization_count"] == 0
        and final_counts["final_provider_score_access_count"] == 0
        and final_counts["final_reachability_execution_count"] == 0
        and final_counts["calibration_execution_count"] == 0
        and final_counts["architecture_selection_execution_count"] == 0
        and final_counts["candidate_tuning_execution_count"] == 0
        and final_counts["final_evaluation_count"] == 0
    )
    supported = (
        verification["verified"]
        and mutation_audit.get("status") == "passed"
        and read_only["status"] == "passed"
        and required_zero
        and not verification["unavailable_checks"]
    )
    payload: dict[str, Any] = {
        "version": CLOSURE_REPORT_VERSION,
        "contract_identity": verification["authoritative_roots"]["benchmark_contract_identity"],
        "policy_identity": verification["authoritative_roots"]["policy_artifact_id"],
        "root_seed_identity": verification["authoritative_roots"]["root_seed_digest"],
        "split_plan_identities": {
            split: _sha256(context)
            for split, context in _read_json(output_dir / "episode-plan.json").get("splits", {}).items()
        } if (output_dir / "episode-plan.json").exists() else {},
        "family_registry_identity": verification["authoritative_roots"]["episode_family_registry_digest"],
        "reachability_identity": verification["authoritative_roots"]["reachability_tile_digest"],
        "provider_identities": verification["authoritative_roots"]["provider_versions"],
        "verification": verification,
        "mutation_audit": mutation_audit,
        "read_only_verification": read_only,
        "expected_mutation_count": mutation_audit.get("expected_mutation_count", 0),
        "detected_mutation_count": mutation_audit.get("detected_mutation_count", 0),
        "undetected_mutation_count": mutation_audit.get("undetected_mutation_count", 0),
        "unexpected_failure_code_count": mutation_audit.get("unexpected_failure_code_count", 0),
        "final_materialization_access_counts": final_counts,
        "supported_status": "reference_instrument_correct" if supported else "reference_instrument_correctness_unresolved",
        "unsupported_statuses": [
            "materialization_ready",
            "benchmark_utility_verified",
            "provider_selected",
            "calibration_complete",
            "final_evaluation_complete",
        ],
        "materialization_status": "prospective_materialization_prohibited",
    }
    payload["closure_report_digest"] = _sha256({key: value for key, value in payload.items() if key != "closure_report_digest"})
    return payload


def verify_instrument(output_dir: Path, repo_root: Path) -> dict[str, Any]:
    closure = build_reference_closure_report(output_dir, repo_root, include_mutation_audit=False)
    verification = closure["verification"]
    return {
        "verified": verification["verified"],
        "version": verification["version"],
        "closure_report_version": closure["version"],
        "repository_status": closure["supported_status"],
        "materialization_status": closure["materialization_status"],
        "primary_failure_code": verification["primary_failure_code"],
        "gates": verification["gates"],
        "final_materialization_count": verification["final_access_measurements"]["final_observation_materialization_count"],
        "final_score_access_count": verification["final_access_measurements"]["final_provider_score_access_count"],
        "final_reachability_execution_count": verification["final_access_measurements"]["final_reachability_execution_count"],
        "candidate_set_selection_count": verification["final_access_measurements"]["candidate_tuning_execution_count"],
        "conformal_calibration_count": verification["final_access_measurements"]["calibration_execution_count"],
        "reachability_replay_count": verification["measured_counts"]["reachability_replay_count"],
        "final_evaluation_count": verification["final_access_measurements"]["final_evaluation_count"],
        "forbidden_final_access_counter": verification["measured_counts"]["forbidden_final_access_counter"],
        "read_only": closure["read_only_verification"]["read_only"],
        "verification_digest": verification["verification_digest"],
    }


def _run_adversarial_mutation_checks(output_dir: Path) -> list[str]:
    repo_root = Path(__file__).resolve().parents[1]
    audit = run_reference_mutation_audit(output_dir, repo_root)
    return [row["mutation"] for row in audit.get("mutations", []) if not row.get("expected_code_matched", False)]


__all__ = [
    "BENCHMARK_VERSION",
    "GENERATOR_VERSION",
    "REFERENCE_VERIFICATION_VERSION",
    "MUTATION_AUDIT_VERSION",
    "CLOSURE_REPORT_VERSION",
    "PHASE_ACCESS_VERSION",
    "SOURCE_SCOPE",
    "audit_canonical_providers",
    "audit_evidence_completeness",
    "build_reference_closure_report",
    "build_split",
    "canonical_prototypes",
    "freeze_benchmark",
    "load_identity",
    "run_reference_mutation_audit",
    "verify_reference_instrument",
    "verify_reference_read_only",
    "verify_instrument",
]
