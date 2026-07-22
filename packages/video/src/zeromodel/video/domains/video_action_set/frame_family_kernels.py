from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from zeromodel.video.arcade_policy import (
    ACTIONS,
    CELL_PIXELS,
    COOLDOWN_BLOCKED_VALUE,
    COOLDOWN_READY_VALUE,
    TANK_VALUE,
    TARGET_VALUE,
    ShooterConfig,
    compile_policy_artifact,
    parse_state_row_id,
    state_row_id,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    CANONICAL_OBSERVATION_UNIVERSE_VERSION,
    CRITICAL_COORDINATE_SET_VERSION,
    CRITICAL_REGION_ID,
    FINAL_VISIBLE_TARGET_ACTION_EVIDENCE_VERSION,
    FRAME_SHAPE,
    SPLICE_MASK_VERSION,
    TARGET_REGION_ID,
)
from zeromodel.video.domains.video_action_set.observation_universe import (
    _canonical_collision_rows,
)
from zeromodel.video.domains.video_action_set.pixel_digest import array_digest


def _coordinate_set_digest(coordinates: Sequence[Sequence[int]]) -> str:
    return canonical_sha256({"coordinates": [[int(y), int(x)] for y, x in coordinates]})


def target_signal_coordinates(
    width: int = FRAME_SHAPE[1],
) -> tuple[tuple[int, int], ...]:
    return tuple((y, x) for y in range(2, 5) for x in range(int(width)))


def target_signal_mask(pixels: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    if tuple(array.shape) != FRAME_SHAPE:
        raise VPMValidationError(
            "target evidence mask requires a canonical frame shape"
        )
    mask = np.zeros(array.shape, dtype=bool)
    mask[2:5, :] = array[2:5, :] != 0
    return mask


def splice_evidence_counts(
    primary_pixels: np.ndarray, secondary_pixels: np.ndarray
) -> dict[str, int]:
    primary = np.ascontiguousarray(primary_pixels, dtype=np.uint8)
    secondary = np.ascontiguousarray(secondary_pixels, dtype=np.uint8)
    if primary.shape != secondary.shape:
        raise VPMValidationError("splice sources must have the same shape")
    primary_target = target_signal_mask(primary)
    secondary_target = target_signal_mask(secondary)
    secondary_additive = secondary_target & ~primary_target
    return {
        "primary_target_pixel_count": int(np.count_nonzero(primary_target)),
        "secondary_target_pixel_count": int(np.count_nonzero(secondary_target)),
        "secondary_additive_target_pixel_count": int(
            np.count_nonzero(secondary_additive)
        ),
        "target_overlap_pixel_count": int(
            np.count_nonzero(primary_target & secondary_target)
        ),
    }


def target_slot_signal_coordinates(
    slot: int, *, width: int = 7
) -> tuple[tuple[int, int], ...]:
    centre = int(slot) * CELL_PIXELS + CELL_PIXELS // 2
    return tuple(
        [
            (2, centre - 1),
            (2, centre),
            (2, centre + 1),
            (3, centre - 1),
            (3, centre),
            (3, centre + 1),
            (4, centre),
        ]
    )


def detect_visible_target_slots(
    pixels: np.ndarray, *, config: ShooterConfig = ShooterConfig()
) -> list[int]:
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    slots = []
    for slot in range(config.width):
        coords = target_slot_signal_coordinates(slot, width=config.width)
        values = [int(array[y, x]) for y, x in coords]
        if all(value == TARGET_VALUE for value in values):
            slots.append(slot)
    return slots


def detect_tank_slot(
    pixels: np.ndarray, *, config: ShooterConfig = ShooterConfig()
) -> int | None:
    array = np.ascontiguousarray(pixels, dtype=np.uint8)
    for slot in range(config.width):
        centre = int(slot) * CELL_PIXELS + CELL_PIXELS // 2
        coords = tuple(
            [
                (11, centre),
                (12, centre - 1),
                (12, centre),
                (12, centre + 1),
                (13, centre - 2),
                (13, centre - 1),
                (13, centre),
                (13, centre + 1),
                (13, centre + 2),
            ]
        )
        if all(int(array[y, x]) == TANK_VALUE for y, x in coords):
            return slot
    return None


def detect_cooldown_state(pixels: np.ndarray) -> int | None:
    block = np.ascontiguousarray(pixels, dtype=np.uint8)[7:9, -3:-1]
    values = {int(item) for item in block.reshape(-1)}
    if values == {COOLDOWN_READY_VALUE}:
        return 0
    if values == {COOLDOWN_BLOCKED_VALUE}:
        return 1
    return None


def final_visible_target_action_evidence(
    pixels: np.ndarray,
    row_actions: Mapping[str, str],
    *,
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    target_slots = detect_visible_target_slots(pixels, config=config)
    tank_slot = detect_tank_slot(pixels, config=config)
    cooldown = detect_cooldown_state(pixels)
    action_map: dict[str, str] = {}
    if tank_slot is not None and cooldown is not None:
        for target in target_slots:
            row_id = state_row_id(tank_slot, target, cooldown)
            action_map[str(target)] = str(row_actions[row_id])
    visible_actions = sorted(set(action_map.values()))
    payload = {
        "version": FINAL_VISIBLE_TARGET_ACTION_EVIDENCE_VERSION,
        "final_visible_target_slots": [int(item) for item in target_slots],
        "final_tank_slot": tank_slot,
        "final_cooldown": cooldown,
        "visible_target_action_map": action_map,
        "visible_action_set": visible_actions,
        "conflicting_action_evidence_present": len(visible_actions) >= 2,
    }
    payload["visible_action_evidence_digest"] = canonical_sha256(payload)
    return payload


def splice_pair_has_final_visible_action_conflict(
    primary_row_id: str, secondary_row_id: str, row_actions: Mapping[str, str]
) -> bool:
    tank, primary_target, cooldown = parse_state_row_id(str(primary_row_id))
    _secondary_tank, secondary_target, _secondary_cooldown = parse_state_row_id(
        str(secondary_row_id)
    )
    if (
        primary_target is None
        or secondary_target is None
        or primary_target == secondary_target
    ):
        return False
    implied = {
        row_actions[state_row_id(tank, int(primary_target), cooldown)],
        row_actions[state_row_id(tank, int(secondary_target), cooldown)],
    }
    return len(implied) >= 2


def critical_coordinates() -> tuple[tuple[int, int], ...]:
    return tuple((y, x) for y in range(7, 9) for x in range(25, 27))


def critical_coordinate_manifest(
    coordinates: Sequence[Sequence[int]] | None = None,
) -> dict[str, Any]:
    source_coordinates = critical_coordinates() if coordinates is None else coordinates
    coords = tuple((int(y), int(x)) for y, x in source_coordinates)
    if not coords:
        raise VPMValidationError("critical coordinate set cannot be empty")
    critical = set(critical_coordinates())
    if any(coord not in critical for coord in coords):
        raise VPMValidationError(
            "selected coordinate is not critical under the frozen definition"
        )
    if len(set(coords)) != len(coords):
        raise VPMValidationError("critical coordinate set cannot contain duplicates")
    return {
        "version": CRITICAL_COORDINATE_SET_VERSION,
        "criticality_source": "tiny_arcade_shooter_rendering",
        "criticality_region_id": CRITICAL_REGION_ID,
        "coordinates": [[y, x] for y, x in coords],
        "coordinate_set_digest": _coordinate_set_digest(coords),
    }


def splice_mask_manifest(
    mask_rows: Sequence[int] = (2, 3, 4), *, width: int = 28
) -> dict[str, Any]:
    rows = tuple(int(row) for row in mask_rows)
    if rows != (2, 3, 4):
        raise VPMValidationError(
            "conflicting splice mask is frozen to the rendered target signal rows"
        )
    if int(width) != FRAME_SHAPE[1]:
        raise VPMValidationError(
            "conflicting splice mask width must match the canonical frame shape"
        )
    coordinates = target_signal_coordinates(int(width))
    return {
        "version": SPLICE_MASK_VERSION,
        "mask_kind": "simultaneous_target_evidence",
        "target_region_id": TARGET_REGION_ID,
        "composition_rule": "copy primary observation, then add secondary target pixels where the primary has no target signal",
        "rows": list(rows),
        "coordinates": [[y, x] for y, x in coordinates],
        "coordinate_count": len(coordinates),
        "mask_digest": _coordinate_set_digest(coordinates),
    }


def _validated_splice_sources(
    *,
    primary_pixels: np.ndarray,
    secondary_pixels: np.ndarray,
    primary_action_id: str,
    secondary_action_id: str,
    mask_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if primary_action_id == secondary_action_id:
        raise VPMValidationError(
            "conflicting splice requires different governed actions"
        )
    primary = np.ascontiguousarray(primary_pixels, dtype=np.uint8)
    secondary = np.ascontiguousarray(secondary_pixels, dtype=np.uint8)
    if primary.shape != secondary.shape:
        raise VPMValidationError("splice sources must have the same shape")
    if tuple(primary.shape) != FRAME_SHAPE:
        raise VPMValidationError(
            "conflicting splice requires the canonical frame shape"
        )
    if mask_manifest.get("version") != SPLICE_MASK_VERSION:
        raise VPMValidationError("unsupported conflicting splice mask version")
    if mask_manifest.get("mask_kind") != "simultaneous_target_evidence":
        raise VPMValidationError(
            "conflicting splice requires the simultaneous target-evidence mask"
        )
    coordinates = tuple((int(y), int(x)) for y, x in mask_manifest["coordinates"])
    if coordinates != target_signal_coordinates(primary.shape[1]):
        raise VPMValidationError(
            "conflicting splice target coordinates do not match the frozen renderer geometry"
        )
    for y, x in coordinates:
        if y < 0 or y >= primary.shape[0] or x < 0 or x >= primary.shape[1]:
            raise VPMValidationError("splice coordinate out of bounds")
    return primary, secondary


def _splice_composition(primary: np.ndarray, secondary: np.ndarray) -> dict[str, Any]:
    primary_target = target_signal_mask(primary)
    secondary_target = target_signal_mask(secondary)
    secondary_overlay = secondary_target & ~primary_target
    primary_target_count = int(np.count_nonzero(primary_target))
    secondary_target_count = int(np.count_nonzero(secondary_target))
    secondary_effective = int(np.count_nonzero(secondary_overlay))
    if primary_target_count == 0 or secondary_target_count == 0:
        raise VPMValidationError(
            "splice requires visible target evidence from both sources"
        )
    if secondary_effective == 0:
        raise VPMValidationError("splice requires distinct secondary target evidence")
    output = np.array(primary, copy=True)
    output[secondary_overlay] = secondary[secondary_overlay]
    primary_effective = int(np.count_nonzero((primary != 0) & (output == primary)))
    changed_pixel_count = int(np.count_nonzero(output != primary))
    if changed_pixel_count == 0 or primary_effective == 0:
        raise VPMValidationError(
            "splice requires nonzero effective contribution from both sources"
        )
    if np.array_equal(output, primary) or np.array_equal(output, secondary):
        raise VPMValidationError(
            "splice output must not equal either source observation"
        )
    return {
        "output": output,
        "primary_target": primary_target,
        "secondary_target": secondary_target,
        "primary_target_count": primary_target_count,
        "secondary_target_count": secondary_target_count,
        "secondary_effective": secondary_effective,
        "primary_effective": primary_effective,
        "changed_pixel_count": changed_pixel_count,
    }


def _policy_row_actions() -> dict[str, str]:
    policy = compile_policy_artifact()
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    return {str(row_id): lookup.choose(str(row_id)) for row_id in policy.source.row_ids}


def _validated_final_splice_evidence(output: np.ndarray) -> dict[str, Any]:
    row_actions = _policy_row_actions()
    visible_action_evidence = final_visible_target_action_evidence(output, row_actions)
    if not visible_action_evidence["conflicting_action_evidence_present"]:
        raise VPMValidationError(
            "conflicting splice final visible target evidence does not imply conflicting actions"
        )
    canonical_collisions = _canonical_collision_rows(output)
    if canonical_collisions:
        raise VPMValidationError(
            "conflicting splice output collides with a canonical valid observation"
        )
    return visible_action_evidence


def _critical_region_primary_count(primary: np.ndarray, output: np.ndarray) -> int:
    critical_mask = np.zeros(primary.shape, dtype=bool)
    for y, x in critical_coordinates():
        critical_mask[y, x] = True
    return int(np.count_nonzero(critical_mask & (output == primary)))


def _conflicting_splice_trace_manifest(
    *,
    primary: np.ndarray,
    secondary: np.ndarray,
    primary_row_id: str,
    secondary_row_id: str,
    primary_action_id: str,
    secondary_action_id: str,
    mask_manifest: Mapping[str, Any],
    composition: Mapping[str, Any],
    visible_action_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    primary_target = composition["primary_target"]
    secondary_target = composition["secondary_target"]
    output = composition["output"]
    manifest = {
        "primary_source_row_id": primary_row_id,
        "primary_source_action_id": primary_action_id,
        "secondary_source_row_id": secondary_row_id,
        "secondary_source_action_id": secondary_action_id,
        "primary_source_digest": array_digest(primary),
        "secondary_source_digest": array_digest(secondary),
        "splice_mask_identity": mask_manifest["mask_digest"],
        "splice_mask_version": mask_manifest["version"],
        "target_region_id": TARGET_REGION_ID,
        "composition_rule": mask_manifest["composition_rule"],
        "primary_contributing_pixel_count": composition["primary_effective"],
        "secondary_contributing_pixel_count": composition["secondary_effective"],
        "changed_pixel_count": composition["changed_pixel_count"],
        "action_relevant_region_contribution_counts": {
            "primary_target_pixel_count": composition["primary_target_count"],
            "secondary_target_pixel_count": composition["secondary_target_count"],
            "secondary_additive_target_pixel_count": composition["secondary_effective"],
            "target_overlap_pixel_count": int(
                np.count_nonzero(primary_target & secondary_target)
            ),
        },
        "critical_region_contribution_counts": {
            "primary_pixel_count": _critical_region_primary_count(primary, output),
            "secondary_pixel_count": 0,
        },
        "canonical_universe_version": CANONICAL_OBSERVATION_UNIVERSE_VERSION,
        "final_visible_target_slots": visible_action_evidence[
            "final_visible_target_slots"
        ],
        "final_tank_slot": visible_action_evidence["final_tank_slot"],
        "final_cooldown": visible_action_evidence["final_cooldown"],
        "visible_target_action_map": visible_action_evidence[
            "visible_target_action_map"
        ],
        "visible_action_set": visible_action_evidence["visible_action_set"],
        "visible_action_evidence_digest": visible_action_evidence[
            "visible_action_evidence_digest"
        ],
        "canonical_collision_count": 0,
        "canonical_collision_rows": [],
        "output_observation_digest": array_digest(output),
        "expected_invalid_family_label": "conflicting_action_splice",
    }
    manifest["splice_trace_digest"] = canonical_sha256(manifest)
    return manifest


def apply_conflicting_splice(
    *,
    primary_pixels: np.ndarray,
    secondary_pixels: np.ndarray,
    primary_row_id: str,
    secondary_row_id: str,
    primary_action_id: str,
    secondary_action_id: str,
    mask_manifest: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    primary, secondary = _validated_splice_sources(
        primary_pixels=primary_pixels,
        secondary_pixels=secondary_pixels,
        primary_action_id=primary_action_id,
        secondary_action_id=secondary_action_id,
        mask_manifest=mask_manifest,
    )
    composition = _splice_composition(primary, secondary)
    output = composition["output"]
    visible_action_evidence = _validated_final_splice_evidence(output)
    manifest = _conflicting_splice_trace_manifest(
        primary=primary,
        secondary=secondary,
        primary_row_id=primary_row_id,
        secondary_row_id=secondary_row_id,
        primary_action_id=primary_action_id,
        secondary_action_id=secondary_action_id,
        mask_manifest=mask_manifest,
        composition=composition,
        visible_action_evidence=visible_action_evidence,
    )
    return output, manifest


def apply_critical_corruption(
    source_pixels: np.ndarray, coordinate_manifest: Mapping[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    source = np.ascontiguousarray(source_pixels, dtype=np.uint8)
    coords = tuple((int(y), int(x)) for y, x in coordinate_manifest["coordinates"])
    critical_coordinate_manifest(coords)
    output = np.array(source, copy=True)
    changes = []
    for y, x in coords:
        if y < 0 or y >= output.shape[0] or x < 0 or x >= output.shape[1]:
            raise VPMValidationError("critical coordinate outside image bounds")
        original = int(output[y, x])
        replacement = 255 if original != 255 else 0
        if replacement == original:
            raise VPMValidationError(
                "critical corruption replacement must change the value"
            )
        output[y, x] = replacement
        if int(output[y, x]) == original:
            raise VPMValidationError("critical corruption no-op after assignment")
        changes.append(
            {"y": y, "x": x, "original": original, "replacement": int(output[y, x])}
        )
    if not changes:
        raise VPMValidationError(
            "critical corruption requires at least one changed pixel"
        )
    if np.array_equal(source, output):
        raise VPMValidationError(
            "critical corruption must change the observation digest"
        )
    manifest = {
        "criticality_artifact_identity": CRITICAL_COORDINATE_SET_VERSION,
        "critical_region_id": CRITICAL_REGION_ID,
        "critical_coordinate_set_identity": coordinate_manifest[
            "coordinate_set_digest"
        ],
        "changes": changes,
        "changed_pixel_count": len(changes),
        "source_observation_digest": array_digest(source),
        "output_observation_digest": array_digest(output),
    }
    manifest["critical_corruption_digest"] = canonical_sha256(manifest)
    return output, manifest


__all__ = [
    "apply_conflicting_splice",
    "apply_critical_corruption",
    "critical_coordinate_manifest",
    "critical_coordinates",
    "detect_cooldown_state",
    "detect_tank_slot",
    "detect_visible_target_slots",
    "final_visible_target_action_evidence",
    "splice_evidence_counts",
    "splice_mask_manifest",
    "splice_pair_has_final_visible_action_conflict",
    "target_signal_coordinates",
    "target_signal_mask",
    "target_slot_signal_coordinates",
]
