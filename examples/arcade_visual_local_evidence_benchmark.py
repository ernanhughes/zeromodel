"""Fresh v3 local-evidence benchmark for bounded visual-address research."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from examples.arcade_shooter_policy import ACTIONS, ShooterConfig, compile_policy_artifact  # noqa: E402
from examples.arcade_visual_address_benchmark import ArcadeBenchmarkDataset  # noqa: E402
from examples.arcade_visual_sign_reader import (  # noqa: E402
    CELL_PIXELS,
    enumerate_visual_frames,
)
from zeromodel.video.arcade_policy import (
    COOLDOWN_BLOCKED_VALUE,
    COOLDOWN_READY_VALUE,
    TANK_VALUE,
    TARGET_VALUE,
)
from zeromodel.analysis.policy_properties import decode_key_value_row_id
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation.visual_address import ImageObservation
from research.visual.visual_dataset import (
    CorruptionFamilySpec,
    VisualDatasetManifest,
    VisualExampleRecord,
)
from research.visual.visual_corruptions import (  # noqa: E402
    add_integer_noise,
    mask_box,
    scale_intensity,
    translate_frame,
)
from research.visual.visual_experiment import EXPECTED_ACCEPT, EXPECTED_REJECT, IMPOSSIBILITY_CONTROL  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SOURCE_SCOPE = "arcade-visual-local-evidence-v3"
PROTOCOL_VERSION = "zeromodel-visual-local-evidence-protocol/v1"


def _state_distance(row_id: str, other_row_id: str) -> Tuple[int, int, int]:
    left = decode_key_value_row_id(row_id)
    right = decode_key_value_row_id(other_row_id)
    target_left = -1 if left["target"] is None else int(left["target"])
    target_right = -1 if right["target"] is None else int(right["target"])
    return (
        abs(int(left["tank"]) - int(right["tank"])),
        abs(target_left - target_right),
        abs(int(left["cooldown"]) - int(right["cooldown"])),
    )


def _target_band(frame: np.ndarray, competitor: np.ndarray) -> np.ndarray:
    result = np.array(frame, dtype=np.uint8, order="C", copy=True)
    result[0:6, :] = competitor[0:6, :]
    result.flags.writeable = False
    return result


def _cooldown_patch(frame: np.ndarray, competitor: np.ndarray) -> np.ndarray:
    result = np.array(frame, dtype=np.uint8, order="C", copy=True)
    result[7:9, -3:-1] = competitor[7:9, -3:-1]
    result.flags.writeable = False
    return result


def _tank_band(frame: np.ndarray, competitor: np.ndarray) -> np.ndarray:
    result = np.array(frame, dtype=np.uint8, order="C", copy=True)
    result[11:14, :] = competitor[11:14, :]
    result.flags.writeable = False
    return result


def _critical_target(frame: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    if state["target"] is None:
        raise ValueError("target intervention requires a visible target")
    centre = int(state["target"]) * CELL_PIXELS + CELL_PIXELS // 2
    return mask_box(frame, top=2, left=centre - 1, height=3, width=3, value=0)


def _critical_cooldown(frame: np.ndarray) -> np.ndarray:
    return mask_box(frame, top=7, left=frame.shape[1] - 3, height=2, width=2, value=0)


def _critical_tank(frame: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    centre = int(state["tank"]) * CELL_PIXELS + CELL_PIXELS // 2
    return mask_box(frame, top=11, left=max(0, centre - 2), height=3, width=5, value=0)


def _double_tank(frame: np.ndarray, config: ShooterConfig) -> np.ndarray:
    result = np.array(frame, dtype=np.uint8, order="C", copy=True)
    left_centre = CELL_PIXELS // 2
    right_centre = (config.width - 1) * CELL_PIXELS + CELL_PIXELS // 2
    for centre in (left_centre, right_centre):
        result[11, centre] = TANK_VALUE
        result[12, centre - 1 : centre + 2] = TANK_VALUE
        result[13, centre - 2 : centre + 3] = TANK_VALUE
    result.flags.writeable = False
    return result


def _choose_same_action_competitor(
    row_id: str,
    *,
    lookup: VPMPolicyLookup,
    canonical: Mapping[str, np.ndarray],
) -> str:
    action = lookup.choose(row_id)
    candidates = tuple(
        candidate
        for candidate in canonical
        if candidate != row_id and lookup.choose(candidate) == action
    )
    return min(candidates, key=lambda candidate: (_state_distance(row_id, candidate), candidate))


def _choose_conflicting_action_competitor(
    row_id: str,
    *,
    lookup: VPMPolicyLookup,
    canonical: Mapping[str, np.ndarray],
) -> str:
    action = lookup.choose(row_id)
    candidates = tuple(
        candidate
        for candidate in canonical
        if lookup.choose(candidate) != action
    )
    return min(candidates, key=lambda candidate: (_state_distance(row_id, candidate), candidate))


def _family_specs() -> Tuple[CorruptionFamilySpec, ...]:
    return (
        CorruptionFamilySpec(family_id="prototype-clean-v3", kind="clean"),
        CorruptionFamilySpec(family_id="prototype-palette-v3", kind="palette"),
        CorruptionFamilySpec(family_id="prototype-shift-right-v3", kind="translation"),
        CorruptionFamilySpec(family_id="prototype-brightness-v3", kind="brightness"),
        CorruptionFamilySpec(family_id="benign-calibration-contrast-v3", kind="contrast"),
        CorruptionFamilySpec(family_id="benign-calibration-translation-x-v3", kind="translation"),
        CorruptionFamilySpec(family_id="benign-calibration-photometric-v3", kind="brightness"),
        CorruptionFamilySpec(family_id="benign-calibration-noise-v3", kind="noise"),
        CorruptionFamilySpec(
            family_id="rejection-calibration-same-action-v3",
            kind="same_action_wrong_row",
            parameters={"distinguishable": True, "conflicting_action_evidence": False},
        ),
        CorruptionFamilySpec(
            family_id="rejection-calibration-conflicting-v3",
            kind="conflicting_action_near_neighbour",
            parameters={"distinguishable": True, "conflicting_action_evidence": True},
        ),
        CorruptionFamilySpec(
            family_id="rejection-calibration-compositional-v3",
            kind="compositional_invalid",
            parameters={"distinguishable": True, "conflicting_action_evidence": True},
        ),
        CorruptionFamilySpec(
            family_id="final-translation-heldout-v3",
            kind="translation",
            parameters={"distinguishable": False, "known_displacement": "varies"},
        ),
        CorruptionFamilySpec(
            family_id="final-translation-photometric-v3",
            kind="translation_plus_photometric",
            parameters={"distinguishable": False, "known_displacement": "varies"},
        ),
        CorruptionFamilySpec(
            family_id="final-translation-occlusion-v3",
            kind="translation_plus_local_occlusion",
            parameters={"distinguishable": False, "known_displacement": "varies"},
        ),
        CorruptionFamilySpec(
            family_id="final-translation-critical-v3",
            kind="translation_plus_critical_corruption",
            critical_evidence_removed=True,
            parameters={"distinguishable": True, "known_displacement": "varies"},
        ),
        CorruptionFamilySpec(
            family_id="final-same-action-wrong-row-v3",
            kind="same_action_wrong_row",
            parameters={"distinguishable": True, "conflicting_action_evidence": False},
        ),
        CorruptionFamilySpec(
            family_id="final-conflicting-action-near-v3",
            kind="conflicting_action_near_neighbour",
            parameters={"distinguishable": True, "conflicting_action_evidence": True},
        ),
        CorruptionFamilySpec(
            family_id="final-compositional-invalid-v3",
            kind="compositional_invalid",
            parameters={"distinguishable": True, "conflicting_action_evidence": True},
        ),
        CorruptionFamilySpec(
            family_id="final-information-impossible-v3",
            kind="information_theoretic_control",
            critical_evidence_removed=True,
            parameters={"distinguishable": False, "information_theoretic_impossible": True},
        ),
        CorruptionFamilySpec(
            family_id="final-beyond-bounds-translation-v3",
            kind="translation_out_of_declared_domain",
            parameters={"distinguishable": True, "known_displacement": "varies"},
        ),
    )


def _prototype_variant(frame: np.ndarray, family_id: str, index: int) -> np.ndarray:
    if family_id == "prototype-clean-v3":
        return scale_intensity(frame, numerator=97 + index)
    if family_id == "prototype-palette-v3":
        mapping = {
            TARGET_VALUE: 200 - 4 * index,
            TANK_VALUE: 240 - 2 * index,
            COOLDOWN_READY_VALUE: 60 + 3 * index,
            COOLDOWN_BLOCKED_VALUE: 145 + 2 * index,
        }
        result = np.array(frame, dtype=np.uint8, order="C", copy=True)
        for source, target in mapping.items():
            result[result == source] = target
        result.flags.writeable = False
        return result
    if family_id == "prototype-shift-right-v3":
        return translate_frame(frame, dx=1, fill=2 + index)
    if family_id == "prototype-brightness-v3":
        return scale_intensity(frame, numerator=88 + 4 * index, offset=1 + index)
    raise ValueError("unknown prototype family")


def _benign_variant(frame: np.ndarray, family_id: str, index: int) -> np.ndarray:
    if family_id == "benign-calibration-contrast-v3":
        return scale_intensity(frame, numerator=106 + 3 * index, offset=2)
    if family_id == "benign-calibration-translation-x-v3":
        return translate_frame(frame, dx=1 if index % 2 == 0 else -1, fill=5 + index)
    if family_id == "benign-calibration-photometric-v3":
        return scale_intensity(frame, numerator=92 + 5 * index, offset=3)
    if family_id == "benign-calibration-noise-v3":
        return add_integer_noise(frame, amplitude=1 + index, seed=2000 + index)
    raise ValueError("unknown benign family")


def _accepted_final_variant(frame: np.ndarray, state: Mapping[str, Any], family_id: str, index: int) -> np.ndarray:
    if family_id == "final-translation-heldout-v3":
        offsets = ((2, 0), (-2, 0), (0, -2))
        dx, dy = offsets[index % len(offsets)]
        return translate_frame(frame, dx=dx, dy=dy, fill=7)
    if family_id == "final-translation-photometric-v3":
        offsets = ((2, 1), (-2, 1), (1, -2))
        dx, dy = offsets[index % len(offsets)]
        translated = translate_frame(frame, dx=dx, dy=dy, fill=9)
        return scale_intensity(translated, numerator=83 + 4 * index, offset=4)
    if family_id == "final-translation-occlusion-v3":
        offsets = ((2, 0), (-2, 0), (0, -2))
        dx, dy = offsets[index % len(offsets)]
        translated = translate_frame(frame, dx=dx, dy=dy, fill=6)
        return mask_box(translated, top=0, left=0, height=2 + (index % 2), width=3, value=90 + 10 * index)
    if family_id == "final-translation-critical-v3":
        offsets = ((2, 1), (-2, 1), (1, -2))
        dx, dy = offsets[index % len(offsets)]
        translated = translate_frame(frame, dx=dx, dy=dy, fill=8)
        if state["target"] is not None:
            return _critical_target(translated, state)
        return _critical_cooldown(translated)
    if family_id == "final-information-impossible-v3":
        if state["target"] is None:
            raise ValueError("information-theoretic control requires visible target")
        return _critical_target(frame, state)
    if family_id == "final-beyond-bounds-translation-v3":
        offsets = ((4, 0), (-4, 0), (0, 4))
        dx, dy = offsets[index % len(offsets)]
        return translate_frame(frame, dx=dx, dy=dy, fill=11)
    raise ValueError("unknown accepted final family")


def _fresh_finalize(frame: np.ndarray, *, family_id: str) -> np.ndarray:
    offsets = {
        "final-translation-heldout-v3": 1,
        "final-translation-photometric-v3": 2,
        "final-translation-occlusion-v3": 3,
        "final-translation-critical-v3": 4,
        "final-same-action-wrong-row-v3": 5,
        "final-conflicting-action-near-v3": 6,
        "final-compositional-invalid-v3": 7,
        "final-information-impossible-v3": 8,
        "final-beyond-bounds-translation-v3": 9,
    }
    return scale_intensity(frame, numerator=100, offset=offsets[family_id])


def build_arcade_local_evidence_dataset(
    config: ShooterConfig = ShooterConfig(),
    *,
    variants_per_family: int = 3,
) -> ArcadeBenchmarkDataset:
    if variants_per_family <= 0:
        raise ValueError("variants_per_family must be positive")
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    canonical = enumerate_visual_frames(config)
    observations: Dict[str, ImageObservation] = {}
    records = []

    split_families = {
        "prototype": (
            "prototype-clean-v3",
            "prototype-palette-v3",
            "prototype-shift-right-v3",
            "prototype-brightness-v3",
        ),
        "benign_calibration": (
            "benign-calibration-contrast-v3",
            "benign-calibration-translation-x-v3",
            "benign-calibration-photometric-v3",
            "benign-calibration-noise-v3",
        ),
        "final_evaluation_accept": (
            "final-translation-heldout-v3",
            "final-translation-photometric-v3",
            "final-translation-occlusion-v3",
        ),
        "final_evaluation_reject": (
            "final-translation-critical-v3",
            "final-same-action-wrong-row-v3",
            "final-conflicting-action-near-v3",
            "final-compositional-invalid-v3",
            "final-information-impossible-v3",
            "final-beyond-bounds-translation-v3",
        ),
        "rejection_calibration": (
            "rejection-calibration-same-action-v3",
            "rejection-calibration-conflicting-v3",
            "rejection-calibration-compositional-v3",
        ),
    }

    same_action_cache = {
        row_id: _choose_same_action_competitor(row_id, lookup=lookup, canonical=canonical)
        for row_id in canonical
    }
    conflicting_cache = {
        row_id: _choose_conflicting_action_competitor(row_id, lookup=lookup, canonical=canonical)
        for row_id in canonical
    }

    for row_id, frame in canonical.items():
        state = decode_key_value_row_id(row_id)
        action = lookup.choose(row_id)
        same_action_row = same_action_cache[row_id]
        conflicting_row = conflicting_cache[row_id]
        same_action_frame = canonical[same_action_row]
        conflicting_frame = canonical[conflicting_row]

        for family_id in split_families["prototype"]:
            for index in range(variants_per_family):
                observation_id = f"{family_id}:{row_id}:{index:02d}"
                varied = _prototype_variant(frame, family_id, index)
                metadata = {
                    "family_id": family_id,
                    "row_id": row_id,
                    "expected_disposition": EXPECTED_ACCEPT,
                    "expected_action_id": action,
                    "transformation_parameters": {"variant_index": index},
                    "distinguishable": False,
                    "information_theoretic_impossible": False,
                    "contains_conflicting_action_evidence": False,
                }
                observation = ImageObservation(pixels=varied, source_id=SOURCE_SCOPE, metadata=metadata)
                observations[observation_id] = observation
                records.append(
                    VisualExampleRecord(
                        observation_id=observation_id,
                        observation_digest=observation.raw_digest,
                        split="prototype",
                        family_id=family_id,
                        row_id=row_id,
                        action_id=action,
                        partition="prototype",
                        calibration_role="prototype",
                        metadata=metadata,
                    )
                )

        for family_id in split_families["benign_calibration"]:
            for index in range(variants_per_family):
                observation_id = f"{family_id}:{row_id}:{index:02d}"
                varied = _benign_variant(frame, family_id, index)
                metadata = {
                    "family_id": family_id,
                    "row_id": row_id,
                    "expected_disposition": EXPECTED_ACCEPT,
                    "expected_action_id": action,
                    "transformation_parameters": {"variant_index": index},
                    "distinguishable": False,
                    "information_theoretic_impossible": False,
                    "contains_conflicting_action_evidence": False,
                }
                observation = ImageObservation(pixels=varied, source_id=SOURCE_SCOPE, metadata=metadata)
                observations[observation_id] = observation
                records.append(
                    VisualExampleRecord(
                        observation_id=observation_id,
                        observation_digest=observation.raw_digest,
                        split="benign_calibration",
                        family_id=family_id,
                        row_id=row_id,
                        action_id=action,
                        partition="benign_calibration",
                        calibration_role="benign_calibration",
                        evaluation_role=EXPECTED_ACCEPT,
                        metadata=metadata,
                    )
                )

        for family_id in split_families["rejection_calibration"]:
            for index in range(variants_per_family):
                observation_id = f"{family_id}:{row_id}:{index:02d}"
                if family_id == "rejection-calibration-same-action-v3":
                    varied = _target_band(frame, same_action_frame)
                    critical_region = "target_band"
                    competitor_row = same_action_row
                elif family_id == "rejection-calibration-conflicting-v3":
                    varied = _cooldown_patch(frame, conflicting_frame)
                    critical_region = "cooldown_indicator"
                    competitor_row = conflicting_row
                else:
                    varied = _target_band(_tank_band(frame, same_action_frame), conflicting_frame)
                    critical_region = "composed_target_and_tank"
                    competitor_row = conflicting_row
                metadata = {
                    "family_id": family_id,
                    "row_id": row_id,
                    "expected_disposition": EXPECTED_REJECT,
                    "expected_action_id": action,
                    "distinguishable": True,
                    "information_theoretic_impossible": False,
                    "contains_conflicting_action_evidence": family_id != "rejection-calibration-same-action-v3",
                    "source_row_id": row_id,
                    "competing_row_id": competitor_row,
                    "source_action_id": action,
                    "competing_action_id": lookup.choose(competitor_row),
                    "critical_changed_region": critical_region,
                    "transformation_parameters": {"variant_index": index},
                }
                observation = ImageObservation(pixels=varied, source_id=SOURCE_SCOPE, metadata=metadata)
                observations[observation_id] = observation
                records.append(
                    VisualExampleRecord(
                        observation_id=observation_id,
                        observation_digest=observation.raw_digest,
                        split="rejection_calibration",
                        family_id=family_id,
                        row_id=row_id,
                        action_id=action,
                        partition="rejection_calibration",
                        calibration_role="rejection_calibration",
                        evaluation_role=EXPECTED_REJECT,
                        metadata=metadata,
                    )
                )

        for family_id in split_families["final_evaluation_accept"]:
            for index in range(variants_per_family):
                observation_id = f"{family_id}:{row_id}:{index:02d}"
                varied = _fresh_finalize(
                    _accepted_final_variant(frame, state, family_id, index),
                    family_id=family_id,
                )
                known_displacements = {
                    "final-translation-heldout-v3": ((2, 0), (-2, 0), (0, -2)),
                    "final-translation-photometric-v3": ((2, 1), (-2, 1), (1, -2)),
                    "final-translation-occlusion-v3": ((2, 0), (-2, 0), (0, -2)),
                }
                metadata = {
                    "family_id": family_id,
                    "row_id": row_id,
                    "expected_disposition": EXPECTED_ACCEPT,
                    "expected_action_id": action,
                    "distinguishable": False,
                    "information_theoretic_impossible": False,
                    "contains_conflicting_action_evidence": False,
                    "known_synthetic_displacement": known_displacements[family_id][index % 3],
                    "transformation_parameters": {"variant_index": index},
                }
                observation = ImageObservation(pixels=varied, source_id=SOURCE_SCOPE, metadata=metadata)
                observations[observation_id] = observation
                records.append(
                    VisualExampleRecord(
                        observation_id=observation_id,
                        observation_digest=observation.raw_digest,
                        split="final_evaluation",
                        family_id=family_id,
                        row_id=row_id,
                        action_id=action,
                        partition="final_evaluation",
                        evaluation_role=EXPECTED_ACCEPT,
                        metadata=metadata,
                    )
                )

        for family_id in split_families["final_evaluation_reject"]:
            if family_id == "final-information-impossible-v3" and state["target"] is None:
                continue
            for index in range(variants_per_family):
                observation_id = f"{family_id}:{row_id}:{index:02d}"
                if family_id == "final-translation-critical-v3":
                    varied = _accepted_final_variant(frame, state, family_id, index)
                    critical_region = "target_region" if state["target"] is not None else "cooldown_indicator"
                    metadata = {
                        "known_synthetic_displacement": ((2, 1), (-2, 1), (1, -2))[index % 3],
                        "critical_changed_region": critical_region,
                    }
                elif family_id == "final-same-action-wrong-row-v3":
                    varied = _target_band(frame, same_action_frame)
                    metadata = {
                        "source_row_id": row_id,
                        "competing_row_id": same_action_row,
                        "source_action_id": action,
                        "competing_action_id": lookup.choose(same_action_row),
                        "critical_changed_region": "target_band",
                    }
                elif family_id == "final-conflicting-action-near-v3":
                    varied = _cooldown_patch(frame, conflicting_frame)
                    metadata = {
                        "source_row_id": row_id,
                        "competing_row_id": conflicting_row,
                        "source_action_id": action,
                        "competing_action_id": lookup.choose(conflicting_row),
                        "critical_changed_region": "cooldown_indicator",
                    }
                elif family_id == "final-compositional-invalid-v3":
                    varied = _target_band(_tank_band(frame, same_action_frame), conflicting_frame)
                    metadata = {
                        "source_row_id": row_id,
                        "same_action_row_id": same_action_row,
                        "conflicting_row_id": conflicting_row,
                        "source_action_id": action,
                        "same_action_id": lookup.choose(same_action_row),
                        "conflicting_action_id": lookup.choose(conflicting_row),
                        "critical_changed_region": "mixed_target_tank",
                    }
                elif family_id == "final-information-impossible-v3":
                    varied = _accepted_final_variant(frame, state, family_id, index)
                    metadata = {"critical_changed_region": "target_region"}
                else:
                    varied = _accepted_final_variant(frame, state, family_id, index)
                    metadata = {
                        "known_synthetic_displacement": ((4, 0), (-4, 0), (0, 4))[index % 3],
                        "critical_changed_region": "out_of_declared_domain_translation",
                    }
                varied = _fresh_finalize(varied, family_id=family_id)
                metadata.update(
                    {
                        "family_id": family_id,
                        "row_id": row_id,
                        "expected_disposition": (
                            IMPOSSIBILITY_CONTROL
                            if family_id == "final-information-impossible-v3"
                            else EXPECTED_REJECT
                        ),
                        "expected_action_id": action,
                        "distinguishable": family_id != "final-information-impossible-v3",
                        "information_theoretic_impossible": family_id == "final-information-impossible-v3",
                        "contains_conflicting_action_evidence": family_id in {
                            "final-conflicting-action-near-v3",
                            "final-compositional-invalid-v3",
                        },
                        "transformation_parameters": {"variant_index": index},
                    }
                )
                observation = ImageObservation(pixels=varied, source_id=SOURCE_SCOPE, metadata=metadata)
                observations[observation_id] = observation
                records.append(
                    VisualExampleRecord(
                        observation_id=observation_id,
                        observation_digest=observation.raw_digest,
                        split="final_evaluation",
                        family_id=family_id,
                        row_id=row_id,
                        action_id=action,
                        partition="final_evaluation",
                        evaluation_role=(
                            IMPOSSIBILITY_CONTROL
                            if family_id == "final-information-impossible-v3"
                            else EXPECTED_REJECT
                        ),
                        metadata=metadata,
                    )
                )

    manifest = VisualDatasetManifest(
        source_scope=SOURCE_SCOPE,
        policy_artifact_id=policy.artifact_id,
        families=_family_specs(),
        records=records,
        enforce_family_holdout=True,
        metadata={
            "fixture": "bounded_arcade_shooter",
            "protocol_version": PROTOCOL_VERSION,
            "variants_per_family": variants_per_family,
            "observation_count": len(records),
            "fresh_final_schedule": "v3-heldout-local-evidence",
        },
    )
    return ArcadeBenchmarkDataset(
        policy=policy,
        policy_lookup=lookup,
        manifest=manifest,
        observations=observations,
    )


def main() -> None:
    dataset = build_arcade_local_evidence_dataset()
    payload = {
        "dataset_digest": dataset.manifest.digest,
        "observation_count": len(dataset.manifest.records),
        "split_counts": {
            split: sum(record.split == split for record in dataset.manifest.records)
            for split in ("prototype", "benign_calibration", "rejection_calibration", "final_evaluation")
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
