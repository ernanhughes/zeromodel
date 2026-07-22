"""Phase 1 held-out visual-address benchmark over the bounded arcade policy.

The default run executes the current deterministic reader (A) and normalized
pixel template matching (B). ``--encoder dinov2`` additionally downloads or loads
a pinned DINOv2-small checkpoint and executes medoid retrieval (C), raw k-NN
(D), and a rejection-equipped linear probe (G).

Run:

    python examples/arcade_visual_address_benchmark.py
    python examples/arcade_visual_address_benchmark.py --encoder dinov2
    python examples/arcade_visual_address_benchmark.py --output-dir build/visual-phase-one
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_shooter_policy import ACTIONS, ShooterConfig, compile_policy_artifact  # noqa: E402
from examples.arcade_visual_sign_reader import (  # noqa: E402
    CELL_PIXELS,
    COOLDOWN_BLOCKED_VALUE,
    COOLDOWN_READY_VALUE,
    FRAME_HEIGHT,
    TARGET_VALUE,
    TANK_VALUE,
    compile_visual_index_artifact,
    enumerate_visual_frames,
    make_visual_reader,
)
from zeromodel.analysis.policy_properties import decode_key_value_row_id
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.observation.visual_address import ImageObservation
from zeromodel.vision.visual_policy import DeterministicVisualAddressProvider
from zeromodel.vision.visual_dataset import (
    CorruptionFamilySpec,
    VisualDatasetManifest,
    VisualExampleRecord,
)
from zeromodel.vision.visual_corruptions import (  # noqa: E402
    add_integer_noise,
    checkerboard_frame,
    mask_box,
    overlay_background_patch,
    remap_levels,
    scale_intensity,
    translate_frame,
)
from zeromodel.vision.visual_encoder import HuggingFaceDinoV2Encoder  # noqa: E402
from research.visual.visual_experiment import (  # noqa: E402
    EXPECTED_ACCEPT,
    EXPECTED_REJECT,
    IMPOSSIBILITY_CONTROL,
    build_research_report,
    encode_observations,
    evaluate_visual_provider,
    records_for_split,
    vectors_for_records,
)
from zeromodel.vision.visual_retrieval import (  # noqa: E402
    FrozenVectorAddressProvider,
    LinearProbeIndex,
    NormalizedPixelEncoder,
    VectorAddressIndex,
    build_linear_probe,
    build_vector_address,
)


SOURCE_SCOPE = "arcade-visual-system-b-adjudication/v2"


@dataclass(frozen=True)
class ArcadeBenchmarkDataset:
    policy: Any
    policy_lookup: VPMPolicyLookup
    manifest: VisualDatasetManifest
    observations: Mapping[str, ImageObservation]


def _family_specs() -> Tuple[CorruptionFamilySpec, ...]:
    return (
        CorruptionFamilySpec(family_id="prototype-clean", kind="clean"),
        CorruptionFamilySpec(family_id="prototype-brightness", kind="brightness"),
        CorruptionFamilySpec(family_id="prototype-shift-up", kind="translation"),
        CorruptionFamilySpec(family_id="prototype-palette-a", kind="palette"),
        CorruptionFamilySpec(family_id="benign-calibration-contrast", kind="contrast"),
        CorruptionFamilySpec(family_id="benign-calibration-shift-down", kind="translation"),
        CorruptionFamilySpec(family_id="benign-calibration-palette-b", kind="palette"),
        CorruptionFamilySpec(family_id="benign-calibration-noise", kind="noise"),
        CorruptionFamilySpec(family_id="rejection-calibration-blank-band", kind="invalid_geometry"),
        CorruptionFamilySpec(family_id="rejection-calibration-checker", kind="structured_non_state"),
        CorruptionFamilySpec(family_id="rejection-calibration-double-tank-left", kind="impossible_object_count"),
        CorruptionFamilySpec(family_id="final-brightness-unseen", kind="brightness"),
        CorruptionFamilySpec(family_id="final-shift-two", kind="translation"),
        CorruptionFamilySpec(family_id="final-palette-c", kind="palette"),
        CorruptionFamilySpec(family_id="final-noncritical-patch", kind="structured_occlusion"),
        CorruptionFamilySpec(
            family_id="final-information-target",
            kind="information_theoretic_control",
            critical_evidence_removed=True,
            parameters={
                "evaluation_role": IMPOSSIBILITY_CONTROL,
                "reason": (
                    "removing the target yields pixels equivalent to a valid "
                    "no-target state, so a single frame cannot distinguish hidden "
                    "presence from true absence"
                ),
            },
        ),
        CorruptionFamilySpec(
            family_id="final-critical-tank",
            kind="critical_intervention",
            critical_evidence_removed=True,
        ),
        CorruptionFamilySpec(
            family_id="final-critical-cooldown",
            kind="critical_intervention",
            critical_evidence_removed=True,
        ),
        CorruptionFamilySpec(family_id="final-ood-blank", kind="out_of_domain"),
        CorruptionFamilySpec(family_id="final-ood-checkerboard", kind="out_of_domain"),
        CorruptionFamilySpec(family_id="final-ood-impossible-state", kind="out_of_domain"),
    )


def _palette(frame: np.ndarray, *, variant: str, index: int) -> np.ndarray:
    if variant == "a":
        mapping = {
            TARGET_VALUE: 190 + 5 * index,
            TANK_VALUE: 235 - 3 * index,
            COOLDOWN_READY_VALUE: 55 + 4 * index,
            COOLDOWN_BLOCKED_VALUE: 145 + 4 * index,
        }
    elif variant == "b":
        mapping = {
            TARGET_VALUE: 175 - 4 * index,
            TANK_VALUE: 210 + 5 * index,
            COOLDOWN_READY_VALUE: 70 + 3 * index,
            COOLDOWN_BLOCKED_VALUE: 130 - 3 * index,
        }
    else:
        mapping = {
            TARGET_VALUE: 130 + 5 * index,
            TANK_VALUE: 185 - 4 * index,
            COOLDOWN_READY_VALUE: 90 + 2 * index,
            COOLDOWN_BLOCKED_VALUE: 115 - 2 * index,
        }
    return remap_levels(frame, mapping)


def _critical_target(frame: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    target = state["target"]
    if target is None:
        raise ValueError("target intervention requires a visible target")
    centre = int(target) * CELL_PIXELS + CELL_PIXELS // 2
    return mask_box(frame, top=2, left=centre - 1, height=3, width=3, value=0)


def _critical_tank(frame: np.ndarray, state: Mapping[str, Any]) -> np.ndarray:
    centre = int(state["tank"]) * CELL_PIXELS + CELL_PIXELS // 2
    return mask_box(frame, top=11, left=max(0, centre - 2), height=3, width=5, value=0)


def _critical_cooldown(frame: np.ndarray) -> np.ndarray:
    return mask_box(frame, top=7, left=frame.shape[1] - 3, height=2, width=2, value=0)


def _second_tank_frame(frame: np.ndarray, config: ShooterConfig) -> np.ndarray:
    """Return a frame with two spatially separated tanks, impossible in policy state."""

    result = np.array(frame, dtype=np.uint8, order="C", copy=True)
    centre = (config.width - 1) * CELL_PIXELS + CELL_PIXELS // 2
    result[11, centre] = TANK_VALUE
    result[12, centre - 1 : centre + 2] = TANK_VALUE
    result[13, centre - 2 : centre + 3] = TANK_VALUE
    result.flags.writeable = False
    return result


def _variant(
    frame: np.ndarray,
    state: Mapping[str, Any],
    family_id: str,
    index: int,
) -> np.ndarray:
    if family_id == "prototype-clean":
        return scale_intensity(frame, numerator=98 + 2 * index)
    if family_id == "prototype-brightness":
        return scale_intensity(frame, numerator=86 + 4 * index)
    if family_id == "prototype-shift-up":
        return translate_frame(frame, dy=-1, fill=index)
    if family_id == "prototype-palette-a":
        return _palette(frame, variant="a", index=index)
    if family_id == "benign-calibration-contrast":
        return scale_intensity(frame, numerator=104 + 4 * index, offset=2 + index)
    if family_id == "benign-calibration-shift-down":
        return translate_frame(frame, dy=1, fill=2 * index)
    if family_id == "benign-calibration-palette-b":
        return _palette(frame, variant="b", index=index)
    if family_id == "benign-calibration-noise":
        return add_integer_noise(frame, amplitude=2 + index, seed=1000 + index)
    if family_id == "final-brightness-unseen":
        return scale_intensity(frame, numerator=68 + 5 * index, offset=4)
    if family_id == "final-shift-two":
        return translate_frame(frame, dy=2 if index % 2 == 0 else -2, fill=3)
    if family_id == "final-palette-c":
        return _palette(frame, variant="c", index=index)
    if family_id == "final-noncritical-patch":
        return overlay_background_patch(
            frame,
            height=2 + (index % 2),
            width=3,
            value=80 + 10 * index,
        )
    if family_id == "final-information-target":
        return _critical_target(frame, state)
    if family_id == "final-critical-tank":
        return _critical_tank(frame, state)
    if family_id == "final-critical-cooldown":
        return _critical_cooldown(frame)
    if family_id == "rejection-calibration-blank-band":
        return mask_box(frame, top=5, left=4 + index, height=4, width=8, value=0)
    if family_id == "rejection-calibration-checker":
        return checkerboard_frame(
            height=frame.shape[0],
            width=frame.shape[1],
            low=30 + index,
            high=210 - index,
            cell=2,
        )
    if family_id == "rejection-calibration-double-tank-left":
        result = np.array(frame, dtype=np.uint8, order="C", copy=True)
        centre = CELL_PIXELS // 2
        result[11, centre] = TANK_VALUE
        result[12, centre - 1 : centre + 2] = TANK_VALUE
        result[13, centre - 2 : centre + 3] = TANK_VALUE
        result.flags.writeable = False
        return result
    raise ValueError("unknown benchmark family: %s" % family_id)


def build_arcade_benchmark_dataset(
    config: ShooterConfig = ShooterConfig(),
    *,
    variants_per_family: int = 3,
    ood_examples_per_family: int = 8,
) -> ArcadeBenchmarkDataset:
    if variants_per_family <= 0 or ood_examples_per_family <= 0:
        raise ValueError("benchmark variant counts must be positive")
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    canonical = enumerate_visual_frames(config)
    observations: Dict[str, ImageObservation] = {}
    records = []

    split_families = {
        "prototype": (
            "prototype-clean",
            "prototype-brightness",
            "prototype-shift-up",
            "prototype-palette-a",
        ),
        "benign_calibration": (
            "benign-calibration-contrast",
            "benign-calibration-shift-down",
            "benign-calibration-palette-b",
            "benign-calibration-noise",
        ),
        "final_evaluation": (
            "final-brightness-unseen",
            "final-shift-two",
            "final-palette-c",
            "final-noncritical-patch",
            "final-information-target",
            "final-critical-tank",
            "final-critical-cooldown",
        ),
    }

    for row_id, frame in canonical.items():
        state = decode_key_value_row_id(row_id)
        action = lookup.choose(row_id)
        for split, family_ids in split_families.items():
            for family_id in family_ids:
                if family_id == "final-information-target" and state["target"] is None:
                    continue
                count = 1 if family_id.startswith("final-critical-") else variants_per_family
                for index in range(count):
                    observation_id = "%s:%s:%02d" % (family_id, row_id, index)
                    varied = _variant(frame, state, family_id, index)
                    metadata: Dict[str, Any] = {
                        "family_id": family_id,
                        "row_id": row_id,
                        "variant_index": index,
                    }
                    evaluation_role = None
                    calibration_role = None
                    if split == "prototype":
                        calibration_role = "prototype"
                    elif split == "benign_calibration":
                        calibration_role = "benign_calibration"
                    if family_id == "final-information-target":
                        evaluation_role = IMPOSSIBILITY_CONTROL
                        metadata["evaluation_role"] = IMPOSSIBILITY_CONTROL
                        metadata["counterfactual_source_row_id"] = row_id
                    elif split == "final_evaluation":
                        evaluation_role = (
                            EXPECTED_REJECT
                            if family_id.startswith("final-critical-")
                            else EXPECTED_ACCEPT
                        )
                    observation = ImageObservation(
                        pixels=varied,
                        source_id=SOURCE_SCOPE,
                        metadata=metadata,
                    )
                    observations[observation_id] = observation
                    records.append(
                        VisualExampleRecord(
                            observation_id=observation_id,
                            observation_digest=observation.raw_digest,
                            split=split,
                            family_id=family_id,
                            row_id=row_id,
                            action_id=action,
                            partition=split,
                            calibration_role=calibration_role,
                            evaluation_role=evaluation_role,
                            metadata=metadata,
                        )
                    )

        for family_id in (
            "rejection-calibration-blank-band",
            "rejection-calibration-checker",
            "rejection-calibration-double-tank-left",
        ):
            for index in range(variants_per_family):
                observation_id = "%s:%s:%02d" % (family_id, row_id, index)
                varied = _variant(frame, state, family_id, index)
                metadata = {
                    "family_id": family_id,
                    "row_id": row_id,
                    "variant_index": index,
                }
                observation = ImageObservation(
                    pixels=varied,
                    source_id=SOURCE_SCOPE,
                    metadata=metadata,
                )
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

    height = FRAME_HEIGHT
    width = config.width * CELL_PIXELS
    first_canonical = next(iter(canonical.values()))
    for family_id in ("final-ood-blank", "final-ood-checkerboard", "final-ood-impossible-state"):
        for index in range(ood_examples_per_family):
            if family_id == "final-ood-blank":
                frame = np.full((height, width), index % 3, dtype=np.uint8)
            elif family_id == "final-ood-checkerboard":
                frame = checkerboard_frame(
                    height=height,
                    width=width,
                    low=20 + index,
                    high=220 - index,
                    cell=1 + index % 3,
                )
            else:
                frame = _second_tank_frame(first_canonical, config)
            observation_id = "%s:%02d" % (family_id, index)
            observation = ImageObservation(
                pixels=frame,
                source_id=SOURCE_SCOPE,
                metadata={"family_id": family_id, "variant_index": index},
            )
            observations[observation_id] = observation
            records.append(
                VisualExampleRecord(
                    observation_id=observation_id,
                    observation_digest=observation.raw_digest,
                    split="final_evaluation",
                    family_id=family_id,
                    partition="final_evaluation",
                    evaluation_role=EXPECTED_REJECT,
                    metadata={"variant_index": index},
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
            "variants_per_family": variants_per_family,
            "ood_examples_per_family": ood_examples_per_family,
            "observation_count": len(records),
        },
    )
    return ArcadeBenchmarkDataset(
        policy=policy,
        policy_lookup=lookup,
        manifest=manifest,
        observations=observations,
    )


class _LinearProbeProvider:
    def __init__(self, encoder: Any, index: LinearProbeIndex) -> None:
        if encoder.manifest().manifest_id != index.build.encoder_manifest_id:
            raise ValueError("linear probe encoder manifest mismatch")
        self.encoder = encoder
        self.index = index

    def contract(self):
        return self.index.contract()

    def read(self, observation: ImageObservation):
        vector = self.encoder.encode_batch((observation,))[0]
        return self.index.match_vector(vector, observation_digest=observation.raw_digest)


def _build_vector_system(
    dataset: ArcadeBenchmarkDataset,
    encoder: Any,
    vectors: Mapping[str, np.ndarray],
    *,
    strategy: str,
):
    prototype = vectors_for_records(records_for_split(dataset.manifest, "prototype"), vectors)
    calibration = vectors_for_records(
        records_for_split(dataset.manifest, "benign_calibration"),
        vectors,
    )
    build = build_vector_address(
        prototype_vectors=prototype[0],
        prototype_row_ids=prototype[1],
        prototype_action_ids=prototype[2],
        prototype_observation_ids=prototype[3],
        calibration_vectors=calibration[0],
        calibration_row_ids=calibration[1],
        calibration_action_ids=calibration[2],
        calibration_observation_ids=calibration[3],
        policy_artifact_id=dataset.policy.artifact_id,
        source_scope=SOURCE_SCOPE,
        representation_spec_digest=encoder.manifest().manifest_id,
        encoder_manifest_id=encoder.manifest().manifest_id,
        strategy=strategy,
        calibration_quantile=0.0,
        deployment_status="research",
    )
    return build, FrozenVectorAddressProvider(encoder, VectorAddressIndex(build))


def run_benchmark(
    *,
    variants_per_family: int = 3,
    encoder_name: str = "none",
    device: str = "cpu",
    local_files_only: bool = False,
    include_traces: bool = False,
) -> Tuple[ArcadeBenchmarkDataset, Any, Mapping[str, Any]]:
    dataset = build_arcade_benchmark_dataset(
        variants_per_family=variants_per_family,
    )
    systems = []
    artifacts: Dict[str, Any] = {}

    deterministic_build = compile_visual_index_artifact(
        policy_artifact=dataset.policy,
    )
    deterministic = DeterministicVisualAddressProvider(
        make_visual_reader(dataset.policy, deterministic_build),
        source_scope=SOURCE_SCOPE,
    )
    result_a, trace_a = evaluate_visual_provider(
        provider=deterministic,
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        system_id="A",
        system_name="current_deterministic_reader",
        splits=("final_evaluation",),
        include_traces=include_traces,
    )
    systems.append(result_a)
    if include_traces:
        artifacts["traces_A"] = [item.to_dict() for item in trace_a]

    pixel_encoder = NormalizedPixelEncoder(
        height=FRAME_HEIGHT,
        width=ShooterConfig().width * CELL_PIXELS,
    )
    all_ids = tuple(record.observation_id for record in dataset.manifest.records)
    pixel_vectors = encode_observations(
        pixel_encoder,
        all_ids,
        dataset.observations,
        batch_size=128,
    )
    build_b, provider_b = _build_vector_system(
        dataset,
        pixel_encoder,
        pixel_vectors,
        strategy="medoid",
    )
    result_b, trace_b = evaluate_visual_provider(
        provider=provider_b,
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        system_id="B",
        system_name="normalized_template_matching",
        splits=("final_evaluation",),
        include_traces=include_traces,
    )
    systems.append(result_b)
    artifacts["pixel_encoder_manifest"] = pixel_encoder.manifest().to_dict()
    artifacts["address_B"] = {
        "manifest": build_b.manifest.to_dict(),
        "calibration": build_b.calibration.to_dict(),
        "matrix_blob": build_b.matrix_blob.to_dict(),
    }
    if include_traces:
        artifacts["traces_B"] = [item.to_dict() for item in trace_b]

    if encoder_name == "dinov2":
        encoder = HuggingFaceDinoV2Encoder(
            device=device,
            local_files_only=local_files_only,
        )
        learned_vectors = encode_observations(
            encoder,
            all_ids,
            dataset.observations,
            batch_size=32,
        )
        for system_id, strategy, name in (
            ("C", "medoid", "frozen_embedding_medoids"),
            ("D", "all", "raw_embedding_knn"),
        ):
            build, provider = _build_vector_system(
                dataset,
                encoder,
                learned_vectors,
                strategy=strategy,
            )
            result, traces = evaluate_visual_provider(
                provider=provider,
                dataset_manifest=dataset.manifest,
                observations=dataset.observations,
                policy_lookup=dataset.policy_lookup,
                system_id=system_id,
                system_name=name,
                splits=("final_evaluation",),
                include_traces=include_traces,
            )
            systems.append(result)
            artifacts["address_%s" % system_id] = {
                "manifest": build.manifest.to_dict(),
                "calibration": build.calibration.to_dict(),
                "matrix_blob": build.matrix_blob.to_dict(),
            }
            if include_traces:
                artifacts["traces_%s" % system_id] = [item.to_dict() for item in traces]

        prototype = vectors_for_records(
            records_for_split(dataset.manifest, "prototype"), learned_vectors
        )
        calibration = vectors_for_records(
            records_for_split(dataset.manifest, "benign_calibration"), learned_vectors
        )
        probe_build = build_linear_probe(
            prototype_vectors=prototype[0],
            prototype_row_ids=prototype[1],
            prototype_action_ids=prototype[2],
            calibration_vectors=calibration[0],
            calibration_row_ids=calibration[1],
            calibration_action_ids=calibration[2],
            policy_artifact_id=dataset.policy.artifact_id,
            source_scope=SOURCE_SCOPE,
            representation_spec_digest=encoder.manifest().manifest_id,
            encoder_manifest_id=encoder.manifest().manifest_id,
        )
        probe_provider = _LinearProbeProvider(encoder, LinearProbeIndex(probe_build))
        result_g, trace_g = evaluate_visual_provider(
            provider=probe_provider,
            dataset_manifest=dataset.manifest,
            observations=dataset.observations,
            policy_lookup=dataset.policy_lookup,
            system_id="G",
            system_name="rejection_equipped_linear_probe",
            splits=("final_evaluation",),
            include_traces=include_traces,
        )
        systems.append(result_g)
        artifacts["encoder_manifest"] = encoder.manifest().to_dict()
        artifacts["linear_probe_G"] = {
            "model_id": probe_build.model_id,
            "weights_blob": probe_build.weights_blob.to_dict(),
            "calibration": probe_build.calibration.to_dict(),
            "row_ids": list(probe_build.row_ids),
            "action_ids": list(probe_build.action_ids),
        }
        if include_traces:
            artifacts["traces_G"] = [item.to_dict() for item in trace_g]
    elif encoder_name != "none":
        raise ValueError("encoder must be 'none' or 'dinov2'")

    report = build_research_report(
        dataset_manifest=dataset.manifest,
        system_results=systems,
        metadata={
            "fixture": "bounded_arcade_shooter",
            "encoder": encoder_name,
            "variants_per_family": variants_per_family,
            "claim_status": "research",
        },
    )
    return dataset, report, artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder", choices=("none", "dinov2"), default="none")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--variants-per-family", type=int, default=3)
    parser.add_argument("--include-traces", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    dataset, report, artifacts = run_benchmark(
        variants_per_family=args.variants_per_family,
        encoder_name=args.encoder,
        device=args.device,
        local_files_only=args.local_files_only,
        include_traces=args.include_traces,
    )
    payload = {
        "dataset_manifest": dataset.manifest.to_dict(),
        "benchmark_report": report.to_dict(),
        "artifacts": artifacts,
    }
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "arcade-visual-phase-one.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    summary = {
        "dataset_digest": dataset.manifest.digest,
        "observation_count": len(dataset.manifest.records),
        "report_digest": report.digest,
        "systems": {
            system.system_id: system.metrics.to_dict() for system in report.systems
        },
        "validation_status": report.validation_status,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
