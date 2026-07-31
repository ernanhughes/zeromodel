from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from zeromodel.video.arcade_policy import (
    ShooterConfig,
    compile_policy_artifact,
    parse_state_row_id,
    render_state_frame,
)
from zeromodel.vision import (
    VisualAcceptanceProfile,
    VisualDecision,
    VisualFeatureSpec,
    VisualSignReader,
    build_visual_index,
    extract_visual_features,
    visual_feature_digest,
    visual_input_digest,
    visual_raw_input_digest,
)

from visual_transition_benchmark.alias_discovery._json import digest
from visual_transition_benchmark.alias_discovery.registry import registry_id
from visual_transition_benchmark.alias_discovery.transforms import (
    TransformSpec,
    changed_stats,
    transform_frame,
)

PROFILES = (
    VisualAcceptanceProfile.CANONICAL_ONLY,
    VisualAcceptanceProfile.EXACT_CODEWORD,
    VisualAcceptanceProfile.CALIBRATED_NEAREST,
    VisualAcceptanceProfile.EVIDENCE_ONLY,
)
SEEDS = (0, 17)


@dataclass(frozen=True)
class VisualAliasCase:
    case_id: str
    split: str
    source_row_id: str
    source_action: str
    source_observation_raw_digest: str
    source_observation_canonical_digest: str
    source_feature_digest: str
    transformed_observation_raw_digest: str
    transformed_observation_canonical_digest: str
    transformed_feature_digest: str
    transform_registry_id: str
    transform_chain_id: str
    transform_id: str
    transform_family: str
    transform_version: str
    transform_parameters: dict[str, Any]
    transform_seed: int | None
    family_severity: float
    severity_rank: int
    changed_pixel_fraction: float
    mean_absolute_pixel_difference: float
    acceptance_profile: str
    reader_version: str
    visual_index_artifact_id: str
    policy_artifact_id: str
    feature_spec_digest: str
    calibration_digest: str
    accepted: bool
    policy_executed: bool
    reason: str
    canonical_input_match: bool
    exact_feature_match: bool
    nearest_row_id: str
    second_nearest_row_id: str
    nearest_distance: float
    second_nearest_distance: float
    distance_margin: float
    matched_row_id: str | None
    matched_action: str | None
    alias_status: str
    action_equivalent: bool | None
    source_row_in_top_two: bool

    def identity_payload(self) -> dict[str, Any]:
        data = self._base_dict()
        for key in ("alias_status", "action_equivalent"):
            data.pop(key, None)
        return data

    @property
    def identity(self) -> str:
        return digest(self.identity_payload())

    def _base_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "split": self.split,
            "source_row_id": self.source_row_id,
            "source_action": self.source_action,
            "source_observation_raw_digest": self.source_observation_raw_digest,
            "source_observation_canonical_digest": self.source_observation_canonical_digest,
            "source_feature_digest": self.source_feature_digest,
            "transformed_observation_raw_digest": self.transformed_observation_raw_digest,
            "transformed_observation_canonical_digest": self.transformed_observation_canonical_digest,
            "transformed_feature_digest": self.transformed_feature_digest,
            "transform_registry_id": self.transform_registry_id,
            "transform_chain_id": self.transform_chain_id,
            "transform_id": self.transform_id,
            "transform_family": self.transform_family,
            "transform_version": self.transform_version,
            "transform_parameters": dict(self.transform_parameters),
            "transform_seed": self.transform_seed,
            "family_severity": self.family_severity,
            "severity_rank": self.severity_rank,
            "changed_pixel_fraction": self.changed_pixel_fraction,
            "mean_absolute_pixel_difference": self.mean_absolute_pixel_difference,
            "acceptance_profile": self.acceptance_profile,
            "reader_version": self.reader_version,
            "visual_index_artifact_id": self.visual_index_artifact_id,
            "policy_artifact_id": self.policy_artifact_id,
            "feature_spec_digest": self.feature_spec_digest,
            "calibration_digest": self.calibration_digest,
            "accepted": self.accepted,
            "policy_executed": self.policy_executed,
            "reason": self.reason,
            "canonical_input_match": self.canonical_input_match,
            "exact_feature_match": self.exact_feature_match,
            "nearest_row_id": self.nearest_row_id,
            "second_nearest_row_id": self.second_nearest_row_id,
            "nearest_distance": self.nearest_distance,
            "second_nearest_distance": self.second_nearest_distance,
            "distance_margin": self.distance_margin,
            "matched_row_id": self.matched_row_id,
            "matched_action": self.matched_action,
            "alias_status": self.alias_status,
            "action_equivalent": self.action_equivalent,
            "source_row_in_top_two": self.source_row_in_top_two,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._base_dict(), "case_identity": self.identity}


@dataclass(frozen=True)
class ReaderContext:
    config: ShooterConfig
    feature_spec: VisualFeatureSpec
    policy: Any
    reader: VisualSignReader
    frames_by_row_id: dict[str, np.ndarray]
    actions_by_row_id: dict[str, str]


def build_context() -> ReaderContext:
    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    feature_spec = VisualFeatureSpec(
        input_height=16,
        input_width=config.width * 4,
        target_height=4,
        target_width=7,
        quantization_levels=16,
    )
    frames = {str(row_id): frame_for_row(str(row_id), config) for row_id in policy.source.row_ids}
    visual_index = build_visual_index(policy, frames, feature_spec)
    reader = VisualSignReader(
        visual_index.artifact,
        policy,
        action_metric_ids=("LEFT", "RIGHT", "STAY", "FIRE"),
        acceptance_profile=VisualAcceptanceProfile.CALIBRATED_NEAREST,
    )
    actions = {
        row_id: str(reader.read(frame, acceptance_profile=VisualAcceptanceProfile.CANONICAL_ONLY).action)
        for row_id, frame in frames.items()
    }
    return ReaderContext(config, feature_spec, policy, reader, frames, actions)


def frame_for_row(row_id: str, config: ShooterConfig | None = None) -> np.ndarray:
    config = config or ShooterConfig()
    tank, target, cooldown = parse_state_row_id(row_id)
    return np.array(render_state_frame(tank, target, cooldown, width=config.width), copy=True)


def source_rows_for_mode(rows: tuple[str, ...], mode: str) -> tuple[str, ...]:
    if mode == "smoke":
        return tuple(rows[:8])
    if mode == "discovery":
        return tuple(row for index, row in enumerate(rows) if index % 4 == 0)
    return rows


def split_for_row(row_id: str) -> str:
    return "discovery" if digest({"split": row_id})[-1] in "01234567" else "confirmation"


def classify_case(source_row: str, source_action: str, decision: VisualDecision) -> tuple[str, bool | None]:
    if decision.acceptance_profile == VisualAcceptanceProfile.EVIDENCE_ONLY:
        if decision.nearest_row_id != source_row:
            return "evidence_only_wrong_nearest", None
        return "noncanonical_correct_row", None
    if not decision.accepted or not decision.policy_executed:
        if decision.reason == "canonical_input_mismatch":
            return "rejected_canonical_mismatch", None
        if decision.reason == "feature_not_exact":
            return "rejected_feature_mismatch", None
        return "rejected_calibration_threshold", None
    if decision.matched_row_id == source_row:
        if decision.canonical_input_match:
            return "canonical_exact", False
        if decision.exact_feature_match:
            return "feature_collision_correct_row", False
        return "noncanonical_correct_row", False
    same_action = decision.action == source_action
    if decision.reason == "accepted_calibrated_nearest":
        return (
            "nearest_wrong_row_same_action" if same_action else "nearest_wrong_row_different_action",
            bool(same_action),
        )
    return (
        "accepted_wrong_row_same_action" if same_action else "accepted_wrong_row_different_action",
        bool(same_action),
    )


def build_case(
    *,
    context: ReaderContext,
    split: str,
    source_row_id: str,
    transform: TransformSpec,
    seed: int | None,
    severity_rank: int,
    profile: str,
) -> tuple[VisualAliasCase, np.ndarray]:
    source = context.frames_by_row_id[source_row_id]
    transformed = transform_frame(source, transform, seed=seed)
    decision = context.reader.read(transformed, acceptance_profile=profile)
    changed_fraction, mean_abs, max_abs = changed_stats(source, transformed)
    family_severity = float(max_abs if transform.family in {"photometric", "noise"} else changed_fraction)
    source_features = extract_visual_features(source, context.feature_spec)
    transformed_features = extract_visual_features(transformed, context.feature_spec)
    source_action = context.actions_by_row_id[source_row_id]
    alias_status, action_equivalent = classify_case(source_row_id, source_action, decision)
    chain_payload = {
        "source_row_id": source_row_id,
        "source_raw_digest": visual_raw_input_digest(source, context.feature_spec),
        "transform": transform.to_dict(),
        "seed": seed,
    }
    transform_chain_id = digest(chain_payload)
    case_payload = {
        "split": split,
        "source_row_id": source_row_id,
        "transform_chain_id": transform_chain_id,
        "acceptance_profile": profile,
    }
    case = VisualAliasCase(
        case_id=digest(case_payload),
        split=split,
        source_row_id=source_row_id,
        source_action=source_action,
        source_observation_raw_digest=visual_raw_input_digest(source, context.feature_spec),
        source_observation_canonical_digest=visual_input_digest(source, context.feature_spec),
        source_feature_digest=visual_feature_digest(source_features, context.feature_spec),
        transformed_observation_raw_digest=visual_raw_input_digest(transformed, context.feature_spec),
        transformed_observation_canonical_digest=visual_input_digest(transformed, context.feature_spec),
        transformed_feature_digest=visual_feature_digest(transformed_features, context.feature_spec),
        transform_registry_id=registry_id(),
        transform_chain_id=transform_chain_id,
        transform_id=transform.transform_id,
        transform_family=transform.family,
        transform_version=transform.version,
        transform_parameters=dict(transform.parameters),
        transform_seed=seed,
        family_severity=family_severity,
        severity_rank=severity_rank,
        changed_pixel_fraction=changed_fraction,
        mean_absolute_pixel_difference=mean_abs,
        acceptance_profile=profile,
        reader_version=decision.reader_version,
        visual_index_artifact_id=decision.visual_index_artifact_id,
        policy_artifact_id=decision.policy_artifact_id,
        feature_spec_digest=decision.feature_spec_digest,
        calibration_digest=decision.calibration_digest,
        accepted=decision.accepted,
        policy_executed=decision.policy_executed,
        reason=decision.reason,
        canonical_input_match=decision.canonical_input_match,
        exact_feature_match=decision.exact_feature_match,
        nearest_row_id=str(decision.nearest_row_id),
        second_nearest_row_id=str(decision.second_nearest_row_id),
        nearest_distance=float(decision.nearest_distance),
        second_nearest_distance=float(decision.second_nearest_distance),
        distance_margin=float(decision.distance_margin),
        matched_row_id=decision.matched_row_id,
        matched_action=decision.action,
        alias_status=alias_status,
        action_equivalent=action_equivalent,
        source_row_in_top_two=source_row_id
        in {str(decision.nearest_row_id), str(decision.second_nearest_row_id)},
    )
    return case, transformed


def generate_cases(
    *, mode: str, registry: tuple[TransformSpec, ...] | None = None
) -> tuple[list[VisualAliasCase], dict[str, np.ndarray], ReaderContext]:
    context = build_context()
    specs = registry or ()
    rows = source_rows_for_mode(tuple(context.frames_by_row_id), mode)
    if mode == "confirmation":
        rows = tuple(row for row in context.frames_by_row_id if split_for_row(row) == "confirmation")
    elif mode == "discovery":
        rows = tuple(row for row in context.frames_by_row_id if split_for_row(row) == "discovery")[:28]
    specs = specs or __import__(
        "visual_transition_benchmark.alias_discovery.registry",
        fromlist=["default_registry"],
    ).default_registry()
    cases: list[VisualAliasCase] = []
    observations: dict[str, np.ndarray] = {}
    for row_id in rows:
        split = "smoke" if mode == "smoke" else split_for_row(row_id)
        for severity_rank, transform in enumerate(specs, start=1):
            seeds = SEEDS if transform.family in {"noise", "negative_control"} and transform.transform_id in {"salt_pepper", "uniform_noise", "gaussian_noise", "dropout"} else (None,)
            for seed in seeds:
                for profile in PROFILES:
                    case, obs = build_case(
                        context=context,
                        split=split,
                        source_row_id=row_id,
                        transform=transform,
                        seed=seed,
                        severity_rank=severity_rank,
                        profile=profile,
                    )
                    cases.append(case)
                    observations[case.case_id] = obs
    return cases, observations, context
