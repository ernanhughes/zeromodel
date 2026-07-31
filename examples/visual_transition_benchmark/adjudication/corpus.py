from __future__ import annotations

from dataclasses import dataclass

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
)

from visual_transition_benchmark.dataset import ArcadeState, render, true_next_state


@dataclass(frozen=True)
class AddressAliasCase:
    case_id: str
    true_row_id: str
    supplied_row_id: str
    observation_source_row_id: str
    observation_transform_id: str
    alias_class: str
    profile: str
    candidate_universe: str
    evidence_mode: str
    observed_frame: np.ndarray
    true_before_frame: np.ndarray
    true_after_frame: np.ndarray
    visual_decision: VisualDecision
    feature_spec: VisualFeatureSpec
    true_action: str
    addressed_action: str | None


def _state_from_row(row_id: str) -> ArcadeState:
    tank, target, cooldown = parse_state_row_id(row_id)
    return ArcadeState(
        tank_x=tank, aliens=() if target is None else (target,), cooldown=cooldown
    )


def _frame_for_row(row_id: str, *, config: ShooterConfig) -> np.ndarray:
    tank, target, cooldown = parse_state_row_id(row_id)
    return np.array(
        render_state_frame(tank, target, cooldown, width=config.width), copy=True
    )


def _noncanonical_exact(frame: np.ndarray) -> np.ndarray:
    return np.stack((frame, frame, frame), axis=-1).astype(np.uint8, copy=False)


def _canonical_changed_background_pixel(frame: np.ndarray) -> np.ndarray:
    mutated = np.array(frame, dtype=np.uint8, copy=True)
    mutated[0, 0] = 1 if mutated[0, 0] == 0 else mutated[0, 0]
    return mutated


def build_reader(
    profile: str = VisualAcceptanceProfile.EXACT_CODEWORD,
) -> VisualSignReader:
    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    feature_spec = VisualFeatureSpec(
        input_height=16,
        input_width=config.width * 4,
        target_height=4,
        target_width=7,
        quantization_levels=16,
    )
    frames = {
        row_id: _frame_for_row(str(row_id), config=config)
        for row_id in policy.source.row_ids
    }
    visual_index = build_visual_index(policy, frames, feature_spec)
    return VisualSignReader(
        visual_index.artifact,
        policy,
        action_metric_ids=("LEFT", "RIGHT", "STAY", "FIRE"),
        acceptance_profile=profile,
    )


def build_feature_spec() -> VisualFeatureSpec:
    config = ShooterConfig()
    return VisualFeatureSpec(
        input_height=16,
        input_width=config.width * 4,
        target_height=4,
        target_width=7,
        quantization_levels=16,
    )


def _make_case(
    *,
    case_id: str,
    true_row_id: str,
    supplied_row_id: str,
    transform: str,
    alias_class: str,
    profile: str,
    candidate_universe: str,
    evidence_mode: str,
) -> AddressAliasCase:
    config = ShooterConfig()
    reader = build_reader(profile)
    true_frame = _frame_for_row(true_row_id, config=config)
    if transform == "canonical":
        observed = true_frame
    elif transform == "noncanonical_exact_background_pixel":
        observed = _noncanonical_exact(true_frame)
    elif transform == "canonical_changed_background_pixel":
        observed = _canonical_changed_background_pixel(true_frame)
    elif transform == "distant_inversion":
        observed = np.array(255 - true_frame, dtype=np.uint8, copy=True)
    else:
        raise ValueError(f"unsupported observation transform: {transform}")
    decision = reader.read(observed, acceptance_profile=profile)
    true_state = _state_from_row(true_row_id)
    addressed_action = decision.action if decision.policy_executed else None
    true_action = (
        build_reader(VisualAcceptanceProfile.CANONICAL_ONLY)
        .read(
            _frame_for_row(true_row_id, config=config),
            acceptance_profile=VisualAcceptanceProfile.CANONICAL_ONLY,
        )
        .action
    )
    action_to_execute = addressed_action or "STAY"
    true_after = true_next_state(true_state, action_to_execute)
    return AddressAliasCase(
        case_id=case_id,
        true_row_id=true_row_id,
        supplied_row_id=supplied_row_id,
        observation_source_row_id=true_row_id,
        observation_transform_id=transform,
        alias_class=alias_class,
        profile=profile,
        candidate_universe=candidate_universe,
        evidence_mode=evidence_mode,
        observed_frame=observed,
        true_before_frame=render(true_state),
        true_after_frame=render(true_after),
        visual_decision=decision,
        feature_spec=reader.feature_spec,
        true_action=str(true_action),
        addressed_action=addressed_action,
    )


def build_case_corpus() -> tuple[AddressAliasCase, ...]:
    rows = {
        "canonical_fire_hit": "tank=0|target=0|cooldown=0",
        "wait_no_effect_a": "tank=0|target=none|cooldown=0",
        "blocked_left": "tank=0|target=3|cooldown=0",
    }
    specs = [
        (
            "canonical-only-accepted",
            rows["canonical_fire_hit"],
            rows["canonical_fire_hit"],
            "canonical",
            "canonical exact",
            VisualAcceptanceProfile.CANONICAL_ONLY,
        ),
        (
            "exact-codeword-canonical",
            rows["canonical_fire_hit"],
            rows["canonical_fire_hit"],
            "canonical",
            "canonical exact",
            VisualAcceptanceProfile.EXACT_CODEWORD,
        ),
        (
            "exact-codeword-noncanonical-correct",
            rows["canonical_fire_hit"],
            rows["canonical_fire_hit"],
            "noncanonical_exact_background_pixel",
            "exact feature codeword, correct row",
            VisualAcceptanceProfile.EXACT_CODEWORD,
        ),
        (
            "calibrated-nearest-correct",
            rows["blocked_left"],
            rows["blocked_left"],
            "noncanonical_exact_background_pixel",
            "nearest accepted, correct row",
            VisualAcceptanceProfile.CALIBRATED_NEAREST,
        ),
        (
            "no-effect-unresolved",
            rows["wait_no_effect_a"],
            rows["wait_no_effect_a"],
            "canonical",
            "canonical exact",
            VisualAcceptanceProfile.EXACT_CODEWORD,
        ),
        (
            "insufficient-observability",
            rows["blocked_left"],
            rows["blocked_left"],
            "canonical",
            "insufficient observability",
            VisualAcceptanceProfile.EXACT_CODEWORD,
        ),
    ]
    cases = []
    for mode in ("component", "value"):
        for universe in ("reader_local", "policy_action"):
            for case_id, true_row, supplied_row, transform, alias_class, profile in specs:
                cases.append(
                    _make_case(
                        case_id=f"{case_id}-{universe}-{mode}",
                        true_row_id=true_row,
                        supplied_row_id=supplied_row,
                        transform=transform,
                        alias_class=alias_class,
                        profile=profile,
                        candidate_universe=universe,
                        evidence_mode=mode,
                    )
                )
    cases.append(
        _make_case(
            case_id="reader-rejected-canonical-only",
            true_row_id=rows["canonical_fire_hit"],
            supplied_row_id=rows["canonical_fire_hit"],
            transform="canonical_changed_background_pixel",
            alias_class="reader rejected",
            profile=VisualAcceptanceProfile.CANONICAL_ONLY,
            candidate_universe="reader_local",
            evidence_mode="component",
        )
    )
    cases.append(
        _make_case(
            case_id="evidence-only-no-execution",
            true_row_id=rows["canonical_fire_hit"],
            supplied_row_id=rows["canonical_fire_hit"],
            transform="noncanonical_exact_background_pixel",
            alias_class="exact feature codeword, correct row",
            profile=VisualAcceptanceProfile.EVIDENCE_ONLY,
            candidate_universe="reader_local",
            evidence_mode="component",
        )
    )
    cases.append(
        _make_case(
            case_id="calibrated-nearest-rejected-distant",
            true_row_id=rows["canonical_fire_hit"],
            supplied_row_id=rows["canonical_fire_hit"],
            transform="distant_inversion",
            alias_class="reader rejected",
            profile=VisualAcceptanceProfile.CALIBRATED_NEAREST,
            candidate_universe="reader_local",
            evidence_mode="component",
        )
    )
    return tuple(cases)
