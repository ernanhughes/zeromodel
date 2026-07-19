from __future__ import annotations

import numpy as np
import pytest

from examples.arcade_visual_video_local_correlation_benchmark import (
    build_video_cases,
    run_calibrate,
    run_evaluate,
    run_verify,
)
from examples.arcade_shooter_policy import ACTIONS, ShooterConfig, compile_policy_artifact
from examples.arcade_visual_local_evidence_benchmark import SOURCE_SCOPE
from examples.arcade_visual_sign_reader import render_state_frame
from zeromodel import VPMPolicyLookup
from zeromodel.artifact import VPMValidationError
from zeromodel.video_local_correlation import (
    LocalCorrelationCalibration,
    LocalCorrelationVideoAddressProvider,
    LocalRegionSpec,
    local_region_digest,
)
from zeromodel.visual_address import ImageObservation
from zeromodel.visual_registration import RegistrationConfig


def _regions() -> tuple[LocalRegionSpec, ...]:
    registration = RegistrationConfig(max_dx=2, max_dy=2, minimum_overlap_fraction=0.5)
    return (
        LocalRegionSpec("target_band", top=0, left=0, height=6, width=28, weight=2.0, registration_config=registration, critical=True),
        LocalRegionSpec("cooldown_indicator", top=7, left=25, height=2, width=2, weight=1.5, registration_config=registration, critical=True),
        LocalRegionSpec("tank_band", top=10, left=0, height=4, width=28, weight=2.0, registration_config=registration, critical=True),
    )


def _provider() -> LocalCorrelationVideoAddressProvider:
    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    lookup = VPMPolicyLookup(policy, action_metric_ids=ACTIONS)
    prototypes = {}
    for tank_x in range(config.width):
        for target_x in (None, *range(config.width)):
            for cooldown in (0, 1):
                row_id = f"tank={tank_x}|target={'none' if target_x is None else target_x}|cooldown={cooldown}"
                observation_id = row_id
                frame = render_state_frame(tank_x, target_x, cooldown, width=config.width)
                observation = ImageObservation(frame, source_id=observation_id)
                prototypes[observation_id] = (
                    row_id,
                    lookup.choose(row_id),
                    observation.raw_digest,
                    observation,
                )
    calibration = LocalCorrelationCalibration(
        winner_threshold=0.2,
        runner_up_margin=0.01,
        conflicting_action_margin=0.01,
        minimum_visible_fraction=0.5,
        region_spec_digest=local_region_digest(_regions()),
        prototype_digest="sha256:prototype",
        benign_calibration_digest="sha256:benign",
        rejection_calibration_digest="sha256:reject",
        policy_artifact_id=policy.artifact_id,
        source_scope=SOURCE_SCOPE,
    )
    return LocalCorrelationVideoAddressProvider(
        prototypes=prototypes,
        calibration=calibration,
        regions=_regions(),
    )


def test_local_correlation_provider_is_deterministic() -> None:
    provider = _provider()
    observation = ImageObservation(render_state_frame(3, 0, 0), source_id="obs-a")
    first = provider.read(observation)
    second = provider.read(observation)
    assert first.to_dict() == second.to_dict()
    assert first.trace["best_candidate"]["region_evidence"]


def test_local_correlation_provider_separates_identical_pixels_across_clips_in_cache_identity() -> None:
    provider = _provider()
    frame = render_state_frame(3, 0, 0)
    left = provider.read(ImageObservation(frame, source_id="clip-a:frame-0"))
    right = provider.read(ImageObservation(frame, source_id="clip-b:frame-0"))
    assert left.representation_digest != right.representation_digest


def test_local_correlation_provider_rejects_geometry_mismatch() -> None:
    provider = _provider()
    with pytest.raises(VPMValidationError, match="geometry"):
        provider.read(ImageObservation(np.zeros((8, 8), dtype=np.uint8), source_id="bad"))


def test_local_correlation_provider_retains_conflicting_action_candidate() -> None:
    provider = _provider()
    observation = ImageObservation(render_state_frame(3, 0, 0), source_id="obs-a")
    decision = provider.read(observation)
    assert decision.trace["nearest_conflicting_action_candidate"] is not None
    assert decision.trace["raw_top1_row_id"] is not None


def test_video_benchmark_generation_is_deterministic() -> None:
    left = build_video_cases()
    right = build_video_cases()
    assert [case.case_id for case in left] == [case.case_id for case in right]
    assert [case.source.manifest().manifest_id for case in left] == [case.source.manifest().manifest_id for case in right]


def test_video_benchmark_reordered_case_has_distinct_identity() -> None:
    cases = {case.case_id: case for case in build_video_cases()}
    assert cases["video-final-negative-reordered"].source.manifest().manifest_id != cases["video-final-benign-exact"].source.manifest().manifest_id


@pytest.mark.slow
def test_stage2_harness_freezes_and_verifies_negative_result(tmp_path) -> None:
    calibrate = run_calibrate(output_dir=tmp_path)
    assert calibrate["v2_selection_status"] == "no_feasible_operating_point"
    assert calibrate["v2_safe_nonzero_operating_point_exists"] is False

    evaluate = run_evaluate(output_dir=tmp_path)
    assert evaluate["claim_category"] == "No feasible V2"
    assert evaluate["kill_condition"] == "A"
    assert evaluate["paired_v2_v3"]["rejected_by_both"] > 0

    verify = run_verify(output_dir=tmp_path)
    assert verify["verified"] is True
