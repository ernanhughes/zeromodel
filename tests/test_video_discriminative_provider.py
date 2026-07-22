from __future__ import annotations

import numpy as np

from zeromodel.observation.visual_address import ImageObservation
from research.evidence.video_discriminative_evidence import video_discriminative_evidence as zde
from research.evidence.video_discriminative_evidence import (
    DiscriminativeEvidenceCalibration,
    DiscriminativeEvidenceProvider,
    DiscriminativeMask,
    DiscriminativeMaskSpec,
    DiscriminativeRegionSpec,
)
from research.visual.visual_registration import RegistrationConfig


def _region() -> DiscriminativeRegionSpec:
    return DiscriminativeRegionSpec(
        region_id="tiny",
        top=0,
        left=0,
        height=1,
        width=2,
        weight=1.0,
        critical=True,
        registration_config=RegistrationConfig(max_dx=1, max_dy=0, minimum_overlap_fraction=0.5),
    )


def _mask(*, row_id: str, action_id: str, pixels: np.ndarray) -> DiscriminativeMask:
    spec = DiscriminativeMaskSpec(
        mask_id=f"{row_id}|mask",
        row_id=row_id,
        action_id=action_id,
        shape=tuple(int(item) for item in pixels.shape),
        informative_pixel_count=int(np.count_nonzero(pixels > 0)),
        action_conflict_pixel_count=int(np.count_nonzero(pixels > 0)),
        stable_pixel_count=int(np.count_nonzero(np.ones_like(pixels))),
        prototype_digest=f"sha256:{row_id}",
        development_digest="sha256:dev",
        derivation_contract="unit-test",
        intensity_tolerance=0,
    )
    weights = (pixels > 0).astype(np.float32)
    return DiscriminativeMask(
        spec=spec,
        row_informative_weights=weights,
        action_conflict_weights=weights,
        stable_weights=np.ones_like(weights, dtype=np.float32),
        separation_weights=np.ones_like(weights, dtype=np.float32),
    )


def _provider(
    *,
    prototypes: dict[str, tuple[str, str, str, ImageObservation]],
    masks: dict[str, DiscriminativeMask],
    architecture_id: str = "C",
    maximum_candidate_set_size: int = 3,
    exact_winner_margin: float = 0.2,
    exact_winner_threshold: float = 0.8,
    minimum_support: float = 0.5,
    minimum_supporting_regions: int = 1,
) -> DiscriminativeEvidenceProvider:
    regions = (_region(),)
    calibration = DiscriminativeEvidenceCalibration(
        architecture_id=architecture_id,
        minimum_available_mass=1.0,
        minimum_available_fraction=0.5,
        minimum_support=minimum_support,
        maximum_contradiction=0.5,
        maximum_critical_contradiction=0.5,
        exact_winner_threshold=exact_winner_threshold,
        exact_winner_margin=exact_winner_margin,
        candidate_relative_margin=0.0,
        conflicting_action_separation=0.0,
        minimum_supporting_regions=minimum_supporting_regions,
        maximum_candidate_set_size=maximum_candidate_set_size,
        prototype_digest=zde._prototype_payload_digest(prototypes),
        region_spec_digest=zde.discriminative_region_digest(regions),
        mask_spec_digest=zde.discriminative_mask_digest(tuple(mask.spec for mask in masks.values())),
        policy_artifact_id="sha256:policy",
        source_scope="stage3-test",
    )
    return DiscriminativeEvidenceProvider(
        prototypes=prototypes,
        masks=masks,
        regions=regions,
        calibration=calibration,
        policy_artifact_id="sha256:policy",
        source_scope="stage3-test",
    )


def test_v4_exact_accepts_tiny_region_pattern_without_stage2_global_visibility_veto() -> None:
    observation_pixels = np.array([[0, 255]], dtype=np.uint8)
    prototypes = {
        "obs-a": ("row-a", "LEFT", "sha256:row-a", ImageObservation(observation_pixels, source_id="row-a")),
        "obs-b": ("row-b", "RIGHT", "sha256:row-b", ImageObservation(np.array([[0, 0]], dtype=np.uint8), source_id="row-b")),
    }
    masks = {
        "row-a": _mask(row_id="row-a", action_id="LEFT", pixels=observation_pixels),
        "row-b": _mask(row_id="row-b", action_id="RIGHT", pixels=np.array([[255, 255]], dtype=np.uint8)),
    }
    provider = _provider(prototypes=prototypes, masks=masks)
    decision = provider.evaluate(ImageObservation(observation_pixels, source_id="frame-0"))
    assert decision.evidence_state == "exact_row_accepted"
    assert decision.candidate_set.exact_row_id == "row-a"
    assert decision.exact_address_decision.accepted is True
    assert decision.exact_address_decision.matched_row_id == "row-a"


def test_v4_same_action_ambiguity_produces_candidate_set_not_exact() -> None:
    shared = np.array([[0, 255]], dtype=np.uint8)
    prototypes = {
        "obs-a": ("row-a", "LEFT", "sha256:row-a", ImageObservation(shared, source_id="row-a")),
        "obs-b": ("row-b", "LEFT", "sha256:row-b", ImageObservation(shared, source_id="row-b")),
    }
    masks = {
        "row-a": _mask(row_id="row-a", action_id="LEFT", pixels=shared),
        "row-b": _mask(row_id="row-b", action_id="LEFT", pixels=shared),
    }
    provider = _provider(
        prototypes=prototypes,
        masks=masks,
        exact_winner_margin=0.5,
        minimum_support=0.0,
        minimum_supporting_regions=0,
        exact_winner_threshold=0.0,
    )
    decision = provider.evaluate(ImageObservation(shared, source_id="frame-0"))
    assert decision.evidence_state == "candidate_set_available"
    assert decision.candidate_set.rows == ("row-a", "row-b")
    assert decision.candidate_set.unique_action_candidate_set is True
    assert decision.exact_address_decision.accepted is False
    assert decision.exact_address_decision.reason == "candidate_set_available"


def test_v4_conflicting_action_ambiguity_produces_candidate_set_not_exact() -> None:
    shared = np.array([[0, 255]], dtype=np.uint8)
    prototypes = {
        "obs-a": ("row-a", "LEFT", "sha256:row-a", ImageObservation(shared, source_id="row-a")),
        "obs-b": ("row-b", "RIGHT", "sha256:row-b", ImageObservation(shared, source_id="row-b")),
    }
    masks = {
        "row-a": _mask(row_id="row-a", action_id="LEFT", pixels=shared),
        "row-b": _mask(row_id="row-b", action_id="RIGHT", pixels=shared),
    }
    provider = _provider(
        prototypes=prototypes,
        masks=masks,
        exact_winner_margin=0.5,
        minimum_support=0.0,
        minimum_supporting_regions=0,
        exact_winner_threshold=0.0,
    )
    decision = provider.evaluate(ImageObservation(shared, source_id="frame-0"))
    assert decision.evidence_state == "candidate_set_available"
    assert set(decision.candidate_set.actions) == {"LEFT", "RIGHT"}
    assert decision.candidate_set.unique_action_candidate_set is False
    assert decision.exact_address_decision.accepted is False


def test_v4_oversized_candidate_set_is_rejected_without_truncation() -> None:
    shared = np.array([[0, 255]], dtype=np.uint8)
    prototypes = {
        f"obs-{index}": (f"row-{index}", "LEFT", f"sha256:row-{index}", ImageObservation(shared, source_id=f"row-{index}"))
        for index in range(4)
    }
    masks = {
        f"row-{index}": _mask(row_id=f"row-{index}", action_id="LEFT", pixels=shared)
        for index in range(4)
    }
    provider = _provider(
        prototypes=prototypes,
        masks=masks,
        maximum_candidate_set_size=3,
        exact_winner_margin=0.5,
        minimum_support=0.0,
        minimum_supporting_regions=0,
        exact_winner_threshold=0.0,
    )
    decision = provider.evaluate(ImageObservation(shared, source_id="frame-0"))
    assert decision.evidence_state == "no_sufficient_evidence"
    assert decision.candidate_set.rejection_reason == "candidate_set_too_large"
    assert len(decision.candidate_set.excluded_nearby_rows) == 4
    assert decision.exact_address_decision.accepted is False


def test_v4_cache_identity_separates_identical_pixels_across_source_ids() -> None:
    pixels = np.array([[0, 255]], dtype=np.uint8)
    prototypes = {
        "obs-a": ("row-a", "LEFT", "sha256:row-a", ImageObservation(pixels, source_id="row-a")),
        "obs-b": ("row-b", "RIGHT", "sha256:row-b", ImageObservation(np.array([[255, 0]], dtype=np.uint8), source_id="row-b")),
    }
    masks = {
        "row-a": _mask(row_id="row-a", action_id="LEFT", pixels=pixels),
        "row-b": _mask(row_id="row-b", action_id="RIGHT", pixels=np.array([[255, 0]], dtype=np.uint8)),
    }
    provider = _provider(prototypes=prototypes, masks=masks)
    left = provider.read(ImageObservation(pixels, source_id="clip-a:frame-0"))
    right = provider.read(ImageObservation(pixels, source_id="clip-b:frame-0"))
    assert left.representation_digest != right.representation_digest
