from __future__ import annotations

import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.video_discriminative_evidence import (
    DiscriminativeCandidateSet,
    DiscriminativeEvidenceCalibration,
    DiscriminativeMaskSpec,
    DiscriminativeRegionSpec,
    RegionDiscriminativeEvidence,
    discriminative_mask_digest,
    discriminative_provider_contract,
    discriminative_region_digest,
)
from zeromodel.visual_registration import RegistrationConfig


def _registration() -> RegistrationConfig:
    return RegistrationConfig(max_dx=2, max_dy=1, minimum_overlap_fraction=0.5)


def _region(region_id: str = "target") -> DiscriminativeRegionSpec:
    return DiscriminativeRegionSpec(
        region_id=region_id,
        top=0,
        left=0,
        height=4,
        width=5,
        weight=1.0,
        critical=True,
        registration_config=_registration(),
    )


def _mask(mask_id: str = "mask-a", row_id: str = "row-a") -> DiscriminativeMaskSpec:
    return DiscriminativeMaskSpec(
        mask_id=mask_id,
        row_id=row_id,
        action_id="LEFT",
        shape=(16, 28),
        informative_pixel_count=12,
        action_conflict_pixel_count=4,
        stable_pixel_count=10,
        prototype_digest="sha256:prototype",
        development_digest="sha256:development",
        derivation_contract="pixel-disagreement",
        intensity_tolerance=2,
    )


def _calibration() -> DiscriminativeEvidenceCalibration:
    return DiscriminativeEvidenceCalibration(
        architecture_id="A",
        minimum_available_mass=1.0,
        minimum_available_fraction=0.2,
        minimum_support=0.1,
        maximum_contradiction=0.3,
        maximum_critical_contradiction=0.1,
        exact_winner_threshold=0.9,
        exact_winner_margin=0.2,
        candidate_relative_margin=0.3,
        conflicting_action_separation=0.2,
        minimum_supporting_regions=1,
        maximum_candidate_set_size=3,
        prototype_digest="sha256:prototype",
        region_spec_digest=discriminative_region_digest((_region(),)),
        mask_spec_digest=discriminative_mask_digest((_mask(),)),
        policy_artifact_id="sha256:policy",
        source_scope="stage3-test",
    )


def test_discriminative_region_digest_is_deterministic() -> None:
    left = discriminative_region_digest((_region("a"), _region("b")))
    right = discriminative_region_digest((_region("a"), _region("b")))
    assert left == right


def test_discriminative_region_digest_rejects_duplicates() -> None:
    with pytest.raises(VPMValidationError, match="duplicate"):
        discriminative_region_digest((_region("dup"), _region("dup")))


def test_discriminative_mask_digest_is_order_independent() -> None:
    left = discriminative_mask_digest((_mask("a", "row-a"), _mask("b", "row-b")))
    right = discriminative_mask_digest((_mask("b", "row-b"), _mask("a", "row-a")))
    assert left == right


def test_discriminative_mask_rejects_invalid_shape() -> None:
    with pytest.raises(VPMValidationError, match="two-dimensional"):
        DiscriminativeMaskSpec(
            mask_id="mask-a",
            row_id="row-a",
            action_id="LEFT",
            shape=(16, 28, 1),  # type: ignore[arg-type]
            informative_pixel_count=1,
            action_conflict_pixel_count=1,
            stable_pixel_count=1,
            prototype_digest="sha256:p",
            development_digest="sha256:d",
            derivation_contract="test",
            intensity_tolerance=1,
        )


def test_discriminative_provider_contract_binds_calibration_and_digests() -> None:
    calibration = _calibration()
    contract = discriminative_provider_contract(
        calibration=calibration,
        region_spec_digest=calibration.region_spec_digest,
        mask_spec_digest=calibration.mask_spec_digest,
    )
    assert contract.provider_kind == "deterministic_discriminative_evidence"
    assert contract.metadata["architecture_id"] == "A"


def test_discriminative_provider_contract_rejects_digest_mismatch() -> None:
    calibration = _calibration()
    with pytest.raises(VPMValidationError, match="region_spec_digest"):
        discriminative_provider_contract(
            calibration=calibration,
            region_spec_digest="sha256:other",
            mask_spec_digest=calibration.mask_spec_digest,
        )


def test_region_discriminative_evidence_requires_bounded_fractions() -> None:
    with pytest.raises(VPMValidationError, match="available_informative_fraction"):
        RegionDiscriminativeEvidence(
            region_id="target",
            expected_informative_mass=1.0,
            available_informative_mass=1.0,
            available_informative_fraction=1.5,
            geometric_overlap=0.8,
            valid_pixel_count=10,
            support_mass=1.0,
            contradiction_mass=0.0,
            critical_contradiction_mass=0.0,
            conflicting_action_support_mass=0.0,
            conflicting_action_contradiction_mass=0.0,
            registration_succeeded=True,
            registration_dx=0,
            registration_dy=0,
            registration_tie_break_reason="overlap",
        )


def test_discriminative_candidate_set_requires_exact_row_only_for_exact_outcome() -> None:
    candidate_set = DiscriminativeCandidateSet(
        observation_digest="sha256:obs",
        provider_digest="sha256:provider",
        architecture_id="A",
        outcome="candidate_set_available",
        candidate_set_limit=3,
        rows=("row-a", "row-b"),
        actions=("LEFT", "RIGHT"),
        candidate_digest="sha256:candidates",
    )
    assert candidate_set.digest.startswith("sha256:")
    with pytest.raises(VPMValidationError, match="exact_row_accepted requires exact_row_id"):
        DiscriminativeCandidateSet(
            observation_digest="sha256:obs",
            provider_digest="sha256:provider",
            architecture_id="A",
            outcome="exact_row_accepted",
            candidate_set_limit=3,
            rows=("row-a",),
            actions=("LEFT",),
            candidate_digest="sha256:candidates",
        )
