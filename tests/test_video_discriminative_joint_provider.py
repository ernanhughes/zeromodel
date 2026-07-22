from __future__ import annotations

import numpy as np

from research.evidence.video_discriminative_joint_evidence import (
    JointEvidenceCalibration,
    JointEvidenceProvider,
    JointEvidenceRegionSpec,
    build_joint_candidate_masks,
    build_joint_candidate_set,
    build_pairwise_discriminative_masks,
)
from zeromodel.observation.visual_address import ImageObservation
from research.visual.visual_registration import RegistrationConfig


def _provider_fixture():
    exact = ImageObservation(np.array([[0, 255], [255, 0]], dtype=np.uint8), source_id="exact")
    rival = ImageObservation(np.array([[0, 255], [255, 0]], dtype=np.uint8), source_id="rival")
    tie = ImageObservation(np.array([[0, 255], [255, 0]], dtype=np.uint8), source_id="tie")
    prototypes = {
        "row-a": ("row-a", "LEFT", exact.raw_digest, exact),
        "row-b": ("row-b", "RIGHT", rival.raw_digest, rival),
    }
    development = {
        "row-a": (exact, ImageObservation(np.array([[0, 255], [255, 1]], dtype=np.uint8), source_id="a-dev")),
        "row-b": (rival, ImageObservation(np.array([[255, 0], [254, 0]], dtype=np.uint8), source_id="b-dev")),
    }
    regions = (
        JointEvidenceRegionSpec(
            region_id="full",
            top=0,
            left=0,
            height=2,
            width=2,
            weight=1.0,
            critical=True,
            registration_config=RegistrationConfig(max_dx=0, max_dy=0, minimum_overlap_fraction=1.0),
        ),
    )
    masks = build_joint_candidate_masks(
        prototypes=prototypes,
        development_observations=development,
        intensity_tolerance=8,
        stability_tolerance=12,
        amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
        operational_contract_digest="sha256:test",
        source_scope="toy",
    )
    pairwise = build_pairwise_discriminative_masks(
        prototypes=prototypes,
        candidate_masks=masks,
        intensity_tolerance=8,
        amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
        operational_contract_digest="sha256:test",
        source_scope="toy",
    )
    calibration = JointEvidenceCalibration(
        architecture_id="B3",
        minimum_actual_scored_mass=0.0,
        minimum_available_candidate_fit_fraction=0.0,
        minimum_candidate_joint_fit=0.0,
        minimum_pairwise_margin=-1.0,
        minimum_conflicting_action_margin=-1.0,
        exact_winner_threshold=0.0,
        exact_winner_margin=0.0,
        candidate_relative_margin=0.0,
        maximum_candidate_set_size=3,
        prototype_digest="sha256:proto",
        region_spec_digest="sha256:region",
        candidate_mask_digest="sha256:candidate",
        pairwise_mask_digest="sha256:pairwise",
        policy_artifact_id="policy",
        source_scope="toy",
        amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
        operational_contract_digest="sha256:test",
    )
    provider = JointEvidenceProvider(
        prototypes=prototypes,
        candidate_masks=masks,
        pairwise_masks=pairwise,
        regions=regions,
        calibration=calibration,
        policy_artifact_id="policy",
        source_scope="toy",
    )
    return provider, tie


def test_exact_tie_safety_rejects_equal_strength_rows() -> None:
    provider, observation = _provider_fixture()
    ranked = provider._rank(observation)
    candidate_set = build_joint_candidate_set(ranked_candidates=ranked, calibration=provider._calibration)
    assert candidate_set.outcome != "exact_row_accepted"
    assert ranked[0].eligible_for_exact is False


def test_provider_ordering_is_deterministic() -> None:
    provider, observation = _provider_fixture()
    first = provider._rank(observation)
    second = provider._rank(observation)
    assert [item.to_dict() for item in first] == [item.to_dict() for item in second]
