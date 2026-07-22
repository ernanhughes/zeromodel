from __future__ import annotations

from pathlib import Path

import numpy as np

from research.evidence.video_discriminative_joint_evidence import (
    JointEvidenceRegionSpec,
    build_joint_candidate_masks,
    build_pairwise_discriminative_masks,
)
from zeromodel.observation.visual_address import ImageObservation
from research.visual.visual_registration import RegistrationConfig


def _toy_inputs() -> tuple[dict, dict, tuple[JointEvidenceRegionSpec, ...]]:
    a = ImageObservation(np.array([[0, 255], [255, 0]], dtype=np.uint8), source_id="a")
    b = ImageObservation(np.array([[255, 0], [255, 0]], dtype=np.uint8), source_id="b")
    c = ImageObservation(np.array([[0, 255], [0, 255]], dtype=np.uint8), source_id="c")
    prototypes = {
        "row-a": ("row-a", "LEFT", a.raw_digest, a),
        "row-b": ("row-b", "LEFT", b.raw_digest, b),
        "row-c": ("row-c", "RIGHT", c.raw_digest, c),
    }
    development = {
        "row-a": (a, ImageObservation(np.array([[0, 255], [255, 1]], dtype=np.uint8), source_id="a-dev")),
        "row-b": (b, ImageObservation(np.array([[255, 0], [254, 0]], dtype=np.uint8), source_id="b-dev")),
        "row-c": (c, ImageObservation(np.array([[1, 255], [0, 255]], dtype=np.uint8), source_id="c-dev")),
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
    return prototypes, development, regions


def test_pairwise_masks_are_symmetric_and_nonnegative() -> None:
    prototypes, development, _regions = _toy_inputs()
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
    assert len(pairwise) == 3
    assert all(mask.spec.row_a < mask.spec.row_b for mask in pairwise.values())
    assert all(np.all(mask.pairwise_weights >= 0.0) for mask in pairwise.values())


def test_candidate_fit_mask_drops_v2_zeroing_rule() -> None:
    prototypes, development, _regions = _toy_inputs()
    masks = build_joint_candidate_masks(
        prototypes=prototypes,
        development_observations=development,
        intensity_tolerance=8,
        stability_tolerance=12,
        amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
        operational_contract_digest="sha256:test",
        source_scope="toy",
    )
    assert any(float(mask.candidate_fit_weights.sum()) > 0.0 for mask in masks.values())
    assert all(float(mask.row_informative_weights.sum()) >= float(mask.candidate_fit_weights.sum()) for mask in masks.values())
