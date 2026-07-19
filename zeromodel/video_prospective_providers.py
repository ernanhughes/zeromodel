from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from examples.arcade_shooter_policy import ACTIONS
from .artifact import VPMValidationError
from .video_complete_row_evidence import CompleteRowEvidence, build_complete_row_evidence
from .video_discriminative_joint_evidence import (
    JointEvidenceCalibration,
    JointEvidenceProvider,
    JointEvidenceRegionSpec,
    build_joint_candidate_masks,
    build_joint_row_candidates,
    build_pairwise_discriminative_masks,
)
from .video_local_correlation import (
    LocalCorrelationCalibration,
    LocalCorrelationVideoAddressProvider,
    LocalRegionSpec,
    local_region_digest,
)
from .visual_address import ImageObservation
from .visual_registration import RegistrationConfig


PROSPECTIVE_P1_VERSION = "zeromodel-video-prospective-normalized-pixel/v1"
PROSPECTIVE_P2_VERSION = "zeromodel-video-prospective-local-correlation/v1"
PROSPECTIVE_P3_VERSION = "zeromodel-video-prospective-b3-joint-fit/v1"


@dataclass(frozen=True)
class ProspectiveProviderResult:
    provider_id: str
    provider_version: str
    evidence: CompleteRowEvidence
    winner_row_id: str
    winner_action_id: str
    maximum_tie_size: int
    diagnostics: Mapping[str, Any]


_P2_PROVIDER_CACHE: dict[tuple[int, str, str], LocalCorrelationVideoAddressProvider] = {}
_P3_STATE_CACHE: dict[tuple[int, str, str], dict[str, Any]] = {}


def _row_action_map(
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
) -> dict[str, str]:
    return {row_id: action_id for _observation_id, (row_id, action_id, _digest, _obs) in prototypes.items()}


def _canonical_regions() -> tuple[LocalRegionSpec, ...]:
    registration = RegistrationConfig(max_dx=2, max_dy=2, minimum_overlap_fraction=0.5)
    return (
        LocalRegionSpec("target_band", top=0, left=0, height=6, width=28, weight=2.0, registration_config=registration, critical=True),
        LocalRegionSpec("cooldown_indicator", top=7, left=25, height=2, width=2, weight=1.5, registration_config=registration, critical=True),
        LocalRegionSpec("tank_band", top=10, left=0, height=4, width=28, weight=2.0, registration_config=registration, critical=True),
    )


def _bounded_similarity_from_distance(distance: float) -> float:
    if not np.isfinite(float(distance)):
        raise VPMValidationError("distance must be finite")
    return max(0.0, min(1.0, 1.0 - float(distance) / 2.0))


def score_normalized_pixel(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
) -> ProspectiveProviderResult:
    row_action = _row_action_map(prototypes)
    rows = []
    for row_id, _action_id, _digest, proto in prototypes.values():
        diff = np.abs(observation.pixels.astype(np.float32) - proto.pixels.astype(np.float32))
        mae = float(np.sum(diff) / (255.0 * diff.size))
        similarity = max(0.0, min(1.0, 1.0 - mae))
        rows.append((row_id, similarity))
    evidence = build_complete_row_evidence(
        row_scores=rows,
        policy_artifact_id=policy_artifact_id,
        provider_id="P1",
        provider_version=PROSPECTIVE_P1_VERSION,
    )
    winner_row = evidence.ranking.ranked_row_ids[0]
    winner_action = row_action[winner_row]
    return ProspectiveProviderResult(
        provider_id="P1",
        provider_version=PROSPECTIVE_P1_VERSION,
        evidence=evidence,
        winner_row_id=winner_row,
        winner_action_id=winner_action,
        maximum_tie_size=max(len(group.row_ids) for group in evidence.ranking.tie_groups),
        diagnostics={"score_count": len(rows)},
    )


def score_registered_local_correlation(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    source_scope: str,
) -> ProspectiveProviderResult:
    cache_key = (id(prototypes), policy_artifact_id, source_scope)
    provider = _P2_PROVIDER_CACHE.get(cache_key)
    if provider is None:
        regions = _canonical_regions()
        calibration = LocalCorrelationCalibration(
            winner_threshold=1.0,
            runner_up_margin=0.0,
            conflicting_action_margin=0.0,
            minimum_visible_fraction=0.5,
            region_spec_digest=local_region_digest(regions),
            prototype_digest="sha256:prospective-prototypes",
            benign_calibration_digest="sha256:prospective-calibration",
            rejection_calibration_digest="sha256:prospective-selection",
            policy_artifact_id=policy_artifact_id,
            source_scope=source_scope,
        )
        provider = LocalCorrelationVideoAddressProvider(
            prototypes={observation_id: (row_id, action_id, digest, proto) for observation_id, (row_id, action_id, digest, proto) in prototypes.items()},
            calibration=calibration,
            regions=regions,
        )
        _P2_PROVIDER_CACHE[cache_key] = provider
    ranked = provider._rank(observation)
    similarities = [(candidate.row_id, _bounded_similarity_from_distance(candidate.total_distance)) for candidate in ranked]
    evidence = build_complete_row_evidence(
        row_scores=similarities,
        policy_artifact_id=policy_artifact_id,
        provider_id="P2",
        provider_version=PROSPECTIVE_P2_VERSION,
    )
    winner = ranked[0]
    return ProspectiveProviderResult(
        provider_id="P2",
        provider_version=PROSPECTIVE_P2_VERSION,
        evidence=evidence,
        winner_row_id=winner.row_id,
        winner_action_id=winner.action_id,
        maximum_tie_size=max(len(group.row_ids) for group in evidence.ranking.tie_groups),
        diagnostics={"candidate_count": len(ranked)},
    )


def _joint_regions() -> tuple[JointEvidenceRegionSpec, ...]:
    registration = RegistrationConfig(max_dx=2, max_dy=2, minimum_overlap_fraction=0.5)
    return (
        JointEvidenceRegionSpec("target_band", 0, 0, 6, 28, 2.0, True, registration),
        JointEvidenceRegionSpec("cooldown_indicator", 7, 25, 2, 2, 1.5, True, registration),
        JointEvidenceRegionSpec("tank_band", 10, 0, 4, 28, 2.0, True, registration),
    )


def _joint_calibration(policy_artifact_id: str, source_scope: str) -> JointEvidenceCalibration:
    return JointEvidenceCalibration(
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
        prototype_digest="sha256:prospective-prototypes",
        region_spec_digest="sha256:prospective-joint-regions",
        candidate_mask_digest="sha256:prospective-joint-candidate-masks",
        pairwise_mask_digest="sha256:prospective-joint-pairwise-masks",
        policy_artifact_id=policy_artifact_id,
        source_scope=source_scope,
        amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
        operational_contract_digest="sha256:prospective-b3-wrapper",
    )


def score_b3_joint_fit(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    source_scope: str,
) -> ProspectiveProviderResult:
    cache_key = (id(prototypes), policy_artifact_id, source_scope)
    state = _P3_STATE_CACHE.get(cache_key)
    if state is None:
        joint_prototypes = {row_id: (row_id, action_id, digest, proto) for _obs_id, (row_id, action_id, digest, proto) in prototypes.items()}
        development = {row_id: (proto, proto) for row_id, (_row, _action, _digest, proto) in joint_prototypes.items()}
        regions = _joint_regions()
        candidate_masks = build_joint_candidate_masks(
            prototypes=joint_prototypes,
            development_observations=development,
            intensity_tolerance=8,
            stability_tolerance=12,
            amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
            operational_contract_digest="sha256:prospective-b3-wrapper",
            source_scope=source_scope,
        )
        pairwise_masks = build_pairwise_discriminative_masks(
            prototypes=joint_prototypes,
            candidate_masks=candidate_masks,
            intensity_tolerance=8,
            amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
            operational_contract_digest="sha256:prospective-b3-wrapper",
            source_scope=source_scope,
        )
        provider = JointEvidenceProvider(
            prototypes=joint_prototypes,
            candidate_masks=candidate_masks,
            pairwise_masks=pairwise_masks,
            regions=regions,
            calibration=_joint_calibration(policy_artifact_id, source_scope),
            policy_artifact_id=policy_artifact_id,
            source_scope=source_scope,
        )
        state = {
            "joint_prototypes": joint_prototypes,
            "regions": regions,
            "candidate_masks": candidate_masks,
            "pairwise_masks": pairwise_masks,
            "provider_contract_digest": provider.contract().digest,
            "row_action": _row_action_map(prototypes),
        }
        _P3_STATE_CACHE[cache_key] = state
    ranked = build_joint_row_candidates(
        observation=observation,
        prototypes=state["joint_prototypes"],
        candidate_masks=state["candidate_masks"],
        pairwise_masks=state["pairwise_masks"],
        regions=state["regions"],
        architecture_id="B3",
    )
    rows = [(candidate.row_id, float(candidate.candidate_strength)) for candidate in ranked]
    evidence = build_complete_row_evidence(
        row_scores=rows,
        policy_artifact_id=policy_artifact_id,
        provider_id="P3",
        provider_version=PROSPECTIVE_P3_VERSION,
    )
    winner_row = evidence.ranking.ranked_row_ids[0]
    winner_action = state["row_action"][winner_row]
    return ProspectiveProviderResult(
        provider_id="P3",
        provider_version=PROSPECTIVE_P3_VERSION,
        evidence=evidence,
        winner_row_id=winner_row,
        winner_action_id=winner_action,
        maximum_tie_size=max(len(group.row_ids) for group in evidence.ranking.tie_groups),
        diagnostics={"candidate_count": len(rows), "provider_contract_digest": state["provider_contract_digest"]},
    )


__all__ = [
    "PROSPECTIVE_P1_VERSION",
    "PROSPECTIVE_P2_VERSION",
    "PROSPECTIVE_P3_VERSION",
    "ProspectiveProviderResult",
    "score_b3_joint_fit",
    "score_normalized_pixel",
    "score_registered_local_correlation",
]
