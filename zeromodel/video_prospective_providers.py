from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Generic, Mapping, Sequence, TypeVar

import numpy as np

from .artifact import VPMValidationError
from .content_identity import PrototypeUniverseIdentity, UnresolvedArtifactIdentity, prototype_universe_identity, sha256_digest
from .video_complete_row_evidence import CompleteRowEvidence, SemanticTopSetOutcome, build_complete_row_evidence, build_semantic_top_set_outcome
from .video_discriminative_joint_evidence import (
    JointEvidenceCalibration,
    JointEvidenceProvider,
    JointEvidenceRegionSpec,
    build_joint_candidate_masks,
    build_joint_row_candidates,
    build_pairwise_discriminative_masks,
    joint_candidate_mask_digest,
    joint_region_digest,
    pairwise_mask_digest,
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
PROSPECTIVE_PROVIDER_IDS = ("P1", "P2", "P3")
PROSPECTIVE_PROVIDER_VERSIONS = {
    "P1": PROSPECTIVE_P1_VERSION,
    "P2": PROSPECTIVE_P2_VERSION,
    "P3": PROSPECTIVE_P3_VERSION,
}
_DEFAULT_CACHE_CAPACITY = 8


T = TypeVar("T")


class _LRUCache(Generic[T]):
    def __init__(self, capacity: int) -> None:
        self._capacity = int(capacity)
        self._data: OrderedDict[tuple[Any, ...], T] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def get(self, key: tuple[Any, ...]) -> T | None:
        with self._lock:
            if key in self._data:
                self._hits += 1
                value = self._data.pop(key)
                self._data[key] = value
                return value
            self._misses += 1
            return None

    def put(self, key: tuple[Any, ...], value: T) -> T:
        with self._lock:
            if key in self._data:
                self._data.pop(key)
            self._data[key] = value
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def info(self) -> dict[str, int]:
        with self._lock:
            return {"capacity": self._capacity, "size": len(self._data), "hits": self._hits, "misses": self._misses}


@dataclass(frozen=True)
class ProspectiveProviderResult:
    provider_id: str
    provider_version: str
    evidence: CompleteRowEvidence
    winner_row_id: str | None
    winner_action_id: str | None
    maximum_tie_size: int
    semantic_top_set_outcome: SemanticTopSetOutcome
    diagnostics: Mapping[str, Any]


@dataclass(frozen=True)
class ProviderScoreVector:
    provider_id: str
    provider_version: str
    row_ids: tuple[str, ...]
    raw_scores: tuple[float, ...]
    quantized_scores: tuple[int, ...]
    evidence: CompleteRowEvidence


_P2_PROVIDER_CACHE: _LRUCache[LocalCorrelationVideoAddressProvider] = _LRUCache(_DEFAULT_CACHE_CAPACITY)
_P3_STATE_CACHE: _LRUCache[dict[str, Any]] = _LRUCache(_DEFAULT_CACHE_CAPACITY)


def clear_prospective_provider_caches() -> None:
    _P2_PROVIDER_CACHE.clear()
    _P3_STATE_CACHE.clear()


def prospective_provider_cache_info() -> dict[str, dict[str, int]]:
    return {"P2": _P2_PROVIDER_CACHE.info(), "P3": _P3_STATE_CACHE.info()}


def _policy_row_ids(prototypes: Mapping[str, tuple[str, str, str, ImageObservation]]) -> tuple[str, ...]:
    return tuple(row_id for row_id, *_rest in (value for _key, value in sorted(prototypes.items())))


def _prototype_identity(
    *,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    source_scope: str,
) -> PrototypeUniverseIdentity:
    return prototype_universe_identity(prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=source_scope)


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


def _build_provider_score_vector(
    *,
    provider_id: str,
    provider_version: str,
    policy_artifact_id: str,
    policy_row_ids: Sequence[str],
    rows: Sequence[tuple[str, float]],
) -> ProviderScoreVector:
    evidence = build_complete_row_evidence(
        row_scores=rows,
        policy_artifact_id=policy_artifact_id,
        provider_id=provider_id,
        provider_version=provider_version,
        policy_row_ids=policy_row_ids,
    )
    row_scores = evidence.row_scores
    return ProviderScoreVector(
        provider_id=provider_id,
        provider_version=provider_version,
        row_ids=tuple(item.row_id for item in row_scores),
        raw_scores=tuple(float(item.raw_score) for item in row_scores),
        quantized_scores=tuple(int(item.quantized_score) for item in row_scores),
        evidence=evidence,
    )


def _p2_cache_key(
    *,
    prototype_identity: PrototypeUniverseIdentity,
    policy_artifact_id: str,
    source_scope: str,
    region_digest: str,
    registration_config_digest: str,
    scoring_config_digest: str,
) -> tuple[Any, ...]:
    return (PROSPECTIVE_P2_VERSION, policy_artifact_id, source_scope, prototype_identity.digest, region_digest, registration_config_digest, scoring_config_digest)


def _p3_cache_key(
    *,
    prototype_identity: PrototypeUniverseIdentity,
    policy_artifact_id: str,
    source_scope: str,
    development_digest: str,
    region_digest: str,
    candidate_mask_digest_value: str,
    pairwise_mask_digest_value: str,
    calibration_digest: str,
) -> tuple[Any, ...]:
    return (
        PROSPECTIVE_P3_VERSION,
        policy_artifact_id,
        source_scope,
        prototype_identity.digest,
        development_digest,
        region_digest,
        candidate_mask_digest_value,
        pairwise_mask_digest_value,
        calibration_digest,
    )


def score_all_rows_reference(
    *,
    provider_id: str,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    source_scope: str,
) -> ProviderScoreVector:
    policy_row_ids = _policy_row_ids(prototypes)
    if provider_id == "P1":
        rows = []
        for row_id, _action_id, _digest, proto in prototypes.values():
            diff = np.abs(observation.pixels.astype(np.float32) - proto.pixels.astype(np.float32))
            mae = float(np.sum(diff) / (255.0 * diff.size))
            similarity = max(0.0, min(1.0, 1.0 - mae))
            rows.append((row_id, similarity))
        return _build_provider_score_vector(
            provider_id="P1",
            provider_version=PROSPECTIVE_P1_VERSION,
            policy_artifact_id=policy_artifact_id,
            policy_row_ids=policy_row_ids,
            rows=rows,
        )
    if provider_id == "P2":
        prototype_id = _prototype_identity(prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=source_scope)
        regions = _canonical_regions()
        region_digest = local_region_digest(regions)
        registration_config_digest = sha256_digest([region.registration_config.to_dict() for region in regions])
        scoring_config_digest = sha256_digest(
            {
                "winner_threshold": 1.0,
                "runner_up_margin": 0.0,
                "conflicting_action_margin": 0.0,
                "minimum_visible_fraction": 0.5,
            }
        )
        cache_key = _p2_cache_key(
            prototype_identity=prototype_id,
            policy_artifact_id=policy_artifact_id,
            source_scope=source_scope,
            region_digest=region_digest,
            registration_config_digest=registration_config_digest,
            scoring_config_digest=scoring_config_digest,
        )
        provider = _P2_PROVIDER_CACHE.get(cache_key)
        if provider is None:
            calibration = LocalCorrelationCalibration(
                winner_threshold=1.0,
                runner_up_margin=0.0,
                conflicting_action_margin=0.0,
                minimum_visible_fraction=0.5,
                region_spec_digest=region_digest,
                prototype_digest=prototype_id.digest,
                benign_calibration_digest=UnresolvedArtifactIdentity("label:prospective-calibration", "prospective calibration evidence not yet materialized").label,
                rejection_calibration_digest=UnresolvedArtifactIdentity("label:prospective-selection", "prospective selection evidence not yet materialized").label,
                policy_artifact_id=policy_artifact_id,
                source_scope=source_scope,
            )
            provider = _P2_PROVIDER_CACHE.put(
                cache_key,
                LocalCorrelationVideoAddressProvider(
                    prototypes={observation_id: (row_id, action_id, digest, proto) for observation_id, (row_id, action_id, digest, proto) in prototypes.items()},
                    calibration=calibration,
                    regions=regions,
                ),
            )
        ranked = provider._rank(observation)
        return _build_provider_score_vector(
            provider_id="P2",
            provider_version=PROSPECTIVE_P2_VERSION,
            policy_artifact_id=policy_artifact_id,
            policy_row_ids=policy_row_ids,
            rows=[(candidate.row_id, _bounded_similarity_from_distance(candidate.total_distance)) for candidate in ranked],
        )
    if provider_id == "P3":
        prototype_id = _prototype_identity(prototypes=prototypes, policy_artifact_id=policy_artifact_id, source_scope=source_scope)
        joint_prototypes = {row_id: (row_id, action_id, digest, proto) for _obs_id, (row_id, action_id, digest, proto) in prototypes.items()}
        development = {row_id: (proto, proto) for row_id, (_row, _action, _digest, proto) in joint_prototypes.items()}
        development_digest = sha256_digest({row_id: [left.raw_digest, right.raw_digest] for row_id, (left, right) in sorted(development.items())})
        regions = _joint_regions()
        region_digest = joint_region_digest(regions)
        candidate_masks = build_joint_candidate_masks(
            prototypes=joint_prototypes,
            development_observations=development,
            intensity_tolerance=8,
            stability_tolerance=12,
            amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
            operational_contract_digest=UnresolvedArtifactIdentity("label:prospective-b3-wrapper", "prospective B3 wrapper contract not yet closed").label,
            source_scope=source_scope,
        )
        pairwise_masks = build_pairwise_discriminative_masks(
            prototypes=joint_prototypes,
            candidate_masks=candidate_masks,
            intensity_tolerance=8,
            amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
            operational_contract_digest=UnresolvedArtifactIdentity("label:prospective-b3-wrapper", "prospective B3 wrapper contract not yet closed").label,
            source_scope=source_scope,
        )
        candidate_mask_digest_value = joint_candidate_mask_digest([mask.spec for mask in candidate_masks.values()])
        pairwise_mask_digest_value = pairwise_mask_digest([mask.spec for mask in pairwise_masks.values()])
        calibration = _joint_calibration(
            policy_artifact_id=policy_artifact_id,
            source_scope=source_scope,
            prototype_digest=prototype_id.digest,
            region_digest=region_digest,
            candidate_mask_digest_value=candidate_mask_digest_value,
            pairwise_mask_digest_value=pairwise_mask_digest_value,
        )
        cache_key = _p3_cache_key(
            prototype_identity=prototype_id,
            policy_artifact_id=policy_artifact_id,
            source_scope=source_scope,
            development_digest=development_digest,
            region_digest=region_digest,
            candidate_mask_digest_value=candidate_mask_digest_value,
            pairwise_mask_digest_value=pairwise_mask_digest_value,
            calibration_digest=calibration.digest,
        )
        state = _P3_STATE_CACHE.get(cache_key)
        if state is None:
            provider = JointEvidenceProvider(
                prototypes=joint_prototypes,
                candidate_masks=candidate_masks,
                pairwise_masks=pairwise_masks,
                regions=regions,
                calibration=calibration,
                policy_artifact_id=policy_artifact_id,
                source_scope=source_scope,
            )
            state = _P3_STATE_CACHE.put(
                cache_key,
                {
                    "joint_prototypes": joint_prototypes,
                    "regions": regions,
                    "candidate_masks": candidate_masks,
                    "pairwise_masks": pairwise_masks,
                    "provider_contract_digest": provider.contract().digest,
                    "row_action": _row_action_map(prototypes),
                },
            )
        ranked = build_joint_row_candidates(
            observation=observation,
            prototypes=state["joint_prototypes"],
            candidate_masks=state["candidate_masks"],
            pairwise_masks=state["pairwise_masks"],
            regions=state["regions"],
            architecture_id="B3",
        )
        return _build_provider_score_vector(
            provider_id="P3",
            provider_version=PROSPECTIVE_P3_VERSION,
            policy_artifact_id=policy_artifact_id,
            policy_row_ids=policy_row_ids,
            rows=[(candidate.row_id, float(candidate.candidate_strength)) for candidate in ranked],
        )
    raise VPMValidationError("unsupported provider_id")


def score_all_rows_optimized(
    *,
    provider_id: str,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    source_scope: str,
) -> ProviderScoreVector:
    return score_all_rows_reference(
        provider_id=provider_id,
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=policy_artifact_id,
        source_scope=source_scope,
    )


def score_normalized_pixel(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
) -> ProspectiveProviderResult:
    row_action = _row_action_map(prototypes)
    vector = score_all_rows_optimized(
        provider_id="P1",
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=policy_artifact_id,
        source_scope="",
    )
    evidence = vector.evidence
    semantic = build_semantic_top_set_outcome(evidence=evidence, row_action=row_action)
    return ProspectiveProviderResult(
        provider_id="P1",
        provider_version=PROSPECTIVE_P1_VERSION,
        evidence=evidence,
        winner_row_id=semantic.resolved_row_id,
        winner_action_id=semantic.resolved_action_id,
        maximum_tie_size=max(len(group.row_ids) for group in evidence.ranking.tie_groups),
        semantic_top_set_outcome=semantic,
        diagnostics={"score_count": len(vector.row_ids)},
    )


def score_registered_local_correlation(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    source_scope: str,
) -> ProspectiveProviderResult:
    vector = score_all_rows_reference(
        provider_id="P2",
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=policy_artifact_id,
        source_scope=source_scope,
    )
    evidence = vector.evidence
    row_action = _row_action_map(prototypes)
    semantic = build_semantic_top_set_outcome(evidence=evidence, row_action=row_action)
    return ProspectiveProviderResult(
        provider_id="P2",
        provider_version=PROSPECTIVE_P2_VERSION,
        evidence=evidence,
        winner_row_id=semantic.resolved_row_id,
        winner_action_id=semantic.resolved_action_id,
        maximum_tie_size=max(len(group.row_ids) for group in evidence.ranking.tie_groups),
        semantic_top_set_outcome=semantic,
        diagnostics={"candidate_count": len(vector.row_ids)},
    )


def _joint_regions() -> tuple[JointEvidenceRegionSpec, ...]:
    registration = RegistrationConfig(max_dx=2, max_dy=2, minimum_overlap_fraction=0.5)
    return (
        JointEvidenceRegionSpec("target_band", 0, 0, 6, 28, 2.0, True, registration),
        JointEvidenceRegionSpec("cooldown_indicator", 7, 25, 2, 2, 1.5, True, registration),
        JointEvidenceRegionSpec("tank_band", 10, 0, 4, 28, 2.0, True, registration),
    )


def _joint_calibration(
    *,
    policy_artifact_id: str,
    source_scope: str,
    prototype_digest: str,
    region_digest: str,
    candidate_mask_digest_value: str,
    pairwise_mask_digest_value: str,
) -> JointEvidenceCalibration:
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
        prototype_digest=prototype_digest,
        region_spec_digest=region_digest,
        candidate_mask_digest=candidate_mask_digest_value,
        pairwise_mask_digest=pairwise_mask_digest_value,
        policy_artifact_id=policy_artifact_id,
        source_scope=source_scope,
        amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
        operational_contract_digest=UnresolvedArtifactIdentity("label:prospective-b3-wrapper", "prospective B3 wrapper contract not yet closed").label,
    )


def score_b3_joint_fit(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, tuple[str, str, str, ImageObservation]],
    policy_artifact_id: str,
    source_scope: str,
) -> ProspectiveProviderResult:
    vector = score_all_rows_reference(
        provider_id="P3",
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=policy_artifact_id,
        source_scope=source_scope,
    )
    evidence = vector.evidence
    row_action = _row_action_map(prototypes)
    semantic = build_semantic_top_set_outcome(evidence=evidence, row_action=row_action)
    return ProspectiveProviderResult(
        provider_id="P3",
        provider_version=PROSPECTIVE_P3_VERSION,
        evidence=evidence,
        winner_row_id=semantic.resolved_row_id,
        winner_action_id=semantic.resolved_action_id,
        maximum_tie_size=max(len(group.row_ids) for group in evidence.ranking.tie_groups),
        semantic_top_set_outcome=semantic,
        diagnostics={"candidate_count": len(vector.row_ids)},
    )


__all__ = [
    "PROSPECTIVE_P1_VERSION",
    "PROSPECTIVE_P2_VERSION",
    "PROSPECTIVE_P3_VERSION",
    "PROSPECTIVE_PROVIDER_IDS",
    "PROSPECTIVE_PROVIDER_VERSIONS",
    "ProviderScoreVector",
    "ProspectiveProviderResult",
    "clear_prospective_provider_caches",
    "prospective_provider_cache_info",
    "score_all_rows_optimized",
    "score_all_rows_reference",
    "score_b3_joint_fit",
    "score_normalized_pixel",
    "score_registered_local_correlation",
]
