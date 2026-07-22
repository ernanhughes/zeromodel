from __future__ import annotations

from types import MappingProxyType
from typing import Any

import numpy as np
import pytest

import research.evidence.video_discriminative_joint_evidence as joint
import research.video.video_prospective_providers as providers
from research.evidence.video_discriminative_joint_evidence import (
    JointEvidenceRegionSpec,
    build_joint_candidate_masks,
    build_pairwise_discriminative_masks,
)
from zeromodel.observation.visual_address import ImageObservation
from research.evidence.video_complete_row_evidence import build_semantic_top_set_outcome
from research.evidence.video_complete_row_evidence import build_complete_row_evidence
from research.video.video_prospective_providers import (
    PROSPECTIVE_P3_VERSION,
    clear_prospective_provider_caches,
    score_all_rows_optimized,
    score_all_rows_reference,
)
from research.visual.visual_registration import RegistrationConfig


POLICY_ARTIFACT_ID = "sha256:policy"
SOURCE_SCOPE = "cache-kernel"
ACTIONS = ("LEFT", "RIGHT", "STAY", "FIRE")
ARCHITECTURES = ("A3", "B3", "C3", "D3")


@pytest.fixture(autouse=True)
def reset_candidate_caches():
    joint._reset_base_candidate_cache()
    clear_prospective_provider_caches()
    yield
    joint._reset_base_candidate_cache()
    clear_prospective_provider_caches()


def _tiny_observation(index: int, *, source_id: str) -> ImageObservation:
    pixels = np.array(
        [
            [index % 256, (index * 3 + 7) % 256],
            [(index * 5 + 11) % 256, (index * 7 + 13) % 256],
        ],
        dtype=np.uint8,
    )
    return ImageObservation(pixels, source_id=source_id)


def _region() -> tuple[JointEvidenceRegionSpec, ...]:
    return (
        JointEvidenceRegionSpec(
            region_id="full",
            top=0,
            left=0,
            height=2,
            width=2,
            weight=1.0,
            critical=True,
            registration_config=RegistrationConfig(
                max_dx=0,
                max_dy=0,
                minimum_overlap_fraction=1.0,
            ),
        ),
    )


def _joint_inputs(
    row_count: int = 3,
) -> tuple[
    dict[str, tuple[str, str, str, ImageObservation]],
    dict[str, tuple[ImageObservation, ImageObservation]],
    tuple[JointEvidenceRegionSpec, ...],
]:
    prototypes: dict[str, tuple[str, str, str, ImageObservation]] = {}
    development: dict[str, tuple[ImageObservation, ImageObservation]] = {}
    for index in range(row_count):
        row_id = f"row-{index:03d}"
        observation = _tiny_observation(index, source_id=f"prototype-{row_id}")
        prototypes[row_id] = (
            row_id,
            ACTIONS[index % len(ACTIONS)],
            observation.raw_digest,
            observation,
        )
        development[row_id] = (observation, observation)
    return prototypes, development, _region()


def _masks(
    prototypes: dict[str, tuple[str, str, str, ImageObservation]],
    development: dict[str, tuple[ImageObservation, ImageObservation]],
):
    candidate_masks = build_joint_candidate_masks(
        prototypes=prototypes,
        development_observations=development,
        intensity_tolerance=8,
        stability_tolerance=12,
        amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
        operational_contract_digest="sha256:test",
        source_scope=SOURCE_SCOPE,
    )
    pairwise_masks = build_pairwise_discriminative_masks(
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        intensity_tolerance=8,
        amendment_commit_sha="ad2093590cde95ad1dc984f0573f452693002717",
        operational_contract_digest="sha256:test",
        source_scope=SOURCE_SCOPE,
    )
    return candidate_masks, pairwise_masks


def _candidate_snapshot(
    observation_index: int,
    *,
    row_count: int = 3,
    architecture_id: str = "B3",
) -> bytes:
    prototypes, development, regions = _joint_inputs(row_count)
    candidate_masks, pairwise_masks = _masks(prototypes, development)
    observation = _tiny_observation(
        observation_index, source_id=f"observation-{observation_index}"
    )
    candidates = joint.build_joint_row_candidates(
        observation=observation,
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=regions,
        architecture_id=architecture_id,
    )
    ranked = joint.rank_joint_row_candidates(candidates)
    return joint._json_bytes([candidate.to_dict() for candidate in ranked])


def _provider_fixture() -> tuple[
    dict[str, tuple[str, str, str, ImageObservation]],
    ImageObservation,
    dict[str, str],
]:
    prototypes, _development, _regions = _joint_inputs(112)
    observation = _tiny_observation(37, source_id="score-vector-observation")
    row_actions = {
        row_id: action_id for row_id, action_id, _digest, _obs in prototypes.values()
    }
    return prototypes, observation, row_actions


def _score_vector_snapshot(vector: Any, row_actions: dict[str, str]) -> dict[str, Any]:
    semantic = build_semantic_top_set_outcome(
        evidence=vector.evidence,
        row_action=row_actions,
    )
    return {
        "raw_score_vectors": list(vector.raw_scores),
        "quantized_score_vectors": list(vector.quantized_scores),
        "rankings": list(vector.evidence.ranking.ranked_row_ids),
        "tie_groups": [group.to_dict() for group in vector.evidence.ranking.tie_groups],
        "winners": {
            "row": semantic.resolved_row_id,
            "action": semantic.resolved_action_id,
        },
        "semantic_outcomes": semantic.to_dict(),
        "evidence_digests": {
            "score_vector": vector.evidence.score_vector_digest,
            "quantized": vector.evidence.quantized_score_vector_digest,
            "raw": vector.evidence.raw_score_diagnostic_digest,
            "ranking": vector.evidence.ranking.ranking_digest,
        },
    }


def _base_snapshot(base: Any) -> dict[str, Any]:
    return {
        key: (
            [item.to_dict() for item in value]
            if key in {"region_evidence", "pairwise_evidence"}
            else value
        )
        for key, value in base.items()
    }


def _architecture_snapshot(
    *,
    architecture_id: str,
    prototypes: dict[str, tuple[str, str, str, ImageObservation]],
    candidate_masks: Any,
    pairwise_masks: Any,
    regions: tuple[JointEvidenceRegionSpec, ...],
    observation: ImageObservation,
    row_actions: dict[str, str],
) -> dict[str, Any]:
    candidates = joint.rank_joint_row_candidates(
        joint.build_joint_row_candidates(
            observation=observation,
            prototypes=prototypes,
            candidate_masks=candidate_masks,
            pairwise_masks=pairwise_masks,
            regions=regions,
            architecture_id=architecture_id,
        )
    )
    evidence = build_complete_row_evidence(
        row_scores=[
            (candidate.row_id, float(candidate.candidate_strength))
            for candidate in candidates
        ],
        policy_artifact_id=POLICY_ARTIFACT_ID,
        provider_id="P3",
        provider_version=PROSPECTIVE_P3_VERSION,
        policy_row_ids=tuple(sorted(prototypes)),
    )
    semantic = build_semantic_top_set_outcome(
        evidence=evidence,
        row_action=row_actions,
    )
    return {
        "candidate_strength": {
            candidate.row_id: float(candidate.candidate_strength)
            for candidate in candidates
        },
        "complete_candidate_serialization": [
            candidate.to_dict() for candidate in candidates
        ],
        "raw_score_vector": [item.raw_score for item in evidence.row_scores],
        "quantized_score_vector": [
            item.quantized_score for item in evidence.row_scores
        ],
        "ranking": list(evidence.ranking.ranked_row_ids),
        "tie_groups": [group.to_dict() for group in evidence.ranking.tie_groups],
        "winner": {
            "row": semantic.resolved_row_id,
            "action": semantic.resolved_action_id,
        },
        "semantic_outcome": semantic.to_dict(),
        "evidence_digests": {
            "score_vector": evidence.score_vector_digest,
            "quantized": evidence.quantized_score_vector_digest,
            "raw": evidence.raw_score_diagnostic_digest,
            "ranking": evidence.ranking.ranking_digest,
            "semantic": semantic.semantic_outcome_digest,
        },
    }


def test_base_candidate_cache_starts_clean_for_each_test() -> None:
    assert joint._base_candidate_cache_info()["size"] == 0
    _candidate_snapshot(0)
    assert joint._base_candidate_cache_info()["size"] == 1


def test_base_candidate_cache_capacity_and_eviction_are_deterministic() -> None:
    capacity = joint._base_candidate_cache_info()["capacity"]
    inserted_keys = []

    for index in range(capacity + 3):
        _candidate_snapshot(index)
        info = joint._base_candidate_cache_info()
        assert info["size"] <= capacity
        inserted_keys.append(info["keys"][-1])

    info = joint._base_candidate_cache_info()
    assert info["size"] == capacity
    assert info["keys"] == inserted_keys[-capacity:]


def test_base_candidate_cache_hits_move_to_mru_and_info_is_snapshot() -> None:
    capacity = joint._base_candidate_cache_info()["capacity"]
    for index in range(capacity):
        _candidate_snapshot(index)

    original_info = joint._base_candidate_cache_info()
    original_oldest = original_info["keys"][0]
    original_second = original_info["keys"][1]
    original_info["keys"].clear()

    assert joint._base_candidate_cache_info()["size"] == capacity

    _candidate_snapshot(0)
    after_hit = joint._base_candidate_cache_info()

    assert after_hit["hits"] == 1
    assert after_hit["keys"][-1] == original_oldest

    _candidate_snapshot(capacity)
    after_insert = joint._base_candidate_cache_info()

    assert after_insert["size"] == capacity
    assert original_oldest in after_insert["keys"]
    assert original_second not in after_insert["keys"]


def test_base_candidate_cache_identity_includes_region_specs() -> None:
    prototypes, development, regions = _joint_inputs()
    candidate_masks, pairwise_masks = _masks(prototypes, development)
    observation = _tiny_observation(0, source_id="observation-0")

    joint.build_joint_row_candidates(
        observation=observation,
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=regions,
        architecture_id="B3",
    )
    alternate_regions = (
        JointEvidenceRegionSpec(
            region_id="full",
            top=0,
            left=0,
            height=2,
            width=2,
            weight=2.0,
            critical=True,
            registration_config=RegistrationConfig(
                max_dx=0,
                max_dy=0,
                minimum_overlap_fraction=1.0,
            ),
        ),
    )

    joint.build_joint_row_candidates(
        observation=observation,
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=alternate_regions,
        architecture_id="B3",
    )
    info = joint._base_candidate_cache_info()

    assert info["size"] == 2
    assert info["misses"] == 2


def test_base_candidate_cache_identity_includes_prototype_metadata() -> None:
    prototypes, development, regions = _joint_inputs()
    candidate_masks, pairwise_masks = _masks(prototypes, development)
    observation = _tiny_observation(0, source_id="observation-0")

    first = joint.build_joint_row_candidates(
        observation=observation,
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=regions,
        architecture_id="B3",
    )
    renamed_prototypes = {
        row_id: (f"renamed-{prototype_observation_id}", action_id, digest, prototype)
        for row_id, (
            prototype_observation_id,
            action_id,
            digest,
            prototype,
        ) in prototypes.items()
    }
    second = joint.build_joint_row_candidates(
        observation=observation,
        prototypes=renamed_prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=regions,
        architecture_id="B3",
    )

    info = joint._base_candidate_cache_info()
    assert info["size"] == 2
    assert info["misses"] == 2
    assert {candidate.prototype_observation_id for candidate in first} == set(
        prototypes
    )
    assert {
        candidate.prototype_observation_id for candidate in second
    } == {f"renamed-{row_id}" for row_id in prototypes}


def test_base_candidate_cache_values_are_immutable_and_materialization_is_pure() -> None:
    prototypes, development, regions = _joint_inputs()
    candidate_masks, pairwise_masks = _masks(prototypes, development)
    observation = _tiny_observation(0, source_id="observation-0")
    candidates = joint.build_joint_row_candidates(
        observation=observation,
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=regions,
        architecture_id="B3",
    )
    key = joint._base_candidate_cache_info()["keys"][0]
    cached = joint._BASE_CANDIDATE_CACHE.get(key)

    assert cached is not None
    assert isinstance(cached[0], MappingProxyType)
    with pytest.raises(TypeError):
        cached[0]["row_id"] = "mutated"

    before = tuple(_base_snapshot(base) for base in cached)
    materialized = joint._materialize_candidate(cached[0], "B3")

    assert tuple(_base_snapshot(base) for base in cached) == before
    assert materialized.to_dict() == candidates[0].to_dict()


def test_rescoring_after_eviction_is_byte_identical() -> None:
    first = _candidate_snapshot(0)
    first_key = joint._base_candidate_cache_info()["keys"][0]
    capacity = joint._base_candidate_cache_info()["capacity"]

    for index in range(1, capacity + 2):
        _candidate_snapshot(index)

    info = joint._base_candidate_cache_info()
    assert info["size"] == capacity
    assert first_key not in info["keys"]
    after_eviction = _candidate_snapshot(0)

    assert after_eviction == first


def test_cached_uncached_and_reference_optimized_provider_vectors_are_identical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(providers, "_joint_regions", _region)
    prototypes, observation, row_actions = _provider_fixture()

    uncached_vector = score_all_rows_reference(
        provider_id="P3",
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=POLICY_ARTIFACT_ID,
        source_scope=SOURCE_SCOPE,
    )
    uncached = _score_vector_snapshot(uncached_vector, row_actions)

    cached_vector = score_all_rows_reference(
        provider_id="P3",
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=POLICY_ARTIFACT_ID,
        source_scope=SOURCE_SCOPE,
    )
    cached = _score_vector_snapshot(cached_vector, row_actions)

    assert cached == uncached
    optimized = score_all_rows_optimized(
        provider_id="P3",
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=POLICY_ARTIFACT_ID,
        source_scope=SOURCE_SCOPE,
    )

    assert _score_vector_snapshot(optimized, row_actions) == uncached
    assert joint._base_candidate_cache_info()["hits"] >= 2
    assert optimized.provider_id == "P3"
    assert optimized.provider_version == PROSPECTIVE_P3_VERSION


@pytest.mark.parametrize("architecture_id", ARCHITECTURES)
def test_architecture_candidate_and_score_parity_across_cache_states(
    architecture_id: str,
) -> None:
    prototypes, development, regions = _joint_inputs(112)
    candidate_masks, pairwise_masks = _masks(prototypes, development)
    observation = _tiny_observation(37, source_id="architecture-parity-observation")
    row_actions = {
        row_id: action_id for row_id, action_id, _digest, _obs in prototypes.values()
    }

    uncached = _architecture_snapshot(
        architecture_id=architecture_id,
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=regions,
        observation=observation,
        row_actions=row_actions,
    )
    cached = _architecture_snapshot(
        architecture_id=architecture_id,
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=regions,
        observation=observation,
        row_actions=row_actions,
    )
    original_key = joint._base_candidate_cache_info()["keys"][-1]
    assert cached == uncached
    assert joint._base_candidate_cache_info()["hits"] >= 1

    capacity = joint._base_candidate_cache_info()["capacity"]
    for index in range(capacity + 1):
        _candidate_snapshot(index, row_count=3, architecture_id="B3")

    assert original_key not in joint._base_candidate_cache_info()["keys"]
    rescored = _architecture_snapshot(
        architecture_id=architecture_id,
        prototypes=prototypes,
        candidate_masks=candidate_masks,
        pairwise_masks=pairwise_masks,
        regions=regions,
        observation=observation,
        row_actions=row_actions,
    )

    assert rescored == uncached
