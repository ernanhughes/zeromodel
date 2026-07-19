from __future__ import annotations

import pytest

from zeromodel.video_action_set_benchmark import SOURCE_SCOPE, canonical_prototypes
from zeromodel.video_complete_row_evidence import VIDEO_COMPLETE_ROW_EVIDENCE_VERSION
from zeromodel.video_prospective_providers import (
    clear_prospective_provider_caches,
    prospective_provider_cache_info,
    score_b3_joint_fit,
    score_normalized_pixel,
    score_registered_local_correlation,
)


def test_prospective_providers_emit_complete_112_row_evidence() -> None:
    clear_prospective_provider_caches()
    prototypes = canonical_prototypes()
    observation_id, (row_id, action_id, _digest, observation) = next(iter(prototypes.items()))
    policy_artifact_id = "policy-artifact"
    p1 = score_normalized_pixel(
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=policy_artifact_id,
    )
    p2 = score_registered_local_correlation(
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=policy_artifact_id,
        source_scope=SOURCE_SCOPE,
    )
    p3 = score_b3_joint_fit(
        observation=observation,
        prototypes=prototypes,
        policy_artifact_id=policy_artifact_id,
        source_scope=SOURCE_SCOPE,
    )
    for result in (p1, p2, p3):
        assert result.evidence.policy_artifact_id == policy_artifact_id
        assert result.evidence.version == VIDEO_COMPLETE_ROW_EVIDENCE_VERSION
        assert len(result.evidence.row_scores) == 112
        assert len(result.evidence.ranking.ranked_row_ids) == 112
        assert result.winner_row_id in {item.row_id for item in result.evidence.row_scores}
    assert p3.winner_row_id == row_id
    assert p3.winner_action_id == action_id


@pytest.mark.slow
def test_prospective_provider_caches_are_bounded_and_clearable() -> None:
    clear_prospective_provider_caches()
    prototypes = canonical_prototypes()
    _, (_row_id, _action_id, _digest, observation) = next(iter(prototypes.items()))
    for scope in [f"scope-{index}" for index in range(10)]:
        score_registered_local_correlation(
            observation=observation,
            prototypes=prototypes,
            policy_artifact_id="policy-artifact",
            source_scope=scope,
        )
        score_b3_joint_fit(
            observation=observation,
            prototypes=prototypes,
            policy_artifact_id="policy-artifact",
            source_scope=scope,
        )
    info = prospective_provider_cache_info()
    assert info["P2"]["capacity"] == 8
    assert info["P3"]["capacity"] == 8
    assert info["P2"]["size"] <= 8
    assert info["P3"]["size"] <= 8
    clear_prospective_provider_caches()
    cleared = prospective_provider_cache_info()
    assert cleared["P2"]["size"] == 0
    assert cleared["P3"]["size"] == 0

