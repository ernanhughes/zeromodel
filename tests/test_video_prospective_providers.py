from __future__ import annotations

from zeromodel.video_action_set_benchmark import SOURCE_SCOPE, canonical_prototypes
from zeromodel.video_prospective_providers import (
    score_b3_joint_fit,
    score_normalized_pixel,
    score_registered_local_correlation,
)


def test_prospective_providers_emit_complete_112_row_evidence() -> None:
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
        assert len(result.evidence.row_scores) == 112
        assert len(result.evidence.ranking.ranked_row_ids) == 112
        assert result.winner_row_id in {item.row_id for item in result.evidence.row_scores}
    assert p3.winner_row_id == row_id
    assert p3.winner_action_id == action_id

