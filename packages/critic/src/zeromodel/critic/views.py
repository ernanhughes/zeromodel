from __future__ import annotations

from zeromodel.core import LayoutRecipe, ScoreTable, VPMArtifact, build_vpm

from zeromodel.critic.dto import CriticScoreResultDTO


def build_critic_score_vpm(*, result: CriticScoreResultDTO) -> VPMArtifact:
    metric_ids = (
        "critic_score",
        "decision_margin",
        "feature_coverage",
        "positive_contribution_strength",
        "negative_contribution_strength",
    )
    values = []
    row_ids = []
    for item in result.items:
        row_ids.append(item.artifact_ref.artifact_id)
        values.append(
            [
                item.score,
                item.decision_margin,
                item.feature_coverage,
                item.positive_contribution_strength,
                item.negative_contribution_strength,
            ]
        )
    if not values:
        row_ids = ["empty-result"]
        values = [[0.0 for _ in metric_ids]]
    table = ScoreTable(
        values=values,
        row_ids=row_ids,
        metric_ids=metric_ids,
        metadata={"kind": "critic_score_result", "result_id": result.result_id},
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "critic-score-desc",
            "row_order": {
                "kind": "lexicographic",
                "keys": [{"metric_id": "critic_score", "direction": "desc"}],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(
        table,
        recipe,
        provenance={
            "kind": "critic_score_vpm",
            "result_id": result.result_id,
            "request_ref": result.request_ref.artifact_id,
            "readout_ref": result.readout_ref.artifact_id,
            "feature_batch_ref": result.feature_batch_ref.artifact_id,
            "parents": [result.result_id],
        },
    )
