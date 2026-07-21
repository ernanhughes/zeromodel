from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.domains.video_action_set import provider_measurement as measurement


pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]

PROVIDER_ROW_KEYS = [
    "benchmark_version",
    "generator_version",
    "split",
    "episode_id",
    "clip_id",
    "frame_id",
    "sequence_number",
    "event_type",
    "family",
    "expected_disposition",
    "episode_family",
    "episode_disposition",
    "frame_disposition",
    "denominator_class",
    "expected_row",
    "expected_action",
    "actual_executed_action",
    "action_known",
    "gap_declaration",
    "observation_pixel_digest",
    "metadata",
    "provider_id",
    "provider_version",
    "policy_artifact_id",
    "reachability_tile_digest",
    "all_112_row_ids",
    "all_112_raw_scores",
    "all_112_quantized_scores",
    "complete_ordered_ranking",
    "tie_groups",
    "semantic_top_set_outcome",
    "semantic_status",
    "resolved_row",
    "resolved_action",
    "top_quantized_score",
    "top_row_ids",
    "top_action_ids",
    "semantic_outcome_digest",
    "reachability_composition_trace",
    "winner_row",
    "winner_action",
    "winner_quantized_score",
    "runner_up_row",
    "runner_up_quantized_score",
    "policy_row_universe_digest",
    "quantized_score_vector_digest",
    "raw_score_diagnostic_digest",
    "score_vector_digest",
    "ranking_digest",
    "observation_digest",
    "provider_observation_descriptor",
    "provider_observation_digest",
    "episode_seed",
    "generator_identity",
    "provider_diagnostics",
]


def _digest(value: Any) -> str:
    return benchmark._sha256(value)


def _real_measurement_context() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    plans = benchmark._episode_plans_for_split(
        identity,
        "selection",
        row_ids,
        row_actions,
    )
    valid_plan = next(
        plan for plan in plans if str(plan["family_label"]) == "valid"
    )
    record = deepcopy(benchmark._materialize_plan(valid_plan, identity, tile)[0])
    rows = measurement.score_record(
        record,
        benchmark.canonical_prototypes(),
        policy.artifact_id,
    )
    return record, rows


def test_real_score_record_p1_p2_p3_evidence_row_goldens() -> None:
    record, rows = _real_measurement_context()

    assert [row["provider_id"] for row in rows] == ["P1", "P2", "P3"]
    assert [list(row) for row in rows] == [PROVIDER_ROW_KEYS] * 3
    assert [row["reachability_composition_trace"] for row in rows] == [None] * 3
    assert [row["provider_observation_digest"] for row in rows] == [
        "sha256:bb6de887ea70ccd7c410547b46967c331b19186365a4d0eac3accf7240926e7b"
    ] * 3
    assert [row["score_vector_digest"] for row in rows] == [
        "sha256:408790b154e367f9219d6c0642a6f1898fdac81a4debfd393e6776e275345068",
        "sha256:92ceb62f8c2f34ecae3201f4c5f50f8808f7cd46052c2afb32460ca1cb1306ec",
        "sha256:6661e1d699dc3ad69f6f7a762ed69d8337968c64657b4217d5484fbf298e86cd",
    ]
    assert [row["ranking_digest"] for row in rows] == [
        "sha256:5b3de33e450fe82f2c1f84c33e01ad0385574a6316f570f3c9b408c4b84b289e",
        "sha256:acea571945f2cc3b9380a9c4e69c1a6cc24a3b3fd401be5a81b13a4f2bf63cdd",
        "sha256:727c629453acc104b613247e981133edad6791a75d91e21f26f215cf015b2325",
    ]
    assert [row["semantic_outcome_digest"] for row in rows] == [
        "sha256:07c17a5055aac46a737fb2ae70ca9999986b7890b4e9f490e85cf3533998fb98",
        "sha256:58b3cee4f9a7fa91fe472e327a0f69c5345ef3c017d5cb26ba860f55f4a926c0",
        "sha256:3025f3b2ab75402d7dc71b76221354094e4733886949b8c09c5e40db1411ddb4",
    ]
    assert [_digest(row) for row in rows] == [
        "sha256:398166f89a64f92dce703a4508a48b709bd170a165658c16b4892b90fa48bd1f",
        "sha256:e2e803531537df79d052d86faae6560954a92c730240266ff740ee0e6f676d35",
        "sha256:43ee89557951e23b2fc30d15d2664e8e660c5dbbdc4879bdf70e4c37b23af90b",
    ]
    assert rows[1]["semantic_status"] == "action_unanimous_tie"
    assert rows[1]["resolved_action"] == "STAY"
    assert rows[1]["resolved_row"] is None
    assert rows[1]["winner_quantized_score"] is None
    assert rows[0]["metadata"] is record["metadata"]
