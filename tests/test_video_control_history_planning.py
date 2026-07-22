from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set import control_histories
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]

CONTROL_SOURCE_ROWS = [
    "tank=0|target=none|cooldown=0",
    "tank=0|target=0|cooldown=0",
    "tank=0|target=1|cooldown=0",
]
FIRST_CANDIDATE = {
    "actual_executed_action": "LEFT",
    "normalized_causal_tuple_digest": "sha256:c03a8f6e753cdecb1e9f94c8c50db5131726d08a0f6d54616f90e42fc2a1734e",
    "predecessor_row_id": "tank=0|target=none|cooldown=0",
    "resulting_row_id": "tank=0|target=none|cooldown=0",
    "transition_choice_index": 0,
}
SELECTED_HISTORY_IDS = [
    "sha256:c03a8f6e753cdecb1e9f94c8c50db5131726d08a0f6d54616f90e42fc2a1734e",
    "sha256:e35c5faeab0bfff99af9b629ff473149507b7f73c6137b41bc479379cac877a3",
    "sha256:0bbfa71389d9c6b97646c62edb0effce87f2229ce5cbde902cdaa6b685cd4bb0",
    "sha256:405d893d7e9456f025c932ffb468148babbc95b34d6d16b1680bb88514a6f28b",
]


def _row_ids_actions() -> tuple[list[str], dict[str, str]]:
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    return row_ids, {row_id: lookup.choose(row_id) for row_id in row_ids}


def test_transition_identity_and_causal_tuple_goldens() -> None:
    row_ids, row_actions = _row_ids_actions()
    source_row_id = row_ids[0]
    action = row_actions[source_row_id]
    causal_tuple = control_histories.normalized_control_causal_tuple(
        predecessor_row_id=source_row_id,
        actual_executed_action=FIRST_CANDIDATE["actual_executed_action"],
        transition_choice_index=0,
        resulting_row_id=source_row_id,
    )

    assert control_histories.transition_identity()["transition_identity_digest"] == (
        "sha256:6e96c50329265b21ecb96db7fc6e5b020afaa8487cb4d93c4ab556be40063403"
    )
    assert control_histories.transition_input_digest(source_row_id, action) == (
        "sha256:4dc004f18d5b5675b04dd4714693b28fbbbefc35eee02bf842d28b41a4cfd668"
    )
    assert control_histories.transition_result_digest(source_row_id, action, 0, source_row_id) == (
        "sha256:0f1816adedc21735c89854d8e91f922de335892b7fe97def1941f59c2c2a3bca"
    )
    assert canonical_sha256(causal_tuple) == FIRST_CANDIDATE["normalized_causal_tuple_digest"]


def test_control_history_enumeration_selection_and_reconstruction_are_frozen() -> None:
    row_ids, _row_actions = _row_ids_actions()

    assert control_histories.control_source_rows(row_ids, count=3, frame_count=4) == CONTROL_SOURCE_ROWS
    candidates = control_histories.grounded_control_histories_for_current_row(CONTROL_SOURCE_ROWS[0], row_ids)
    assert len(candidates) == 6
    assert candidates[0] == FIRST_CANDIDATE
    assert canonical_sha256(candidates) == "sha256:fe5bd93b70a20d87c12f0e543acdc56f0b2372e3ebdf68ae2d06df6cbe30e8fe"

    selected = control_histories.select_grounded_control_histories(
        CONTROL_SOURCE_ROWS[0],
        row_ids,
        control_group_id="stage5-control-group",
        frame_count=4,
    )
    assert [history["history_id"] for history in selected] == SELECTED_HISTORY_IDS
    assert canonical_sha256(selected) == "sha256:77e821ed82496125338322aa62a1c61321c54c68acb8d5b04a14ac18903183e8"
    assert control_histories.reconstructed_control_causal_tuple_digest(selected[0]) == SELECTED_HISTORY_IDS[0]

    tampered = deepcopy(selected[0])
    tampered["transition_result_digest"] = "sha256:" + "0" * 64
    with pytest.raises(VPMValidationError, match="result digest mismatch"):
        control_histories.reconstructed_control_causal_tuple_digest(tampered)


def test_benchmark_control_history_aliases_are_direct() -> None:
    assert benchmark._transition_identity is control_histories.transition_identity
    assert benchmark._transition_input_digest is control_histories.transition_input_digest
    assert benchmark._transition_result_digest is control_histories.transition_result_digest
    assert benchmark._normalized_control_causal_tuple is control_histories.normalized_control_causal_tuple
    assert benchmark._grounded_control_history is control_histories.grounded_control_history
    assert benchmark._reconstructed_control_causal_tuple_digest is (
        control_histories.reconstructed_control_causal_tuple_digest
    )
    assert benchmark._grounded_control_histories_for_current_row is (
        control_histories.grounded_control_histories_for_current_row
    )
    assert benchmark._select_grounded_control_histories is control_histories.select_grounded_control_histories
    assert benchmark._control_source_rows is control_histories.control_source_rows
