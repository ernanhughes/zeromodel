from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set import materialization_reachability as reach
from zeromodel.domains.video_action_set.canonical_json import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def _first_valid_plan():
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    plan = benchmark._episode_plans_for_split(
        identity, "selection", row_ids, row_actions
    )[0]
    return lookup, plan


def test_reachability_tile_identity_and_next_row_trace_are_frozen() -> None:
    lookup, plan = _first_valid_plan()
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    frame_plan = plan["frame_plans"][0]

    next_row, action, index, trace = reach.next_materialized_row(
        lookup,
        str(plan["source_row_id"]),
        choice_seed=int(frame_plan["transition_choice_seed"]),
        config=benchmark.ShooterConfig(),
        reachability_tile=tile,
    )

    assert reach.reachability_tile_digest(tile) == (
        "sha256:fef2bc5fd795bb92d3bd564bccdc2d32e1b23319aba55dffed5e0391e795a5df"
    )
    assert (next_row, action, index) == ("tank=0|target=none|cooldown=0", "STAY", 0)
    assert trace == {
        "reachability_tile_digest": "sha256:fef2bc5fd795bb92d3bd564bccdc2d32e1b23319aba55dffed5e0391e795a5df",
        "candidate_row_id": "tank=0|target=none|cooldown=0",
        "candidate_action_set": ["STAY"],
        "top_action_set": ["STAY"],
        "tile_action_id": "STAY",
        "reachable_row_ids": [
            "tank=0|target=none|cooldown=0",
            "tank=0|target=none|cooldown=1",
            "tank=1|target=none|cooldown=0",
        ],
        "chosen_reachable_index": 0,
        "executed_action": "STAY",
        "rejected": False,
    }
    assert canonical_sha256(trace) == (
        "sha256:155d6f93bd739a363cbcd5ebc271c7577ae170efcca9dedc026236229bf410d2"
    )


def test_reachability_rejects_foreign_tile_identity() -> None:
    lookup, plan = _first_valid_plan()
    tile = deepcopy(benchmark._load_reachability_tile(REPO_ROOT))
    tile["tile_digest"] = "sha256:" + "0" * 64

    with pytest.raises(VPMValidationError, match="foreign reachability tile digest"):
        reach.next_materialized_row(
            lookup,
            str(plan["source_row_id"]),
            choice_seed=int(plan["frame_plans"][0]["transition_choice_seed"]),
            config=benchmark.ShooterConfig(),
            reachability_tile=tile,
        )


def test_foreign_reachability_identity_precedes_missing_edge() -> None:
    lookup, plan = _first_valid_plan()
    tile = deepcopy(benchmark._load_reachability_tile(REPO_ROOT))
    tile["edges"] = []
    tile["tile_digest"] = "sha256:" + "0" * 64

    with pytest.raises(VPMValidationError, match="foreign reachability tile digest"):
        reach.next_materialized_row(
            lookup,
            str(plan["source_row_id"]),
            choice_seed=int(plan["frame_plans"][0]["transition_choice_seed"]),
            config=benchmark.ShooterConfig(),
            reachability_tile=tile,
        )


def test_missing_reachability_edge_is_rejected_after_identity_validation() -> None:
    lookup, plan = _first_valid_plan()
    tile = deepcopy(benchmark._load_reachability_tile(REPO_ROOT))
    row_id = str(plan["source_row_id"])
    action = lookup.choose(row_id)
    tile["edges"] = [
        edge
        for edge in tile["edges"]
        if not (edge["source_row_id"] == row_id and edge["action_id"] == action)
    ]
    tile["tile_digest"] = reach.reachability_tile_digest(tile)

    with pytest.raises(VPMValidationError, match="missing reachability edge"):
        reach.next_materialized_row(
            lookup,
            row_id,
            choice_seed=int(plan["frame_plans"][0]["transition_choice_seed"]),
            config=benchmark.ShooterConfig(),
            reachability_tile=tile,
        )


def test_unadmitted_authoritative_destination_is_rejected_with_valid_tile_digest() -> (
    None
):
    lookup, plan = _first_valid_plan()
    tile = deepcopy(benchmark._load_reachability_tile(REPO_ROOT))
    row_id = str(plan["source_row_id"])
    action = lookup.choose(row_id)
    frame_plan = plan["frame_plans"][0]
    tank, target, cooldown = benchmark.parse_state_row_id(row_id)
    rows = benchmark.next_rows(
        tank, target, cooldown, action, width=benchmark.ShooterConfig().width
    )
    selected = str(rows[int(frame_plan["transition_choice_seed"]) % len(rows)])
    edge = next(
        edge
        for edge in tile["edges"]
        if edge["source_row_id"] == row_id and edge["action_id"] == action
    )
    edge["reachable_row_ids"] = [
        reachable for reachable in edge["reachable_row_ids"] if reachable != selected
    ]
    tile["tile_digest"] = reach.reachability_tile_digest(tile)

    with pytest.raises(VPMValidationError, match="does not admit chosen transition"):
        reach.next_materialized_row(
            lookup,
            row_id,
            choice_seed=int(frame_plan["transition_choice_seed"]),
            config=benchmark.ShooterConfig(),
            reachability_tile=tile,
        )


def test_reachability_aliases_are_direct() -> None:
    assert benchmark._tile_edge is reach.tile_edge
    assert benchmark._reachability_tile_digest is reach.reachability_tile_digest
    assert (
        benchmark._validate_reachability_tile_identity
        is reach.validate_reachability_tile_identity
    )
    assert benchmark._next_row is reach.next_materialized_row
