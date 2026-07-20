from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set import reachability_composition as reach


REPO_ROOT = Path(__file__).resolve().parents[1]

TRACE_KEYS = [
    "version",
    "composition_version",
    "frame_id",
    "semantic_outcome_digest",
    "input_candidate_rows",
    "input_candidate_actions",
    "prior_reachable_rows",
    "prior_state_status",
    "reachability_tile_identity",
    "consulted_edges",
    "reachable_candidate_pairs",
    "removed_rows",
    "removed_actions",
    "retained_rows",
    "retained_actions",
    "resulting_candidate_set",
    "resolved_row_id",
    "resolved_action_id",
    "rejection_reason",
    "executed_action",
    "status",
    "trace_digest",
]


@pytest.fixture(scope="module")
def reachability_context() -> dict[str, Any]:
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    return {"row_ids": row_ids, "row_actions": row_actions, "tile": tile}


def _outcome(
    rows: list[str],
    row_actions: dict[str, str],
    *,
    status: str | None = None,
) -> dict[str, object]:
    status = status or ("unique_row" if len(rows) == 1 else "conflicting_action_tie")
    return {
        "semantic_outcome_digest": benchmark._sha256({"rows": rows, "status": status}),
        "status": status,
        "top_row_ids": rows,
        "top_row_actions": [
            {"row_id": row_id, "action_id": row_actions[row_id]} for row_id in rows
        ],
    }


def _reachable_fixture_cases(
    row_ids: list[str],
    row_actions: dict[str, str],
    tile: dict[str, Any],
) -> tuple[tuple[str, list[str]], tuple[str, list[str]]]:
    conflict_case = None
    same_action_case = None
    for prior in row_ids:
        reachable = list(
            benchmark._tile_edge(tile, prior, row_actions[prior])["reachable_row_ids"]
        )
        by_action: dict[str, list[str]] = {}
        for row in reachable:
            by_action.setdefault(row_actions[row], []).append(row)
        if same_action_case is None:
            same = next(
                (rows[:2] for rows in by_action.values() if len(rows) >= 2), None
            )
            if same:
                same_action_case = (prior, same)
        if conflict_case is None and len(by_action) >= 2:
            actions = list(by_action)
            conflict_case = (
                prior,
                [by_action[actions[0]][0], by_action[actions[1]][0]],
            )
        if conflict_case and same_action_case:
            break
    assert conflict_case is not None
    assert same_action_case is not None
    return conflict_case, same_action_case


def _trace_cases(context: dict[str, Any]) -> dict[str, dict[str, Any]]:
    row_ids = context["row_ids"]
    row_actions = context["row_actions"]
    tile = context["tile"]
    source = row_ids[0]
    edge = benchmark._tile_edge(tile, source, row_actions[source])
    ordinary_dest = edge["reachable_row_ids"][0]
    missing_dest = next(
        row for row in row_ids if row not in set(edge["reachable_row_ids"])
    )
    conflict_case, same_action_case = _reachable_fixture_cases(
        row_ids, row_actions, tile
    )

    return {
        "first_frame": reach.compose_reachability_trace(
            frame_id="frame-first",
            semantic_outcome=_outcome([ordinary_dest], row_actions),
            previous_state=None,
            reachability_tile=tile,
            row_actions=row_actions,
        ),
        "ordinary_reachable": reach.compose_reachability_trace(
            frame_id="frame-ordinary",
            semantic_outcome=_outcome([ordinary_dest], row_actions),
            previous_state={"status": "resolved", "candidate_rows": (source,)},
            reachability_tile=tile,
            row_actions=row_actions,
        ),
        "prior_unresolved": reach.compose_reachability_trace(
            frame_id="frame-prior-unresolved",
            semantic_outcome=_outcome([ordinary_dest], row_actions),
            previous_state={
                "status": "unresolved",
                "candidate_rows": tuple(),
                "reason": "typed_gap_event",
            },
            reachability_tile=tile,
            row_actions=row_actions,
        ),
        "no_reachable_candidate": reach.compose_reachability_trace(
            frame_id="frame-no-reach",
            semantic_outcome=_outcome([missing_dest], row_actions),
            previous_state={"status": "resolved", "candidate_rows": (source,)},
            reachability_tile=tile,
            row_actions=row_actions,
        ),
        "conflicting_reachable_actions": reach.compose_reachability_trace(
            frame_id="frame-conflict",
            semantic_outcome=_outcome(
                conflict_case[1], row_actions, status="conflicting_action_tie"
            ),
            previous_state={
                "status": "resolved",
                "candidate_rows": (conflict_case[0],),
            },
            reachability_tile=tile,
            row_actions=row_actions,
        ),
        "multiple_retained_one_action": reach.compose_reachability_trace(
            frame_id="frame-same-action",
            semantic_outcome=_outcome(
                same_action_case[1], row_actions, status="action_unanimous_tie"
            ),
            previous_state={
                "status": "resolved",
                "candidate_rows": (same_action_case[0],),
            },
            reachability_tile=tile,
            row_actions=row_actions,
        ),
        "empty_top_set": reach.compose_reachability_trace(
            frame_id="frame-empty",
            semantic_outcome=_outcome([], row_actions, status="unresolved"),
            previous_state=None,
            reachability_tile=tile,
            row_actions=row_actions,
        ),
    }


def _ordinary_validation_fixture(
    reachability_context: dict[str, Any],
) -> tuple[
    dict[str, Any], dict[str, object], dict[str, object], dict[str, Any], dict[str, str]
]:
    row_actions = reachability_context["row_actions"]
    tile = reachability_context["tile"]
    source = reachability_context["row_ids"][0]
    destination = benchmark._tile_edge(tile, source, row_actions[source])[
        "reachable_row_ids"
    ][0]
    semantic = _outcome([destination], row_actions)
    previous = {"status": "resolved", "candidate_rows": (source,)}
    trace = reach.compose_reachability_trace(
        frame_id="frame-ordinary",
        semantic_outcome=semantic,
        previous_state=previous,
        reachability_tile=tile,
        row_actions=row_actions,
    )
    return trace, semantic, previous, tile, row_actions


def _validate_trace(
    trace: dict[str, Any],
    semantic: dict[str, object],
    previous: dict[str, object],
    tile: dict[str, Any],
    row_actions: dict[str, str],
) -> str:
    return reach.validate_reachability_trace(
        trace,
        semantic_outcome=semantic,
        previous_state=previous,
        reachability_tile=tile,
        row_actions=row_actions,
    )


def test_reachability_aliases_are_direct() -> None:
    assert benchmark._trace_digest is reach.trace_digest
    assert benchmark._top_row_action_map is reach.top_row_action_map
    assert benchmark.compose_reachability_trace is reach.compose_reachability_trace
    assert benchmark.validate_reachability_trace is reach.validate_reachability_trace
    assert benchmark._state_from_trace is reach.state_from_trace
    assert benchmark._gap_reachability_state is reach.gap_reachability_state
    assert (
        benchmark.REACHABILITY_COMPOSITION_VERSION
        == reach.REACHABILITY_COMPOSITION_VERSION
    )
    assert benchmark.REACHABILITY_TRACE_VERSION == reach.REACHABILITY_TRACE_VERSION


def test_trace_variants_and_digests_are_frozen(
    reachability_context: dict[str, Any],
) -> None:
    traces = _trace_cases(reachability_context)

    assert {name: trace["trace_digest"] for name, trace in traces.items()} == {
        "first_frame": "sha256:ffaf0407aa1ff213b0c5203b8fab43f8a0231e5f55c61bcf3516a7d6cf81208b",
        "ordinary_reachable": "sha256:3380df273b057f6667369f05a9864a683aebecb219b93e5c711bfccda8e95155",
        "prior_unresolved": "sha256:727dbbaf8ddb85bc12ba5d309d042269167bd8160e4360a00a418295ecd8bec9",
        "no_reachable_candidate": "sha256:e25068a215bae51f7ce66717e39c2879b0ea8b5da1dc235de3646c1651481dd0",
        "conflicting_reachable_actions": "sha256:aa2eca16e15dc6cb4898ca12fc07fa5e0df197ed81ec368e5c6eeefab8728dfd",
        "multiple_retained_one_action": "sha256:7fbe2d435e73d7908c2adf7366218f7640ae6e917d3e0b82d732dcd95f84a987",
        "empty_top_set": "sha256:39107f8e7351f3f404f46a0a4bd9802e8807bc481d4ff90565a0083cbc5c3cf5",
    }
    assert list(traces["ordinary_reachable"]) == TRACE_KEYS
    assert traces["first_frame"]["consulted_edges"] == []
    assert traces["first_frame"]["reachable_candidate_pairs"] == []
    assert traces["prior_unresolved"]["rejection_reason"] == "prior_state_unresolved"
    assert (
        traces["no_reachable_candidate"]["rejection_reason"] == "no_reachable_candidate"
    )
    assert traces["conflicting_reachable_actions"]["rejection_reason"] == (
        "conflicting_reachable_actions"
    )
    assert traces["multiple_retained_one_action"]["resolved_action_id"] == "STAY"
    assert traces["multiple_retained_one_action"]["resolved_row_id"] is None
    assert traces["empty_top_set"]["status"] == "unresolved"
    assert reach.state_from_trace(traces["empty_top_set"]) == {
        "status": "resolved",
        "candidate_rows": tuple(),
    }


def test_trace_validation_status_precedence_is_frozen(
    reachability_context: dict[str, Any],
) -> None:
    trace, semantic, previous, tile, row_actions = _ordinary_validation_fixture(
        reachability_context
    )

    assert _validate_trace(trace, semantic, previous, tile, row_actions) == "ok"

    foreign_tile = deepcopy(tile)
    foreign_tile["tile_digest"] = "sha256:" + "0" * 64
    assert (
        _validate_trace(trace, semantic, previous, foreign_tile, row_actions)
        == "foreign_reachability_tile"
    )

    bad_semantic = dict(semantic)
    bad_semantic["top_row_actions"] = []
    assert (
        _validate_trace(trace, bad_semantic, previous, tile, row_actions)
        == "reachability_trace_recompute_failed"
    )

    tampered = deepcopy(trace)
    tampered["trace_digest"] = "sha256:" + "1" * 64
    assert (
        _validate_trace(tampered, semantic, previous, tile, row_actions)
        == "foreign_reachability_trace_digest"
    )

    tampered = deepcopy(trace)
    tampered["consulted_edges"] = []
    tampered["trace_digest"] = reach.trace_digest(tampered)
    assert (
        _validate_trace(tampered, semantic, previous, tile, row_actions)
        == "consulted_edge_mismatch"
    )

    tampered = deepcopy(trace)
    tampered["reachable_candidate_pairs"] = []
    tampered["trace_digest"] = reach.trace_digest(tampered)
    assert (
        _validate_trace(tampered, semantic, previous, tile, row_actions)
        == "reachable_pair_mismatch"
    )

    tampered = deepcopy(trace)
    tampered["removed_rows"] = ["synthetic-row"]
    tampered["trace_digest"] = reach.trace_digest(tampered)
    assert (
        _validate_trace(tampered, semantic, previous, tile, row_actions)
        == "reachability_trace_mismatch"
    )

    with pytest.raises(KeyError):
        reach.validate_reachability_trace(
            {},
            semantic_outcome=semantic,
            previous_state=previous,
            reachability_tile=tile,
            row_actions=row_actions,
        )


def test_malformed_composition_order_and_gap_state(
    reachability_context: dict[str, Any],
) -> None:
    row_actions = reachability_context["row_actions"]
    tile = reachability_context["tile"]
    top_row = reachability_context["row_ids"][0]
    malformed = {
        "semantic_outcome_digest": "sha256:" + "0" * 64,
        "top_row_ids": [top_row],
        "top_row_actions": [],
    }
    foreign_tile = deepcopy(tile)
    foreign_tile["tile_digest"] = "sha256:" + "0" * 64
    with pytest.raises(VPMValidationError, match="reachability tile digest"):
        reach.compose_reachability_trace(
            frame_id="frame",
            semantic_outcome=malformed,
            previous_state=None,
            reachability_tile=foreign_tile,
            row_actions=row_actions,
        )
    with pytest.raises(
        VPMValidationError, match="semantic top rows and row/action mapping disagree"
    ):
        reach.compose_reachability_trace(
            frame_id="frame",
            semantic_outcome=malformed,
            previous_state=None,
            reachability_tile=tile,
            row_actions=row_actions,
        )
    assert reach.gap_reachability_state(
        {"gap_declaration": "declared_gap", "metadata": {"gap_declaration": "wrong"}}
    ) == {"status": "unresolved", "candidate_rows": tuple(), "reason": "declared_gap"}
