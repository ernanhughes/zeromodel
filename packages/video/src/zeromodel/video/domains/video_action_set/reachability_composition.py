from __future__ import annotations

from typing import Any, Mapping

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.materialization_reachability import (
    tile_edge,
    validate_reachability_tile_identity,
)


REACHABILITY_COMPOSITION_VERSION = (
    "zeromodel-video-action-set-reachability-composition/v1"
)
REACHABILITY_TRACE_VERSION = "zeromodel-video-action-set-reachability-trace/v1"


def trace_digest(payload: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in payload.items() if key != "trace_digest"}
    )


def top_row_action_map(outcome: Mapping[str, Any]) -> dict[str, str]:
    return {
        str(item["row_id"]): str(item["action_id"])
        for item in outcome.get("top_row_actions", [])
    }


def _prior_rows_and_status(
    previous_state: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], str | None]:
    prior_rows = tuple(
        str(row_id) for row_id in (previous_state or {}).get("candidate_rows", ())
    )
    prior_status = (
        None
        if previous_state is None
        else str(previous_state.get("status", "unresolved"))
    )
    return prior_rows, prior_status


def _reachable_candidate_filter(
    *,
    top_rows: tuple[str, ...],
    prior_rows: tuple[str, ...],
    prior_status: str | None,
    previous_state: Mapping[str, Any] | None,
    reachability_tile: Mapping[str, Any],
    row_actions: Mapping[str, str],
) -> tuple[
    list[dict[str, Any]], list[dict[str, str]], list[str], tuple[str, ...], str | None
]:
    consulted_edges: list[dict[str, Any]] = []
    reachable_pairs: list[dict[str, str]] = []
    removed_rows: list[str] = []
    rejection_reason = None
    if previous_state is not None and prior_status == "unresolved":
        return (
            consulted_edges,
            reachable_pairs,
            list(top_rows),
            (),
            "prior_state_unresolved",
        )
    if previous_state is None:
        return (
            consulted_edges,
            reachable_pairs,
            removed_rows,
            tuple(sorted(top_rows)),
            None,
        )

    retained = set()
    for prior_row in prior_rows:
        prior_action = row_actions[prior_row]
        edge = tile_edge(reachability_tile, prior_row, prior_action)
        consulted_edges.append(
            {
                "source_row_id": edge["source_row_id"],
                "action_id": edge["action_id"],
                "reachable_row_ids": list(edge["reachable_row_ids"]),
            }
        )
        reachable = set(edge["reachable_row_ids"])
        for current_row in top_rows:
            if current_row in reachable:
                retained.add(current_row)
                reachable_pairs.append(
                    {
                        "source_row_id": prior_row,
                        "action_id": prior_action,
                        "destination_row_id": current_row,
                    }
                )
    retained_rows = tuple(sorted(retained))
    removed_rows = sorted(set(top_rows) - set(retained_rows))
    if not retained_rows:
        rejection_reason = "no_reachable_candidate"
    return (
        consulted_edges,
        reachable_pairs,
        removed_rows,
        retained_rows,
        rejection_reason,
    )


def _resolution_fields(
    *,
    semantic_outcome: Mapping[str, Any],
    retained_rows: tuple[str, ...],
    rejection_reason: str | None,
    row_actions: Mapping[str, str],
) -> tuple[tuple[str, ...], str | None, str | None, str | None, str]:
    retained_actions = tuple(sorted({row_actions[row_id] for row_id in retained_rows}))
    if (
        rejection_reason is None
        and semantic_outcome.get("status") == "conflicting_action_tie"
        and len(retained_actions) > 1
    ):
        rejection_reason = "conflicting_reachable_actions"
    resolved_action = (
        retained_actions[0]
        if rejection_reason is None and len(retained_actions) == 1
        else None
    )
    resolved_row = (
        retained_rows[0]
        if rejection_reason is None and len(retained_rows) == 1
        else None
    )
    status = (
        "rejected"
        if rejection_reason
        else ("resolved" if resolved_action is not None else "unresolved")
    )
    return retained_actions, rejection_reason, resolved_row, resolved_action, status


def compose_reachability_trace(
    *,
    frame_id: str,
    semantic_outcome: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None,
    reachability_tile: Mapping[str, Any],
    row_actions: Mapping[str, str],
) -> dict[str, Any]:
    validate_reachability_tile_identity(reachability_tile)
    top_rows = tuple(str(row_id) for row_id in semantic_outcome.get("top_row_ids", ()))
    top_row_actions = top_row_action_map(semantic_outcome)
    if set(top_rows) != set(top_row_actions):
        raise VPMValidationError("semantic top rows and row/action mapping disagree")
    if any(row_actions[row_id] != top_row_actions[row_id] for row_id in top_rows):
        raise VPMValidationError(
            "semantic outcome row/action mapping is inconsistent with policy"
        )
    prior_rows, prior_status = _prior_rows_and_status(previous_state)
    (
        consulted_edges,
        reachable_pairs,
        removed_rows,
        retained_rows,
        rejection_reason,
    ) = _reachable_candidate_filter(
        top_rows=top_rows,
        prior_rows=prior_rows,
        prior_status=prior_status,
        previous_state=previous_state,
        reachability_tile=reachability_tile,
        row_actions=row_actions,
    )
    (
        retained_actions,
        rejection_reason,
        resolved_row,
        resolved_action,
        status,
    ) = _resolution_fields(
        semantic_outcome=semantic_outcome,
        retained_rows=retained_rows,
        rejection_reason=rejection_reason,
        row_actions=row_actions,
    )
    trace = {
        "version": REACHABILITY_TRACE_VERSION,
        "composition_version": REACHABILITY_COMPOSITION_VERSION,
        "frame_id": frame_id,
        "semantic_outcome_digest": semantic_outcome["semantic_outcome_digest"],
        "input_candidate_rows": list(top_rows),
        "input_candidate_actions": sorted(
            {top_row_actions[row_id] for row_id in top_rows}
        ),
        "prior_reachable_rows": list(prior_rows),
        "prior_state_status": prior_status,
        "reachability_tile_identity": reachability_tile["tile_digest"],
        "consulted_edges": consulted_edges,
        "reachable_candidate_pairs": reachable_pairs,
        "removed_rows": removed_rows,
        "removed_actions": sorted({row_actions[row_id] for row_id in removed_rows}),
        "retained_rows": list(retained_rows),
        "retained_actions": list(retained_actions),
        "resulting_candidate_set": list(retained_rows),
        "resolved_row_id": resolved_row,
        "resolved_action_id": resolved_action,
        "rejection_reason": rejection_reason,
        "executed_action": resolved_action,
        "status": status,
    }
    trace["trace_digest"] = trace_digest(trace)
    return trace


def validate_reachability_trace(
    trace: Mapping[str, Any],
    *,
    semantic_outcome: Mapping[str, Any],
    previous_state: Mapping[str, Any] | None,
    reachability_tile: Mapping[str, Any],
    row_actions: Mapping[str, str],
) -> str:
    try:
        expected = compose_reachability_trace(
            frame_id=str(trace["frame_id"]),
            semantic_outcome=semantic_outcome,
            previous_state=previous_state,
            reachability_tile=reachability_tile,
            row_actions=row_actions,
        )
    except VPMValidationError as exc:
        if "reachability tile digest" in str(exc):
            return "foreign_reachability_tile"
        return "reachability_trace_recompute_failed"
    if str(trace.get("reachability_tile_identity")) != str(
        reachability_tile["tile_digest"]
    ):
        return "foreign_reachability_tile"
    if str(trace.get("trace_digest")) != trace_digest(trace):
        return "foreign_reachability_trace_digest"
    if list(trace.get("consulted_edges", [])) != expected["consulted_edges"]:
        return "consulted_edge_mismatch"
    if (
        list(trace.get("reachable_candidate_pairs", []))
        != expected["reachable_candidate_pairs"]
    ):
        return "reachable_pair_mismatch"
    if dict(trace) != expected:
        return "reachability_trace_mismatch"
    return "ok"


def state_from_trace(trace: Mapping[str, Any]) -> dict[str, Any]:
    if trace["status"] == "rejected":
        return {
            "status": "unresolved",
            "candidate_rows": tuple(),
            "reason": trace["rejection_reason"],
        }
    return {
        "status": "resolved",
        "candidate_rows": tuple(str(row_id) for row_id in trace["retained_rows"]),
    }


def gap_reachability_state(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "status": "unresolved",
        "candidate_rows": tuple(),
        "reason": record.get("gap_declaration") or "typed_gap_event",
    }


__all__ = [
    "REACHABILITY_COMPOSITION_VERSION",
    "REACHABILITY_TRACE_VERSION",
    "compose_reachability_trace",
    "gap_reachability_state",
    "state_from_trace",
    "top_row_action_map",
    "trace_digest",
    "validate_reachability_trace",
]
