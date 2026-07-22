from __future__ import annotations

import hashlib
import json
from typing import Any

VIDEO_POLICY_REACHABILITY_TILE_VERSION = "zeromodel-video-policy-reachability-tile/v1"


def _sha256(value: Any) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    )


def compile_reachability_tile(
    *, policy_artifact_id: str, transition_spec: Any
) -> dict[str, Any]:
    row_ids = tuple(str(row_id) for row_id in transition_spec.row_ids)
    actions = ("LEFT", "RIGHT", "STAY", "FIRE")
    allowed = {
        str(source): {str(dest) for dest in destinations}
        for source, destinations in transition_spec.allowed_row_transitions.items()
    }
    edges = []
    for row_id in row_ids:
        for action in actions:
            destinations = sorted(allowed.get(row_id, set()))
            status = "known_reachable_set" if destinations else "unknown_transition"
            edges.append(
                {
                    "source_row_id": row_id,
                    "action_id": action,
                    "status": status,
                    "reachable_row_ids": destinations,
                }
            )
    tile = {
        "tile_version": VIDEO_POLICY_REACHABILITY_TILE_VERSION,
        "policy_artifact_id": policy_artifact_id,
        "row_universe_size": len(row_ids),
        "action_universe": list(actions),
        "source_action_pair_count": len(edges),
        "unknown_transition_semantics": "row-union transition scope without action-conditioned branching",
        "gap_semantics": {
            "maximum_frame_gap": transition_spec.maximum_frame_gap,
            "transition_scope": transition_spec.transition_scope,
        },
        "edges": edges,
    }
    tile["tile_digest"] = _sha256(tile)
    return tile


def verify_reachability_tile(tile: dict[str, Any]) -> dict[str, Any]:
    pair_count = len(tile["edges"])
    row_count = tile["row_universe_size"]
    action_count = len(tile["action_universe"])
    violations = []
    row_universe = {edge["source_row_id"] for edge in tile["edges"]}
    for edge in tile["edges"]:
        for destination in edge["reachable_row_ids"]:
            if destination not in row_universe:
                violations.append(
                    {
                        "source_row_id": edge["source_row_id"],
                        "action_id": edge["action_id"],
                        "destination": destination,
                    }
                )
    return {
        "verified": pair_count == row_count * action_count and not violations,
        "row_count": row_count,
        "action_count": action_count,
        "pair_count": pair_count,
        "violations": violations,
        "tile_digest": tile["tile_digest"],
    }


__all__ = ["compile_reachability_tile", "verify_reachability_tile"]
