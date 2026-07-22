from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zeromodel.video.arcade_policy import ShooterConfig, next_rows, parse_state_row_id
from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256


def tile_edge(
    tile: Mapping[str, Any], row_id: str, action_id: str
) -> Mapping[str, Any]:
    for edge in tile["edges"]:
        if edge["source_row_id"] == row_id and edge["action_id"] == action_id:
            return edge
    raise VPMValidationError("missing reachability edge")


def reachability_tile_digest(tile: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {key: value for key, value in tile.items() if key != "tile_digest"}
    )


def validate_reachability_tile_identity(tile: Mapping[str, Any]) -> None:
    if str(tile.get("tile_digest")) != reachability_tile_digest(tile):
        raise VPMValidationError("foreign reachability tile digest")


def next_materialized_row(
    policy_lookup: VPMPolicyLookup,
    row_id: str,
    *,
    choice_seed: int,
    config: ShooterConfig,
    reachability_tile: Mapping[str, Any],
) -> tuple[str, str, int, dict[str, Any]]:
    validate_reachability_tile_identity(reachability_tile)
    action = policy_lookup.choose(row_id)
    tank, target, cooldown = parse_state_row_id(str(row_id))
    rows = next_rows(tank, target, cooldown, action, width=config.width)
    index = choice_seed % len(rows)
    edge = tile_edge(reachability_tile, row_id, action)
    if str(rows[index]) not in set(edge["reachable_row_ids"]):
        raise VPMValidationError("reachability tile does not admit chosen transition")
    trace = {
        "reachability_tile_digest": reachability_tile["tile_digest"],
        "candidate_row_id": row_id,
        "candidate_action_set": [action],
        "top_action_set": [action],
        "tile_action_id": action,
        "reachable_row_ids": list(edge["reachable_row_ids"]),
        "chosen_reachable_index": index,
        "executed_action": action,
        "rejected": False,
    }
    return str(rows[index]), action, index, trace


__all__ = [
    "next_materialized_row",
    "reachability_tile_digest",
    "tile_edge",
    "validate_reachability_tile_identity",
]
