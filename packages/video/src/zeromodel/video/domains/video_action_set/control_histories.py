from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from zeromodel.video.arcade_policy import ACTIONS, ShooterConfig, next_rows, parse_state_row_id
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    AUTHORITATIVE_TRANSITION_FUNCTION_VERSION,
    GROUNDED_CONTROL_HISTORY_VERSION,
)


def _transition_config_payload(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    return {
        "width": int(config.width),
        "wave": [int(item) for item in config.wave],
        "max_steps": int(config.max_steps),
    }


def transition_identity(config: ShooterConfig = ShooterConfig()) -> dict[str, Any]:
    payload = {
        "version": AUTHORITATIVE_TRANSITION_FUNCTION_VERSION,
        "function": "zeromodel.arcade_policy.transitions.next_rows",
        "action_universe": list(ACTIONS),
        "config": _transition_config_payload(config),
    }
    return payload | {"transition_identity_digest": canonical_sha256(payload)}


def transition_input_digest(
    predecessor_row_id: str,
    action: str,
    *,
    config: ShooterConfig = ShooterConfig(),
) -> str:
    tank, target, cooldown = parse_state_row_id(str(predecessor_row_id))
    return canonical_sha256(
        {
            "version": AUTHORITATIVE_TRANSITION_FUNCTION_VERSION,
            "predecessor_row_id": str(predecessor_row_id),
            "parsed_state": {"tank_x": tank, "target_x": target, "cooldown": cooldown},
            "actual_executed_action": str(action),
            "config_digest": canonical_sha256(_transition_config_payload(config)),
        }
    )


def transition_result_digest(
    predecessor_row_id: str,
    action: str,
    transition_choice_index: int,
    resulting_row_id: str,
    *,
    config: ShooterConfig = ShooterConfig(),
) -> str:
    tank, target, cooldown = parse_state_row_id(str(predecessor_row_id))
    destinations = list(
        next_rows(tank, target, cooldown, str(action), width=config.width)
    )
    return canonical_sha256(
        {
            "version": AUTHORITATIVE_TRANSITION_FUNCTION_VERSION,
            "predecessor_row_id": str(predecessor_row_id),
            "actual_executed_action": str(action),
            "reachable_row_ids": destinations,
            "transition_choice_index": int(transition_choice_index),
            "resulting_row_id": str(resulting_row_id),
            "config_digest": canonical_sha256(_transition_config_payload(config)),
        }
    )


def normalized_control_causal_tuple(
    *,
    predecessor_row_id: str,
    actual_executed_action: str,
    transition_choice_index: int,
    resulting_row_id: str,
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    identity = transition_identity(config)
    return {
        "version": GROUNDED_CONTROL_HISTORY_VERSION,
        "predecessor_row_id": str(predecessor_row_id),
        "actual_executed_action": str(actual_executed_action),
        "transition_choice_index": int(transition_choice_index),
        "resulting_row_id": str(resulting_row_id),
        "transition_identity": identity,
        "transition_input_digest": transition_input_digest(
            str(predecessor_row_id),
            str(actual_executed_action),
            config=config,
        ),
        "transition_result_digest": transition_result_digest(
            str(predecessor_row_id),
            str(actual_executed_action),
            int(transition_choice_index),
            str(resulting_row_id),
            config=config,
        ),
    }


def grounded_control_history(
    *,
    control_group_id: str,
    hidden_history_index: int,
    predecessor_row_id: str,
    actual_executed_action: str,
    transition_choice_index: int,
    resulting_row_id: str,
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    causal_tuple = normalized_control_causal_tuple(
        predecessor_row_id=predecessor_row_id,
        actual_executed_action=actual_executed_action,
        transition_choice_index=transition_choice_index,
        resulting_row_id=resulting_row_id,
        config=config,
    )
    tuple_digest = canonical_sha256(causal_tuple)
    payload = {
        "version": GROUNDED_CONTROL_HISTORY_VERSION,
        "control_group_id": str(control_group_id),
        "hidden_history_index": int(hidden_history_index),
        "history_id": tuple_digest,
        "hidden_history_id": tuple_digest,
        "predecessor_row_id": str(predecessor_row_id),
        "actual_executed_action": str(actual_executed_action),
        "transition_choice_index": int(transition_choice_index),
        "resulting_row_id": str(resulting_row_id),
        "transition_identity": causal_tuple["transition_identity"],
        "transition_input_digest": causal_tuple["transition_input_digest"],
        "transition_result_digest": causal_tuple["transition_result_digest"],
        "normalized_causal_tuple_digest": tuple_digest,
    }
    payload["history_digest"] = canonical_sha256(payload)
    payload["hidden_source_label_digest"] = payload["history_digest"]
    return payload


def reconstructed_control_causal_tuple_digest(
    history: Mapping[str, Any],
    *,
    config: ShooterConfig = ShooterConfig(),
) -> str:
    if history.get("version") != GROUNDED_CONTROL_HISTORY_VERSION:
        raise VPMValidationError("unsupported grounded control history version")
    action = str(history.get("actual_executed_action"))
    if action not in ACTIONS:
        raise VPMValidationError("control history action is not declared")
    predecessor = str(history.get("predecessor_row_id"))
    resulting = str(history.get("resulting_row_id"))
    choice_index = int(history.get("transition_choice_index", -1))
    tank, target, cooldown = parse_state_row_id(predecessor)
    destinations = list(next_rows(tank, target, cooldown, action, width=config.width))
    if choice_index < 0 or choice_index >= len(destinations):
        raise VPMValidationError("control history transition choice is invalid")
    if str(destinations[choice_index]) != resulting:
        raise VPMValidationError(
            "control history result does not match the declared choice"
        )
    causal_tuple = normalized_control_causal_tuple(
        predecessor_row_id=predecessor,
        actual_executed_action=action,
        transition_choice_index=choice_index,
        resulting_row_id=resulting,
        config=config,
    )
    tuple_digest = canonical_sha256(causal_tuple)
    if history.get("transition_identity") != causal_tuple["transition_identity"]:
        raise VPMValidationError("control history transition identity mismatch")
    if (
        history.get("transition_input_digest")
        != causal_tuple["transition_input_digest"]
    ):
        raise VPMValidationError("control history input digest mismatch")
    if (
        history.get("transition_result_digest")
        != causal_tuple["transition_result_digest"]
    ):
        raise VPMValidationError("control history result digest mismatch")
    if history.get("normalized_causal_tuple_digest") != tuple_digest:
        raise VPMValidationError("control history causal tuple digest mismatch")
    return tuple_digest


def grounded_control_histories_for_current_row(
    current_row_id: str,
    row_ids: Sequence[str],
    *,
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    histories: list[dict[str, Any]] = []
    for predecessor in row_ids:
        tank, target, cooldown = parse_state_row_id(str(predecessor))
        for action in ACTIONS:
            destinations = list(
                next_rows(tank, target, cooldown, action, width=config.width)
            )
            for index, destination in enumerate(destinations):
                if str(destination) != str(current_row_id):
                    continue
                histories.append(
                    {
                        "predecessor_row_id": str(predecessor),
                        "actual_executed_action": str(action),
                        "transition_choice_index": int(index),
                        "resulting_row_id": str(current_row_id),
                        "normalized_causal_tuple_digest": canonical_sha256(
                            normalized_control_causal_tuple(
                                predecessor_row_id=str(predecessor),
                                actual_executed_action=str(action),
                                transition_choice_index=int(index),
                                resulting_row_id=str(current_row_id),
                                config=config,
                            )
                        ),
                    }
                )
    return sorted(
        histories,
        key=lambda item: (
            str(item["actual_executed_action"]),
            str(item["predecessor_row_id"]),
            int(item["transition_choice_index"]),
        ),
    )


def select_grounded_control_histories(
    current_row_id: str,
    row_ids: Sequence[str],
    *,
    control_group_id: str,
    frame_count: int,
    config: ShooterConfig = ShooterConfig(),
) -> list[dict[str, Any]]:
    candidates = grounded_control_histories_for_current_row(
        current_row_id,
        row_ids,
        config=config,
    )
    if len({item["normalized_causal_tuple_digest"] for item in candidates}) < 2:
        raise VPMValidationError(
            "information control current row lacks grounded causal ambiguity"
        )
    preferred: list[dict[str, Any]] = []
    used_predecessors: set[str] = set()
    used_actions: set[str] = set()
    for candidate in candidates:
        predecessor = str(candidate["predecessor_row_id"])
        action = str(candidate["actual_executed_action"])
        if predecessor in used_predecessors or action in used_actions:
            continue
        preferred.append(candidate)
        used_predecessors.add(predecessor)
        used_actions.add(action)
        if len(preferred) == int(frame_count):
            break
    for candidate in candidates:
        if len(preferred) == int(frame_count):
            break
        if candidate in preferred:
            continue
        predecessor = str(candidate["predecessor_row_id"])
        if predecessor in used_predecessors:
            continue
        preferred.append(candidate)
        used_predecessors.add(predecessor)
    for candidate in candidates:
        if len(preferred) == int(frame_count):
            break
        if candidate not in preferred:
            preferred.append(candidate)
    if len(preferred) < min(2, int(frame_count)):
        raise VPMValidationError(
            "information control needs at least two grounded causal histories"
        )
    return [
        grounded_control_history(
            control_group_id=control_group_id,
            hidden_history_index=index,
            predecessor_row_id=str(item["predecessor_row_id"]),
            actual_executed_action=str(item["actual_executed_action"]),
            transition_choice_index=int(item["transition_choice_index"]),
            resulting_row_id=str(current_row_id),
            config=config,
        )
        for index, item in enumerate(preferred)
    ]


def control_source_rows(
    row_ids: Sequence[str],
    *,
    count: int,
    frame_count: int,
    config: ShooterConfig = ShooterConfig(),
) -> list[str]:
    selected = []
    for row_id in row_ids:
        histories = grounded_control_histories_for_current_row(
            str(row_id),
            row_ids,
            config=config,
        )
        if len({item["normalized_causal_tuple_digest"] for item in histories}) >= int(
            frame_count
        ):
            selected.append(str(row_id))
        if len(selected) == int(count):
            return selected
    raise VPMValidationError(
        "unable to select enough grounded information-control rows"
    )


__all__ = [
    "control_source_rows",
    "grounded_control_histories_for_current_row",
    "grounded_control_history",
    "normalized_control_causal_tuple",
    "reconstructed_control_causal_tuple_digest",
    "select_grounded_control_histories",
    "transition_identity",
    "transition_input_digest",
    "transition_result_digest",
]
