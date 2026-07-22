from __future__ import annotations

from typing import Dict, Optional, Tuple

from zeromodel.core.policy_transitions import (
    ROW_UNION_TRANSITION_SCOPE,
    PolicyTransitionSpec,
)
from zeromodel.video.arcade_policy.model import (
    ACTIONS,
    ShooterConfig,
    compile_policy_artifact,
    parse_state_row_id,
    state_row_id,
)


def next_rows(
    tank_x: int,
    target_x: Optional[int],
    cooldown: int,
    action: str,
    *,
    width: int,
) -> Tuple[str, ...]:
    action = str(action)
    next_tank = tank_x
    if action == "LEFT":
        next_tank = max(0, tank_x - 1)
    elif action == "RIGHT":
        next_tank = min(width - 1, tank_x + 1)

    if action == "FIRE":
        next_cooldown = 1 if cooldown == 0 else cooldown
    else:
        next_cooldown = max(0, cooldown - 1)

    successful_fire = (
        action == "FIRE"
        and cooldown == 0
        and target_x is not None
        and tank_x == target_x
    )
    if successful_fire:
        next_targets: Tuple[Optional[int], ...] = (None,) + tuple(range(width))
    else:
        next_targets = (target_x,)
    return tuple(
        state_row_id(next_tank, next_target, next_cooldown)
        for next_target in next_targets
    )


def arcade_transition_spec(
    config: ShooterConfig = ShooterConfig(),
    *,
    maximum_frame_gap: int = 2,
) -> PolicyTransitionSpec:
    policy = compile_policy_artifact(config)
    transitions: Dict[str, Tuple[str, ...]] = {}
    for row_id in policy.source.row_ids:
        tank_x, target_x, cooldown = parse_state_row_id(str(row_id))
        destinations: set[str] = set()
        for action in ACTIONS:
            destinations.update(
                next_rows(tank_x, target_x, cooldown, action, width=config.width)
            )
        transitions[str(row_id)] = tuple(sorted(destinations))
    return PolicyTransitionSpec(
        allowed_row_transitions=transitions,
        maximum_frame_gap=maximum_frame_gap,
        maximum_position_delta=1,
        transition_scope=ROW_UNION_TRANSITION_SCOPE,
        metadata={
            "world": "tiny_arcade_shooter",
            "derivation": "declared_environment_dynamics",
        },
    )
