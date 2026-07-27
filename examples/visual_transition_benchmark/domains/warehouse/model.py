"""A small, deterministic Sokoban-like warehouse environment.

Written fresh for this experiment (there is no pre-existing warehouse engine
to reuse, unlike the arcade domain). Deliberately small: a 3x3 interior room
inside a walled 5x5 grid, up to 3 identity-marked crates, one door, one goal,
one battery counter.

Adaptation note (documented once, mirrors the arcade domain's own adaptation
section): the prompt's generic ``PUSH`` action is implemented as four
direction-specific actions (``PUSH_UP/DOWN/LEFT/RIGHT``), exactly mirroring
``MOVE_UP/DOWN/LEFT/RIGHT``, because a push without a declared direction is
not a deterministic action in a 2D grid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

GRID_SIZE = 5  # 0..4; rows/cols 0 and 4 are always walls
INTERIOR = (1, 2, 3)
DOOR_POSITION: Tuple[int, int] = (2, 2)
GOAL_POSITION: Tuple[int, int] = (3, 3)
MAX_BATTERY = 3
MAX_CRATES = 3
CRATE_LABELS: Tuple[str, ...] = ("A", "B", "C")

DIRECTIONS: Dict[str, Tuple[int, int]] = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}
MOVE_ACTIONS: Tuple[str, ...] = tuple(f"MOVE_{d}" for d in DIRECTIONS)
PUSH_ACTIONS: Tuple[str, ...] = tuple(f"PUSH_{d}" for d in DIRECTIONS)
ACTIONS: Tuple[str, ...] = MOVE_ACTIONS + PUSH_ACTIONS + ("OPEN_DOOR", "WAIT")


class WarehouseError(ValueError):
    pass


@dataclass(frozen=True)
class WarehouseState:
    robot: Tuple[int, int]
    crates: Tuple[
        Tuple[int, int], ...
    ]  # ordered; index is the crate's visible identity (0=A, 1=B, 2=C)
    door_open: bool
    battery: int

    def __post_init__(self) -> None:
        if not (0 <= len(self.crates) <= MAX_CRATES):
            raise WarehouseError("crates must number between 0 and 3")
        if not (0 <= self.battery <= MAX_BATTERY):
            raise WarehouseError("battery out of range")

    def crate_at(self, position: Tuple[int, int]) -> Optional[int]:
        for index, position_of_crate in enumerate(self.crates):
            if position_of_crate == position:
                return index
        return None

    def as_dict(self) -> dict:
        return {
            "robot": list(self.robot),
            "crates": [list(c) for c in self.crates],
            "door_open": self.door_open,
            "battery": self.battery,
        }


def is_wall(position: Tuple[int, int]) -> bool:
    row, col = position
    if not (0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE):
        return True
    return row in (0, GRID_SIZE - 1) or col in (0, GRID_SIZE - 1)


def _passable(
    state: WarehouseState, position: Tuple[int, int], *, for_crate: bool
) -> bool:
    if is_wall(position):
        return False
    if position == DOOR_POSITION:
        if for_crate:
            return False  # crates never enter the door cell; keeps door semantics unambiguous
        return state.door_open
    return state.crate_at(position) is None


def step(state: WarehouseState, action: str) -> WarehouseState:
    """The real environment rule. Every "true" transition calls this; fault
    injection (faults.py) only ever substitutes the *rendered* post-state,
    exactly mirroring the arcade domain's dataset.py discipline."""

    if action not in ACTIONS:
        raise WarehouseError(f"unknown action: {action}")
    if action == "WAIT":
        return state
    if action == "OPEN_DOOR":
        return WarehouseState(
            robot=state.robot,
            crates=state.crates,
            door_open=True,
            battery=state.battery,
        )
    if action in MOVE_ACTIONS:
        direction = DIRECTIONS[action[len("MOVE_") :]]
        target = (state.robot[0] + direction[0], state.robot[1] + direction[1])
        if not _passable(state, target, for_crate=False):
            return state
        return WarehouseState(
            robot=target,
            crates=state.crates,
            door_open=state.door_open,
            battery=max(0, state.battery - 1),
        )
    # PUSH_*
    direction = DIRECTIONS[action[len("PUSH_") :]]
    target = (state.robot[0] + direction[0], state.robot[1] + direction[1])
    crate_index = state.crate_at(target)
    if crate_index is None:
        return state
    beyond = (target[0] + direction[0], target[1] + direction[1])
    if not _passable(state, beyond, for_crate=True):
        return state
    new_crates = list(state.crates)
    new_crates[crate_index] = beyond
    return WarehouseState(
        robot=target,
        crates=tuple(new_crates),
        door_open=state.door_open,
        battery=max(0, state.battery - 1),
    )
