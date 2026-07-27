"""Warehouse transition categories: ordinary + presence/value/relation/identity
faults, mirroring ``visual_transition_benchmark.dataset``'s discipline: every
"true" transition is produced by actually calling ``model.step``; every fault
only ever substitutes the *rendered* post-state or pokes a documented pixel.

Z-order note (an honest, documented environmental constraint, not a bug):
the renderer draws crates, then the robot, on top. A successful push always
lands the robot exactly on the crate's *pre-push* cell, so a crate that fails
to follow the robot is pixel-hidden by the robot standing on top of it --
this is unrecoverable from pixels alone, by any system, including the
privileged baseline if it were pixel-based (it is not: System B reads state
directly). This is why the identity/relation faults below are constructed to
manifest at the crate's *destination* cell (never occluded), not at the
robot's landing cell (always occluded when a crate fails to follow).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Tuple


from visual_transition_benchmark.domains.warehouse import dataset as wds
from visual_transition_benchmark.domains.warehouse import model as wm
from visual_transition_benchmark.domains.warehouse import rendering as wr

ORDINARY_CATEGORIES: Tuple[str, ...] = (
    "robot_moves_left",
    "robot_moves_right",
    "robot_blocked_by_wall",
    "push_crate_moves_target",
    "push_crate_reaches_goal",
    "push_attempt_with_no_crate_is_noop",
    "door_opens",
    "wait_no_op",
)

PRESENCE_FAULT_CATEGORIES: Tuple[str, ...] = (
    "robot_moves_during_wait",
    "wall_changes_unexpectedly",
    "door_changes_during_move",
    "unrelated_crate_moves_during_robot_move",
    "push_fails_silently",
)
VALUE_FAULT_CATEGORIES: Tuple[str, ...] = (
    "robot_moves_wrong_direction",
    "robot_moves_too_far",
    "battery_decreases_by_wrong_amount",
    "door_changes_to_wrong_visual_state",
)
RELATION_FAULT_CATEGORIES: Tuple[str, ...] = (
    "push_advances_robot_without_crate",
    "crate_moves_without_robot_adjacency",
    "two_crates_move_during_single_push",
)
IDENTITY_FAULT_CATEGORIES: Tuple[str, ...] = (
    "wrong_crate_moves",
    "two_crates_swap_identities",
    "expected_crate_remains_while_another_moves",
)

FAULT_CATEGORIES: Tuple[str, ...] = (
    PRESENCE_FAULT_CATEGORIES
    + VALUE_FAULT_CATEGORIES
    + RELATION_FAULT_CATEGORIES
    + IDENTITY_FAULT_CATEGORIES
)
ALL_CATEGORIES: Tuple[str, ...] = ORDINARY_CATEGORIES + FAULT_CATEGORIES

FAULT_FAMILY_OF = {}
for _category in PRESENCE_FAULT_CATEGORIES:
    FAULT_FAMILY_OF[_category] = "presence"
for _category in VALUE_FAULT_CATEGORIES:
    FAULT_FAMILY_OF[_category] = "value"
for _category in RELATION_FAULT_CATEGORIES:
    FAULT_FAMILY_OF[_category] = "relation"
for _category in IDENTITY_FAULT_CATEGORIES:
    FAULT_FAMILY_OF[_category] = "identity"


class WarehouseFaultError(ValueError):
    pass


def _extra_pixel_edits(
    *edits: Tuple[int, int, int],
) -> Tuple[Tuple[int, int, int], ...]:
    return tuple(edits)


def _door_bar_edit(height_px: int) -> Tuple[Tuple[int, int, int], ...]:
    door_y0, door_x0 = wr.cell_origin(*wm.DOOR_POSITION)
    edits = []
    for dy in range(wr.CELL_PIXELS):
        for dx in (2, 3):
            value = wr.DOOR_VALUE if dy < height_px else 0
            edits.append((door_y0 + dy, door_x0 + dx, value))
    return tuple(edits)


# --------------------------------------------------------------------------- #
# Ordinary category builders: (rng) -> (robot, crates, door_open, battery, action)
# --------------------------------------------------------------------------- #

_SAFE_LEFT = (
    (1, 3),
    (3, 3),
    (3, 2),
)  # (start, target) pairs where MOVE_LEFT succeeds cleanly
_SAFE_LEFT_TARGETS = {(1, 3): (1, 2), (3, 3): (3, 2), (3, 2): (3, 1)}
_SAFE_RIGHT_TARGETS = {(1, 1): (1, 2), (3, 1): (3, 2), (3, 2): (3, 3)}


def _extra_crates(
    rng: random.Random, count: int, exclude: Tuple[Tuple[int, int], ...]
) -> Tuple[Tuple[int, int], ...]:
    pool = [cell for cell in wds.PLACEABLE_CELLS if cell not in exclude]
    rng.shuffle(pool)
    if len(pool) < count:
        raise WarehouseFaultError("not enough distinct cells for extra crates")
    return tuple(pool[:count])


def _build_robot_moves_left(rng: random.Random):
    start = rng.choice(list(_SAFE_LEFT_TARGETS))
    target = _SAFE_LEFT_TARGETS[start]
    crates = _extra_crates(
        rng, rng.choice((0, 1)), exclude=(start, target, wm.DOOR_POSITION)
    )
    return start, crates, False, 3, "MOVE_LEFT"


def _build_robot_moves_right(rng: random.Random):
    start = rng.choice(list(_SAFE_RIGHT_TARGETS))
    target = _SAFE_RIGHT_TARGETS[start]
    crates = _extra_crates(
        rng, rng.choice((0, 1)), exclude=(start, target, wm.DOOR_POSITION)
    )
    return start, crates, False, 3, "MOVE_RIGHT"


def _build_robot_blocked_by_wall(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1, 2)), exclude=((1, 1), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "MOVE_UP"


def _build_push_crate_moves_target(rng: random.Random):
    crates = ((1, 2),) + _extra_crates(
        rng, rng.choice((0, 1)), exclude=((1, 1), (1, 2), (1, 3), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "PUSH_RIGHT"


def _build_push_crate_reaches_goal(rng: random.Random):
    crates = ((3, 2),) + _extra_crates(
        rng, rng.choice((0, 1)), exclude=((3, 1), (3, 2), (3, 3), wm.DOOR_POSITION)
    )
    return (3, 1), crates, False, 3, "PUSH_RIGHT"


def _build_push_attempt_with_no_crate_is_noop(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((1, 2)), exclude=((1, 1), (1, 2), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "PUSH_RIGHT"


def _build_door_opens(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1, 2)), exclude=((1, 1), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "OPEN_DOOR"


def _build_wait_no_op(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1, 2)), exclude=((1, 1), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "WAIT"


ORDINARY_BUILDERS = {
    "robot_moves_left": _build_robot_moves_left,
    "robot_moves_right": _build_robot_moves_right,
    "robot_blocked_by_wall": _build_robot_blocked_by_wall,
    "push_crate_moves_target": _build_push_crate_moves_target,
    "push_crate_reaches_goal": _build_push_crate_reaches_goal,
    "push_attempt_with_no_crate_is_noop": _build_push_attempt_with_no_crate_is_noop,
    "door_opens": _build_door_opens,
    "wait_no_op": _build_wait_no_op,
}


# --------------------------------------------------------------------------- #
# Fault builders (preconditions) and fault functions (post-state corruption)
# --------------------------------------------------------------------------- #


def _build_robot_moves_during_wait(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1)), exclude=((1, 1), (1, 2), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "WAIT"


def _fault_robot_moves_during_wait(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    return (
        (1, 2),
        true_after.crates,
        true_after.door_open,
        true_after.battery,
        (),
        "robot rendered as moved despite WAIT",
    )


def _build_wall_changes_unexpectedly(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1, 2)), exclude=((1, 1), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "WAIT"


def _fault_wall_changes_unexpectedly(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    wall_y, wall_x = wr.cell_origin(0, 1)
    edits = _extra_pixel_edits((wall_y + 1, wall_x + 1, 150))
    return (
        true_after.robot,
        true_after.crates,
        true_after.door_open,
        true_after.battery,
        edits,
        "a wall pixel changed despite no environment rule touching it",
    )


def _build_door_changes_during_move(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1)), exclude=((3, 3), (3, 2), wm.DOOR_POSITION)
    )
    return (3, 3), crates, False, 3, "MOVE_LEFT"


def _fault_door_changes_during_move(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    return (
        true_after.robot,
        true_after.crates,
        True,
        true_after.battery,
        (),
        "door rendered as opened despite a plain MOVE action",
    )


def _build_unrelated_crate_moves_during_robot_move(rng: random.Random):
    crates = ((1, 3),) + _extra_crates(
        rng,
        rng.choice((0, 1)),
        exclude=((3, 1), (3, 2), (1, 3), (1, 2), wm.DOOR_POSITION),
    )
    return (3, 1), crates, False, 3, "MOVE_RIGHT"


def _fault_unrelated_crate_moves_during_robot_move(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    new_crates = ((1, 2),) + true_after.crates[1:]
    return (
        true_after.robot,
        new_crates,
        true_after.door_open,
        true_after.battery,
        (),
        "an uninvolved crate rendered as moved alongside a legitimate robot move",
    )


def _build_push_fails_silently(rng: random.Random):
    crates = ((1, 2),) + _extra_crates(
        rng, rng.choice((0, 1)), exclude=((1, 1), (1, 2), (1, 3), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "PUSH_RIGHT"


def _fault_push_fails_silently(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    # A valid push precondition holds (crate ahead, landing cell clear), so
    # the true transition moves robot+crate and spends battery -- but the
    # render is the *exact* pre-push scene: nothing visibly changes. This is
    # the warehouse analogue of the arcade domain's fire_no_projectile: the
    # expected region shows zero pixel difference, not a wrong one.
    return (
        before.robot,
        before.crates,
        before.door_open,
        before.battery,
        (),
        "a valid push produced no rendered change at all -- the expected robot/crate/battery "
        "change is completely absent from the frames",
    )


def _build_robot_moves_wrong_direction(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1)), exclude=((3, 2), (3, 1), (3, 3), wm.DOOR_POSITION)
    )
    return (3, 2), crates, False, 3, "MOVE_LEFT"


def _fault_robot_moves_wrong_direction(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    return (
        (3, 3),
        true_after.crates,
        true_after.door_open,
        true_after.battery,
        (),
        "robot moved right instead of the commanded left",
    )


def _build_robot_moves_too_far(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1)), exclude=((3, 3), (3, 2), (3, 1), wm.DOOR_POSITION)
    )
    return (3, 3), crates, False, 3, "MOVE_LEFT"


def _fault_robot_moves_too_far(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    return (
        (3, 1),
        true_after.crates,
        true_after.door_open,
        true_after.battery,
        (),
        "robot moved two cells instead of one in the commanded direction",
    )


def _build_battery_decreases_by_wrong_amount(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1)), exclude=((1, 1), (1, 2), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "MOVE_RIGHT"


def _fault_battery_decreases_by_wrong_amount(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    return (
        true_after.robot,
        true_after.crates,
        true_after.door_open,
        max(0, before.battery - 2),
        (),
        "battery dropped by two instead of one",
    )


def _build_door_changes_to_wrong_visual_state(rng: random.Random):
    crates = _extra_crates(
        rng, rng.choice((0, 1, 2)), exclude=((1, 1), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "OPEN_DOOR"


def _fault_door_changes_to_wrong_visual_state(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    edits = _door_bar_edit(2)  # neither closed (6px) nor open (3px)
    return (
        true_after.robot,
        true_after.crates,
        true_after.door_open,
        true_after.battery,
        edits,
        "door rendered at an out-of-contract bar height",
    )


def _build_push_advances_robot_without_crate(rng: random.Random):
    crates = ((1, 2),) + _extra_crates(
        rng, rng.choice((0, 1)), exclude=((1, 1), (1, 2), (1, 3), wm.DOOR_POSITION)
    )
    return (1, 1), crates, False, 3, "PUSH_RIGHT"


def _fault_push_advances_robot_without_crate(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    unaffected = before.crates[1:]
    return (
        true_after.robot,
        (before.crates[0],) + unaffected,
        true_after.door_open,
        true_after.battery,
        (),
        "robot rendered as advancing into the crate's cell, but the crate did not follow "
        "(the crate is z-order-hidden under the robot at that cell -- an honest, "
        "documented environmental blind spot, not a system-specific one)",
    )


def _build_crate_moves_without_robot_adjacency(rng: random.Random):
    crates = ((1, 3),) + _extra_crates(
        rng, rng.choice((0, 1)), exclude=((3, 1), (1, 3), (1, 2), wm.DOOR_POSITION)
    )
    return (3, 1), crates, False, 3, "WAIT"


def _fault_crate_moves_without_robot_adjacency(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    new_crates = ((1, 2),) + true_after.crates[1:]
    return (
        true_after.robot,
        new_crates,
        true_after.door_open,
        true_after.battery,
        (),
        "a crate rendered as moved though the robot was never adjacent to it",
    )


def _build_two_crates_move_during_single_push(rng: random.Random):
    crates = ((1, 2), (3, 1)) + _extra_crates(
        rng,
        rng.choice((0, 1)),
        exclude=((1, 1), (1, 2), (1, 3), (3, 1), (3, 2), wm.DOOR_POSITION),
    )
    return (1, 1), crates, False, 3, "PUSH_RIGHT"


def _fault_two_crates_move_during_single_push(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    new_crates = (true_after.crates[0], (3, 2)) + true_after.crates[2:]
    return (
        true_after.robot,
        new_crates,
        true_after.door_open,
        true_after.battery,
        (),
        "a second, uninvolved crate rendered as moved during a single push",
    )


def _build_wrong_crate_moves(rng: random.Random):
    return (1, 1), ((1, 2), (3, 1)), False, 3, "PUSH_RIGHT"


def _fault_wrong_crate_moves(true_after: wm.WarehouseState, before: wm.WarehouseState):
    # true_after.crates == ((1, 3), (3, 1)): crate A (index 0) pushed to (1, 3).
    # Render crate B's marker (2 dots) at (1, 3) instead of crate A's (1 dot);
    # crate A stays nominally at (1, 2) (hidden under the robot, same
    # z-order caveat as the relation faults above).
    return (
        true_after.robot,
        ((1, 2), (1, 3)),
        true_after.door_open,
        true_after.battery,
        (),
        "the wrong crate's identity marker appears at the pushed-to cell",
    )


def _build_two_crates_swap_identities(rng: random.Random):
    return (1, 1), ((1, 2), (3, 1)), False, 3, "PUSH_RIGHT"


def _fault_two_crates_swap_identities(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    # true_after.crates == ((1, 3), (3, 1)) -- render with the tuple order
    # swapped: since identity == tuple index, swapping order swaps which
    # position shows which dot-count without moving anything.
    return (
        true_after.robot,
        (true_after.crates[1], true_after.crates[0]),
        true_after.door_open,
        true_after.battery,
        (),
        "the two crates' identity markers are swapped between their true positions",
    )


def _build_expected_crate_remains_while_another_moves(rng: random.Random):
    return (1, 1), ((1, 2), (3, 1)), False, 3, "PUSH_RIGHT"


def _fault_expected_crate_remains_while_another_moves(
    true_after: wm.WarehouseState, before: wm.WarehouseState
):
    # crate A never appears at (1, 3) (stays nominally at (1, 2), hidden);
    # crate B wrongly moves from (3, 1) to (3, 2) though nothing commanded it.
    return (
        true_after.robot,
        ((1, 2), (3, 2)),
        true_after.door_open,
        true_after.battery,
        (),
        "the expected crate shows no change at its target while an uninvolved crate moves",
    )


FAULT_BUILDERS = {
    "robot_moves_during_wait": _build_robot_moves_during_wait,
    "wall_changes_unexpectedly": _build_wall_changes_unexpectedly,
    "door_changes_during_move": _build_door_changes_during_move,
    "unrelated_crate_moves_during_robot_move": _build_unrelated_crate_moves_during_robot_move,
    "push_fails_silently": _build_push_fails_silently,
    "robot_moves_wrong_direction": _build_robot_moves_wrong_direction,
    "robot_moves_too_far": _build_robot_moves_too_far,
    "battery_decreases_by_wrong_amount": _build_battery_decreases_by_wrong_amount,
    "door_changes_to_wrong_visual_state": _build_door_changes_to_wrong_visual_state,
    "push_advances_robot_without_crate": _build_push_advances_robot_without_crate,
    "crate_moves_without_robot_adjacency": _build_crate_moves_without_robot_adjacency,
    "two_crates_move_during_single_push": _build_two_crates_move_during_single_push,
    "wrong_crate_moves": _build_wrong_crate_moves,
    "two_crates_swap_identities": _build_two_crates_swap_identities,
    "expected_crate_remains_while_another_moves": _build_expected_crate_remains_while_another_moves,
}

FAULT_FUNCTIONS = {
    "robot_moves_during_wait": _fault_robot_moves_during_wait,
    "wall_changes_unexpectedly": _fault_wall_changes_unexpectedly,
    "door_changes_during_move": _fault_door_changes_during_move,
    "unrelated_crate_moves_during_robot_move": _fault_unrelated_crate_moves_during_robot_move,
    "push_fails_silently": _fault_push_fails_silently,
    "robot_moves_wrong_direction": _fault_robot_moves_wrong_direction,
    "robot_moves_too_far": _fault_robot_moves_too_far,
    "battery_decreases_by_wrong_amount": _fault_battery_decreases_by_wrong_amount,
    "door_changes_to_wrong_visual_state": _fault_door_changes_to_wrong_visual_state,
    "push_advances_robot_without_crate": _fault_push_advances_robot_without_crate,
    "crate_moves_without_robot_adjacency": _fault_crate_moves_without_robot_adjacency,
    "two_crates_move_during_single_push": _fault_two_crates_move_during_single_push,
    "wrong_crate_moves": _fault_wrong_crate_moves,
    "two_crates_swap_identities": _fault_two_crates_swap_identities,
    "expected_crate_remains_while_another_moves": _fault_expected_crate_remains_while_another_moves,
}


def _category_seed(seed: int, episode_id: str, category: str) -> int:
    digest = 0
    for part in (str(seed), episode_id, category):
        for ch in part:
            digest = (digest * 1_000_003 + ord(ch)) % (2**32)
    return digest


def build_transition(
    *, episode_id: str, step_number: int, seed: int, category: str
) -> wds.WarehouseTransitionRecord:
    if category not in ALL_CATEGORIES:
        raise WarehouseFaultError(f"unknown category: {category}")
    rng = random.Random(_category_seed(seed, episode_id, category))
    is_faulty = category in FAULT_CATEGORIES

    if is_faulty:
        robot, crates, door_open, battery, action = FAULT_BUILDERS[category](rng)
    else:
        robot, crates, door_open, battery, action = ORDINARY_BUILDERS[category](rng)

    before = wm.WarehouseState(
        robot=robot, crates=crates, door_open=door_open, battery=battery
    )
    true_after = wm.step(before, action)
    frame_before = wds.render(before)

    if is_faulty:
        (
            render_robot,
            render_crates,
            render_door_open,
            render_battery,
            extra_edits,
            notes,
        ) = FAULT_FUNCTIONS[category](true_after, before)
        rendered_state = wm.WarehouseState(
            robot=render_robot,
            crates=render_crates,
            door_open=render_door_open,
            battery=render_battery,
        )
        frame_after = wds.render(rendered_state)
        for row, col, value in extra_edits:
            frame_after[row, col] = value
        fault_type = category
    else:
        rendered_state = true_after
        frame_after = wds.render(true_after)
        fault_type = None
        notes = "ordinary transition; rendered from the true post-state"

    expected_changed = wds._changed_components_from_states(before, true_after)
    observed_changed = wds._changed_components_from_pixels(
        frame_before, frame_after, before, rendered_state
    )
    annotations = wds.transition_component_masks(before, rendered_state)

    transition_id = f"{episode_id}-{step_number:04d}"
    return wds.WarehouseTransitionRecord(
        transition_id=transition_id,
        episode_id=episode_id,
        step_number=step_number,
        seed=seed,
        action=action,
        category=category,
        frame_before=frame_before,
        frame_after=frame_after,
        state_before=before.as_dict(),
        state_after=true_after.as_dict(),
        rendered_state=rendered_state.as_dict(),
        component_annotations=annotations,
        expected_changed_components=expected_changed,
        observed_changed_components=observed_changed,
        fault_type=fault_type,
        is_faulty=is_faulty,
        notes=notes,
    )


def generate_episode(
    episode_id: str, seed: int
) -> Tuple[wds.WarehouseTransitionRecord, ...]:
    return tuple(
        build_transition(
            episode_id=episode_id, step_number=step_number, seed=seed, category=category
        )
        for step_number, category in enumerate(ALL_CATEGORIES)
    )


@dataclass(frozen=True)
class WarehouseDatasetSplit:
    name: str
    episode_ids: Tuple[str, ...]
    records: Tuple[wds.WarehouseTransitionRecord, ...]


def generate_split(
    *, prefix: str, episode_count: int, seed_offset: int
) -> WarehouseDatasetSplit:
    episode_ids = tuple(f"{prefix}-{index:04d}" for index in range(episode_count))
    records: list = []
    for index, episode_id in enumerate(episode_ids):
        records.extend(generate_episode(episode_id, seed_offset + index))
    return WarehouseDatasetSplit(
        name=prefix, episode_ids=episode_ids, records=tuple(records)
    )
