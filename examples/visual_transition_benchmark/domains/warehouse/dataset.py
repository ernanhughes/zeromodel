"""Warehouse transition generator: mirrors ``visual_transition_benchmark.dataset``'s
structure and discipline for the arcade domain, applied to a new environment.

Every "true" transition is produced by actually calling
``domains.warehouse.model.step`` -- never reimplemented. Fault injection only
ever substitutes the *rendered* post-state (or pokes a documented pixel).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from visual_transition_benchmark.domains.warehouse import model as wm
from visual_transition_benchmark.domains.warehouse import rendering as wr

COMPONENT_NAMES: Tuple[str, ...] = ("robot", "crate", "door", "battery", "wall", "background")

# A fixed floor cell used by background/wall probe faults; builders that need
# a *guaranteed-empty* pixel exclude this cell from robot/crate placement in
# that specific scenario (mirrors dataset.py's BACKGROUND_PROBE_PIXEL, which
# is likewise a scenario-local guarantee, not a globally reserved cell).
PROBE_CELL: Tuple[int, int] = (1, 2)
BACKGROUND_PROBE_PIXEL: Tuple[int, int] = (
    wr.cell_origin(*PROBE_CELL)[0] + 1,
    wr.cell_origin(*PROBE_CELL)[1] + 1,
)
BACKGROUND_PROBE_VALUE = 175  # matches no canonical glyph level (0/50/60/120/200/220/255)

# Interior cells available for robot/crate placement (excludes only the door;
# the goal is placeable since a crate legitimately ends up there).
PLACEABLE_CELLS: Tuple[Tuple[int, int], ...] = tuple(
    (row, col) for row in wm.INTERIOR for col in wm.INTERIOR if (row, col) != wm.DOOR_POSITION
)


class WarehouseDatasetError(ValueError):
    pass


def render(state: wm.WarehouseState) -> np.ndarray:
    return np.array(wr.render_state_frame(state), dtype=np.uint8, copy=True)


# --------------------------------------------------------------------------- #
# Privileged, exact ground-truth component masks (formula-derived, not
# detected). Used only by the dataset generator and the privileged baseline;
# never by a deployable analyzer.
# --------------------------------------------------------------------------- #


def robot_mask(position: Tuple[int, int]) -> np.ndarray:
    mask = np.zeros((wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH), dtype=bool)
    y0, x0 = wr.cell_origin(*position)
    mask[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS] = True
    return mask


def crate_mask(positions: Tuple[Tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros((wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH), dtype=bool)
    for position in positions:
        y0, x0 = wr.cell_origin(*position)
        mask[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS] = True
    return mask


def door_mask() -> np.ndarray:
    mask = np.zeros((wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH), dtype=bool)
    y0, x0 = wr.cell_origin(*wm.DOOR_POSITION)
    mask[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS] = True
    return mask


def battery_mask() -> np.ndarray:
    mask = np.zeros((wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH), dtype=bool)
    mask[wm.GRID_SIZE * wr.CELL_PIXELS :, :] = True
    return mask


def wall_mask() -> np.ndarray:
    mask = np.zeros((wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH), dtype=bool)
    for row in range(wm.GRID_SIZE):
        for col in range(wm.GRID_SIZE):
            if row in (0, wm.GRID_SIZE - 1) or col in (0, wm.GRID_SIZE - 1):
                y0, x0 = wr.cell_origin(row, col)
                mask[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS] = True
    return mask


def transition_component_masks(before: wm.WarehouseState, after: wm.WarehouseState) -> dict:
    """Exact partition of the canvas across one transition (same discipline as
    ``dataset.transition_component_masks``: background is the complement of
    the *combined* before/after footprint, not two independently-complemented
    masks)."""

    robot = robot_mask(before.robot) | robot_mask(after.robot)
    crate = crate_mask(before.crates) | crate_mask(after.crates)
    door = door_mask()
    battery = battery_mask()
    wall = wall_mask()
    background = ~(robot | crate | door | battery | wall)
    return {"robot": robot, "crate": crate, "door": door, "battery": battery, "wall": wall, "background": background}


def _changed_components_from_pixels(
    frame_before: np.ndarray, frame_after: np.ndarray, before: wm.WarehouseState, after: wm.WarehouseState
) -> Tuple[str, ...]:
    masks = transition_component_masks(before, after)
    changed = []
    for name in COMPONENT_NAMES:
        region = masks[name]
        if region.any() and np.any(frame_before[region] != frame_after[region]):
            changed.append(name)
    return tuple(changed)


def _changed_components_from_states(before: wm.WarehouseState, after: wm.WarehouseState) -> Tuple[str, ...]:
    changed = []
    if before.robot != after.robot:
        changed.append("robot")
    if before.crates != after.crates:
        changed.append("crate")
    if before.door_open != after.door_open:
        changed.append("door")
    if before.battery != after.battery:
        changed.append("battery")
    return tuple(changed)


@dataclass(frozen=True)
class WarehouseTransitionRecord:
    transition_id: str
    episode_id: str
    step_number: int
    seed: int
    action: str
    category: str
    frame_before: np.ndarray
    frame_after: np.ndarray
    state_before: dict
    state_after: dict  # true post-state, regardless of rendering faults
    rendered_state: dict  # what was actually rendered
    component_annotations: dict
    expected_changed_components: Tuple[str, ...]
    observed_changed_components: Tuple[str, ...]
    fault_type: Optional[str]
    is_faulty: bool
    notes: str
