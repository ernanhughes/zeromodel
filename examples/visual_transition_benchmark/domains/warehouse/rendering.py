"""Geometric-glyph renderer for the warehouse domain. No external assets.

Canvas: a 5x5 grid at 6px/cell (30x30) plus a 4-row battery strip appended
below (total 34x30), grayscale uint8 -- the same representation style as the
arcade renderer (``zeromodel.video.arcade_policy.rendering``), just a
different layout.
"""

from __future__ import annotations

import numpy as np

from visual_transition_benchmark.domains.warehouse.model import (
    DOOR_POSITION,
    GOAL_POSITION,
    GRID_SIZE,
    MAX_BATTERY,
    WarehouseState,
)

CELL_PIXELS = 6
BATTERY_STRIP_HEIGHT = 4
CANVAS_HEIGHT = GRID_SIZE * CELL_PIXELS + BATTERY_STRIP_HEIGHT  # 34
CANVAS_WIDTH = GRID_SIZE * CELL_PIXELS  # 30

WALL_VALUE = 60
GOAL_RING_VALUE = 50
ROBOT_VALUE = 200
CRATE_BODY_VALUE = 120
CRATE_DOT_VALUE = 255
DOOR_VALUE = 220
BATTERY_ON_VALUE = 200

# Fixed, declared sub-positions for crate identity dots (relative to the
# crate's cell origin). Crate identity N (0-indexed) shows exactly N+1 dots,
# always the same N+1 positions from this ordered list -- never a different
# combination, so "how many of these three fixed spots are lit" is a
# complete, deterministic identity signature.
_DOT_OFFSETS = ((0, 0), (0, CELL_PIXELS - 2), (CELL_PIXELS - 2, 0))


def cell_origin(row: int, col: int) -> tuple:
    return row * CELL_PIXELS, col * CELL_PIXELS


def render_state_frame(state: WarehouseState) -> np.ndarray:
    frame = np.zeros((CANVAS_HEIGHT, CANVAS_WIDTH), dtype=np.uint8)

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            if row in (0, GRID_SIZE - 1) or col in (0, GRID_SIZE - 1):
                y0, x0 = cell_origin(row, col)
                frame[y0 : y0 + CELL_PIXELS, x0 : x0 + CELL_PIXELS] = WALL_VALUE

    if state.crate_at(GOAL_POSITION) is None:
        y0, x0 = cell_origin(*GOAL_POSITION)
        frame[y0, x0 : x0 + CELL_PIXELS] = GOAL_RING_VALUE
        frame[y0 + CELL_PIXELS - 1, x0 : x0 + CELL_PIXELS] = GOAL_RING_VALUE
        frame[y0 : y0 + CELL_PIXELS, x0] = GOAL_RING_VALUE
        frame[y0 : y0 + CELL_PIXELS, x0 + CELL_PIXELS - 1] = GOAL_RING_VALUE

    door_y0, door_x0 = cell_origin(*DOOR_POSITION)
    door_height = CELL_PIXELS // 2 if state.door_open else CELL_PIXELS
    frame[door_y0 : door_y0 + door_height, door_x0 + 2 : door_x0 + 4] = DOOR_VALUE

    for identity, position in enumerate(state.crates):
        y0, x0 = cell_origin(*position)
        frame[y0 : y0 + CELL_PIXELS, x0 : x0 + CELL_PIXELS] = CRATE_BODY_VALUE
        for dy, dx in _DOT_OFFSETS[: identity + 1]:
            frame[y0 + dy : y0 + dy + 2, x0 + dx : x0 + dx + 2] = CRATE_DOT_VALUE

    robot_y0, robot_x0 = cell_origin(*state.robot)
    frame[robot_y0 : robot_y0 + CELL_PIXELS, robot_x0 : robot_x0 + CELL_PIXELS] = (
        ROBOT_VALUE
    )

    strip_y0 = GRID_SIZE * CELL_PIXELS
    segment_width = CANVAS_WIDTH // MAX_BATTERY
    for index in range(MAX_BATTERY):
        if index < state.battery:
            frame[
                strip_y0 : strip_y0 + BATTERY_STRIP_HEIGHT,
                index * segment_width : (index + 1) * segment_width,
            ] = BATTERY_ON_VALUE

    frame.flags.writeable = False
    return frame
