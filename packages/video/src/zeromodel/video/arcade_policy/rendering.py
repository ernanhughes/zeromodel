from __future__ import annotations

from typing import Mapping, Optional, Tuple

import numpy as np

from zeromodel.video.arcade_policy.model import ShooterConfig, state_row_id


FRAME_HEIGHT = 16
CELL_PIXELS = 4
TARGET_VALUE = 220
TANK_VALUE = 255
COOLDOWN_READY_VALUE = 40
COOLDOWN_BLOCKED_VALUE = 160


def render_state_frame(
    tank_x: int,
    target_x: Optional[int],
    cooldown: int,
    *,
    width: int = 7,
) -> np.ndarray:
    if width <= 1:
        raise ValueError("width must be greater than one")
    if not (0 <= int(tank_x) < width):
        raise ValueError("tank_x must be inside the screen")
    if target_x is not None and not (0 <= int(target_x) < width):
        raise ValueError("target_x must be inside the screen")
    if int(cooldown) not in {0, 1}:
        raise ValueError("cooldown must be 0 or 1")

    frame = np.zeros((FRAME_HEIGHT, width * CELL_PIXELS), dtype=np.uint8)

    if target_x is not None:
        centre = int(target_x) * CELL_PIXELS + CELL_PIXELS // 2
        frame[2:4, centre - 1 : centre + 2] = TARGET_VALUE
        frame[4, centre] = TARGET_VALUE

    centre = int(tank_x) * CELL_PIXELS + CELL_PIXELS // 2
    frame[11, centre] = TANK_VALUE
    frame[12, centre - 1 : centre + 2] = TANK_VALUE
    frame[13, centre - 2 : centre + 3] = TANK_VALUE

    frame[7:9, -3:-1] = COOLDOWN_BLOCKED_VALUE if int(cooldown) else COOLDOWN_READY_VALUE
    frame.flags.writeable = False
    return frame


def enumerate_visual_frames(config: ShooterConfig = ShooterConfig()) -> Mapping[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    targets: Tuple[Optional[int], ...] = (None,) + tuple(range(config.width))
    for tank_x in range(config.width):
        for target_x in targets:
            for cooldown in (0, 1):
                row_id = state_row_id(tank_x, target_x, cooldown)
                frames[row_id] = render_state_frame(tank_x, target_x, cooldown, width=config.width)
    return frames
