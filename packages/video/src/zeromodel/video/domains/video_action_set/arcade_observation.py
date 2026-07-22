from __future__ import annotations

from typing import Any

import numpy as np

from zeromodel.video.arcade_policy import (
    CELL_PIXELS,
    COOLDOWN_BLOCKED_VALUE,
    COOLDOWN_READY_VALUE,
    TANK_VALUE,
    TARGET_VALUE,
    ShooterConfig,
    parse_state_row_id,
    render_state_frame,
)
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    ARCADE_RENDERER_IDENTITY_VERSION,
    FRAME_SHAPE,
)


def shooter_config_payload(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    return {
        "width": int(config.width),
        "wave": [int(item) for item in config.wave],
        "max_steps": int(config.max_steps),
    }


def render_row_frame(
    row_id: str,
    *,
    config: ShooterConfig = ShooterConfig(),
) -> np.ndarray:
    tank, target, cooldown = parse_state_row_id(str(row_id))
    return render_state_frame(tank, target, cooldown, width=config.width)


def renderer_identity(
    config: ShooterConfig = ShooterConfig(),
) -> dict[str, Any]:
    payload = {
        "version": ARCADE_RENDERER_IDENTITY_VERSION,
        "function": "zeromodel.arcade_policy.render_state_frame",
        "frame_shape": list(FRAME_SHAPE),
        "cell_pixels": int(CELL_PIXELS),
        "target_value": int(TARGET_VALUE),
        "tank_value": int(TANK_VALUE),
        "cooldown_ready_value": int(COOLDOWN_READY_VALUE),
        "cooldown_blocked_value": int(COOLDOWN_BLOCKED_VALUE),
        "config": shooter_config_payload(config),
    }
    return payload | {"renderer_identity_digest": canonical_sha256(payload)}


__all__ = [
    "render_row_frame",
    "renderer_identity",
    "shooter_config_payload",
]
