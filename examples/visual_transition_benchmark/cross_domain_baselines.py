"""Domain-neutral System A (pixel diff) and System B (privileged), for the
cross-domain runner. Not a modification of ``baselines.py`` (stage 1's own
pixel-diff/privileged baselines are untouched and still used for the arcade-
only reports) -- this is a small, separately-maintained equivalent that takes
declared component band masks as a parameter instead of importing arcade's.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from visual_transition_benchmark.domains.protocol import DomainTransition

PIXEL_THRESHOLD = 8
MIN_COMPONENT_SIZE = 2


@dataclass(frozen=True)
class SimpleSystemOutput:
    predicted_components: Tuple[str, ...]
    missing_components: Tuple[str, ...]
    unexpected_components: Tuple[str, ...]
    predicted_region_mask: np.ndarray


def _connected_components(changed_mask: np.ndarray) -> np.ndarray:
    labels = np.zeros(changed_mask.shape, dtype=np.int32)
    next_label = 0
    height, width = changed_mask.shape
    for start_r in range(height):
        for start_c in range(width):
            if not changed_mask[start_r, start_c] or labels[start_r, start_c] != 0:
                continue
            next_label += 1
            stack = [(start_r, start_c)]
            labels[start_r, start_c] = next_label
            while stack:
                r, c = stack.pop()
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and changed_mask[nr, nc]
                        and labels[nr, nc] == 0
                    ):
                        labels[nr, nc] = next_label
                        stack.append((nr, nc))
    return labels


def pixel_diff_baseline(
    frame_before: np.ndarray,
    frame_after: np.ndarray,
    band_masks: Mapping[str, np.ndarray],
) -> SimpleSystemOutput:
    diff = np.abs(frame_after.astype(np.int16) - frame_before.astype(np.int16))
    changed_mask = diff >= PIXEL_THRESHOLD
    labels = _connected_components(changed_mask)
    counts = np.bincount(labels.reshape(-1))
    keep_mask = np.zeros_like(changed_mask)
    for label_id in range(1, len(counts)):
        if counts[label_id] >= MIN_COMPONENT_SIZE:
            keep_mask |= labels == label_id

    predicted = tuple(
        sorted(name for name, mask in band_masks.items() if keep_mask[mask].any())
    )
    return SimpleSystemOutput(
        predicted_components=predicted,
        missing_components=(),  # no expectation model, same limitation as stage 1's System A
        unexpected_components=(),
        predicted_region_mask=keep_mask,
    )


def privileged_baseline(transition: DomainTransition) -> SimpleSystemOutput:
    observed = set(transition.observed_changed_components)
    expected = set(transition.expected_changed_components)
    return SimpleSystemOutput(
        predicted_components=tuple(sorted(observed)),
        missing_components=tuple(sorted(expected - observed)),
        unexpected_components=tuple(sorted(observed - expected)),
        predicted_region_mask=np.zeros(transition.frame_before.shape, dtype=bool),
    )


def declared_band_masks_arcade() -> Dict[str, np.ndarray]:
    from visual_transition_benchmark import zeromodel_adapter as zm

    return dict(zm.BAND_MASKS)


def declared_band_masks_warehouse() -> Dict[str, np.ndarray]:
    from visual_transition_benchmark.domains.warehouse import model as wm
    from visual_transition_benchmark.domains.warehouse import rendering as wr

    height, width = wr.CANVAS_HEIGHT, wr.CANVAS_WIDTH
    interior = np.zeros((height, width), dtype=bool)
    for row in range(1, wm.GRID_SIZE - 1):
        for col in range(1, wm.GRID_SIZE - 1):
            y0, x0 = wr.cell_origin(row, col)
            interior[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS] = True

    door_mask = np.zeros((height, width), dtype=bool)
    door_y0, door_x0 = wr.cell_origin(*wm.DOOR_POSITION)
    door_mask[
        door_y0 : door_y0 + wr.CELL_PIXELS, door_x0 : door_x0 + wr.CELL_PIXELS
    ] = True

    wall_mask = np.zeros((height, width), dtype=bool)
    for row in range(wm.GRID_SIZE):
        for col in range(wm.GRID_SIZE):
            if row in (0, wm.GRID_SIZE - 1) or col in (0, wm.GRID_SIZE - 1):
                y0, x0 = wr.cell_origin(row, col)
                wall_mask[y0 : y0 + wr.CELL_PIXELS, x0 : x0 + wr.CELL_PIXELS] = True

    battery_mask = np.zeros((height, width), dtype=bool)
    battery_mask[wm.GRID_SIZE * wr.CELL_PIXELS :, :] = True

    background_mask = ~(interior | door_mask | wall_mask | battery_mask)

    # A naive, domain-unaware pixel-diff operator only knows "where these
    # objects could ever appear" -- it cannot resolve robot vs. crate by
    # region alone, since both share the full interior. This is intentional,
    # not a bug: it is exactly the point being measured.
    return {
        "robot": interior,
        "crate": interior,
        "door": door_mask,
        "battery": battery_mask,
        "wall": wall_mask,
        "background": background_mask,
    }
