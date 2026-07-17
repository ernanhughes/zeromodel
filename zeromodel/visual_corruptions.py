"""Deterministic uint8 image transformations for visual-address benchmarks."""
from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError


def canonical_uint8_frame(frame: Any) -> np.ndarray:
    array = np.asarray(frame)
    if array.dtype != np.uint8:
        raise VPMValidationError("visual corruption inputs must use uint8 samples")
    if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in {3, 4})):
        raise VPMValidationError("visual corruption inputs must be HxW or HxWx3/4")
    if array.size == 0:
        raise VPMValidationError("visual corruption inputs cannot be empty")
    result = np.array(array, dtype=np.uint8, order="C", copy=True)
    result.flags.writeable = False
    return result


def _owned(frame: Any) -> np.ndarray:
    result = np.array(canonical_uint8_frame(frame), dtype=np.uint8, order="C", copy=True)
    result.flags.writeable = True
    return result


def scale_intensity(
    frame: Any,
    *,
    numerator: int,
    denominator: int = 100,
    offset: int = 0,
) -> np.ndarray:
    if denominator <= 0 or numerator < 0:
        raise VPMValidationError("intensity scale requires non-negative numerator and positive denominator")
    values = _owned(frame).astype(np.int32)
    values = (values * int(numerator) + int(denominator) // 2) // int(denominator)
    values = np.clip(values + int(offset), 0, 255).astype(np.uint8)
    values.flags.writeable = False
    return values


def translate_frame(frame: Any, *, dx: int = 0, dy: int = 0, fill: int = 0) -> np.ndarray:
    source = canonical_uint8_frame(frame)
    if not (0 <= int(fill) <= 255):
        raise VPMValidationError("translation fill must be in [0, 255]")
    height, width = source.shape[:2]
    if abs(int(dx)) >= width or abs(int(dy)) >= height:
        raise VPMValidationError("translation must leave some source pixels visible")
    result = np.full(source.shape, int(fill), dtype=np.uint8)
    src_y0 = max(0, -int(dy))
    src_y1 = min(height, height - int(dy))
    dst_y0 = max(0, int(dy))
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    src_x0 = max(0, -int(dx))
    src_x1 = min(width, width - int(dx))
    dst_x0 = max(0, int(dx))
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    result[dst_y0:dst_y1, dst_x0:dst_x1] = source[src_y0:src_y1, src_x0:src_x1]
    result.flags.writeable = False
    return result


def remap_levels(frame: Any, mapping: Mapping[int, int]) -> np.ndarray:
    result = _owned(frame)
    for source, target in sorted((int(key), int(value)) for key, value in mapping.items()):
        if not (0 <= source <= 255 and 0 <= target <= 255):
            raise VPMValidationError("level remapping values must be in [0, 255]")
        result[result == source] = target
    result.flags.writeable = False
    return result


def add_integer_noise(frame: Any, *, amplitude: int, seed: int) -> np.ndarray:
    if not (0 <= int(amplitude) <= 255):
        raise VPMValidationError("noise amplitude must be in [0, 255]")
    source = canonical_uint8_frame(frame).astype(np.int16)
    generator = np.random.default_rng(int(seed))
    noise = generator.integers(
        -int(amplitude),
        int(amplitude) + 1,
        size=source.shape,
        dtype=np.int16,
    )
    result = np.clip(source + noise, 0, 255).astype(np.uint8)
    result.flags.writeable = False
    return result


def mask_box(
    frame: Any,
    *,
    top: int,
    left: int,
    height: int,
    width: int,
    value: int = 0,
) -> np.ndarray:
    """Mask the visible intersection of a declared box and the frame."""

    result = _owned(frame)
    frame_height, frame_width = result.shape[:2]
    if height <= 0 or width <= 0:
        raise VPMValidationError("mask dimensions must be positive")
    if not (0 <= top < frame_height and 0 <= left < frame_width):
        raise VPMValidationError("mask origin must be inside the frame")
    if not (0 <= int(value) <= 255):
        raise VPMValidationError("mask value must be in [0, 255]")
    bottom = min(frame_height, int(top) + int(height))
    right = min(frame_width, int(left) + int(width))
    result[int(top) : bottom, int(left) : right] = int(value)
    result.flags.writeable = False
    return result


def overlay_background_patch(
    frame: Any,
    *,
    height: int = 3,
    width: int = 3,
    value: int = 96,
    candidate_order: Sequence[Tuple[int, int]] = (),
) -> np.ndarray:
    """Overlay one deterministic patch only where the source is all zero.

    The operation is useful for non-critical corruption fixtures. It refuses to
    cover existing non-zero evidence rather than guessing which pixels matter.
    """

    source = canonical_uint8_frame(frame)
    frame_height, frame_width = source.shape[:2]
    if height <= 0 or width <= 0 or height > frame_height or width > frame_width:
        raise VPMValidationError("background patch dimensions must fit the frame")
    if not (0 <= int(value) <= 255):
        raise VPMValidationError("background patch value must be in [0, 255]")
    candidates = tuple(candidate_order) or tuple(
        (top, left)
        for top in range(frame_height - height + 1)
        for left in range(frame_width - width + 1)
    )
    for top, left in candidates:
        region = source[top : top + height, left : left + width]
        if region.shape[:2] == (height, width) and not np.any(region):
            return mask_box(
                source,
                top=int(top),
                left=int(left),
                height=height,
                width=width,
                value=int(value),
            )
    raise VPMValidationError("no all-zero background patch fits the requested dimensions")


def checkerboard_frame(
    *,
    height: int,
    width: int,
    low: int = 32,
    high: int = 224,
    cell: int = 2,
) -> np.ndarray:
    if height <= 0 or width <= 0 or cell <= 0:
        raise VPMValidationError("checkerboard dimensions and cell must be positive")
    if not (0 <= low <= 255 and 0 <= high <= 255):
        raise VPMValidationError("checkerboard values must be in [0, 255]")
    rows = np.arange(height)[:, None] // int(cell)
    columns = np.arange(width)[None, :] // int(cell)
    mask = (rows + columns) % 2
    result = np.where(mask == 0, int(low), int(high)).astype(np.uint8)
    result.flags.writeable = False
    return result
