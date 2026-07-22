from __future__ import annotations

import numpy as np
import pytest

from zeromodel.core.artifact import VPMValidationError
from research.visual.visual_registration import RegistrationConfig, register_integer_translation


def _frame() -> np.ndarray:
    return np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 255, 0, 0],
            [0, 255, 255, 255, 0],
            [0, 0, 255, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )


def _shift(frame: np.ndarray, *, dx: int = 0, dy: int = 0) -> np.ndarray:
    result = np.zeros_like(frame)
    height, width = frame.shape
    x0 = max(0, dx)
    x1 = min(width, width + dx)
    y0 = max(0, dy)
    y1 = min(height, height + dy)
    result[y0:y1, x0:x1] = frame[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return result


def test_zero_displacement_round_trips() -> None:
    config = RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6)
    result = register_integer_translation(_frame(), _frame(), config=config)
    assert (result.dx, result.dy) == (0, 0)
    assert result.distance_after == pytest.approx(0.0)
    assert result.registration_succeeded is True


@pytest.mark.parametrize(
    ("dx", "dy"),
    [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-2, 0)],
)
def test_integer_translation_recovery(dx: int, dy: int) -> None:
    frame = _frame()
    shifted = _shift(frame, dx=dx, dy=dy)
    config = RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6)
    result = register_integer_translation(frame, shifted, config=config)
    assert (result.dx, result.dy) == (-dx, -dy)
    assert result.distance_after <= result.distance_before + 1e-12


def test_shift_outside_configured_range_does_not_recover_exact_displacement() -> None:
    frame = _frame()
    shifted = _shift(frame, dx=4, dy=0)
    config = RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6)
    result = register_integer_translation(frame, shifted, config=config)
    assert (result.dx, result.dy) != (4, 0)


def test_no_wraparound_from_left_edge() -> None:
    frame = np.zeros((4, 4), dtype=np.uint8)
    frame[1, 0] = 255
    shifted = _shift(frame, dx=1, dy=0)
    config = RegistrationConfig(max_dx=1, max_dy=0, minimum_overlap_fraction=0.5)
    result = register_integer_translation(frame, shifted, config=config)
    assert result.dx == -1
    assert result.distance_after < result.distance_before


def test_minimum_overlap_constraint_keeps_overlap_fraction_explicit() -> None:
    config = RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.95)
    result = register_integer_translation(_frame(), _shift(_frame(), dx=3), config=config)
    assert result.overlap_fraction >= 0.95


def test_constant_value_images_stay_finite() -> None:
    frame = np.full((4, 4), 7, dtype=np.uint8)
    config = RegistrationConfig(max_dx=1, max_dy=1, minimum_overlap_fraction=0.5)
    result = register_integer_translation(frame, frame, config=config)
    assert np.isfinite(result.distance_after)
    assert result.distance_after == pytest.approx(0.0)


def test_shape_mismatch_is_rejected() -> None:
    config = RegistrationConfig(max_dx=1, max_dy=1, minimum_overlap_fraction=0.5)
    with pytest.raises(VPMValidationError, match="identical shape"):
        register_integer_translation(np.zeros((4, 4), dtype=np.uint8), np.zeros((4, 5), dtype=np.uint8), config=config)


def test_input_arrays_remain_immutable_to_caller() -> None:
    left = _frame()
    right = _shift(left, dx=1)
    left_before = left.copy()
    right_before = right.copy()
    config = RegistrationConfig(max_dx=2, max_dy=2, minimum_overlap_fraction=0.5)
    _ = register_integer_translation(left, right, config=config)
    assert np.array_equal(left, left_before)
    assert np.array_equal(right, right_before)


def test_deterministic_displacement_tie_break_prefers_smallest_manhattan() -> None:
    frame = np.zeros((4, 4), dtype=np.uint8)
    config = RegistrationConfig(max_dx=1, max_dy=1, minimum_overlap_fraction=0.5)
    result = register_integer_translation(frame, frame, config=config)
    assert (result.dx, result.dy) == (0, 0)


def test_stage2_registration_keeps_tiny_region_overlap_pathology_unchanged() -> None:
    prototype = np.array(
        [
            [255, 0],
            [255, 0],
        ],
        dtype=np.uint8,
    )
    observation = np.array(
        [
            [0, 255],
            [0, 255],
        ],
        dtype=np.uint8,
    )
    config = RegistrationConfig(max_dx=1, max_dy=0, minimum_overlap_fraction=0.5)
    result = register_integer_translation(prototype, observation, config=config)
    assert (result.dx, result.dy) == (-1, 0)
    assert result.distance_after == pytest.approx(0.0)
    assert result.overlap_fraction == pytest.approx(0.5)


def test_config_digest_is_stable() -> None:
    left = RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6)
    right = RegistrationConfig(max_dx=3, max_dy=3, minimum_overlap_fraction=0.6)
    assert left.to_dict() == right.to_dict()
    assert left.digest == right.digest
