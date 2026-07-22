"""Deterministic integer-translation registration for local visual baselines."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np

from zeromodel.core.artifact import VPMValidationError


REGISTRATION_CONFIG_VERSION = "zeromodel-registration-config/v1"
REGISTRATION_RESULT_VERSION = "zeromodel-registration-result/v1"


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("registration values must be JSON-serializable") from exc


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _grayscale(image: Any) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        raise VPMValidationError("registration inputs must be uint8")
    if array.ndim == 2:
        gray = array.astype(np.float32)
    elif array.ndim == 3 and array.shape[2] in {3, 4}:
        rgb = array[:, :, :3].astype(np.float32)
        gray = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    else:
        raise VPMValidationError("registration inputs must be HxW or HxWx3/4")
    owned = np.ascontiguousarray(gray, dtype=np.float32)
    owned /= 255.0
    owned.flags.writeable = False
    return owned


def _normalized_overlap_distance(
    prototype: np.ndarray,
    observation: np.ndarray,
    *,
    dx: int,
    dy: int,
) -> Tuple[float, float, int]:
    height, width = prototype.shape
    x0 = max(0, dx)
    x1 = min(width, width + dx)
    y0 = max(0, dy)
    y1 = min(height, height + dy)
    valid_width = x1 - x0
    valid_height = y1 - y0
    if valid_width <= 0 or valid_height <= 0:
        return (float("inf"), 0.0, 0)

    prototype_region = prototype[y0:y1, x0:x1]
    observation_region = observation[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    valid_count = int(valid_width * valid_height)
    overlap_fraction = float(valid_count) / float(height * width)

    left = prototype_region.astype(np.float32, copy=False)
    right = observation_region.astype(np.float32, copy=False)
    pixel_count = float(valid_count)
    left_sum = float(left.sum(dtype=np.float64))
    right_sum = float(right.sum(dtype=np.float64))
    left_sum_sq = float(np.multiply(left, left, dtype=np.float32).sum(dtype=np.float64))
    right_sum_sq = float(np.multiply(right, right, dtype=np.float32).sum(dtype=np.float64))
    cross_sum = float(np.multiply(left, right, dtype=np.float32).sum(dtype=np.float64))

    left_centered_sum_sq = max(0.0, left_sum_sq - (left_sum * left_sum) / pixel_count)
    right_centered_sum_sq = max(0.0, right_sum_sq - (right_sum * right_sum) / pixel_count)
    if left_centered_sum_sq <= 1e-12 and right_centered_sum_sq <= 1e-12:
        distance = 0.0
    elif left_centered_sum_sq <= 1e-12 or right_centered_sum_sq <= 1e-12:
        distance = 1.0
    else:
        centered_cross_sum = cross_sum - (left_sum * right_sum) / pixel_count
        cosine = centered_cross_sum / np.sqrt(left_centered_sum_sq * right_centered_sum_sq)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        distance = float(np.sqrt(max(0.0, 2.0 - (2.0 * cosine))))
    return (distance, overlap_fraction, valid_count)


def _displacement_order(dx: int, dy: int) -> Tuple[int, int, int, int, int]:
    return (abs(dx) + abs(dy), abs(dy), abs(dx), int(dy), int(dx))


@dataclass(frozen=True)
class RegistrationConfig:
    max_dx: int
    max_dy: int
    minimum_overlap_fraction: float
    metric: str = "normalized_l2"
    padding_value: int = 0
    version: str = REGISTRATION_CONFIG_VERSION

    def __post_init__(self) -> None:
        if self.version != REGISTRATION_CONFIG_VERSION:
            raise VPMValidationError("unsupported registration config version")
        if int(self.max_dx) < 0 or int(self.max_dy) < 0:
            raise VPMValidationError("registration bounds must be non-negative")
        if not np.isfinite(float(self.minimum_overlap_fraction)) or not (
            0.0 < float(self.minimum_overlap_fraction) <= 1.0
        ):
            raise VPMValidationError("minimum_overlap_fraction must be in (0, 1]")
        if self.metric != "normalized_l2":
            raise VPMValidationError("registration metric must be normalized_l2")
        if int(self.padding_value) != 0:
            raise VPMValidationError("padding_value must remain zero for invalid-region exclusion")

    @property
    def digest(self) -> str:
        return _sha256_json(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "max_dx": int(self.max_dx),
            "max_dy": int(self.max_dy),
            "minimum_overlap_fraction": float(self.minimum_overlap_fraction),
            "metric": self.metric,
            "padding_value": int(self.padding_value),
        }


@dataclass(frozen=True)
class RegistrationResult:
    dx: int
    dy: int
    distance_before: float
    distance_after: float
    distance_improvement: float
    overlap_fraction: float
    valid_pixel_count: int
    score_before: float
    score_after: float
    registration_succeeded: bool
    rejection_reason: Optional[str] = None
    version: str = REGISTRATION_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.version != REGISTRATION_RESULT_VERSION:
            raise VPMValidationError("unsupported registration result version")
        for name in (
            "distance_before",
            "distance_after",
            "distance_improvement",
            "overlap_fraction",
            "score_before",
            "score_after",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise VPMValidationError("%s must be finite" % name)
        if self.valid_pixel_count < 0:
            raise VPMValidationError("valid_pixel_count cannot be negative")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "dx": int(self.dx),
            "dy": int(self.dy),
            "distance_before": float(self.distance_before),
            "distance_after": float(self.distance_after),
            "distance_improvement": float(self.distance_improvement),
            "overlap_fraction": float(self.overlap_fraction),
            "valid_pixel_count": int(self.valid_pixel_count),
            "score_before": float(self.score_before),
            "score_after": float(self.score_after),
            "registration_succeeded": bool(self.registration_succeeded),
            "rejection_reason": self.rejection_reason,
        }


def register_integer_translation(
    prototype: np.ndarray,
    observation: np.ndarray,
    *,
    config: RegistrationConfig,
) -> RegistrationResult:
    left = _grayscale(prototype)
    right = _grayscale(observation)
    if left.shape != right.shape:
        raise VPMValidationError("registration inputs must have identical shape")

    distance_before, overlap_before, valid_before = _normalized_overlap_distance(
        left,
        right,
        dx=0,
        dy=0,
    )
    best: Optional[Tuple[float, Tuple[int, int, int, int, int], int, int, float]] = None
    for dy in range(-int(config.max_dy), int(config.max_dy) + 1):
        for dx in range(-int(config.max_dx), int(config.max_dx) + 1):
            distance, overlap_fraction, valid_count = _normalized_overlap_distance(
                left,
                right,
                dx=dx,
                dy=dy,
            )
            if overlap_fraction + 1e-12 < float(config.minimum_overlap_fraction):
                continue
            candidate = (
                float(distance),
                _displacement_order(dx, dy),
                int(dx),
                int(dy),
                float(overlap_fraction),
            )
            if best is None or candidate < best:
                best = candidate
                best_valid = int(valid_count)
    if best is None:
        return RegistrationResult(
            dx=0,
            dy=0,
            distance_before=float(distance_before),
            distance_after=float(distance_before),
            distance_improvement=0.0,
            overlap_fraction=float(overlap_before),
            valid_pixel_count=int(valid_before),
            score_before=-float(distance_before),
            score_after=-float(distance_before),
            registration_succeeded=False,
            rejection_reason="insufficient_overlap",
        )

    best_distance, _order, best_dx, best_dy, best_overlap = best
    return RegistrationResult(
        dx=int(best_dx),
        dy=int(best_dy),
        distance_before=float(distance_before),
        distance_after=float(best_distance),
        distance_improvement=float(distance_before - best_distance),
        overlap_fraction=float(best_overlap),
        valid_pixel_count=int(best_valid),
        score_before=-float(distance_before),
        score_after=-float(best_distance),
        registration_succeeded=True,
        rejection_reason=None,
    )
