"""Stage 3 discriminative current-frame evidence contracts and mechanics."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError
from .visual_address import ImageObservation
from .visual_address import VisualAddressContract
from .visual_registration import RegistrationConfig


VIDEO_DISCRIMINATIVE_REGION_SPEC_VERSION = "zeromodel-video-discriminative-region-spec/v1"
VIDEO_DISCRIMINATIVE_MASK_SPEC_VERSION = "zeromodel-video-discriminative-mask-spec/v1"
VIDEO_DISCRIMINATIVE_CALIBRATION_VERSION = "zeromodel-video-discriminative-calibration/v1"
VIDEO_DISCRIMINATIVE_CANDIDATE_SET_VERSION = "zeromodel-video-discriminative-candidate-set/v1"
VIDEO_DISCRIMINATIVE_PROVIDER_VERSION = "zeromodel-video-discriminative-provider/v1"
VIDEO_DISCRIMINATIVE_ARCHITECTURE_SELECTION_VERSION = "zeromodel-video-discriminative-architecture-selection/v1"
VIDEO_DISCRIMINATIVE_OPERATING_POINT_SELECTION_VERSION = "zeromodel-video-discriminative-operating-point-selection/v1"
VIDEO_DISCRIMINATIVE_MASK_PAYLOAD_VERSION = "zeromodel-video-discriminative-mask-payload/v1"
VIDEO_DISCRIMINATIVE_REGISTRATION_VERSION = "zeromodel-video-discriminative-registration/v1"
VIDEO_DISCRIMINATIVE_EVIDENCE_MECHANICS_VERSION = "zeromodel-video-discriminative-evidence-mechanics/v1"

REGISTRATION_DISTANCE_TIE_EPSILON = 1e-12

_ARCHITECTURES = {"A", "B", "C", "D"}
_OUTCOMES = {"exact_row_accepted", "candidate_set_available", "no_sufficient_evidence"}


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("discriminative-evidence values must be JSON-serializable") from exc


def _json_digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _freeze(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _array_descriptor(array: np.ndarray) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "shape": list(contiguous.shape),
        "dtype": str(contiguous.dtype),
        "bytes": hashlib.sha256(contiguous.view(np.uint8).tobytes()).hexdigest(),
    }


def _coerce_observation_pixels(value: Any) -> np.ndarray:
    if isinstance(value, ImageObservation):
        pixels = value.pixels
    else:
        pixels = value
    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        raise VPMValidationError("discriminative evidence inputs must be uint8")
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.ndim == 3 and array.shape[2] in {3, 4}:
        rgb = array[:, :, :3].astype(np.float32)
        gray = np.round(0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.uint8)
        return np.ascontiguousarray(gray)
    raise VPMValidationError("discriminative evidence inputs must be HxW or HxWx3/4")


def _immutable_weight_array(array: Any, *, shape: Tuple[int, int], name: str, clip: bool = False) -> np.ndarray:
    owned = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    if owned.shape != shape:
        raise VPMValidationError(f"{name} must match mask shape")
    if owned.ndim != 2:
        raise VPMValidationError(f"{name} must be two-dimensional")
    if not np.isfinite(owned).all():
        raise VPMValidationError(f"{name} must be finite")
    if (owned < 0.0).any():
        raise VPMValidationError(f"{name} must be non-negative")
    if clip:
        owned = np.clip(owned, 0.0, 1.0)
    owned.flags.writeable = False
    return owned


def _normalized_gray(image: Any) -> np.ndarray:
    array = _coerce_observation_pixels(image).astype(np.float32)
    array /= 255.0
    array.flags.writeable = False
    return array


def _crop(array: np.ndarray, *, top: int, left: int, height: int, width: int) -> np.ndarray:
    return np.ascontiguousarray(array[top : top + height, left : left + width])


def _aligned_bounds(shape: Tuple[int, int], *, dx: int, dy: int) -> Tuple[int, int, int, int]:
    height, width = shape
    x0 = max(0, dx)
    x1 = min(width, width + dx)
    y0 = max(0, dy)
    y1 = min(height, height + dy)
    return (x0, x1, y0, y1)


def _aligned_slices(array: np.ndarray, *, dx: int, dy: int, observation: bool) -> np.ndarray:
    x0, x1, y0, y1 = _aligned_bounds(array.shape, dx=dx, dy=dy)
    if observation:
        return array[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    return array[y0:y1, x0:x1]


def _normalized_overlap_distance(prototype: np.ndarray, observation: np.ndarray, *, dx: int, dy: int) -> Tuple[float, float, int]:
    x0, x1, y0, y1 = _aligned_bounds(prototype.shape, dx=dx, dy=dy)
    valid_width = x1 - x0
    valid_height = y1 - y0
    if valid_width <= 0 or valid_height <= 0:
        return (float("inf"), 0.0, 0)
    prototype_region = prototype[y0:y1, x0:x1]
    observation_region = observation[y0 - dy : y1 - dy, x0 - dx : x1 - dx]
    valid_count = int(valid_width * valid_height)
    overlap_fraction = float(valid_count) / float(prototype.shape[0] * prototype.shape[1])

    pixel_count = float(valid_count)
    left_sum = float(prototype_region.sum(dtype=np.float64))
    right_sum = float(observation_region.sum(dtype=np.float64))
    left_sum_sq = float(np.multiply(prototype_region, prototype_region, dtype=np.float32).sum(dtype=np.float64))
    right_sum_sq = float(np.multiply(observation_region, observation_region, dtype=np.float32).sum(dtype=np.float64))
    cross_sum = float(np.multiply(prototype_region, observation_region, dtype=np.float32).sum(dtype=np.float64))

    left_centered_sum_sq = max(0.0, left_sum_sq - (left_sum * left_sum) / pixel_count)
    right_centered_sum_sq = max(0.0, right_sum_sq - (right_sum * right_sum) / pixel_count)
    if left_centered_sum_sq <= REGISTRATION_DISTANCE_TIE_EPSILON and right_centered_sum_sq <= REGISTRATION_DISTANCE_TIE_EPSILON:
        distance = 0.0
    elif left_centered_sum_sq <= REGISTRATION_DISTANCE_TIE_EPSILON or right_centered_sum_sq <= REGISTRATION_DISTANCE_TIE_EPSILON:
        distance = 1.0
    else:
        centered_cross_sum = cross_sum - (left_sum * right_sum) / pixel_count
        cosine = centered_cross_sum / np.sqrt(left_centered_sum_sq * right_centered_sum_sq)
        cosine = float(np.clip(cosine, -1.0, 1.0))
        distance = float(np.sqrt(max(0.0, 2.0 - (2.0 * cosine))))
    return (distance, overlap_fraction, valid_count)


def _signed_translation_order(dx: int, dy: int) -> Tuple[int, int]:
    return (int(dy), int(dx))


def _registration_contract_digest(config: RegistrationConfig) -> str:
    return _json_digest(
        {
            "version": VIDEO_DISCRIMINATIVE_REGISTRATION_VERSION,
            "tie_epsilon": REGISTRATION_DISTANCE_TIE_EPSILON,
            "signed_offset_ordering": "ascending(dy,dx)",
            "registration_config": config.to_dict(),
        }
    )


def _difference_uint8(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
    return delta.astype(np.int16, copy=False)


def _prototype_payload_digest(prototypes: Mapping[str, Tuple[str, str, str, Any]]) -> str:
    entries = []
    for observation_id, (row_id, action_id, digest, observation) in sorted(prototypes.items()):
        pixels = _coerce_observation_pixels(observation)
        entries.append(
            {
                "observation_id": observation_id,
                "row_id": row_id,
                "action_id": action_id,
                "prototype_digest": digest,
                "pixels": _array_descriptor(pixels),
            }
        )
    return _json_digest(entries)


def _development_payload_digest(development_observations: Mapping[str, Sequence[Any]]) -> str:
    entries = []
    for row_id, observations in sorted(development_observations.items()):
        payloads = []
        for item in observations:
            pixels = _coerce_observation_pixels(item)
            payloads.append(_array_descriptor(pixels))
        entries.append({"row_id": row_id, "observations": payloads})
    return _json_digest(entries or [{"row_id": "__none__", "observations": []}])


@dataclass(frozen=True)
class DiscriminativeRegionSpec:
    region_id: str
    top: int
    left: int
    height: int
    width: int
    weight: float
    critical: bool
    registration_config: RegistrationConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_DISCRIMINATIVE_REGION_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_DISCRIMINATIVE_REGION_SPEC_VERSION:
            raise VPMValidationError("unsupported discriminative region spec version")
        if not str(self.region_id):
            raise VPMValidationError("region_id cannot be empty")
        if int(self.top) < 0 or int(self.left) < 0:
            raise VPMValidationError("region origin must be non-negative")
        if int(self.height) <= 0 or int(self.width) <= 0:
            raise VPMValidationError("region dimensions must be positive")
        if not np.isfinite(float(self.weight)) or float(self.weight) <= 0.0:
            raise VPMValidationError("region weight must be finite and positive")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        _json_bytes(self.metadata)

    @property
    def digest(self) -> str:
        return _json_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "region_id": self.region_id,
            "top": int(self.top),
            "left": int(self.left),
            "height": int(self.height),
            "width": int(self.width),
            "weight": float(self.weight),
            "critical": bool(self.critical),
            "registration_config": self.registration_config.to_dict(),
            "metadata": _freeze(self.metadata),
        }


@dataclass(frozen=True)
class DiscriminativeMaskSpec:
    mask_id: str
    row_id: str
    action_id: str
    shape: Tuple[int, int]
    informative_pixel_count: int
    action_conflict_pixel_count: int
    stable_pixel_count: int
    prototype_digest: str
    development_digest: str
    derivation_contract: str
    intensity_tolerance: int
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_DISCRIMINATIVE_MASK_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_DISCRIMINATIVE_MASK_SPEC_VERSION:
            raise VPMValidationError("unsupported discriminative mask spec version")
        for name in ("mask_id", "row_id", "action_id", "prototype_digest", "development_digest", "derivation_contract"):
            if not str(getattr(self, name)):
                raise VPMValidationError(f"{name} cannot be empty")
        if len(tuple(self.shape)) != 2:
            raise VPMValidationError("mask shape must be two-dimensional")
        if any(int(item) <= 0 for item in self.shape):
            raise VPMValidationError("mask shape dimensions must be positive")
        for name in ("informative_pixel_count", "action_conflict_pixel_count", "stable_pixel_count", "intensity_tolerance"):
            if int(getattr(self, name)) < 0:
                raise VPMValidationError(f"{name} must be non-negative")
        object.__setattr__(self, "shape", (int(self.shape[0]), int(self.shape[1])))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        _json_bytes(self.metadata)

    @property
    def digest(self) -> str:
        return _json_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mask_id": self.mask_id,
            "row_id": self.row_id,
            "action_id": self.action_id,
            "shape": list(self.shape),
            "informative_pixel_count": int(self.informative_pixel_count),
            "action_conflict_pixel_count": int(self.action_conflict_pixel_count),
            "stable_pixel_count": int(self.stable_pixel_count),
            "prototype_digest": self.prototype_digest,
            "development_digest": self.development_digest,
            "derivation_contract": self.derivation_contract,
            "intensity_tolerance": int(self.intensity_tolerance),
            "metadata": _freeze(self.metadata),
        }


@dataclass(frozen=True)
class DiscriminativeMask:
    spec: DiscriminativeMaskSpec
    row_informative_weights: np.ndarray
    action_conflict_weights: np.ndarray
    stable_weights: np.ndarray
    separation_weights: np.ndarray
    payload_digest: str = ""
    version: str = VIDEO_DISCRIMINATIVE_MASK_PAYLOAD_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_DISCRIMINATIVE_MASK_PAYLOAD_VERSION:
            raise VPMValidationError("unsupported discriminative mask payload version")
        shape = tuple(self.spec.shape)
        row_informative = _immutable_weight_array(self.row_informative_weights, shape=shape, name="row_informative_weights")
        action_conflict = _immutable_weight_array(self.action_conflict_weights, shape=shape, name="action_conflict_weights")
        stable = _immutable_weight_array(self.stable_weights, shape=shape, name="stable_weights")
        separation = _immutable_weight_array(self.separation_weights, shape=shape, name="separation_weights", clip=True)
        if int(np.count_nonzero(row_informative > 0.0)) != int(self.spec.informative_pixel_count):
            raise VPMValidationError("row_informative_weights count must match spec")
        if int(np.count_nonzero(action_conflict > 0.0)) != int(self.spec.action_conflict_pixel_count):
            raise VPMValidationError("action_conflict_weights count must match spec")
        if int(np.count_nonzero(stable > 0.0)) != int(self.spec.stable_pixel_count):
            raise VPMValidationError("stable_weights count must match spec")
        object.__setattr__(self, "row_informative_weights", row_informative)
        object.__setattr__(self, "action_conflict_weights", action_conflict)
        object.__setattr__(self, "stable_weights", stable)
        object.__setattr__(self, "separation_weights", separation)
        digest = self.payload_digest or _json_digest(
            {
                "version": self.version,
                "spec_digest": self.spec.digest,
                "row_informative_weights": _array_descriptor(row_informative),
                "action_conflict_weights": _array_descriptor(action_conflict),
                "stable_weights": _array_descriptor(stable),
                "separation_weights": _array_descriptor(separation),
            }
        )
        object.__setattr__(self, "payload_digest", digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "spec": self.spec.to_dict(),
            "payload_digest": self.payload_digest,
            "row_informative_weights": _array_descriptor(self.row_informative_weights),
            "action_conflict_weights": _array_descriptor(self.action_conflict_weights),
            "stable_weights": _array_descriptor(self.stable_weights),
            "separation_weights": _array_descriptor(self.separation_weights),
        }


@dataclass(frozen=True)
class DiscriminativeEvidenceCalibration:
    architecture_id: str
    minimum_available_mass: float
    minimum_available_fraction: float
    minimum_support: float
    maximum_contradiction: float
    maximum_critical_contradiction: float
    exact_winner_threshold: float
    exact_winner_margin: float
    candidate_relative_margin: float
    conflicting_action_separation: float
    minimum_supporting_regions: int
    maximum_candidate_set_size: int
    prototype_digest: str
    region_spec_digest: str
    mask_spec_digest: str
    policy_artifact_id: str
    source_scope: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_DISCRIMINATIVE_CALIBRATION_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_DISCRIMINATIVE_CALIBRATION_VERSION:
            raise VPMValidationError("unsupported discriminative calibration version")
        if self.architecture_id not in _ARCHITECTURES:
            raise VPMValidationError("unsupported architecture_id")
        for name in (
            "minimum_available_mass",
            "minimum_available_fraction",
            "minimum_support",
            "maximum_contradiction",
            "maximum_critical_contradiction",
            "exact_winner_threshold",
            "exact_winner_margin",
            "candidate_relative_margin",
            "conflicting_action_separation",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise VPMValidationError(f"{name} must be finite")
            if value < 0.0:
                raise VPMValidationError(f"{name} must be non-negative")
        if not (0.0 <= float(self.minimum_available_fraction) <= 1.0):
            raise VPMValidationError("minimum_available_fraction must be in [0, 1]")
        if int(self.minimum_supporting_regions) < 0:
            raise VPMValidationError("minimum_supporting_regions must be non-negative")
        if int(self.maximum_candidate_set_size) < 1:
            raise VPMValidationError("maximum_candidate_set_size must be positive")
        for name in ("prototype_digest", "region_spec_digest", "mask_spec_digest", "policy_artifact_id", "source_scope"):
            if not str(getattr(self, name)):
                raise VPMValidationError(f"{name} cannot be empty")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        _json_bytes(self.metadata)

    @property
    def digest(self) -> str:
        return _json_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "architecture_id": self.architecture_id,
            "minimum_available_mass": float(self.minimum_available_mass),
            "minimum_available_fraction": float(self.minimum_available_fraction),
            "minimum_support": float(self.minimum_support),
            "maximum_contradiction": float(self.maximum_contradiction),
            "maximum_critical_contradiction": float(self.maximum_critical_contradiction),
            "exact_winner_threshold": float(self.exact_winner_threshold),
            "exact_winner_margin": float(self.exact_winner_margin),
            "candidate_relative_margin": float(self.candidate_relative_margin),
            "conflicting_action_separation": float(self.conflicting_action_separation),
            "minimum_supporting_regions": int(self.minimum_supporting_regions),
            "maximum_candidate_set_size": int(self.maximum_candidate_set_size),
            "prototype_digest": self.prototype_digest,
            "region_spec_digest": self.region_spec_digest,
            "mask_spec_digest": self.mask_spec_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "source_scope": self.source_scope,
            "metadata": _freeze(self.metadata),
        }


@dataclass(frozen=True)
class RegionDiscriminativeEvidence:
    region_id: str
    expected_informative_mass: float
    available_informative_mass: float
    available_informative_fraction: float
    geometric_overlap: float
    valid_pixel_count: int
    support_mass: float
    contradiction_mass: float
    critical_contradiction_mass: float
    conflicting_action_support_mass: float
    conflicting_action_contradiction_mass: float
    registration_succeeded: bool
    registration_dx: int
    registration_dy: int
    registration_tie_break_reason: str
    rejection_reasons: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not str(self.region_id):
            raise VPMValidationError("region evidence requires region_id")
        for name in (
            "expected_informative_mass",
            "available_informative_mass",
            "available_informative_fraction",
            "geometric_overlap",
            "support_mass",
            "contradiction_mass",
            "critical_contradiction_mass",
            "conflicting_action_support_mass",
            "conflicting_action_contradiction_mass",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise VPMValidationError(f"{name} must be finite")
            if value < 0.0:
                raise VPMValidationError(f"{name} must be non-negative")
        if int(self.valid_pixel_count) < 0:
            raise VPMValidationError("valid_pixel_count must be non-negative")
        if not str(self.registration_tie_break_reason):
            raise VPMValidationError("registration_tie_break_reason cannot be empty")
        if not (0.0 <= float(self.available_informative_fraction) <= 1.0):
            raise VPMValidationError("available_informative_fraction must be in [0, 1]")
        if not (0.0 <= float(self.geometric_overlap) <= 1.0):
            raise VPMValidationError("geometric_overlap must be in [0, 1]")
        object.__setattr__(self, "rejection_reasons", tuple(str(item) for item in self.rejection_reasons))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "region_id": self.region_id,
            "expected_informative_mass": float(self.expected_informative_mass),
            "available_informative_mass": float(self.available_informative_mass),
            "available_informative_fraction": float(self.available_informative_fraction),
            "geometric_overlap": float(self.geometric_overlap),
            "valid_pixel_count": int(self.valid_pixel_count),
            "support_mass": float(self.support_mass),
            "contradiction_mass": float(self.contradiction_mass),
            "critical_contradiction_mass": float(self.critical_contradiction_mass),
            "conflicting_action_support_mass": float(self.conflicting_action_support_mass),
            "conflicting_action_contradiction_mass": float(self.conflicting_action_contradiction_mass),
            "registration_succeeded": bool(self.registration_succeeded),
            "registration_dx": int(self.registration_dx),
            "registration_dy": int(self.registration_dy),
            "registration_tie_break_reason": self.registration_tie_break_reason,
            "rejection_reasons": list(self.rejection_reasons),
        }


@dataclass(frozen=True)
class InformativeRegistrationResult:
    prototype_shape: Tuple[int, int]
    observation_shape: Tuple[int, int]
    region_id: str
    dx: int
    dy: int
    distance: float
    score: float
    valid_pixel_count: int
    geometric_overlap: float
    expected_informative_mass: float
    available_informative_mass: float
    available_informative_fraction: float
    registration_succeeded: bool
    rejection_reason: Optional[str]
    tie_break_reason: str
    evaluated_translation_count: int
    registration_contract_digest: str
    runner_up_dx: Optional[int] = None
    runner_up_dy: Optional[int] = None
    runner_up_available_informative_mass: Optional[float] = None
    runner_up_valid_pixel_count: Optional[int] = None
    version: str = VIDEO_DISCRIMINATIVE_REGISTRATION_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_DISCRIMINATIVE_REGISTRATION_VERSION:
            raise VPMValidationError("unsupported informative registration version")
        for name in ("distance", "score", "geometric_overlap", "expected_informative_mass", "available_informative_mass", "available_informative_fraction"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise VPMValidationError(f"{name} must be finite")
        if int(self.valid_pixel_count) < 0 or int(self.evaluated_translation_count) < 0:
            raise VPMValidationError("registration counts must be non-negative")
        if not (0.0 <= float(self.geometric_overlap) <= 1.0):
            raise VPMValidationError("geometric_overlap must be in [0, 1]")
        if not (0.0 <= float(self.available_informative_fraction) <= 1.0):
            raise VPMValidationError("available_informative_fraction must be in [0, 1]")
        if not str(self.region_id):
            raise VPMValidationError("region_id cannot be empty")
        if not str(self.tie_break_reason):
            raise VPMValidationError("tie_break_reason cannot be empty")
        if not str(self.registration_contract_digest):
            raise VPMValidationError("registration_contract_digest cannot be empty")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "prototype_shape": list(self.prototype_shape),
            "observation_shape": list(self.observation_shape),
            "region_id": self.region_id,
            "dx": int(self.dx),
            "dy": int(self.dy),
            "distance": float(self.distance),
            "score": float(self.score),
            "valid_pixel_count": int(self.valid_pixel_count),
            "geometric_overlap": float(self.geometric_overlap),
            "expected_informative_mass": float(self.expected_informative_mass),
            "available_informative_mass": float(self.available_informative_mass),
            "available_informative_fraction": float(self.available_informative_fraction),
            "registration_succeeded": bool(self.registration_succeeded),
            "rejection_reason": self.rejection_reason,
            "tie_break_reason": self.tie_break_reason,
            "evaluated_translation_count": int(self.evaluated_translation_count),
            "registration_contract_digest": self.registration_contract_digest,
            "runner_up_dx": self.runner_up_dx,
            "runner_up_dy": self.runner_up_dy,
            "runner_up_available_informative_mass": self.runner_up_available_informative_mass,
            "runner_up_valid_pixel_count": self.runner_up_valid_pixel_count,
        }


@dataclass(frozen=True)
class PixelEvidenceTotals:
    expected_informative_mass: float
    available_informative_mass: float
    available_informative_fraction: float
    support_mass: float
    contradiction_mass: float
    critical_contradiction_mass: float
    conflicting_action_support_mass: float
    conflicting_action_contradiction_mass: float

    def __post_init__(self) -> None:
        for name in (
            "expected_informative_mass",
            "available_informative_mass",
            "available_informative_fraction",
            "support_mass",
            "contradiction_mass",
            "critical_contradiction_mass",
            "conflicting_action_support_mass",
            "conflicting_action_contradiction_mass",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise VPMValidationError(f"{name} must be finite")
            if value < 0.0:
                raise VPMValidationError(f"{name} must be non-negative")
        if not (0.0 <= float(self.available_informative_fraction) <= 1.0):
            raise VPMValidationError("available_informative_fraction must be in [0, 1]")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "expected_informative_mass": float(self.expected_informative_mass),
            "available_informative_mass": float(self.available_informative_mass),
            "available_informative_fraction": float(self.available_informative_fraction),
            "support_mass": float(self.support_mass),
            "contradiction_mass": float(self.contradiction_mass),
            "critical_contradiction_mass": float(self.critical_contradiction_mass),
            "conflicting_action_support_mass": float(self.conflicting_action_support_mass),
            "conflicting_action_contradiction_mass": float(self.conflicting_action_contradiction_mass),
        }


@dataclass(frozen=True)
class DiscriminativeRowCandidate:
    row_id: str
    action_id: str
    prototype_observation_id: str
    prototype_digest: str
    observation_digest: str
    architecture_id: str
    mask_digest: str
    region_spec_digest: str
    provider_digest: str
    aggregate_support: float
    aggregate_contradiction: float
    aggregate_critical_contradiction: float
    available_informative_mass: float
    available_informative_fraction: float
    supporting_region_count: int
    exact_winner_margin: Optional[float]
    conflicting_action_separation: Optional[float]
    eligible_for_exact: bool
    eligible_for_candidate_set: bool
    ineligibility_reasons: Tuple[str, ...]
    regional_evidence: Tuple[RegionDiscriminativeEvidence, ...]

    def __post_init__(self) -> None:
        for name in ("row_id", "action_id", "prototype_observation_id", "prototype_digest", "observation_digest", "architecture_id", "mask_digest", "region_spec_digest", "provider_digest"):
            if not str(getattr(self, name)):
                raise VPMValidationError(f"{name} cannot be empty")
        if self.architecture_id not in _ARCHITECTURES:
            raise VPMValidationError("unsupported architecture_id")
        for name in ("aggregate_support", "aggregate_contradiction", "aggregate_critical_contradiction", "available_informative_mass", "available_informative_fraction"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not np.isfinite(value):
                    raise VPMValidationError(f"{name} must be finite when present")
                if value < 0.0:
                    raise VPMValidationError(f"{name} must be non-negative")
        if int(self.supporting_region_count) < 0:
            raise VPMValidationError("supporting_region_count must be non-negative")
        if len(self.regional_evidence) == 0:
            raise VPMValidationError("regional_evidence cannot be empty")
        object.__setattr__(self, "ineligibility_reasons", tuple(str(item) for item in self.ineligibility_reasons))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_id": self.row_id,
            "action_id": self.action_id,
            "prototype_observation_id": self.prototype_observation_id,
            "prototype_digest": self.prototype_digest,
            "observation_digest": self.observation_digest,
            "architecture_id": self.architecture_id,
            "mask_digest": self.mask_digest,
            "region_spec_digest": self.region_spec_digest,
            "provider_digest": self.provider_digest,
            "aggregate_support": float(self.aggregate_support),
            "aggregate_contradiction": float(self.aggregate_contradiction),
            "aggregate_critical_contradiction": float(self.aggregate_critical_contradiction),
            "available_informative_mass": float(self.available_informative_mass),
            "available_informative_fraction": float(self.available_informative_fraction),
            "supporting_region_count": int(self.supporting_region_count),
            "exact_winner_margin": None if self.exact_winner_margin is None else float(self.exact_winner_margin),
            "conflicting_action_separation": None if self.conflicting_action_separation is None else float(self.conflicting_action_separation),
            "eligible_for_exact": bool(self.eligible_for_exact),
            "eligible_for_candidate_set": bool(self.eligible_for_candidate_set),
            "ineligibility_reasons": list(self.ineligibility_reasons),
            "regional_evidence": [item.to_dict() for item in self.regional_evidence],
        }


@dataclass(frozen=True)
class DiscriminativeCandidateSet:
    observation_digest: str
    provider_digest: str
    architecture_id: str
    outcome: str
    candidate_set_limit: int
    rows: Tuple[str, ...]
    actions: Tuple[str, ...]
    candidate_digest: str
    exact_row_id: Optional[str] = None
    rejection_reason: Optional[str] = None
    version: str = VIDEO_DISCRIMINATIVE_CANDIDATE_SET_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_DISCRIMINATIVE_CANDIDATE_SET_VERSION:
            raise VPMValidationError("unsupported discriminative candidate set version")
        if self.outcome not in _OUTCOMES:
            raise VPMValidationError("unsupported discriminative outcome")
        for name in ("observation_digest", "provider_digest", "architecture_id", "candidate_digest"):
            if not str(getattr(self, name)):
                raise VPMValidationError(f"{name} cannot be empty")
        if self.architecture_id not in _ARCHITECTURES:
            raise VPMValidationError("unsupported architecture_id")
        if int(self.candidate_set_limit) < 1:
            raise VPMValidationError("candidate_set_limit must be positive")
        if len(tuple(self.rows)) != len(tuple(self.actions)):
            raise VPMValidationError("rows and actions must be aligned")
        if self.outcome == "exact_row_accepted":
            if self.exact_row_id is None:
                raise VPMValidationError("exact_row_accepted requires exact_row_id")
        elif self.exact_row_id is not None:
            raise VPMValidationError("only exact_row_accepted may carry exact_row_id")
        object.__setattr__(self, "rows", tuple(str(item) for item in self.rows))
        object.__setattr__(self, "actions", tuple(str(item) for item in self.actions))

    @property
    def digest(self) -> str:
        return _json_digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "observation_digest": self.observation_digest,
            "provider_digest": self.provider_digest,
            "architecture_id": self.architecture_id,
            "outcome": self.outcome,
            "candidate_set_limit": int(self.candidate_set_limit),
            "rows": list(self.rows),
            "actions": list(self.actions),
            "candidate_digest": self.candidate_digest,
            "exact_row_id": self.exact_row_id,
            "rejection_reason": self.rejection_reason,
        }


def discriminative_region_digest(regions: Sequence[DiscriminativeRegionSpec]) -> str:
    values = [region.to_dict() for region in regions]
    ids = [region["region_id"] for region in values]
    if len(set(ids)) != len(ids):
        raise VPMValidationError("duplicate discriminative region IDs are not allowed")
    if not values:
        raise VPMValidationError("discriminative region list cannot be empty")
    return _json_digest(values)


def discriminative_mask_digest(masks: Sequence[DiscriminativeMaskSpec]) -> str:
    values = [mask.to_dict() for mask in sorted(masks, key=lambda item: (item.row_id, item.action_id, item.mask_id))]
    ids = [mask["mask_id"] for mask in values]
    if len(set(ids)) != len(ids):
        raise VPMValidationError("duplicate discriminative mask IDs are not allowed")
    if not values:
        raise VPMValidationError("discriminative mask list cannot be empty")
    return _json_digest(values)


def _registration_candidate(
    *,
    prototype: np.ndarray,
    observation: np.ndarray,
    informative_weights: np.ndarray,
    dx: int,
    dy: int,
) -> Dict[str, Any]:
    distance, overlap_fraction, valid_count = _normalized_overlap_distance(prototype, observation, dx=dx, dy=dy)
    available_mass = float(_aligned_slices(informative_weights, dx=dx, dy=dy, observation=False).sum(dtype=np.float64))
    return {
        "distance": float(distance),
        "available_informative_mass": available_mass,
        "valid_pixel_count": int(valid_count),
        "geometric_overlap": float(overlap_fraction),
        "dx": int(dx),
        "dy": int(dy),
    }


def _compare_registration_candidates(left: Dict[str, Any], right: Dict[str, Any]) -> Tuple[int, str]:
    if left["distance"] + REGISTRATION_DISTANCE_TIE_EPSILON < right["distance"]:
        return (-1, "distance")
    if right["distance"] + REGISTRATION_DISTANCE_TIE_EPSILON < left["distance"]:
        return (1, "distance")
    ordered_fields = (
        ("available_informative_mass", True, "available_informative_mass"),
        ("valid_pixel_count", True, "valid_pixel_count"),
        ("geometric_overlap", True, "geometric_overlap"),
    )
    for field, descending, reason in ordered_fields:
        left_value = left[field]
        right_value = right[field]
        if descending:
            if left_value > right_value + REGISTRATION_DISTANCE_TIE_EPSILON:
                return (-1, reason)
            if right_value > left_value + REGISTRATION_DISTANCE_TIE_EPSILON:
                return (1, reason)
        else:
            if left_value + REGISTRATION_DISTANCE_TIE_EPSILON < right_value:
                return (-1, reason)
            if right_value + REGISTRATION_DISTANCE_TIE_EPSILON < left_value:
                return (1, reason)
    left_manhattan = abs(int(left["dx"])) + abs(int(left["dy"]))
    right_manhattan = abs(int(right["dx"])) + abs(int(right["dy"]))
    if left_manhattan != right_manhattan:
        return (-1 if left_manhattan < right_manhattan else 1, "manhattan_translation")
    left_abs_dy = abs(int(left["dy"]))
    right_abs_dy = abs(int(right["dy"]))
    if left_abs_dy != right_abs_dy:
        return (-1 if left_abs_dy < right_abs_dy else 1, "vertical_shift")
    left_abs_dx = abs(int(left["dx"]))
    right_abs_dx = abs(int(right["dx"]))
    if left_abs_dx != right_abs_dx:
        return (-1 if left_abs_dx < right_abs_dx else 1, "horizontal_shift")
    left_signed = _signed_translation_order(int(left["dx"]), int(left["dy"]))
    right_signed = _signed_translation_order(int(right["dx"]), int(right["dy"]))
    if left_signed != right_signed:
        return (-1 if left_signed < right_signed else 1, "signed_offset_ordering")
    return (0, "complete_tie")


def register_informative_translation(
    prototype: Any,
    observation: Any,
    *,
    informative_weights: Any,
    region_id: str,
    config: RegistrationConfig,
) -> InformativeRegistrationResult:
    left = _normalized_gray(prototype)
    right = _normalized_gray(observation)
    if left.shape != right.shape:
        raise VPMValidationError("informative registration inputs must have identical shape")
    weights = _immutable_weight_array(informative_weights, shape=left.shape, name="informative_weights")
    expected_mass = float(weights.sum(dtype=np.float64))
    evaluated = 0
    best: Optional[Dict[str, Any]] = None
    runner_up: Optional[Dict[str, Any]] = None
    for dy in range(-int(config.max_dy), int(config.max_dy) + 1):
        for dx in range(-int(config.max_dx), int(config.max_dx) + 1):
            candidate = _registration_candidate(
                prototype=left,
                observation=right,
                informative_weights=weights,
                dx=dx,
                dy=dy,
            )
            if candidate["geometric_overlap"] + REGISTRATION_DISTANCE_TIE_EPSILON < float(config.minimum_overlap_fraction):
                continue
            evaluated += 1
            if best is None:
                best = candidate
                continue
            comparison, _reason = _compare_registration_candidates(candidate, best)
            if comparison < 0:
                runner_up = best
                best = candidate
            else:
                if runner_up is None:
                    runner_up = candidate
                else:
                    runner_comparison, _runner_reason = _compare_registration_candidates(candidate, runner_up)
                    if runner_comparison < 0:
                        runner_up = candidate
    contract_digest = _registration_contract_digest(config)
    if best is None:
        return InformativeRegistrationResult(
            prototype_shape=tuple(int(item) for item in left.shape),
            observation_shape=tuple(int(item) for item in right.shape),
            region_id=region_id,
            dx=0,
            dy=0,
            distance=float("inf"),
            score=float("-inf"),
            valid_pixel_count=0,
            geometric_overlap=0.0,
            expected_informative_mass=expected_mass,
            available_informative_mass=0.0,
            available_informative_fraction=0.0,
            registration_succeeded=False,
            rejection_reason="insufficient_overlap",
            tie_break_reason="distance",
            evaluated_translation_count=evaluated,
            registration_contract_digest=contract_digest,
        )
    tie_break_reason = "distance"
    if runner_up is not None:
        _cmp, tie_break_reason = _compare_registration_candidates(best, runner_up)
    available_fraction = 0.0 if expected_mass <= 0.0 else float(best["available_informative_mass"] / expected_mass)
    return InformativeRegistrationResult(
        prototype_shape=tuple(int(item) for item in left.shape),
        observation_shape=tuple(int(item) for item in right.shape),
        region_id=region_id,
        dx=int(best["dx"]),
        dy=int(best["dy"]),
        distance=float(best["distance"]),
        score=-float(best["distance"]),
        valid_pixel_count=int(best["valid_pixel_count"]),
        geometric_overlap=float(best["geometric_overlap"]),
        expected_informative_mass=expected_mass,
        available_informative_mass=float(best["available_informative_mass"]),
        available_informative_fraction=available_fraction,
        registration_succeeded=True,
        rejection_reason=None,
        tie_break_reason=tie_break_reason,
        evaluated_translation_count=evaluated,
        registration_contract_digest=contract_digest,
        runner_up_dx=None if runner_up is None else int(runner_up["dx"]),
        runner_up_dy=None if runner_up is None else int(runner_up["dy"]),
        runner_up_available_informative_mass=None if runner_up is None else float(runner_up["available_informative_mass"]),
        runner_up_valid_pixel_count=None if runner_up is None else int(runner_up["valid_pixel_count"]),
    )


def build_discriminative_masks(
    *,
    prototypes: Mapping[str, Tuple[str, str, str, Any]],
    development_observations: Mapping[str, Sequence[Any]],
    intensity_tolerance: int,
    stability_tolerance: int,
    separation_cap: int,
) -> Mapping[str, DiscriminativeMask]:
    if not prototypes:
        raise VPMValidationError("prototypes cannot be empty")
    if int(intensity_tolerance) < 0 or int(stability_tolerance) < 0:
        raise VPMValidationError("tolerances must be non-negative")
    if int(separation_cap) <= 0:
        raise VPMValidationError("separation_cap must be positive")

    prototype_rows: Dict[str, Dict[str, Any]] = {}
    prototype_ids = set()
    shape: Optional[Tuple[int, int]] = None
    for observation_id, (row_id, action_id, prototype_digest, observation) in sorted(prototypes.items()):
        if observation_id in prototype_ids:
            raise VPMValidationError("duplicate observation IDs are not allowed")
        prototype_ids.add(observation_id)
        if row_id in prototype_rows:
            raise VPMValidationError("duplicate row identities are not allowed")
        if not str(action_id):
            raise VPMValidationError("action_id cannot be empty")
        pixels = _coerce_observation_pixels(observation)
        if shape is None:
            shape = tuple(int(item) for item in pixels.shape)
        elif tuple(int(item) for item in pixels.shape) != shape:
            raise VPMValidationError("all prototypes must share identical geometry")
        prototype_rows[row_id] = {
            "row_id": row_id,
            "action_id": action_id,
            "observation_id": observation_id,
            "prototype_digest": prototype_digest,
            "pixels": pixels,
        }

    assert shape is not None
    prototype_digest = _prototype_payload_digest(prototypes)
    development_digest = _development_payload_digest(development_observations)
    derivation_contract = _json_digest(
        {
            "version": VIDEO_DISCRIMINATIVE_EVIDENCE_MECHANICS_VERSION,
            "prototype_digest": prototype_digest,
            "development_digest": development_digest,
            "intensity_tolerance": int(intensity_tolerance),
            "stability_tolerance": int(stability_tolerance),
            "separation_cap": int(separation_cap),
            "stability_fallback": "zero_stable_mass_without_development",
            "nearest_competitor_rule": "minimum relevant separation",
        }
    )

    results: Dict[str, DiscriminativeMask] = {}
    for row_id, entry in prototype_rows.items():
        candidate = entry["pixels"]
        competitors = [other for other_id, other in prototype_rows.items() if other_id != row_id]
        row_info = np.zeros(shape, dtype=np.float32)
        action_conflict = np.zeros(shape, dtype=np.float32)
        separation = np.zeros(shape, dtype=np.float32)

        if competitors:
            diff_stack = np.stack([_difference_uint8(candidate, other["pixels"]) for other in competitors], axis=0)
            row_info = (diff_stack.max(axis=0) > int(intensity_tolerance)).astype(np.float32)
            nearest_relevant = diff_stack.min(axis=0).astype(np.float32)
            separation = np.clip(nearest_relevant / float(int(separation_cap)), 0.0, 1.0)
            conflict_competitors = [other for other in competitors if other["action_id"] != entry["action_id"]]
            if conflict_competitors:
                conflict_stack = np.stack([_difference_uint8(candidate, other["pixels"]) for other in conflict_competitors], axis=0)
                action_conflict = (conflict_stack.max(axis=0) > int(intensity_tolerance)).astype(np.float32)
        development = tuple(development_observations.get(row_id, ()))
        if development:
            variations = []
            for observation in development:
                observed_pixels = _coerce_observation_pixels(observation)
                if observed_pixels.shape != shape:
                    raise VPMValidationError("development observations must match prototype geometry")
                variations.append(_difference_uint8(candidate, observed_pixels))
            variation_stack = np.stack(variations, axis=0)
            stable = (variation_stack.max(axis=0) <= int(stability_tolerance)).astype(np.float32)
        else:
            stable = np.zeros(shape, dtype=np.float32)

        spec = DiscriminativeMaskSpec(
            mask_id=f"{row_id}|mask",
            row_id=row_id,
            action_id=entry["action_id"],
            shape=shape,
            informative_pixel_count=int(np.count_nonzero(row_info > 0.0)),
            action_conflict_pixel_count=int(np.count_nonzero(action_conflict > 0.0)),
            stable_pixel_count=int(np.count_nonzero(stable > 0.0)),
            prototype_digest=entry["prototype_digest"],
            development_digest=development_digest,
            derivation_contract=derivation_contract,
            intensity_tolerance=int(intensity_tolerance),
            metadata={
                "prototype_observation_id": entry["observation_id"],
                "prototype_payload_digest": prototype_digest,
                "stability_fallback": "zero_stable_mass_without_development" if not development else "development_bounded",
            },
        )
        results[row_id] = DiscriminativeMask(
            spec=spec,
            row_informative_weights=row_info,
            action_conflict_weights=action_conflict,
            stable_weights=stable,
            separation_weights=separation,
        )
    return results


def extract_candidate_region_evidence(
    *,
    candidate_row_id: str,
    candidate_action_id: str,
    candidate_prototype: Any,
    observation: Any,
    mask: DiscriminativeMask,
    competing_prototypes: Mapping[str, Tuple[str, str, str, Any]],
    region: DiscriminativeRegionSpec,
) -> RegionDiscriminativeEvidence:
    if mask.spec.row_id != candidate_row_id:
        raise VPMValidationError("mask row_id must match candidate_row_id")
    candidate_pixels = _coerce_observation_pixels(candidate_prototype)
    observation_pixels = _coerce_observation_pixels(observation)
    if candidate_pixels.shape != observation_pixels.shape:
        raise VPMValidationError("candidate and observation must share identical geometry")
    top = int(region.top)
    left = int(region.left)
    height = int(region.height)
    width = int(region.width)
    candidate_region = _crop(candidate_pixels, top=top, left=left, height=height, width=width)
    observation_region = _crop(observation_pixels, top=top, left=left, height=height, width=width)
    informative_region = _crop(mask.row_informative_weights * mask.stable_weights, top=top, left=left, height=height, width=width)
    action_conflict_region = _crop(mask.action_conflict_weights * mask.stable_weights, top=top, left=left, height=height, width=width)
    separation_region = _crop(mask.separation_weights, top=top, left=left, height=height, width=width)

    registration = register_informative_translation(
        candidate_region,
        observation_region,
        informative_weights=informative_region,
        region_id=region.region_id,
        config=region.registration_config,
    )
    if not registration.registration_succeeded:
        return RegionDiscriminativeEvidence(
            region_id=region.region_id,
            expected_informative_mass=registration.expected_informative_mass,
            available_informative_mass=registration.available_informative_mass,
            available_informative_fraction=registration.available_informative_fraction,
            geometric_overlap=registration.geometric_overlap,
            valid_pixel_count=registration.valid_pixel_count,
            support_mass=0.0,
            contradiction_mass=0.0,
            critical_contradiction_mass=0.0,
            conflicting_action_support_mass=0.0,
            conflicting_action_contradiction_mass=0.0,
            registration_succeeded=False,
            registration_dx=registration.dx,
            registration_dy=registration.dy,
            registration_tie_break_reason=registration.tie_break_reason,
            rejection_reasons=(str(registration.rejection_reason),) if registration.rejection_reason else (),
        )

    dx = int(registration.dx)
    dy = int(registration.dy)
    candidate_aligned = _aligned_slices(candidate_region, dx=dx, dy=dy, observation=False).astype(np.float32) / 255.0
    observation_aligned = _aligned_slices(observation_region, dx=dx, dy=dy, observation=True).astype(np.float32) / 255.0
    informative_aligned = _aligned_slices(informative_region, dx=dx, dy=dy, observation=False).astype(np.float32)
    conflict_aligned = _aligned_slices(action_conflict_region, dx=dx, dy=dy, observation=False).astype(np.float32)
    separation_aligned = _aligned_slices(separation_region, dx=dx, dy=dy, observation=False).astype(np.float32)
    weighted_informative = informative_aligned * separation_aligned
    weighted_conflict = conflict_aligned * separation_aligned

    other_distances = []
    conflicting_distances = []
    for _observation_id, (row_id, action_id, _digest, prototype) in sorted(competing_prototypes.items()):
        prototype_pixels = _coerce_observation_pixels(prototype)
        if prototype_pixels.shape != candidate_pixels.shape:
            raise VPMValidationError("competing prototypes must match candidate geometry")
        prototype_region_other = _crop(prototype_pixels, top=top, left=left, height=height, width=width)
        aligned = _aligned_slices(prototype_region_other, dx=dx, dy=dy, observation=False).astype(np.float32) / 255.0
        distances = np.abs(observation_aligned - aligned)
        other_distances.append(distances)
        if action_id != candidate_action_id:
            conflicting_distances.append(distances)

    if other_distances:
        nearest_other = np.min(np.stack(other_distances, axis=0), axis=0)
    else:
        nearest_other = np.full(candidate_aligned.shape, np.inf, dtype=np.float32)
    if conflicting_distances:
        nearest_conflicting = np.min(np.stack(conflicting_distances, axis=0), axis=0)
    else:
        nearest_conflicting = np.full(candidate_aligned.shape, np.inf, dtype=np.float32)

    candidate_distance = np.abs(observation_aligned - candidate_aligned)
    support_mask = weighted_informative > 0.0
    support = support_mask & (candidate_distance + REGISTRATION_DISTANCE_TIE_EPSILON < nearest_other)
    contradiction = support_mask & (nearest_other + REGISTRATION_DISTANCE_TIE_EPSILON < candidate_distance)
    conflict_support_mask = weighted_conflict > 0.0
    conflict_support = conflict_support_mask & (candidate_distance + REGISTRATION_DISTANCE_TIE_EPSILON < nearest_conflicting)
    conflict_contradiction = conflict_support_mask & (
        nearest_conflicting + REGISTRATION_DISTANCE_TIE_EPSILON < candidate_distance
    )

    totals = PixelEvidenceTotals(
        expected_informative_mass=registration.expected_informative_mass,
        available_informative_mass=registration.available_informative_mass,
        available_informative_fraction=registration.available_informative_fraction,
        support_mass=float(weighted_informative[support].sum(dtype=np.float64)),
        contradiction_mass=float(weighted_informative[contradiction].sum(dtype=np.float64)),
        critical_contradiction_mass=float(weighted_conflict[conflict_contradiction].sum(dtype=np.float64)),
        conflicting_action_support_mass=float(weighted_conflict[conflict_support].sum(dtype=np.float64)),
        conflicting_action_contradiction_mass=float(weighted_conflict[conflict_contradiction].sum(dtype=np.float64)),
    )
    rejection_reasons = ()
    if totals.expected_informative_mass <= 0.0:
        rejection_reasons = ("no_expected_informative_mass",)
    elif totals.available_informative_mass <= 0.0:
        rejection_reasons = ("no_available_informative_mass",)
    return RegionDiscriminativeEvidence(
        region_id=region.region_id,
        expected_informative_mass=totals.expected_informative_mass,
        available_informative_mass=totals.available_informative_mass,
        available_informative_fraction=totals.available_informative_fraction,
        geometric_overlap=registration.geometric_overlap,
        valid_pixel_count=registration.valid_pixel_count,
        support_mass=totals.support_mass,
        contradiction_mass=totals.contradiction_mass,
        critical_contradiction_mass=totals.critical_contradiction_mass,
        conflicting_action_support_mass=totals.conflicting_action_support_mass,
        conflicting_action_contradiction_mass=totals.conflicting_action_contradiction_mass,
        registration_succeeded=True,
        registration_dx=registration.dx,
        registration_dy=registration.dy,
        registration_tie_break_reason=registration.tie_break_reason,
        rejection_reasons=rejection_reasons,
    )


def discriminative_provider_contract(*, calibration: DiscriminativeEvidenceCalibration, region_spec_digest: str, mask_spec_digest: str) -> VisualAddressContract:
    if calibration.region_spec_digest != region_spec_digest:
        raise VPMValidationError("calibration region_spec_digest does not match provider input")
    if calibration.mask_spec_digest != mask_spec_digest:
        raise VPMValidationError("calibration mask_spec_digest does not match provider input")
    provider_id = _json_digest(
        {
            "provider_version": VIDEO_DISCRIMINATIVE_PROVIDER_VERSION,
            "architecture_id": calibration.architecture_id,
            "calibration_digest": calibration.digest,
            "region_spec_digest": region_spec_digest,
            "mask_spec_digest": mask_spec_digest,
        }
    )
    return VisualAddressContract(
        provider_kind="deterministic_discriminative_evidence",
        provider_version=VIDEO_DISCRIMINATIVE_PROVIDER_VERSION,
        score_semantics="similarity",
        observation_spec_digest=region_spec_digest,
        representation_spec_digest=mask_spec_digest,
        address_artifact_id=provider_id,
        calibration_artifact_id=calibration.digest,
        policy_artifact_id=calibration.policy_artifact_id,
        source_scope=calibration.source_scope,
        replay_contract="exact_decision",
        metadata={
            "architecture_id": calibration.architecture_id,
            "region_spec_digest": region_spec_digest,
            "mask_spec_digest": mask_spec_digest,
            "maximum_candidate_set_size": int(calibration.maximum_candidate_set_size),
        },
    )


__all__ = [
    "DiscriminativeMask",
    "DiscriminativeCandidateSet",
    "DiscriminativeEvidenceCalibration",
    "DiscriminativeMaskSpec",
    "DiscriminativeRegionSpec",
    "DiscriminativeRowCandidate",
    "InformativeRegistrationResult",
    "PixelEvidenceTotals",
    "RegionDiscriminativeEvidence",
    "REGISTRATION_DISTANCE_TIE_EPSILON",
    "VIDEO_DISCRIMINATIVE_ARCHITECTURE_SELECTION_VERSION",
    "VIDEO_DISCRIMINATIVE_CALIBRATION_VERSION",
    "VIDEO_DISCRIMINATIVE_CANDIDATE_SET_VERSION",
    "VIDEO_DISCRIMINATIVE_EVIDENCE_MECHANICS_VERSION",
    "VIDEO_DISCRIMINATIVE_MASK_SPEC_VERSION",
    "VIDEO_DISCRIMINATIVE_MASK_PAYLOAD_VERSION",
    "VIDEO_DISCRIMINATIVE_OPERATING_POINT_SELECTION_VERSION",
    "VIDEO_DISCRIMINATIVE_PROVIDER_VERSION",
    "VIDEO_DISCRIMINATIVE_REGISTRATION_VERSION",
    "VIDEO_DISCRIMINATIVE_REGION_SPEC_VERSION",
    "build_discriminative_masks",
    "discriminative_mask_digest",
    "discriminative_provider_contract",
    "discriminative_region_digest",
    "extract_candidate_region_evidence",
    "register_informative_translation",
]
