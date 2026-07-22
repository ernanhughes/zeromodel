"""Stage 3 discriminative current-frame evidence contracts and mechanics."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.observation.visual_address import ImageObservation
from zeromodel.observation.visual_address import VisualAddressContract, VisualAddressDecision
from research.visual.visual_registration import RegistrationConfig


VIDEO_DISCRIMINATIVE_REGION_SPEC_VERSION = "zeromodel-video-discriminative-region-spec/v1"
VIDEO_DISCRIMINATIVE_MASK_SPEC_VERSION = "zeromodel-video-discriminative-mask-spec/v1"
VIDEO_DISCRIMINATIVE_CALIBRATION_VERSION = "zeromodel-video-discriminative-calibration/v1"
VIDEO_DISCRIMINATIVE_CANDIDATE_SET_VERSION = "zeromodel-video-discriminative-candidate-set/v2"
VIDEO_DISCRIMINATIVE_PROVIDER_VERSION = "zeromodel-video-discriminative-provider/v2"
VIDEO_DISCRIMINATIVE_ARCHITECTURE_SELECTION_VERSION = "zeromodel-video-discriminative-architecture-selection/v1"
VIDEO_DISCRIMINATIVE_OPERATING_POINT_SELECTION_VERSION = "zeromodel-video-discriminative-operating-point-selection/v1"
VIDEO_DISCRIMINATIVE_MASK_PAYLOAD_VERSION = "zeromodel-video-discriminative-mask-payload/v1"
VIDEO_DISCRIMINATIVE_REGISTRATION_VERSION = "zeromodel-video-discriminative-registration/v1"
VIDEO_DISCRIMINATIVE_EVIDENCE_MECHANICS_VERSION = "zeromodel-video-discriminative-evidence-mechanics/v1"
VIDEO_DISCRIMINATIVE_DECISION_VERSION = "zeromodel-video-discriminative-decision/v1"

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
    score_semantics: str
    candidate_strength: float
    raw_score: float
    raw_score_kind: str
    aggregate_support: float
    aggregate_contradiction: float
    aggregate_critical_contradiction: float
    aggregate_conflicting_action_support: float
    aggregate_conflicting_action_contradiction: float
    available_informative_mass: float
    available_informative_fraction: float
    supporting_region_count: int
    nearest_competitor_row_id: Optional[str]
    nearest_competitor_strength: Optional[float]
    nearest_same_action_row_id: Optional[str]
    nearest_same_action_strength: Optional[float]
    nearest_conflicting_action_row_id: Optional[str]
    nearest_conflicting_action_strength: Optional[float]
    candidate_relative_margin: Optional[float]
    exact_winner_margin: Optional[float]
    conflicting_action_separation: Optional[float]
    eligible_for_exact: bool
    eligible_for_candidate_set: bool
    ineligibility_reasons: Tuple[str, ...]
    regional_evidence: Tuple[RegionDiscriminativeEvidence, ...]

    def __post_init__(self) -> None:
        for name in ("row_id", "action_id", "prototype_observation_id", "prototype_digest", "observation_digest", "architecture_id", "mask_digest", "region_spec_digest", "provider_digest", "score_semantics", "raw_score_kind"):
            if not str(getattr(self, name)):
                raise VPMValidationError(f"{name} cannot be empty")
        if self.architecture_id not in _ARCHITECTURES:
            raise VPMValidationError("unsupported architecture_id")
        if self.score_semantics != "similarity":
            raise VPMValidationError("discriminative candidates must use similarity score semantics")
        for name in (
            "candidate_strength",
            "raw_score",
            "aggregate_support",
            "aggregate_contradiction",
            "aggregate_critical_contradiction",
            "aggregate_conflicting_action_support",
            "aggregate_conflicting_action_contradiction",
            "available_informative_mass",
            "available_informative_fraction",
        ):
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
            "score_semantics": self.score_semantics,
            "candidate_strength": float(self.candidate_strength),
            "raw_score": float(self.raw_score),
            "raw_score_kind": self.raw_score_kind,
            "aggregate_support": float(self.aggregate_support),
            "aggregate_contradiction": float(self.aggregate_contradiction),
            "aggregate_critical_contradiction": float(self.aggregate_critical_contradiction),
            "aggregate_conflicting_action_support": float(self.aggregate_conflicting_action_support),
            "aggregate_conflicting_action_contradiction": float(self.aggregate_conflicting_action_contradiction),
            "available_informative_mass": float(self.available_informative_mass),
            "available_informative_fraction": float(self.available_informative_fraction),
            "supporting_region_count": int(self.supporting_region_count),
            "nearest_competitor_row_id": self.nearest_competitor_row_id,
            "nearest_competitor_strength": self.nearest_competitor_strength,
            "nearest_same_action_row_id": self.nearest_same_action_row_id,
            "nearest_same_action_strength": self.nearest_same_action_strength,
            "nearest_conflicting_action_row_id": self.nearest_conflicting_action_row_id,
            "nearest_conflicting_action_strength": self.nearest_conflicting_action_strength,
            "candidate_relative_margin": self.candidate_relative_margin,
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
    strongest_candidate_row_id: Optional[str] = None
    weakest_included_row_id: Optional[str] = None
    unique_action_candidate_set: Optional[bool] = None
    excluded_nearby_rows: Tuple[str, ...] = ()
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
        object.__setattr__(self, "excluded_nearby_rows", tuple(str(item) for item in self.excluded_nearby_rows))

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
            "strongest_candidate_row_id": self.strongest_candidate_row_id,
            "weakest_included_row_id": self.weakest_included_row_id,
            "unique_action_candidate_set": self.unique_action_candidate_set,
            "excluded_nearby_rows": list(self.excluded_nearby_rows),
            "exact_row_id": self.exact_row_id,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class RawCandidateAggregate:
    architecture_id: str
    score_semantics: str
    candidate_strength: float
    raw_score: float
    raw_score_kind: str
    available_informative_mass: float
    available_informative_fraction: float
    aggregate_support: float
    aggregate_contradiction: float
    aggregate_critical_contradiction: float
    aggregate_conflicting_action_support: float
    aggregate_conflicting_action_contradiction: float
    supporting_region_count: int
    expected_informative_mass: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "architecture_id": self.architecture_id,
            "score_semantics": self.score_semantics,
            "candidate_strength": float(self.candidate_strength),
            "raw_score": float(self.raw_score),
            "raw_score_kind": self.raw_score_kind,
            "available_informative_mass": float(self.available_informative_mass),
            "available_informative_fraction": float(self.available_informative_fraction),
            "aggregate_support": float(self.aggregate_support),
            "aggregate_contradiction": float(self.aggregate_contradiction),
            "aggregate_critical_contradiction": float(self.aggregate_critical_contradiction),
            "aggregate_conflicting_action_support": float(self.aggregate_conflicting_action_support),
            "aggregate_conflicting_action_contradiction": float(self.aggregate_conflicting_action_contradiction),
            "supporting_region_count": int(self.supporting_region_count),
            "expected_informative_mass": float(self.expected_informative_mass),
        }


@dataclass(frozen=True)
class DiscriminativeEvidenceDecision:
    observation_digest: str
    provider_digest: str
    evidence_state: str
    candidate_set: DiscriminativeCandidateSet
    ranked_candidates: Tuple[DiscriminativeRowCandidate, ...]
    exact_address_decision: VisualAddressDecision
    trace: Mapping[str, Any]
    version: str = VIDEO_DISCRIMINATIVE_DECISION_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_DISCRIMINATIVE_DECISION_VERSION:
            raise VPMValidationError("unsupported discriminative decision version")
        if self.evidence_state not in _OUTCOMES:
            raise VPMValidationError("unsupported evidence_state")
        if not str(self.observation_digest) or not str(self.provider_digest):
            raise VPMValidationError("decision digests cannot be empty")
        object.__setattr__(self, "ranked_candidates", tuple(self.ranked_candidates))
        object.__setattr__(self, "trace", _freeze(self.trace))
        _json_bytes(self.trace)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "observation_digest": self.observation_digest,
            "provider_digest": self.provider_digest,
            "evidence_state": self.evidence_state,
            "candidate_set": self.candidate_set.to_dict(),
            "ranked_candidates": [candidate.to_dict() for candidate in self.ranked_candidates],
            "exact_address_decision": self.exact_address_decision.to_dict(),
            "trace": _freeze(self.trace),
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
    for field_name, descending, reason in ordered_fields:
        left_value = left[field_name]
        right_value = right[field_name]
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


def _architecture_active_gates(architecture_id: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    common = (
        "minimum_available_mass",
        "minimum_available_fraction",
        "candidate_relative_margin",
        "conflicting_action_separation",
        "maximum_candidate_set_size",
    )
    if architecture_id == "A":
        active = common + ("exact_winner_threshold", "exact_winner_margin")
        inactive = (
            "minimum_support",
            "maximum_contradiction",
            "maximum_critical_contradiction",
            "minimum_supporting_regions",
        )
    elif architecture_id == "B":
        active = common + ("minimum_support", "exact_winner_threshold", "exact_winner_margin")
        inactive = (
            "maximum_contradiction",
            "maximum_critical_contradiction",
            "minimum_supporting_regions",
        )
    elif architecture_id == "C":
        active = common + (
            "minimum_support",
            "maximum_contradiction",
            "maximum_critical_contradiction",
            "minimum_supporting_regions",
            "exact_winner_threshold",
            "exact_winner_margin",
        )
        inactive = ()
    elif architecture_id == "D":
        active = common + (
            "minimum_support",
            "maximum_contradiction",
            "maximum_critical_contradiction",
            "minimum_supporting_regions",
            "exact_winner_threshold",
            "exact_winner_margin",
        )
        inactive = ()
    else:
        raise VPMValidationError("unsupported architecture_id")
    return (active, inactive)


def _aggregate_candidate(
    *,
    architecture_id: str,
    regional_evidence: Sequence[RegionDiscriminativeEvidence],
    regions: Sequence[DiscriminativeRegionSpec],
) -> RawCandidateAggregate:
    region_by_id = {region.region_id: region for region in regions}
    total_expected_mass = float(sum(item.expected_informative_mass for item in regional_evidence))
    total_available_mass = float(sum(item.available_informative_mass for item in regional_evidence))
    available_fraction = 0.0 if total_expected_mass <= 0.0 else total_available_mass / total_expected_mass
    weighted_distance_numerator = 0.0
    weighted_distance_denominator = 0.0
    total_support = 0.0
    total_contradiction = 0.0
    total_critical_contradiction = 0.0
    total_conflict_support = 0.0
    total_conflict_contradiction = 0.0
    supporting_region_count = 0
    for item in regional_evidence:
        region = region_by_id[item.region_id]
        # The current regional support/contradiction masses are normalized through available mass later.
        total_support += float(item.support_mass)
        total_contradiction += float(item.contradiction_mass)
        total_critical_contradiction += float(item.critical_contradiction_mass)
        total_conflict_support += float(item.conflicting_action_support_mass)
        total_conflict_contradiction += float(item.conflicting_action_contradiction_mass)
        if item.support_mass > 0.0:
            supporting_region_count += 1
        # Stage 2-style regional distance reconstructed from support/contradiction evidence:
        evidence_total = float(item.support_mass + item.contradiction_mass)
        if evidence_total > 0.0:
            region_distance = min(1.0, max(0.0, float(item.contradiction_mass / evidence_total)))
        else:
            region_distance = 1.0
        weighted_distance_numerator += float(region.weight) * region_distance
        weighted_distance_denominator += float(region.weight)
    support_ratio = 0.0 if total_available_mass <= 0.0 else total_support / total_available_mass
    contradiction_ratio = 0.0 if total_available_mass <= 0.0 else total_contradiction / total_available_mass
    critical_ratio = 0.0 if total_available_mass <= 0.0 else total_critical_contradiction / total_available_mass
    if architecture_id == "A":
        raw_distance = 1.0 if weighted_distance_denominator <= 0.0 else weighted_distance_numerator / weighted_distance_denominator
        candidate_strength = max(0.0, 1.0 - raw_distance) * available_fraction
        return RawCandidateAggregate(
            architecture_id=architecture_id,
            score_semantics="similarity",
            candidate_strength=float(candidate_strength),
            raw_score=float(raw_distance),
            raw_score_kind="distance",
            available_informative_mass=float(total_available_mass),
            available_informative_fraction=float(available_fraction),
            aggregate_support=float(candidate_strength),
            aggregate_contradiction=0.0,
            aggregate_critical_contradiction=0.0,
            aggregate_conflicting_action_support=float(total_conflict_support / total_available_mass) if total_available_mass > 0.0 else 0.0,
            aggregate_conflicting_action_contradiction=float(total_conflict_contradiction / total_available_mass) if total_available_mass > 0.0 else 0.0,
            supporting_region_count=int(supporting_region_count),
            expected_informative_mass=float(total_expected_mass),
        )
    if architecture_id == "B":
        candidate_strength = float(support_ratio)
        return RawCandidateAggregate(
            architecture_id=architecture_id,
            score_semantics="similarity",
            candidate_strength=candidate_strength,
            raw_score=candidate_strength,
            raw_score_kind="support_ratio",
            available_informative_mass=float(total_available_mass),
            available_informative_fraction=float(available_fraction),
            aggregate_support=float(support_ratio),
            aggregate_contradiction=float(contradiction_ratio),
            aggregate_critical_contradiction=float(critical_ratio),
            aggregate_conflicting_action_support=float(total_conflict_support / total_available_mass) if total_available_mass > 0.0 else 0.0,
            aggregate_conflicting_action_contradiction=float(total_conflict_contradiction / total_available_mass) if total_available_mass > 0.0 else 0.0,
            supporting_region_count=int(supporting_region_count),
            expected_informative_mass=float(total_expected_mass),
        )
    if architecture_id == "C":
        candidate_strength = max(0.0, support_ratio - contradiction_ratio - critical_ratio)
        return RawCandidateAggregate(
            architecture_id=architecture_id,
            score_semantics="similarity",
            candidate_strength=float(candidate_strength),
            raw_score=float(candidate_strength),
            raw_score_kind="support_minus_contradiction",
            available_informative_mass=float(total_available_mass),
            available_informative_fraction=float(available_fraction),
            aggregate_support=float(support_ratio),
            aggregate_contradiction=float(contradiction_ratio),
            aggregate_critical_contradiction=float(critical_ratio),
            aggregate_conflicting_action_support=float(total_conflict_support / total_available_mass) if total_available_mass > 0.0 else 0.0,
            aggregate_conflicting_action_contradiction=float(total_conflict_contradiction / total_available_mass) if total_available_mass > 0.0 else 0.0,
            supporting_region_count=int(supporting_region_count),
            expected_informative_mass=float(total_expected_mass),
        )
    if architecture_id == "D":
        candidate_strength = max(0.0, support_ratio - contradiction_ratio - critical_ratio)
        return RawCandidateAggregate(
            architecture_id=architecture_id,
            score_semantics="similarity",
            candidate_strength=float(candidate_strength),
            raw_score=float(candidate_strength),
            raw_score_kind="combined_support_minus_contradiction",
            available_informative_mass=float(total_available_mass),
            available_informative_fraction=float(available_fraction),
            aggregate_support=float(support_ratio),
            aggregate_contradiction=float(contradiction_ratio),
            aggregate_critical_contradiction=float(critical_ratio),
            aggregate_conflicting_action_support=float(total_conflict_support / total_available_mass) if total_available_mass > 0.0 else 0.0,
            aggregate_conflicting_action_contradiction=float(total_conflict_contradiction / total_available_mass) if total_available_mass > 0.0 else 0.0,
            supporting_region_count=int(supporting_region_count),
            expected_informative_mass=float(total_expected_mass),
        )
    raise VPMValidationError("unsupported architecture_id")


def _candidate_rank_key(candidate: DiscriminativeRowCandidate) -> Tuple[Any, ...]:
    return (
        -float(candidate.candidate_strength),
        -float(candidate.available_informative_mass),
        -float(candidate.available_informative_fraction),
        float(candidate.aggregate_contradiction),
        float(candidate.aggregate_critical_contradiction),
        -int(candidate.supporting_region_count),
        candidate.row_id,
        candidate.prototype_observation_id,
    )


def build_raw_discriminative_candidates(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
    masks: Mapping[str, DiscriminativeMask],
    regions: Sequence[DiscriminativeRegionSpec],
    architecture_id: str,
    provider_digest: str,
    region_spec_digest: str,
) -> Tuple[DiscriminativeRowCandidate, ...]:
    rows_seen = set()
    candidates = []
    for observation_id, (row_id, action_id, prototype_digest, prototype_observation) in sorted(prototypes.items(), key=lambda item: (item[1][0], item[0])):
        if row_id in rows_seen:
            raise VPMValidationError("discriminative provider requires exactly one prototype per row")
        rows_seen.add(row_id)
        mask = masks.get(row_id)
        if mask is None:
            raise VPMValidationError("missing discriminative mask for row_id")
        if mask.spec.row_id != row_id or mask.spec.action_id != action_id:
            raise VPMValidationError("mask identity does not match prototype row/action")
        regional = tuple(
            extract_candidate_region_evidence(
                candidate_row_id=row_id,
                candidate_action_id=action_id,
                candidate_prototype=prototype_observation,
                observation=observation,
                mask=mask,
                competing_prototypes={key: value for key, value in prototypes.items() if value[0] != row_id},
                region=region,
            )
            for region in regions
        )
        aggregate = _aggregate_candidate(
            architecture_id=architecture_id,
            regional_evidence=regional,
            regions=regions,
        )
        candidates.append(
            DiscriminativeRowCandidate(
                row_id=row_id,
                action_id=action_id,
                prototype_observation_id=observation_id,
                prototype_digest=prototype_digest,
                observation_digest=observation.raw_digest,
                architecture_id=architecture_id,
                mask_digest=mask.payload_digest,
                region_spec_digest=region_spec_digest,
                provider_digest=provider_digest,
                score_semantics=aggregate.score_semantics,
                candidate_strength=aggregate.candidate_strength,
                raw_score=aggregate.raw_score,
                raw_score_kind=aggregate.raw_score_kind,
                aggregate_support=aggregate.aggregate_support,
                aggregate_contradiction=aggregate.aggregate_contradiction,
                aggregate_critical_contradiction=aggregate.aggregate_critical_contradiction,
                aggregate_conflicting_action_support=aggregate.aggregate_conflicting_action_support,
                aggregate_conflicting_action_contradiction=aggregate.aggregate_conflicting_action_contradiction,
                available_informative_mass=aggregate.available_informative_mass,
                available_informative_fraction=aggregate.available_informative_fraction,
                supporting_region_count=aggregate.supporting_region_count,
                nearest_competitor_row_id=None,
                nearest_competitor_strength=None,
                nearest_same_action_row_id=None,
                nearest_same_action_strength=None,
                nearest_conflicting_action_row_id=None,
                nearest_conflicting_action_strength=None,
                candidate_relative_margin=None,
                exact_winner_margin=None,
                conflicting_action_separation=None,
                eligible_for_exact=False,
                eligible_for_candidate_set=False,
                ineligibility_reasons=(),
                regional_evidence=regional,
            )
        )
    return tuple(candidates)


def rank_discriminative_candidates(candidates: Sequence[DiscriminativeRowCandidate]) -> Tuple[DiscriminativeRowCandidate, ...]:
    ranked = list(sorted(candidates, key=_candidate_rank_key))
    if not ranked:
        raise VPMValidationError("rank_discriminative_candidates requires candidates")
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    strongest_conflicting_by_action: Dict[str, DiscriminativeRowCandidate] = {}
    for candidate in ranked:
        strongest_conflicting_by_action.setdefault(candidate.action_id, candidate)
    enriched = []
    for candidate in ranked:
        nearest_same = next((other for other in ranked if other.row_id != candidate.row_id and other.action_id == candidate.action_id), None)
        nearest_conflicting = next((other for other in ranked if other.action_id != candidate.action_id), None)
        nearest_competitor = next((other for other in ranked if other.row_id != candidate.row_id), None)
        exact_margin = None if candidate.row_id != winner.row_id or runner_up is None else float(candidate.candidate_strength - runner_up.candidate_strength)
        relative_margin = None if candidate.row_id == winner.row_id else float(winner.candidate_strength - candidate.candidate_strength)
        conflicting_separation = None if nearest_conflicting is None else float(candidate.candidate_strength - nearest_conflicting.candidate_strength)
        enriched.append(
            DiscriminativeRowCandidate(
                **{
                    **candidate.__dict__,
                    "nearest_competitor_row_id": None if nearest_competitor is None else nearest_competitor.row_id,
                    "nearest_competitor_strength": None if nearest_competitor is None else float(nearest_competitor.candidate_strength),
                    "nearest_same_action_row_id": None if nearest_same is None else nearest_same.row_id,
                    "nearest_same_action_strength": None if nearest_same is None else float(nearest_same.candidate_strength),
                    "nearest_conflicting_action_row_id": None if nearest_conflicting is None else nearest_conflicting.row_id,
                    "nearest_conflicting_action_strength": None if nearest_conflicting is None else float(nearest_conflicting.candidate_strength),
                    "candidate_relative_margin": relative_margin,
                    "exact_winner_margin": exact_margin,
                    "conflicting_action_separation": conflicting_separation,
                }
            )
        )
    return tuple(sorted(enriched, key=_candidate_rank_key))


def evaluate_candidate_eligibility(
    *,
    candidate: DiscriminativeRowCandidate,
    ranked_candidates: Sequence[DiscriminativeRowCandidate],
    calibration: DiscriminativeEvidenceCalibration,
) -> DiscriminativeRowCandidate:
    active_gates, inactive_gates = _architecture_active_gates(calibration.architecture_id)
    reasons = []
    if candidate.available_informative_mass + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.minimum_available_mass:
        reasons.append("minimum_available_mass")
    if candidate.available_informative_fraction + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.minimum_available_fraction:
        reasons.append("minimum_available_fraction")
    if "minimum_support" in active_gates and candidate.aggregate_support + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.minimum_support:
        reasons.append("minimum_support")
    if "maximum_contradiction" in active_gates and candidate.aggregate_contradiction > calibration.maximum_contradiction + REGISTRATION_DISTANCE_TIE_EPSILON:
        reasons.append("maximum_contradiction")
    if "maximum_critical_contradiction" in active_gates and candidate.aggregate_critical_contradiction > calibration.maximum_critical_contradiction + REGISTRATION_DISTANCE_TIE_EPSILON:
        reasons.append("maximum_critical_contradiction")
    if candidate.candidate_relative_margin is not None and candidate.candidate_relative_margin + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.candidate_relative_margin:
        reasons.append("candidate_relative_margin")
    if candidate.conflicting_action_separation is not None and candidate.conflicting_action_separation + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.conflicting_action_separation:
        reasons.append("conflicting_action_separation")
    if "minimum_supporting_regions" in active_gates and candidate.supporting_region_count < calibration.minimum_supporting_regions:
        reasons.append("minimum_supporting_regions")
    eligible_for_candidate_set = len(reasons) == 0
    exact_reasons = list(reasons)
    if "exact_winner_threshold" in active_gates and candidate.candidate_strength + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.exact_winner_threshold:
        exact_reasons.append("exact_winner_threshold")
    if "exact_winner_margin" in active_gates:
        if candidate.exact_winner_margin is None or candidate.exact_winner_margin + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.exact_winner_margin:
            exact_reasons.append("exact_winner_margin")
    if candidate.aggregate_critical_contradiction > calibration.maximum_critical_contradiction + REGISTRATION_DISTANCE_TIE_EPSILON:
        if "maximum_critical_contradiction" not in exact_reasons:
            exact_reasons.append("maximum_critical_contradiction")
    exact_reasons.extend(f"inactive:{name}" for name in inactive_gates)
    return DiscriminativeRowCandidate(
        **{
            **candidate.__dict__,
            "eligible_for_candidate_set": eligible_for_candidate_set,
            "eligible_for_exact": len([item for item in exact_reasons if not item.startswith("inactive:")]) == 0,
            "ineligibility_reasons": tuple(exact_reasons),
        }
    )


def build_discriminative_candidate_set(
    *,
    ranked_candidates: Sequence[DiscriminativeRowCandidate],
    calibration: DiscriminativeEvidenceCalibration,
    provider_digest: str,
    observation_digest: str,
) -> DiscriminativeCandidateSet:
    eligible = [candidate for candidate in ranked_candidates if candidate.eligible_for_candidate_set]
    eligible_exact = [candidate for candidate in ranked_candidates if candidate.eligible_for_exact]
    if len(eligible_exact) == 1 and ranked_candidates[0].row_id == eligible_exact[0].row_id:
        winner = eligible_exact[0]
        return DiscriminativeCandidateSet(
            observation_digest=observation_digest,
            provider_digest=provider_digest,
            architecture_id=calibration.architecture_id,
            outcome="exact_row_accepted",
            candidate_set_limit=int(calibration.maximum_candidate_set_size),
            rows=(winner.row_id,),
            actions=(winner.action_id,),
            candidate_digest=_json_digest([winner.to_dict()]),
            strongest_candidate_row_id=winner.row_id,
            weakest_included_row_id=winner.row_id,
            unique_action_candidate_set=True,
            excluded_nearby_rows=tuple(candidate.row_id for candidate in ranked_candidates[1:4]),
            exact_row_id=winner.row_id,
            rejection_reason=None,
        )
    if eligible:
        if len(eligible) > int(calibration.maximum_candidate_set_size):
            return DiscriminativeCandidateSet(
                observation_digest=observation_digest,
                provider_digest=provider_digest,
                architecture_id=calibration.architecture_id,
                outcome="no_sufficient_evidence",
                candidate_set_limit=int(calibration.maximum_candidate_set_size),
                rows=(),
                actions=(),
                candidate_digest=_json_digest([candidate.to_dict() for candidate in eligible]),
                strongest_candidate_row_id=eligible[0].row_id,
                weakest_included_row_id=eligible[-1].row_id,
                unique_action_candidate_set=None,
                excluded_nearby_rows=tuple(candidate.row_id for candidate in eligible),
                exact_row_id=None,
                rejection_reason="candidate_set_too_large",
            )
        rows = tuple(candidate.row_id for candidate in eligible)
        actions = tuple(candidate.action_id for candidate in eligible)
        return DiscriminativeCandidateSet(
            observation_digest=observation_digest,
            provider_digest=provider_digest,
            architecture_id=calibration.architecture_id,
            outcome="candidate_set_available",
            candidate_set_limit=int(calibration.maximum_candidate_set_size),
            rows=rows,
            actions=actions,
            candidate_digest=_json_digest([candidate.to_dict() for candidate in eligible]),
            strongest_candidate_row_id=eligible[0].row_id,
            weakest_included_row_id=eligible[-1].row_id,
            unique_action_candidate_set=len(set(actions)) == 1,
            excluded_nearby_rows=tuple(candidate.row_id for candidate in ranked_candidates if candidate.row_id not in rows)[:4],
            exact_row_id=None,
            rejection_reason=None,
        )
    return DiscriminativeCandidateSet(
        observation_digest=observation_digest,
        provider_digest=provider_digest,
        architecture_id=calibration.architecture_id,
        outcome="no_sufficient_evidence",
        candidate_set_limit=int(calibration.maximum_candidate_set_size),
        rows=(),
        actions=(),
        candidate_digest=_json_digest([]),
        strongest_candidate_row_id=None,
        weakest_included_row_id=None,
        unique_action_candidate_set=None,
        excluded_nearby_rows=tuple(candidate.row_id for candidate in ranked_candidates[:4]),
        exact_row_id=None,
        rejection_reason="no_candidate_passed",
    )


class DiscriminativeEvidenceProvider:
    def __init__(
        self,
        *,
        prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
        masks: Mapping[str, DiscriminativeMask],
        regions: Sequence[DiscriminativeRegionSpec],
        calibration: DiscriminativeEvidenceCalibration,
        policy_artifact_id: str,
        source_scope: str,
    ) -> None:
        if calibration.architecture_id not in {"A", "B", "C"}:
            raise VPMValidationError("V4 provider supports architectures A, B, and C only")
        self._prototypes = dict(sorted(prototypes.items()))
        self._masks = dict(sorted(masks.items()))
        self._regions = tuple(regions)
        self._calibration = calibration
        self._policy_artifact_id = str(policy_artifact_id)
        self._source_scope = str(source_scope)
        if not self._prototypes:
            raise VPMValidationError("discriminative provider requires prototypes")
        self._shape = next(iter(self._prototypes.values()))[3].pixels.shape
        self._prototype_collection_digest = _prototype_payload_digest(self._prototypes)
        self._mask_spec_digest = discriminative_mask_digest(tuple(mask.spec for mask in self._masks.values()))
        self._mask_payload_digest = _json_digest([mask.to_dict() for _row_id, mask in sorted(self._masks.items())])
        self._region_spec_digest = discriminative_region_digest(self._regions)
        if calibration.prototype_digest != self._prototype_collection_digest:
            raise VPMValidationError("calibration prototype_digest does not match provider prototypes")
        if calibration.mask_spec_digest != self._mask_spec_digest:
            raise VPMValidationError("calibration mask_spec_digest does not match provider masks")
        if calibration.region_spec_digest != self._region_spec_digest:
            raise VPMValidationError("calibration region_spec_digest does not match provider regions")
        if calibration.policy_artifact_id != self._policy_artifact_id:
            raise VPMValidationError("calibration policy_artifact_id does not match provider input")
        if calibration.source_scope != self._source_scope:
            raise VPMValidationError("calibration source_scope does not match provider input")
        registration_contracts = sorted({region.registration_config.digest for region in self._regions})
        self._provider_digest = _json_digest(
            {
                "provider_version": VIDEO_DISCRIMINATIVE_PROVIDER_VERSION,
                "architecture_id": calibration.architecture_id,
                "calibration_digest": calibration.digest,
                "prototype_collection_digest": self._prototype_collection_digest,
                "mask_payload_digest": self._mask_payload_digest,
                "region_spec_digest": self._region_spec_digest,
                "registration_contracts": registration_contracts,
                "policy_artifact_id": self._policy_artifact_id,
                "source_scope": self._source_scope,
            }
        )
        self._raw_cache: Dict[str, Tuple[DiscriminativeRowCandidate, ...]] = {}
        self._decision_cache: Dict[str, DiscriminativeEvidenceDecision] = {}
        self.cache_hits = 0
        self.cache_misses = 0

    def contract(self) -> VisualAddressContract:
        return VisualAddressContract(
            provider_kind="deterministic_discriminative_evidence",
            provider_version=VIDEO_DISCRIMINATIVE_PROVIDER_VERSION,
            score_semantics="similarity",
            observation_spec_digest=self._region_spec_digest,
            representation_spec_digest=self._mask_spec_digest,
            address_artifact_id=self._provider_digest,
            calibration_artifact_id=self._calibration.digest,
            policy_artifact_id=self._policy_artifact_id,
            source_scope=self._source_scope,
            replay_contract="exact_decision",
            metadata={
                "architecture_id": self._calibration.architecture_id,
                "region_spec_digest": self._region_spec_digest,
                "mask_spec_digest": self._mask_spec_digest,
                "mask_payload_digest": self._mask_payload_digest,
                "prototype_collection_digest": self._prototype_collection_digest,
                "maximum_candidate_set_size": int(self._calibration.maximum_candidate_set_size),
            },
        )

    def _raw_cache_key(self, observation: ImageObservation) -> str:
        registration_contracts = sorted({region.registration_config.digest for region in self._regions})
        return _json_digest(
            {
                "observation_digest": observation.raw_digest,
                "source_id": observation.source_id,
                "shape": list(observation.pixels.shape),
                "prototype_collection_digest": self._prototype_collection_digest,
                "mask_payload_digest": self._mask_payload_digest,
                "region_spec_digest": self._region_spec_digest,
                "registration_contracts": registration_contracts,
                "architecture_id": self._calibration.architecture_id,
                "evidence_mechanics_version": VIDEO_DISCRIMINATIVE_EVIDENCE_MECHANICS_VERSION,
                "policy_artifact_id": self._policy_artifact_id,
                "source_scope": self._source_scope,
            }
        )

    def _decision_cache_key(self, observation: ImageObservation) -> str:
        return _json_digest({"raw_cache_key": self._raw_cache_key(observation), "calibration_digest": self._calibration.digest})

    def _rank(self, observation: ImageObservation) -> Tuple[DiscriminativeRowCandidate, ...]:
        if observation.pixels.shape != self._shape:
            raise VPMValidationError("discriminative provider observation geometry does not match prototypes")
        raw_key = self._raw_cache_key(observation)
        cached = self._raw_cache.get(raw_key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        self.cache_misses += 1
        raw = build_raw_discriminative_candidates(
            observation=observation,
            prototypes=self._prototypes,
            masks=self._masks,
            regions=self._regions,
            architecture_id=self._calibration.architecture_id,
            provider_digest=self._provider_digest,
            region_spec_digest=self._region_spec_digest,
        )
        ranked = rank_discriminative_candidates(raw)
        evaluated = tuple(
            evaluate_candidate_eligibility(candidate=candidate, ranked_candidates=ranked, calibration=self._calibration)
            for candidate in ranked
        )
        evaluated = tuple(sorted(evaluated, key=_candidate_rank_key))
        self._raw_cache[raw_key] = evaluated
        return evaluated

    def evaluate(self, observation: ImageObservation) -> DiscriminativeEvidenceDecision:
        decision_key = self._decision_cache_key(observation)
        cached = self._decision_cache.get(decision_key)
        if cached is not None:
            return cached
        ranked = self._rank(observation)
        candidate_set = build_discriminative_candidate_set(
            ranked_candidates=ranked,
            calibration=self._calibration,
            provider_digest=self._provider_digest,
            observation_digest=observation.raw_digest,
        )
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        visual_reason = candidate_set.outcome if candidate_set.outcome != "exact_row_accepted" else "accepted"
        visual = VisualAddressDecision(
            accepted=candidate_set.outcome == "exact_row_accepted",
            reason=visual_reason if candidate_set.rejection_reason is None else candidate_set.rejection_reason,
            observation_digest=observation.raw_digest,
            representation_digest=decision_key,
            provider_kind="deterministic_discriminative_evidence",
            provider_version=VIDEO_DISCRIMINATIVE_PROVIDER_VERSION,
            score_semantics="similarity",
            address_artifact_id=self._provider_digest,
            calibration_artifact_id=self._calibration.digest,
            policy_artifact_id=self._policy_artifact_id,
            nearest_row_id=best.row_id,
            nearest_score=float(best.candidate_strength),
            second_row_id=None if second is None else second.row_id,
            second_score=None if second is None else float(second.candidate_strength),
            ambiguity_measure=None if best.exact_winner_margin is None else float(best.exact_winner_margin),
            local_evidence_score=float(best.candidate_strength),
            visible_evidence_fraction=float(best.available_informative_fraction),
            critical_evidence_present=bool(best.aggregate_critical_contradiction <= self._calibration.maximum_critical_contradiction + REGISTRATION_DISTANCE_TIE_EPSILON),
            matched_row_id=candidate_set.exact_row_id,
            exact_match=candidate_set.outcome == "exact_row_accepted",
            accepted_by=(
                tuple(
                    name
                    for name in _architecture_active_gates(
                        self._calibration.architecture_id
                    )[0]
                )
                if candidate_set.outcome == "exact_row_accepted"
                else ()
            ),
            trace={},
        )
        trace = {
            "version": VIDEO_DISCRIMINATIVE_DECISION_VERSION,
            "provider_version": VIDEO_DISCRIMINATIVE_PROVIDER_VERSION,
            "architecture_id": self._calibration.architecture_id,
            "provider_digest": self._provider_digest,
            "calibration_digest": self._calibration.digest,
            "region_digest": self._region_spec_digest,
            "mask_digest": self._mask_payload_digest,
            "prototype_digest": self._prototype_collection_digest,
            "observation_digest": observation.raw_digest,
            "cache_key": decision_key,
            "evidence_state": candidate_set.outcome,
            "exact_row_id": candidate_set.exact_row_id,
            "candidate_set_rows": list(candidate_set.rows),
            "candidate_set_actions": list(candidate_set.actions),
            "unique_action_status": candidate_set.unique_action_candidate_set,
            "raw_eligible_rows": [candidate.row_id for candidate in ranked if candidate.eligible_for_candidate_set],
            "exclusion_reasons": {candidate.row_id: list(candidate.ineligibility_reasons) for candidate in ranked if not candidate.eligible_for_candidate_set},
            "ranked_candidate_summaries": [candidate.to_dict() for candidate in ranked],
            "winner": best.to_dict(),
            "runner_up": None if second is None else second.to_dict(),
            "winner_margin": best.exact_winner_margin,
            "nearest_conflicting_action_candidate": None if best.nearest_conflicting_action_row_id is None else best.nearest_conflicting_action_row_id,
            "conflicting_action_separation": best.conflicting_action_separation,
            "active_gates": list(_architecture_active_gates(self._calibration.architecture_id)[0]),
            "inactive_gates": list(_architecture_active_gates(self._calibration.architecture_id)[1]),
            "registration_summary": {
                candidate.row_id: [
                    {
                        "region_id": region.region_id,
                        "dx": region.registration_dx,
                        "dy": region.registration_dy,
                        "tie_break_reason": region.registration_tie_break_reason,
                    }
                    for region in candidate.regional_evidence
                ]
                for candidate in ranked
            },
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
        }
        visual = VisualAddressDecision(**{**visual.to_dict(), "trace": trace})
        decision = DiscriminativeEvidenceDecision(
            observation_digest=observation.raw_digest,
            provider_digest=self._provider_digest,
            evidence_state=candidate_set.outcome,
            candidate_set=candidate_set,
            ranked_candidates=ranked,
            exact_address_decision=visual,
            trace=trace,
        )
        self._decision_cache[decision_key] = decision
        return decision

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        return self.evaluate(observation).exact_address_decision


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
    "DiscriminativeEvidenceDecision",
    "DiscriminativeEvidenceProvider",
    "DiscriminativeMask",
    "DiscriminativeCandidateSet",
    "DiscriminativeEvidenceCalibration",
    "DiscriminativeMaskSpec",
    "DiscriminativeRegionSpec",
    "DiscriminativeRowCandidate",
    "InformativeRegistrationResult",
    "PixelEvidenceTotals",
    "RawCandidateAggregate",
    "RegionDiscriminativeEvidence",
    "REGISTRATION_DISTANCE_TIE_EPSILON",
    "VIDEO_DISCRIMINATIVE_ARCHITECTURE_SELECTION_VERSION",
    "VIDEO_DISCRIMINATIVE_CALIBRATION_VERSION",
    "VIDEO_DISCRIMINATIVE_CANDIDATE_SET_VERSION",
    "VIDEO_DISCRIMINATIVE_DECISION_VERSION",
    "VIDEO_DISCRIMINATIVE_EVIDENCE_MECHANICS_VERSION",
    "VIDEO_DISCRIMINATIVE_MASK_SPEC_VERSION",
    "VIDEO_DISCRIMINATIVE_MASK_PAYLOAD_VERSION",
    "VIDEO_DISCRIMINATIVE_OPERATING_POINT_SELECTION_VERSION",
    "VIDEO_DISCRIMINATIVE_PROVIDER_VERSION",
    "VIDEO_DISCRIMINATIVE_REGISTRATION_VERSION",
    "VIDEO_DISCRIMINATIVE_REGION_SPEC_VERSION",
    "build_discriminative_masks",
    "build_discriminative_candidate_set",
    "build_raw_discriminative_candidates",
    "discriminative_mask_digest",
    "discriminative_provider_contract",
    "discriminative_region_digest",
    "evaluate_candidate_eligibility",
    "extract_candidate_region_evidence",
    "rank_discriminative_candidates",
    "register_informative_translation",
]
