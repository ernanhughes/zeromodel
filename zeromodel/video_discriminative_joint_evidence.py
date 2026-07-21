from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import json
from threading import RLock
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError
from .video_discriminative_evidence import (
    REGISTRATION_DISTANCE_TIE_EPSILON,
    _aligned_slices,
    _array_descriptor,
    _coerce_observation_pixels,
    _difference_uint8,
    _immutable_weight_array,
    register_informative_translation,
)
from .visual_address import ImageObservation, VisualAddressContract, VisualAddressDecision
from .visual_registration import RegistrationConfig


VIDEO_JOINT_EVIDENCE_MECHANICS_VERSION = "zeromodel-video-joint-evidence-mechanics/v1"
VIDEO_JOINT_CANDIDATE_SET_VERSION = "zeromodel-video-joint-candidate-set/v1"
VIDEO_JOINT_DECISION_VERSION = "zeromodel-video-joint-evidence-decision/v1"
VIDEO_JOINT_PROVIDER_VERSION = "zeromodel-video-discriminative-provider/v3"
VIDEO_JOINT_REGION_SPEC_VERSION = "zeromodel-video-joint-region-spec/v1"
VIDEO_JOINT_CANDIDATE_MASK_SPEC_VERSION = "zeromodel-video-joint-candidate-mask-spec/v1"
VIDEO_JOINT_CANDIDATE_MASK_PAYLOAD_VERSION = "zeromodel-video-joint-candidate-mask-payload/v1"
VIDEO_JOINT_PAIRWISE_MASK_SPEC_VERSION = "zeromodel-video-pairwise-mask-spec/v1"
VIDEO_JOINT_PAIRWISE_MASK_PAYLOAD_VERSION = "zeromodel-video-pairwise-mask-payload/v1"
VIDEO_JOINT_CALIBRATION_VERSION = "zeromodel-video-joint-calibration/v1"
_ARCHITECTURES = {"A3", "B3", "C3", "D3"}
_BASE_CANDIDATE_CACHE_CAPACITY = 8
_BaseCandidateCacheKey = Tuple[str, str, str, str, str]
_BaseCandidateCacheValue = Tuple[Dict[str, Any], ...]


class _BoundedBaseCandidateCache:
    """Deterministic LRU for the small window of repeated P3 observation scoring."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise VPMValidationError("base candidate cache capacity must be positive")
        self._capacity = int(capacity)
        self._data: OrderedDict[_BaseCandidateCacheKey, _BaseCandidateCacheValue] = OrderedDict()
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def get(self, key: _BaseCandidateCacheKey) -> _BaseCandidateCacheValue | None:
        with self._lock:
            cached = self._data.pop(key, None)
            if cached is None:
                self._misses += 1
                return None
            self._hits += 1
            self._data[key] = cached
            return cached

    def put(self, key: _BaseCandidateCacheKey, value: _BaseCandidateCacheValue) -> _BaseCandidateCacheValue:
        with self._lock:
            self._data.pop(key, None)
            self._data[key] = value
            while len(self._data) > self._capacity:
                self._data.popitem(last=False)
            return value

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._hits = 0
            self._misses = 0

    def info(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "capacity": self._capacity,
                "size": len(self._data),
                "hits": self._hits,
                "misses": self._misses,
                "keys": list(self._data.keys()),
            }


# Eight entries preserve direct A3/B3/C3/D3 architecture sweeps and
# reference/optimized rescoring of the current observation without retaining a
# long split's mostly unique frame observations indefinitely.
_BASE_CANDIDATE_CACHE = _BoundedBaseCandidateCache(_BASE_CANDIDATE_CACHE_CAPACITY)


def _reset_base_candidate_cache() -> None:
    _BASE_CANDIDATE_CACHE.clear()


def _base_candidate_cache_info() -> Dict[str, Any]:
    return _BASE_CANDIDATE_CACHE.info()


def _freeze(value: Any) -> Any:
    if isinstance(value, np.generic):
        raise VPMValidationError("joint-evidence JSON must use plain scalars")
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(_thaw(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("joint-evidence values must be JSON serializable") from exc


def _digest(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _nonempty(name: str, value: str) -> None:
    if not str(value):
        raise VPMValidationError(f"{name} cannot be empty")


def _finite(name: str, value: float) -> None:
    if not np.isfinite(float(value)):
        raise VPMValidationError(f"{name} must be finite")


def _finite_optional(name: str, value: Optional[float]) -> None:
    if value is not None and not np.isfinite(float(value)):
        raise VPMValidationError(f"{name} must be finite when present")


@dataclass(frozen=True)
class JointEvidenceRegionSpec:
    region_id: str
    top: int
    left: int
    height: int
    width: int
    weight: float
    critical: bool
    registration_config: RegistrationConfig
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_JOINT_REGION_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_JOINT_REGION_SPEC_VERSION:
            raise VPMValidationError("unsupported joint region spec version")
        for name in ("region_id",):
            _nonempty(name, getattr(self, name))
        if int(self.top) < 0 or int(self.left) < 0 or int(self.height) <= 0 or int(self.width) <= 0:
            raise VPMValidationError("joint region geometry must be positive and non-negative")
        _finite("weight", float(self.weight))
        if float(self.weight) <= 0.0:
            raise VPMValidationError("joint region weight must be positive")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        _json_bytes(self.metadata)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

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
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True)
class JointCandidateMaskSpec:
    mask_id: str
    row_id: str
    action_id: str
    shape: Tuple[int, int]
    prototype_digest: str
    development_digest: str
    row_informative_pixel_count: int
    stable_pixel_count: int
    candidate_fit_pixel_count: int
    action_conflict_pixel_count: int
    intensity_tolerance: int
    amendment_commit_sha: str
    operational_contract_digest: str
    source_scope: str
    version: str = VIDEO_JOINT_CANDIDATE_MASK_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_JOINT_CANDIDATE_MASK_SPEC_VERSION:
            raise VPMValidationError("unsupported joint candidate mask spec version")
        for name in (
            "mask_id",
            "row_id",
            "action_id",
            "prototype_digest",
            "development_digest",
            "amendment_commit_sha",
            "operational_contract_digest",
            "source_scope",
        ):
            _nonempty(name, getattr(self, name))
        if len(tuple(self.shape)) != 2 or any(int(item) <= 0 for item in self.shape):
            raise VPMValidationError("candidate mask shape must be positive HxW")
        for name in ("row_informative_pixel_count", "stable_pixel_count", "candidate_fit_pixel_count", "action_conflict_pixel_count", "intensity_tolerance"):
            if int(getattr(self, name)) < 0:
                raise VPMValidationError(f"{name} must be non-negative")

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "mask_id": self.mask_id,
            "row_id": self.row_id,
            "action_id": self.action_id,
            "shape": list(self.shape),
            "prototype_digest": self.prototype_digest,
            "development_digest": self.development_digest,
            "row_informative_pixel_count": int(self.row_informative_pixel_count),
            "stable_pixel_count": int(self.stable_pixel_count),
            "candidate_fit_pixel_count": int(self.candidate_fit_pixel_count),
            "action_conflict_pixel_count": int(self.action_conflict_pixel_count),
            "intensity_tolerance": int(self.intensity_tolerance),
            "amendment_commit_sha": self.amendment_commit_sha,
            "operational_contract_digest": self.operational_contract_digest,
            "source_scope": self.source_scope,
        }


@dataclass(frozen=True)
class JointCandidateMask:
    spec: JointCandidateMaskSpec
    row_informative_weights: np.ndarray
    stable_weights: np.ndarray
    candidate_fit_weights: np.ndarray
    action_conflict_weights: np.ndarray
    payload_digest: str = ""
    version: str = VIDEO_JOINT_CANDIDATE_MASK_PAYLOAD_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_JOINT_CANDIDATE_MASK_PAYLOAD_VERSION:
            raise VPMValidationError("unsupported joint candidate mask payload version")
        shape = tuple(self.spec.shape)
        row_info = _immutable_weight_array(self.row_informative_weights, shape=shape, name="row_informative_weights")
        stable = _immutable_weight_array(self.stable_weights, shape=shape, name="stable_weights")
        fit = _immutable_weight_array(self.candidate_fit_weights, shape=shape, name="candidate_fit_weights", clip=True)
        conflict = _immutable_weight_array(self.action_conflict_weights, shape=shape, name="action_conflict_weights", clip=True)
        if int(np.count_nonzero(row_info > 0.0)) != int(self.spec.row_informative_pixel_count):
            raise VPMValidationError("row_informative_weights count mismatch")
        if int(np.count_nonzero(stable > 0.0)) != int(self.spec.stable_pixel_count):
            raise VPMValidationError("stable_weights count mismatch")
        if int(np.count_nonzero(fit > 0.0)) != int(self.spec.candidate_fit_pixel_count):
            raise VPMValidationError("candidate_fit_weights count mismatch")
        if int(np.count_nonzero(conflict > 0.0)) != int(self.spec.action_conflict_pixel_count):
            raise VPMValidationError("action_conflict_weights count mismatch")
        object.__setattr__(self, "row_informative_weights", row_info)
        object.__setattr__(self, "stable_weights", stable)
        object.__setattr__(self, "candidate_fit_weights", fit)
        object.__setattr__(self, "action_conflict_weights", conflict)
        digest = self.payload_digest or _digest(
            {
                "version": self.version,
                "spec_digest": self.spec.digest,
                "row_informative_weights": _array_descriptor(row_info),
                "stable_weights": _array_descriptor(stable),
                "candidate_fit_weights": _array_descriptor(fit),
                "action_conflict_weights": _array_descriptor(conflict),
            }
        )
        object.__setattr__(self, "payload_digest", digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "spec": self.spec.to_dict(),
            "payload_digest": self.payload_digest,
            "row_informative_weights": _array_descriptor(self.row_informative_weights),
            "stable_weights": _array_descriptor(self.stable_weights),
            "candidate_fit_weights": _array_descriptor(self.candidate_fit_weights),
            "action_conflict_weights": _array_descriptor(self.action_conflict_weights),
        }


@dataclass(frozen=True)
class PairwiseDiscriminativeMaskSpec:
    pair_id: str
    row_a: str
    row_b: str
    action_a: str
    action_b: str
    shape: Tuple[int, int]
    prototype_digest: str
    development_digest: str
    pairwise_pixel_count: int
    intensity_tolerance: int
    amendment_commit_sha: str
    operational_contract_digest: str
    source_scope: str
    version: str = VIDEO_JOINT_PAIRWISE_MASK_SPEC_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_JOINT_PAIRWISE_MASK_SPEC_VERSION:
            raise VPMValidationError("unsupported pairwise mask spec version")
        for name in (
            "pair_id",
            "row_a",
            "row_b",
            "action_a",
            "action_b",
            "prototype_digest",
            "development_digest",
            "amendment_commit_sha",
            "operational_contract_digest",
            "source_scope",
        ):
            _nonempty(name, getattr(self, name))
        if self.row_a >= self.row_b:
            raise VPMValidationError("pairwise row identity must be canonical sorted order")
        if len(tuple(self.shape)) != 2 or any(int(item) <= 0 for item in self.shape):
            raise VPMValidationError("pairwise mask shape must be positive HxW")
        if int(self.pairwise_pixel_count) < 0 or int(self.intensity_tolerance) < 0:
            raise VPMValidationError("pairwise mask counts must be non-negative")

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "pair_id": self.pair_id,
            "row_a": self.row_a,
            "row_b": self.row_b,
            "action_a": self.action_a,
            "action_b": self.action_b,
            "shape": list(self.shape),
            "prototype_digest": self.prototype_digest,
            "development_digest": self.development_digest,
            "pairwise_pixel_count": int(self.pairwise_pixel_count),
            "intensity_tolerance": int(self.intensity_tolerance),
            "amendment_commit_sha": self.amendment_commit_sha,
            "operational_contract_digest": self.operational_contract_digest,
            "source_scope": self.source_scope,
        }


@dataclass(frozen=True)
class PairwiseDiscriminativeMask:
    spec: PairwiseDiscriminativeMaskSpec
    pairwise_weights: np.ndarray
    payload_digest: str = ""
    version: str = VIDEO_JOINT_PAIRWISE_MASK_PAYLOAD_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_JOINT_PAIRWISE_MASK_PAYLOAD_VERSION:
            raise VPMValidationError("unsupported pairwise mask payload version")
        weights = _immutable_weight_array(self.pairwise_weights, shape=tuple(self.spec.shape), name="pairwise_weights", clip=True)
        if int(np.count_nonzero(weights > 0.0)) != int(self.spec.pairwise_pixel_count):
            raise VPMValidationError("pairwise mask count mismatch")
        object.__setattr__(self, "pairwise_weights", weights)
        digest = self.payload_digest or _digest(
            {
                "version": self.version,
                "spec_digest": self.spec.digest,
                "pairwise_weights": _array_descriptor(weights),
            }
        )
        object.__setattr__(self, "payload_digest", digest)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "spec": self.spec.to_dict(),
            "payload_digest": self.payload_digest,
            "pairwise_weights": _array_descriptor(self.pairwise_weights),
        }


@dataclass(frozen=True)
class JointEvidenceCalibration:
    architecture_id: str
    minimum_actual_scored_mass: float
    minimum_available_candidate_fit_fraction: float
    minimum_candidate_joint_fit: float
    minimum_pairwise_margin: float
    minimum_conflicting_action_margin: float
    exact_winner_threshold: float
    exact_winner_margin: float
    candidate_relative_margin: float
    maximum_candidate_set_size: int
    prototype_digest: str
    region_spec_digest: str
    candidate_mask_digest: str
    pairwise_mask_digest: str
    policy_artifact_id: str
    source_scope: str
    amendment_commit_sha: str
    operational_contract_digest: str
    provider_version: str = VIDEO_JOINT_PROVIDER_VERSION
    version: str = VIDEO_JOINT_CALIBRATION_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_JOINT_CALIBRATION_VERSION:
            raise VPMValidationError("unsupported joint calibration version")
        if self.architecture_id not in _ARCHITECTURES:
            raise VPMValidationError("unsupported joint architecture_id")
        for name in (
            "minimum_actual_scored_mass",
            "minimum_available_candidate_fit_fraction",
            "minimum_candidate_joint_fit",
            "minimum_pairwise_margin",
            "minimum_conflicting_action_margin",
            "exact_winner_threshold",
            "exact_winner_margin",
            "candidate_relative_margin",
        ):
            _finite(name, float(getattr(self, name)))
        if not (0.0 <= float(self.minimum_available_candidate_fit_fraction) <= 1.0):
            raise VPMValidationError("minimum_available_candidate_fit_fraction must be in [0, 1]")
        if not (-1.0 <= float(self.minimum_pairwise_margin) <= 1.0):
            raise VPMValidationError("minimum_pairwise_margin must be in [-1, 1]")
        if not (-1.0 <= float(self.minimum_conflicting_action_margin) <= 1.0):
            raise VPMValidationError("minimum_conflicting_action_margin must be in [-1, 1]")
        if int(self.maximum_candidate_set_size) < 1:
            raise VPMValidationError("maximum_candidate_set_size must be positive")
        for name in (
            "prototype_digest",
            "region_spec_digest",
            "candidate_mask_digest",
            "pairwise_mask_digest",
            "policy_artifact_id",
            "source_scope",
            "amendment_commit_sha",
            "operational_contract_digest",
        ):
            _nonempty(name, getattr(self, name))

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "provider_version": self.provider_version,
            "architecture_id": self.architecture_id,
            "minimum_actual_scored_mass": float(self.minimum_actual_scored_mass),
            "minimum_available_candidate_fit_fraction": float(self.minimum_available_candidate_fit_fraction),
            "minimum_candidate_joint_fit": float(self.minimum_candidate_joint_fit),
            "minimum_pairwise_margin": float(self.minimum_pairwise_margin),
            "minimum_conflicting_action_margin": float(self.minimum_conflicting_action_margin),
            "exact_winner_threshold": float(self.exact_winner_threshold),
            "exact_winner_margin": float(self.exact_winner_margin),
            "candidate_relative_margin": float(self.candidate_relative_margin),
            "maximum_candidate_set_size": int(self.maximum_candidate_set_size),
            "prototype_digest": self.prototype_digest,
            "region_spec_digest": self.region_spec_digest,
            "candidate_mask_digest": self.candidate_mask_digest,
            "pairwise_mask_digest": self.pairwise_mask_digest,
            "policy_artifact_id": self.policy_artifact_id,
            "source_scope": self.source_scope,
            "amendment_commit_sha": self.amendment_commit_sha,
            "operational_contract_digest": self.operational_contract_digest,
        }


@dataclass(frozen=True)
class JointRegionFitEvidence:
    region_id: str
    registration_distance: float
    registration_score: float
    geometric_overlap: float
    available_candidate_fit_mass: float
    available_candidate_fit_fraction: float
    actual_scored_mass: float
    direct_similarity: float
    joint_fit: float
    registration_dx: int
    registration_dy: int
    registration_tie_break_reason: Optional[str]
    registration_contract_digest: str

    def __post_init__(self) -> None:
        for name in (
            "registration_distance",
            "registration_score",
            "geometric_overlap",
            "available_candidate_fit_mass",
            "available_candidate_fit_fraction",
            "actual_scored_mass",
            "direct_similarity",
            "joint_fit",
        ):
            _finite(name, float(getattr(self, name)))
        if float(self.available_candidate_fit_mass) < 0.0 or float(self.actual_scored_mass) < 0.0:
            raise VPMValidationError("region masses must be non-negative")
        _nonempty("region_id", self.region_id)
        _nonempty("registration_contract_digest", self.registration_contract_digest)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PairwiseRegionEvidence:
    region_id: str
    competitor_row_id: str
    competitor_action_id: str
    pairwise_mask_digest: str
    pairwise_mass: float
    candidate_fit: float
    competitor_fit: float
    margin: float
    conflicting_action: bool
    neutral_mass_reason: Optional[str]
    registration_dx: int
    registration_dy: int

    def __post_init__(self) -> None:
        for name in ("pairwise_mass", "candidate_fit", "competitor_fit", "margin"):
            _finite(name, float(getattr(self, name)))
        if float(self.pairwise_mass) < 0.0:
            raise VPMValidationError("pairwise_mass must be non-negative")
        _nonempty("region_id", self.region_id)
        _nonempty("competitor_row_id", self.competitor_row_id)
        _nonempty("competitor_action_id", self.competitor_action_id)
        _nonempty("pairwise_mask_digest", self.pairwise_mask_digest)

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class JointRowCandidate:
    row_id: str
    action_id: str
    prototype_observation_id: str
    architecture_id: str
    candidate_strength: float
    candidate_joint_fit: float
    minimum_pairwise_margin: Optional[float]
    minimum_conflicting_action_margin: Optional[float]
    actual_scored_mass: float
    available_candidate_fit_mass: float
    available_candidate_fit_fraction: float
    declared_informative_mass: float
    stable_informative_mass: float
    available_geometric_mass: float
    pairwise_discriminative_mass: float
    candidate_superiority_margin: Optional[float]
    semantic_tie_group_size: int
    semantic_tie_group_rows: Tuple[str, ...]
    winner_selected_by_semantic_strength: bool
    trace_order: int
    exact_winner_margin: Optional[float]
    candidate_relative_margin: Optional[float]
    eligible_for_candidate_set: bool
    eligible_for_exact: bool
    ineligibility_reasons: Tuple[str, ...]
    region_evidence: Tuple[JointRegionFitEvidence, ...]
    pairwise_evidence: Tuple[PairwiseRegionEvidence, ...]

    def __post_init__(self) -> None:
        for name in (
            "candidate_strength",
            "candidate_joint_fit",
            "actual_scored_mass",
            "available_candidate_fit_mass",
            "available_candidate_fit_fraction",
            "declared_informative_mass",
            "stable_informative_mass",
            "available_geometric_mass",
            "pairwise_discriminative_mass",
        ):
            _finite(name, float(getattr(self, name)))
        for name in ("minimum_pairwise_margin", "minimum_conflicting_action_margin", "candidate_superiority_margin", "exact_winner_margin", "candidate_relative_margin"):
            _finite_optional(name, getattr(self, name))
        for name in ("row_id", "action_id", "prototype_observation_id", "architecture_id"):
            _nonempty(name, getattr(self, name))

    def to_dict(self) -> Dict[str, Any]:
        payload = self.__dict__.copy()
        payload["region_evidence"] = [item.to_dict() for item in self.region_evidence]
        payload["pairwise_evidence"] = [item.to_dict() for item in self.pairwise_evidence]
        payload["semantic_tie_group_rows"] = list(self.semantic_tie_group_rows)
        payload["ineligibility_reasons"] = list(self.ineligibility_reasons)
        return payload


@dataclass(frozen=True)
class JointCandidateSet:
    architecture_id: str
    outcome: str
    rows: Tuple[str, ...]
    rejection_reason: Optional[str]
    qualifying_rows: Tuple[str, ...]
    version: str = VIDEO_JOINT_CANDIDATE_SET_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_JOINT_CANDIDATE_SET_VERSION:
            raise VPMValidationError("unsupported joint candidate set version")
        if self.outcome not in {"exact_row_accepted", "candidate_set_available", "no_sufficient_evidence"}:
            raise VPMValidationError("unsupported joint candidate set outcome")
        _nonempty("architecture_id", self.architecture_id)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "architecture_id": self.architecture_id,
            "outcome": self.outcome,
            "rows": list(self.rows),
            "rejection_reason": self.rejection_reason,
            "qualifying_rows": list(self.qualifying_rows),
        }


def joint_candidate_mask_digest(masks: Sequence[JointCandidateMaskSpec]) -> str:
    seen = set()
    rows = []
    for spec in sorted(masks, key=lambda item: item.row_id):
        if spec.row_id in seen:
            raise VPMValidationError("duplicate joint candidate mask row_id")
        seen.add(spec.row_id)
        rows.append(spec.to_dict())
    return _digest(rows)


def pairwise_mask_digest(masks: Sequence[PairwiseDiscriminativeMaskSpec]) -> str:
    seen = set()
    rows = []
    for spec in sorted(masks, key=lambda item: (item.row_a, item.row_b)):
        key = (spec.row_a, spec.row_b)
        if key in seen:
            raise VPMValidationError("duplicate pairwise mask identity")
        seen.add(key)
        rows.append(spec.to_dict())
    return _digest(rows)


def joint_region_digest(regions: Sequence[JointEvidenceRegionSpec]) -> str:
    seen = set()
    rows = []
    for region in sorted(regions, key=lambda item: item.region_id):
        if region.region_id in seen:
            raise VPMValidationError("duplicate joint region_id")
        seen.add(region.region_id)
        rows.append(region.to_dict())
    return _digest(rows)


def _crop(array: np.ndarray, region: JointEvidenceRegionSpec) -> np.ndarray:
    return array[int(region.top) : int(region.top) + int(region.height), int(region.left) : int(region.left) + int(region.width)]


def build_joint_candidate_masks(
    *,
    prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
    development_observations: Mapping[str, Sequence[ImageObservation]],
    intensity_tolerance: int,
    stability_tolerance: int,
    amendment_commit_sha: str,
    operational_contract_digest: str,
    source_scope: str,
) -> Mapping[str, JointCandidateMask]:
    shape = None
    rows = {}
    for row_id, (prototype_observation_id, action_id, prototype_digest, observation) in sorted(prototypes.items()):
        pixels = _coerce_observation_pixels(observation)
        shape = shape or pixels.shape
        if pixels.shape != shape:
            raise VPMValidationError("joint candidate prototypes must share geometry")
        rows[row_id] = {
            "observation_id": prototype_observation_id,
            "action_id": action_id,
            "prototype_digest": prototype_digest,
            "pixels": pixels,
        }
    development_digest = _digest({row_id: [item.raw_digest for item in values] for row_id, values in sorted(development_observations.items())})
    results = {}
    for row_id, entry in rows.items():
        candidate = entry["pixels"]
        competitors = [other for other_id, other in rows.items() if other_id != row_id]
        diff_stack = np.stack([_difference_uint8(candidate, other["pixels"]) for other in competitors], axis=0)
        row_informative = (diff_stack.max(axis=0) > int(intensity_tolerance)).astype(np.float32)
        conflict_competitors = [other for other in competitors if other["action_id"] != entry["action_id"]]
        if conflict_competitors:
            action_conflict = (np.stack([_difference_uint8(candidate, other["pixels"]) for other in conflict_competitors], axis=0).max(axis=0) > int(intensity_tolerance)).astype(np.float32)
        else:
            action_conflict = np.zeros(shape, dtype=np.float32)
        development = tuple(development_observations.get(row_id, ()))
        if not development:
            raise VPMValidationError("joint candidate masks require development observations for every row")
        variation_stack = np.stack([_difference_uint8(candidate, _coerce_observation_pixels(item)) for item in development], axis=0)
        stable = (variation_stack.max(axis=0) <= int(stability_tolerance)).astype(np.float32)
        candidate_fit = row_informative * stable
        spec = JointCandidateMaskSpec(
            mask_id=f"{row_id}|joint-mask",
            row_id=row_id,
            action_id=entry["action_id"],
            shape=shape,
            prototype_digest=entry["prototype_digest"],
            development_digest=development_digest,
            row_informative_pixel_count=int(np.count_nonzero(row_informative > 0.0)),
            stable_pixel_count=int(np.count_nonzero(stable > 0.0)),
            candidate_fit_pixel_count=int(np.count_nonzero(candidate_fit > 0.0)),
            action_conflict_pixel_count=int(np.count_nonzero(action_conflict > 0.0)),
            intensity_tolerance=int(intensity_tolerance),
            amendment_commit_sha=amendment_commit_sha,
            operational_contract_digest=operational_contract_digest,
            source_scope=source_scope,
        )
        results[row_id] = JointCandidateMask(
            spec=spec,
            row_informative_weights=row_informative,
            stable_weights=stable,
            candidate_fit_weights=candidate_fit,
            action_conflict_weights=action_conflict,
        )
    return results


def build_pairwise_discriminative_masks(
    *,
    prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
    candidate_masks: Mapping[str, JointCandidateMask],
    intensity_tolerance: int,
    amendment_commit_sha: str,
    operational_contract_digest: str,
    source_scope: str,
) -> Mapping[Tuple[str, str], PairwiseDiscriminativeMask]:
    prototype_digest = _digest({row_id: value[2] for row_id, value in sorted(prototypes.items())})
    development_digest = _digest({row_id: candidate_masks[row_id].spec.development_digest for row_id in sorted(candidate_masks)})
    results = {}
    row_ids = sorted(prototypes)
    for index, row_a in enumerate(row_ids):
        pixels_a = _coerce_observation_pixels(prototypes[row_a][3])
        mask_a = candidate_masks[row_a]
        for row_b in row_ids[index + 1 :]:
            pixels_b = _coerce_observation_pixels(prototypes[row_b][3])
            mask_b = candidate_masks[row_b]
            pairwise_stable = np.minimum(mask_a.stable_weights, mask_b.stable_weights)
            pairwise_difference = (np.abs(pixels_a.astype(np.int16) - pixels_b.astype(np.int16)) > int(intensity_tolerance)).astype(np.float32)
            weights = pairwise_stable * pairwise_difference
            spec = PairwiseDiscriminativeMaskSpec(
                pair_id=f"{row_a}|{row_b}|pairwise-mask",
                row_a=row_a,
                row_b=row_b,
                action_a=prototypes[row_a][1],
                action_b=prototypes[row_b][1],
                shape=tuple(pixels_a.shape),
                prototype_digest=prototype_digest,
                development_digest=development_digest,
                pairwise_pixel_count=int(np.count_nonzero(weights > 0.0)),
                intensity_tolerance=int(intensity_tolerance),
                amendment_commit_sha=amendment_commit_sha,
                operational_contract_digest=operational_contract_digest,
                source_scope=source_scope,
            )
            results[(row_a, row_b)] = PairwiseDiscriminativeMask(spec=spec, pairwise_weights=weights)
    return results


def _weighted_similarity(candidate: np.ndarray, observation: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
    total = float(weights.sum(dtype=np.float64))
    if total <= 0.0:
        return (0.0, 0.0)
    error = np.abs(observation.astype(np.float32) - candidate.astype(np.float32)) / 255.0
    weighted_error = float((error * weights).sum(dtype=np.float64))
    return (max(0.0, min(1.0, 1.0 - weighted_error / total)), total)


def _candidate_pair_key(row_a: str, row_b: str) -> Tuple[str, str]:
    return (row_a, row_b) if row_a < row_b else (row_b, row_a)


def _build_candidate_base(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
    candidate_masks: Mapping[str, JointCandidateMask],
    pairwise_masks: Mapping[Tuple[str, str], PairwiseDiscriminativeMask],
    regions: Sequence[JointEvidenceRegionSpec],
) -> Tuple[Dict[str, Any], ...]:
    bases = []
    observation_pixels = _coerce_observation_pixels(observation)
    pair_alignment_cache: Dict[Tuple[Tuple[str, str], str], Dict[str, Any]] = {}
    for row_id in sorted(prototypes):
        prototype_observation_id, action_id, _prototype_digest, prototype_observation = prototypes[row_id]
        candidate_pixels = _coerce_observation_pixels(prototype_observation)
        mask = candidate_masks[row_id]
        region_evidence = []
        aligned_by_region: Dict[str, Dict[str, Any]] = {}
        candidate_regions = {region.region_id: _crop(candidate_pixels, region) for region in regions}
        total_declared = float(mask.row_informative_weights.sum(dtype=np.float64))
        total_stable = float((mask.row_informative_weights * mask.stable_weights).sum(dtype=np.float64))
        total_available_geom = 0.0
        total_candidate_fit = 0.0
        weighted_b3_num = 0.0
        weighted_b3_den = 0.0
        weighted_a3_num = 0.0
        weighted_a3_den = 0.0
        for region in regions:
            candidate_region = candidate_regions[region.region_id]
            observation_region = _crop(observation_pixels, region)
            candidate_fit_region = _crop(mask.candidate_fit_weights, region)
            registration = register_informative_translation(
                candidate_region,
                observation_region,
                informative_weights=candidate_fit_region,
                region_id=region.region_id,
                config=region.registration_config,
            )
            if registration.registration_succeeded:
                dx = int(registration.dx)
                dy = int(registration.dy)
                aligned_candidate = _aligned_slices(candidate_region, dx=dx, dy=dy, observation=False)
                aligned_observation = _aligned_slices(observation_region, dx=dx, dy=dy, observation=True)
                aligned_mask = _aligned_slices(candidate_fit_region, dx=dx, dy=dy, observation=False).astype(np.float32)
                joint_fit, scored_mass = _weighted_similarity(aligned_candidate, aligned_observation, aligned_mask)
                direct_similarity = max(0.0, 1.0 - float(registration.distance) / 2.0)
                available_fraction = float(registration.available_informative_fraction)
                available_mass = float(registration.available_informative_mass)
                total_available_geom += available_mass
                total_candidate_fit += scored_mass
                weighted_b3_num += float(region.weight) * (1.0 - joint_fit) * scored_mass
                weighted_b3_den += float(region.weight) * scored_mass
                weighted_a3_num += float(region.weight) * available_fraction * direct_similarity
                weighted_a3_den += float(region.weight) * available_fraction
                aligned_by_region[region.region_id] = {
                    "dx": dx,
                    "dy": dy,
                    "aligned_observation": aligned_observation,
                    "aligned_candidate": aligned_candidate,
                }
                region_evidence.append(
                    JointRegionFitEvidence(
                        region_id=region.region_id,
                        registration_distance=float(registration.distance),
                        registration_score=float(registration.score),
                        geometric_overlap=float(registration.geometric_overlap),
                        available_candidate_fit_mass=available_mass,
                        available_candidate_fit_fraction=available_fraction,
                        actual_scored_mass=scored_mass,
                        direct_similarity=direct_similarity,
                        joint_fit=joint_fit,
                        registration_dx=dx,
                        registration_dy=dy,
                        registration_tie_break_reason=registration.tie_break_reason,
                        registration_contract_digest=region.registration_config.digest,
                    )
                )
            else:
                aligned_by_region[region.region_id] = {
                    "dx": int(registration.dx),
                    "dy": int(registration.dy),
                    "aligned_observation": None,
                    "aligned_candidate": None,
                }
                region_evidence.append(
                    JointRegionFitEvidence(
                        region_id=region.region_id,
                        registration_distance=2.0,
                        registration_score=0.0,
                        geometric_overlap=float(registration.geometric_overlap),
                        available_candidate_fit_mass=float(registration.available_informative_mass),
                        available_candidate_fit_fraction=float(registration.available_informative_fraction),
                        actual_scored_mass=0.0,
                        direct_similarity=0.0,
                        joint_fit=0.0,
                        registration_dx=int(registration.dx),
                        registration_dy=int(registration.dy),
                        registration_tie_break_reason=registration.tie_break_reason,
                        registration_contract_digest=region.registration_config.digest,
                    )
                )
        candidate_joint_fit = 0.0 if weighted_b3_den <= 0.0 else max(0.0, min(1.0, 1.0 - weighted_b3_num / weighted_b3_den))
        a3_strength = 0.0 if weighted_a3_den <= 0.0 else max(0.0, min(1.0, weighted_a3_num / weighted_a3_den))
        pairwise_entries = []
        pairwise_masses = []
        pairwise_margins = []
        conflicting_margins = []
        for competitor_row, (_obs_id, competitor_action, _digest, competitor_observation) in sorted(prototypes.items()):
            if competitor_row == row_id:
                continue
            pair_key = _candidate_pair_key(row_id, competitor_row)
            pairwise_mask = pairwise_masks[pair_key]
            competitor_pixels = _coerce_observation_pixels(competitor_observation)
            competitor_mass = 0.0
            competitor_margin_numerator = 0.0
            for region in regions:
                competitor_region = _crop(competitor_pixels, region)
                weights = _crop(pairwise_mask.pairwise_weights, region)
                cache_id = (pair_key, region.region_id)
                pair_aligned = pair_alignment_cache.get(cache_id)
                if pair_aligned is None:
                    anchor_row = pair_key[0]
                    anchor_pixels = _coerce_observation_pixels(prototypes[anchor_row][3])
                    anchor_region = _crop(anchor_pixels, region)
                    observation_region = _crop(observation_pixels, region)
                    anchor_mask = _crop(candidate_masks[anchor_row].candidate_fit_weights, region)
                    registration = register_informative_translation(
                        anchor_region,
                        observation_region,
                        informative_weights=anchor_mask,
                        region_id=region.region_id,
                        config=region.registration_config,
                    )
                    if registration.registration_succeeded:
                        pair_aligned = {
                            "dx": int(registration.dx),
                            "dy": int(registration.dy),
                            "aligned_observation": _aligned_slices(observation_region, dx=int(registration.dx), dy=int(registration.dy), observation=True),
                        }
                    else:
                        pair_aligned = {
                            "dx": int(registration.dx),
                            "dy": int(registration.dy),
                            "aligned_observation": None,
                        }
                    pair_alignment_cache[cache_id] = pair_aligned
                if pair_aligned["aligned_observation"] is not None:
                    dx = int(pair_aligned["dx"])
                    dy = int(pair_aligned["dy"])
                    aligned_weights = _aligned_slices(weights, dx=dx, dy=dy, observation=False).astype(np.float32)
                    aligned_candidate = _aligned_slices(candidate_regions[region.region_id], dx=dx, dy=dy, observation=False)
                    aligned_competitor = _aligned_slices(competitor_region, dx=dx, dy=dy, observation=False)
                    aligned_observation = pair_aligned["aligned_observation"]
                    fit_c, mass = _weighted_similarity(aligned_candidate, aligned_observation, aligned_weights)
                    fit_j, _ = _weighted_similarity(aligned_competitor, aligned_observation, aligned_weights)
                    neutral_reason = None if mass > 0.0 else "zero_pairwise_mass"
                    margin = fit_c - fit_j if mass > 0.0 else 0.0
                else:
                    dx = int(pair_aligned["dx"])
                    dy = int(pair_aligned["dy"])
                    mass = 0.0
                    fit_c = 0.0
                    fit_j = 0.0
                    margin = 0.0
                    neutral_reason = "registration_failed"
                pairwise_entries.append(
                    PairwiseRegionEvidence(
                        region_id=region.region_id,
                        competitor_row_id=competitor_row,
                        competitor_action_id=competitor_action,
                        pairwise_mask_digest=pairwise_mask.payload_digest,
                        pairwise_mass=float(mass),
                        candidate_fit=float(fit_c),
                        competitor_fit=float(fit_j),
                        margin=float(margin),
                        conflicting_action=bool(competitor_action != action_id),
                        neutral_mass_reason=neutral_reason,
                        registration_dx=dx,
                        registration_dy=dy,
                    )
                )
                competitor_mass += float(mass)
                competitor_margin_numerator += float(margin) * float(mass)
            mean_margin = float(competitor_margin_numerator / competitor_mass) if competitor_mass > 0.0 else 0.0
            pairwise_masses.append(float(competitor_mass))
            pairwise_margins.append(mean_margin)
            if competitor_action != action_id:
                conflicting_margins.append(mean_margin)
        min_pairwise_margin = min(pairwise_margins) if pairwise_margins else None
        min_conflicting_margin = min(conflicting_margins) if conflicting_margins else None
        bases.append(
            {
                "row_id": row_id,
                "action_id": action_id,
                "prototype_observation_id": prototype_observation_id,
                "candidate_joint_fit": float(candidate_joint_fit),
                "a3_strength": float(a3_strength),
                "minimum_pairwise_margin": None if min_pairwise_margin is None else float(min_pairwise_margin),
                "minimum_conflicting_action_margin": None if min_conflicting_margin is None else float(min_conflicting_margin),
                "available_candidate_fit_mass": float(total_candidate_fit),
                "available_candidate_fit_fraction": 0.0 if total_stable <= 0.0 else float(total_candidate_fit / total_stable),
                "declared_informative_mass": float(total_declared),
                "stable_informative_mass": float(total_stable),
                "available_geometric_mass": float(total_available_geom),
                "pairwise_discriminative_mass": float(sum(pairwise_masses)),
                "region_evidence": tuple(region_evidence),
                "pairwise_evidence": tuple(pairwise_entries),
            }
        )
    return tuple(bases)


def _materialize_candidate(base: Mapping[str, Any], architecture_id: str) -> JointRowCandidate:
    c3_component = 0.5 + 0.5 * (0.0 if base["minimum_pairwise_margin"] is None else float(base["minimum_pairwise_margin"]))
    d3_component = min(float(base["candidate_joint_fit"]), max(0.0, min(1.0, c3_component)))
    strength = {
        "A3": float(base["a3_strength"]),
        "B3": float(base["candidate_joint_fit"]),
        "C3": max(0.0, min(1.0, c3_component)),
        "D3": max(0.0, min(1.0, d3_component)),
    }[architecture_id]
    return JointRowCandidate(
        row_id=str(base["row_id"]),
        action_id=str(base["action_id"]),
        prototype_observation_id=str(base["prototype_observation_id"]),
        architecture_id=architecture_id,
        candidate_strength=float(strength),
        candidate_joint_fit=float(base["candidate_joint_fit"]),
        minimum_pairwise_margin=base["minimum_pairwise_margin"],
        minimum_conflicting_action_margin=base["minimum_conflicting_action_margin"],
        actual_scored_mass=float(base["available_candidate_fit_mass"] if architecture_id in {"A3", "B3", "D3"} else base["pairwise_discriminative_mass"]),
        available_candidate_fit_mass=float(base["available_candidate_fit_mass"]),
        available_candidate_fit_fraction=float(base["available_candidate_fit_fraction"]),
        declared_informative_mass=float(base["declared_informative_mass"]),
        stable_informative_mass=float(base["stable_informative_mass"]),
        available_geometric_mass=float(base["available_geometric_mass"]),
        pairwise_discriminative_mass=float(base["pairwise_discriminative_mass"]),
        candidate_superiority_margin=None,
        semantic_tie_group_size=1,
        semantic_tie_group_rows=(str(base["row_id"]),),
        winner_selected_by_semantic_strength=False,
        trace_order=0,
        exact_winner_margin=None,
        candidate_relative_margin=None,
        eligible_for_candidate_set=False,
        eligible_for_exact=False,
        ineligibility_reasons=(),
        region_evidence=tuple(base["region_evidence"]),
        pairwise_evidence=tuple(base["pairwise_evidence"]),
    )


def _build_candidate(
    *,
    observation: ImageObservation,
    row_id: str,
    architecture_id: str,
    prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
    candidate_masks: Mapping[str, JointCandidateMask],
    pairwise_masks: Mapping[Tuple[str, str], PairwiseDiscriminativeMask],
    regions: Sequence[JointEvidenceRegionSpec],
) -> JointRowCandidate:
    base = next(item for item in _build_candidate_base(observation=observation, prototypes=prototypes, candidate_masks=candidate_masks, pairwise_masks=pairwise_masks, regions=regions) if item["row_id"] == row_id)
    return _materialize_candidate(base, architecture_id)


def build_joint_row_candidates(
    *,
    observation: ImageObservation,
    prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
    candidate_masks: Mapping[str, JointCandidateMask],
    pairwise_masks: Mapping[Tuple[str, str], PairwiseDiscriminativeMask],
    regions: Sequence[JointEvidenceRegionSpec],
    architecture_id: str,
) -> Tuple[JointRowCandidate, ...]:
    if architecture_id not in _ARCHITECTURES:
        raise VPMValidationError("unsupported joint architecture_id")
    cache_key = (
        observation.raw_digest,
        _digest({row_id: value[2] for row_id, value in sorted(prototypes.items())}),
        _digest({row_id: mask.payload_digest for row_id, mask in sorted(candidate_masks.items())}),
        _digest({f"{row_a}|{row_b}": mask.payload_digest for (row_a, row_b), mask in sorted(pairwise_masks.items())}),
        _digest([region.digest for region in regions]),
    )
    bases = _BASE_CANDIDATE_CACHE.get(cache_key)
    if bases is None:
        bases = _build_candidate_base(
            observation=observation,
            prototypes=prototypes,
            candidate_masks=candidate_masks,
            pairwise_masks=pairwise_masks,
            regions=regions,
        )
        bases = _BASE_CANDIDATE_CACHE.put(cache_key, bases)
    return tuple(_materialize_candidate(base, architecture_id) for base in bases)


def rank_joint_row_candidates(candidates: Sequence[JointRowCandidate]) -> Tuple[JointRowCandidate, ...]:
    if not candidates:
        raise VPMValidationError("rank_joint_row_candidates requires candidates")
    ranked = sorted(
        candidates,
        key=lambda item: (
            -float(item.candidate_strength),
            -float(item.actual_scored_mass),
            -float(item.available_candidate_fit_mass),
            -(-1.0 if item.minimum_pairwise_margin is None else float(item.minimum_pairwise_margin)),
            -(-1.0 if item.minimum_conflicting_action_margin is None else float(item.minimum_conflicting_action_margin)),
            item.row_id,
            item.prototype_observation_id,
        ),
    )
    top_strength = ranked[0].candidate_strength
    tie_rows = tuple(item.row_id for item in ranked if abs(item.candidate_strength - top_strength) <= REGISTRATION_DISTANCE_TIE_EPSILON)
    winner = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    enriched = []
    for index, candidate in enumerate(ranked):
        superiority = None if candidate.row_id != winner.row_id or runner_up is None else float(candidate.candidate_strength - runner_up.candidate_strength)
        relative_margin = None if candidate.row_id == winner.row_id else float(winner.candidate_strength - candidate.candidate_strength)
        exact_margin = superiority if candidate.row_id == winner.row_id else relative_margin
        enriched.append(
            JointRowCandidate(
                **{
                    **candidate.__dict__,
                    "candidate_superiority_margin": superiority if candidate.row_id == winner.row_id else relative_margin,
                    "semantic_tie_group_size": len(tie_rows) if abs(candidate.candidate_strength - top_strength) <= REGISTRATION_DISTANCE_TIE_EPSILON else 1,
                    "semantic_tie_group_rows": tie_rows if abs(candidate.candidate_strength - top_strength) <= REGISTRATION_DISTANCE_TIE_EPSILON else (candidate.row_id,),
                    "winner_selected_by_semantic_strength": bool(candidate.row_id == winner.row_id and len(tie_rows) == 1),
                    "trace_order": index,
                    "exact_winner_margin": exact_margin,
                    "candidate_relative_margin": relative_margin,
                }
            )
        )
    return tuple(enriched)


def evaluate_joint_candidate_eligibility(
    *,
    candidate: JointRowCandidate,
    ranked_candidates: Sequence[JointRowCandidate],
    calibration: JointEvidenceCalibration,
) -> JointRowCandidate:
    reasons = []
    if candidate.actual_scored_mass + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.minimum_actual_scored_mass:
        reasons.append("minimum_actual_scored_mass")
    if candidate.available_candidate_fit_fraction + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.minimum_available_candidate_fit_fraction:
        reasons.append("minimum_available_candidate_fit_fraction")
    if candidate.candidate_joint_fit + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.minimum_candidate_joint_fit:
        reasons.append("minimum_candidate_joint_fit")
    if candidate.minimum_pairwise_margin is not None and candidate.minimum_pairwise_margin + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.minimum_pairwise_margin:
        reasons.append("minimum_pairwise_margin")
    if candidate.minimum_conflicting_action_margin is not None and candidate.minimum_conflicting_action_margin + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.minimum_conflicting_action_margin:
        reasons.append("minimum_conflicting_action_margin")
    if candidate.candidate_relative_margin is not None and candidate.candidate_relative_margin + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.candidate_relative_margin:
        reasons.append("candidate_relative_margin")
    eligible_for_candidate_set = len(reasons) == 0
    exact_reasons = list(reasons)
    winner = ranked_candidates[0]
    runner_up = ranked_candidates[1] if len(ranked_candidates) > 1 else None
    strict_superiority = bool(
        candidate.row_id == winner.row_id
        and runner_up is not None
        and candidate.candidate_strength > runner_up.candidate_strength + REGISTRATION_DISTANCE_TIE_EPSILON
    )
    if candidate.candidate_strength + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.exact_winner_threshold:
        exact_reasons.append("exact_winner_threshold")
    if not strict_superiority:
        exact_reasons.append("strict_superiority")
    if candidate.candidate_superiority_margin is None or candidate.candidate_superiority_margin + REGISTRATION_DISTANCE_TIE_EPSILON < calibration.exact_winner_margin:
        exact_reasons.append("exact_winner_margin")
    return JointRowCandidate(
        **{
            **candidate.__dict__,
            "eligible_for_candidate_set": eligible_for_candidate_set,
            "eligible_for_exact": len(exact_reasons) == 0,
            "ineligibility_reasons": tuple(exact_reasons),
        }
    )


def build_joint_candidate_set(
    *,
    ranked_candidates: Sequence[JointRowCandidate],
    calibration: JointEvidenceCalibration,
) -> JointCandidateSet:
    qualifying = tuple(candidate.row_id for candidate in ranked_candidates if candidate.eligible_for_candidate_set)
    exact = next((candidate for candidate in ranked_candidates if candidate.eligible_for_exact), None)
    if exact is not None:
        return JointCandidateSet(architecture_id=calibration.architecture_id, outcome="exact_row_accepted", rows=(exact.row_id,), rejection_reason=None, qualifying_rows=qualifying)
    if len(qualifying) == 0:
        return JointCandidateSet(architecture_id=calibration.architecture_id, outcome="no_sufficient_evidence", rows=(), rejection_reason="no_qualifying_candidates", qualifying_rows=qualifying)
    top_strength = max(candidate.candidate_strength for candidate in ranked_candidates if candidate.row_id in qualifying)
    top_rows = tuple(candidate.row_id for candidate in ranked_candidates if candidate.row_id in qualifying and abs(candidate.candidate_strength - top_strength) <= REGISTRATION_DISTANCE_TIE_EPSILON)
    if len(top_rows) > calibration.maximum_candidate_set_size or len(qualifying) > calibration.maximum_candidate_set_size:
        return JointCandidateSet(architecture_id=calibration.architecture_id, outcome="no_sufficient_evidence", rows=(), rejection_reason="candidate_set_too_large", qualifying_rows=qualifying)
    return JointCandidateSet(architecture_id=calibration.architecture_id, outcome="candidate_set_available", rows=tuple(qualifying), rejection_reason=None, qualifying_rows=qualifying)


def joint_evidence_provider_contract(*, calibration: JointEvidenceCalibration) -> VisualAddressContract:
    representation_digest = _digest(
        {
            "provider_version": VIDEO_JOINT_PROVIDER_VERSION,
            "calibration": calibration.to_dict(),
        }
    )
    return VisualAddressContract(
        provider_kind="joint-discriminative-evidence",
        provider_version=VIDEO_JOINT_PROVIDER_VERSION,
        score_semantics="similarity",
        observation_spec_digest="zeromodel-image-observation/v1",
        representation_spec_digest=representation_digest,
        address_artifact_id=representation_digest,
        calibration_artifact_id=calibration.digest,
        policy_artifact_id=calibration.policy_artifact_id,
        source_scope=calibration.source_scope,
        metadata={"architecture_id": calibration.architecture_id, "mechanics_version": VIDEO_JOINT_EVIDENCE_MECHANICS_VERSION},
    )


class JointEvidenceProvider:
    def __init__(
        self,
        *,
        prototypes: Mapping[str, Tuple[str, str, str, ImageObservation]],
        candidate_masks: Mapping[str, JointCandidateMask],
        pairwise_masks: Mapping[Tuple[str, str], PairwiseDiscriminativeMask],
        regions: Sequence[JointEvidenceRegionSpec],
        calibration: JointEvidenceCalibration,
        policy_artifact_id: str,
        source_scope: str,
    ) -> None:
        if calibration.policy_artifact_id != policy_artifact_id:
            raise VPMValidationError("joint calibration policy artifact mismatch")
        if calibration.source_scope != source_scope:
            raise VPMValidationError("joint calibration source scope mismatch")
        self._prototypes = dict(prototypes)
        self._candidate_masks = dict(candidate_masks)
        self._pairwise_masks = dict(pairwise_masks)
        self._regions = tuple(regions)
        self._calibration = calibration
        self._policy_artifact_id = policy_artifact_id
        self._source_scope = source_scope
        self._contract = joint_evidence_provider_contract(calibration=calibration)

    def contract(self) -> VisualAddressContract:
        return self._contract

    def _rank(self, observation: ImageObservation) -> Tuple[JointRowCandidate, ...]:
        raw = build_joint_row_candidates(
            observation=observation,
            prototypes=self._prototypes,
            candidate_masks=self._candidate_masks,
            pairwise_masks=self._pairwise_masks,
            regions=self._regions,
            architecture_id=self._calibration.architecture_id,
        )
        ranked = rank_joint_row_candidates(raw)
        return tuple(evaluate_joint_candidate_eligibility(candidate=item, ranked_candidates=ranked, calibration=self._calibration) for item in ranked)

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        ranked = self._rank(observation)
        candidate_set = build_joint_candidate_set(ranked_candidates=ranked, calibration=self._calibration)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else None
        accepted = candidate_set.outcome == "exact_row_accepted"
        reason = candidate_set.rejection_reason or candidate_set.outcome
        matched = candidate_set.rows[0] if accepted and candidate_set.rows else None
        return VisualAddressDecision(
            accepted=accepted,
            reason=reason,
            observation_digest=observation.raw_digest,
            representation_digest=self._contract.representation_spec_digest,
            provider_kind=self._contract.provider_kind,
            provider_version=self._contract.provider_version,
            score_semantics=self._contract.score_semantics,
            address_artifact_id=self._contract.address_artifact_id,
            calibration_artifact_id=self._contract.calibration_artifact_id,
            policy_artifact_id=self._contract.policy_artifact_id,
            nearest_row_id=best.row_id,
            nearest_score=float(best.candidate_strength),
            second_row_id=None if second is None else second.row_id,
            second_score=None if second is None else float(second.candidate_strength),
            ambiguity_measure=best.candidate_superiority_margin,
            local_evidence_score=float(best.candidate_strength),
            visible_evidence_fraction=float(best.available_candidate_fit_fraction),
            critical_evidence_present=bool((best.minimum_conflicting_action_margin or 0.0) >= self._calibration.minimum_conflicting_action_margin - REGISTRATION_DISTANCE_TIE_EPSILON),
            matched_row_id=matched,
        )


__all__ = [
    "VIDEO_JOINT_EVIDENCE_MECHANICS_VERSION",
    "VIDEO_JOINT_CANDIDATE_SET_VERSION",
    "VIDEO_JOINT_DECISION_VERSION",
    "VIDEO_JOINT_PROVIDER_VERSION",
    "VIDEO_JOINT_REGION_SPEC_VERSION",
    "VIDEO_JOINT_CANDIDATE_MASK_SPEC_VERSION",
    "VIDEO_JOINT_CANDIDATE_MASK_PAYLOAD_VERSION",
    "VIDEO_JOINT_PAIRWISE_MASK_SPEC_VERSION",
    "VIDEO_JOINT_PAIRWISE_MASK_PAYLOAD_VERSION",
    "VIDEO_JOINT_CALIBRATION_VERSION",
    "JointEvidenceRegionSpec",
    "JointCandidateMaskSpec",
    "JointCandidateMask",
    "PairwiseDiscriminativeMaskSpec",
    "PairwiseDiscriminativeMask",
    "JointEvidenceCalibration",
    "JointRegionFitEvidence",
    "PairwiseRegionEvidence",
    "JointRowCandidate",
    "JointCandidateSet",
    "JointEvidenceProvider",
    "build_joint_candidate_masks",
    "build_pairwise_discriminative_masks",
    "joint_candidate_mask_digest",
    "pairwise_mask_digest",
    "joint_region_digest",
    "build_joint_row_candidates",
    "rank_joint_row_candidates",
    "evaluate_joint_candidate_eligibility",
    "build_joint_candidate_set",
    "joint_evidence_provider_contract",
    "REGISTRATION_DISTANCE_TIE_EPSILON",
]
