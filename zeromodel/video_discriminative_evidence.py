"""Stage 3 discriminative current-frame evidence contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError
from .visual_address import VisualAddressContract
from .visual_registration import RegistrationConfig


VIDEO_DISCRIMINATIVE_REGION_SPEC_VERSION = "zeromodel-video-discriminative-region-spec/v1"
VIDEO_DISCRIMINATIVE_MASK_SPEC_VERSION = "zeromodel-video-discriminative-mask-spec/v1"
VIDEO_DISCRIMINATIVE_CALIBRATION_VERSION = "zeromodel-video-discriminative-calibration/v1"
VIDEO_DISCRIMINATIVE_CANDIDATE_SET_VERSION = "zeromodel-video-discriminative-candidate-set/v1"
VIDEO_DISCRIMINATIVE_PROVIDER_VERSION = "zeromodel-video-discriminative-provider/v1"
VIDEO_DISCRIMINATIVE_ARCHITECTURE_SELECTION_VERSION = "zeromodel-video-discriminative-architecture-selection/v1"
VIDEO_DISCRIMINATIVE_OPERATING_POINT_SELECTION_VERSION = "zeromodel-video-discriminative-operating-point-selection/v1"

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
    "DiscriminativeCandidateSet",
    "DiscriminativeEvidenceCalibration",
    "DiscriminativeMaskSpec",
    "DiscriminativeRegionSpec",
    "DiscriminativeRowCandidate",
    "RegionDiscriminativeEvidence",
    "VIDEO_DISCRIMINATIVE_ARCHITECTURE_SELECTION_VERSION",
    "VIDEO_DISCRIMINATIVE_CALIBRATION_VERSION",
    "VIDEO_DISCRIMINATIVE_CANDIDATE_SET_VERSION",
    "VIDEO_DISCRIMINATIVE_MASK_SPEC_VERSION",
    "VIDEO_DISCRIMINATIVE_OPERATING_POINT_SELECTION_VERSION",
    "VIDEO_DISCRIMINATIVE_PROVIDER_VERSION",
    "VIDEO_DISCRIMINATIVE_REGION_SPEC_VERSION",
    "discriminative_mask_digest",
    "discriminative_provider_contract",
    "discriminative_region_digest",
]
