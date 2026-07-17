"""Governed seam between visual observations and independently identified VPM policies."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .artifact import VPMValidationError
from .policy_lookup import PolicyLookupDecision, VPMPolicyLookup
from .visual import VISUAL_READER_VERSION, VisualDecision, VisualSignReader

VISUAL_ADDRESS_CONTRACT_VERSION = "zeromodel-visual-address-contract/v1"
VISUAL_ADDRESS_DECISION_VERSION = "zeromodel-visual-address-decision/v1"
VISUAL_ADDRESS_MANIFEST_VERSION = "zeromodel-visual-address-manifest/v1"
VISUAL_POLICY_DECISION_VERSION = "zeromodel-visual-policy-decision/v1"
IMAGE_OBSERVATION_VERSION = "zeromodel-image-observation/v1"
_SCORE_SEMANTICS = {"distance", "similarity"}
_REPLAY_CONTRACTS = {"exact_bytes", "exact_decision", "tolerance_equivalent"}


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, np.generic):
        raise VPMValidationError("visual-address JSON must use plain scalar types")
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("visual-address values must be JSON-serializable") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _nonempty(name: str, value: str) -> None:
    if not value:
        raise VPMValidationError("%s cannot be empty" % name)


def _finite_optional(name: str, value: Optional[float]) -> None:
    if value is not None and not np.isfinite(float(value)):
        raise VPMValidationError("%s must be finite when present" % name)


@dataclass(frozen=True)
class ImageObservation:
    pixels: np.ndarray
    timestamp: Optional[str] = None
    source_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = IMAGE_OBSERVATION_VERSION

    def __post_init__(self) -> None:
        array = np.asarray(self.pixels)
        if array.dtype != np.uint8:
            raise VPMValidationError("image observations must use uint8 samples")
        if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in {3, 4})):
            raise VPMValidationError("image observations must be HxW or HxWx3/4")
        if array.size == 0:
            raise VPMValidationError("image observations cannot be empty")
        owned = np.array(array, dtype=np.uint8, order="C", copy=True)
        owned.flags.writeable = False
        object.__setattr__(self, "pixels", owned)
        object.__setattr__(self, "source_id", str(self.source_id))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.version != IMAGE_OBSERVATION_VERSION:
            raise VPMValidationError("unsupported image observation version")
        _json_bytes(self.metadata)

    @property
    def raw_digest(self) -> str:
        payload = (
            b"zeromodel.image-observation.raw.v1\0"
            + _json_bytes(list(self.pixels.shape))
            + self.pixels.tobytes(order="C")
        )
        return "sha256:" + hashlib.sha256(payload).hexdigest()

    def to_descriptor(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "raw_digest": self.raw_digest,
            "shape": list(self.pixels.shape),
            "timestamp": self.timestamp,
            "source_id": self.source_id,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True)
class VisualAddressContract:
    provider_kind: str
    provider_version: str
    score_semantics: str
    observation_spec_digest: str
    representation_spec_digest: str
    address_artifact_id: str
    calibration_artifact_id: str
    policy_artifact_id: str
    source_scope: Optional[str] = None
    replay_contract: str = "exact_decision"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VISUAL_ADDRESS_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.version != VISUAL_ADDRESS_CONTRACT_VERSION:
            raise VPMValidationError("unsupported visual address contract version")
        for name in (
            "provider_kind", "provider_version", "observation_spec_digest",
            "representation_spec_digest", "address_artifact_id",
            "calibration_artifact_id", "policy_artifact_id",
        ):
            _nonempty(name, str(getattr(self, name)))
        if self.score_semantics not in _SCORE_SEMANTICS:
            raise VPMValidationError("score_semantics must be distance or similarity")
        if self.replay_contract not in _REPLAY_CONTRACTS:
            raise VPMValidationError("unsupported replay_contract")
        _json_bytes(self.metadata)

    @property
    def digest(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "provider_kind": self.provider_kind,
            "provider_version": self.provider_version,
            "score_semantics": self.score_semantics,
            "observation_spec_digest": self.observation_spec_digest,
            "representation_spec_digest": self.representation_spec_digest,
            "address_artifact_id": self.address_artifact_id,
            "calibration_artifact_id": self.calibration_artifact_id,
            "policy_artifact_id": self.policy_artifact_id,
            "source_scope": self.source_scope,
            "replay_contract": self.replay_contract,
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualAddressContract":
        return cls(**{key: value for key, value in data.items() if key != "digest"})


@dataclass(frozen=True)
class VisualAddressDecision:
    accepted: bool
    reason: str
    observation_digest: str
    representation_digest: str
    provider_kind: str
    provider_version: str
    score_semantics: str
    address_artifact_id: str
    calibration_artifact_id: str
    policy_artifact_id: str
    nearest_row_id: Optional[str]
    nearest_score: Optional[float]
    second_row_id: Optional[str]
    second_score: Optional[float]
    ambiguity_measure: Optional[float]
    local_evidence_score: Optional[float] = None
    visible_evidence_fraction: Optional[float] = None
    critical_evidence_present: Optional[bool] = None
    matched_row_id: Optional[str] = None
    exact_match: bool = False
    accepted_by: Tuple[str, ...] = ()
    trace: Mapping[str, Any] = field(default_factory=dict)
    version: str = VISUAL_ADDRESS_DECISION_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "accepted_by", tuple(self.accepted_by))
        object.__setattr__(self, "trace", _freeze(self.trace))
        if self.version != VISUAL_ADDRESS_DECISION_VERSION:
            raise VPMValidationError("unsupported visual address decision version")
        for name in (
            "reason", "observation_digest", "representation_digest", "provider_kind",
            "provider_version", "address_artifact_id", "calibration_artifact_id",
            "policy_artifact_id",
        ):
            _nonempty(name, str(getattr(self, name)))
        if self.score_semantics not in _SCORE_SEMANTICS:
            raise VPMValidationError("score_semantics must be distance or similarity")
        for name in (
            "nearest_score", "second_score", "ambiguity_measure",
            "local_evidence_score", "visible_evidence_fraction",
        ):
            _finite_optional(name, getattr(self, name))
        if self.visible_evidence_fraction is not None and not (
            0.0 <= self.visible_evidence_fraction <= 1.0
        ):
            raise VPMValidationError("visible_evidence_fraction must be in [0, 1]")
        if self.accepted != (self.matched_row_id is not None):
            raise VPMValidationError("matched_row_id must exist exactly when accepted")
        if len(set(self.accepted_by)) != len(self.accepted_by):
            raise VPMValidationError("accepted_by entries must be unique")
        _json_bytes(self.trace)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "accepted": self.accepted,
            "reason": self.reason,
            "observation_digest": self.observation_digest,
            "representation_digest": self.representation_digest,
            "provider_kind": self.provider_kind,
            "provider_version": self.provider_version,
            "score_semantics": self.score_semantics,
            "address_artifact_id": self.address_artifact_id,
            "calibration_artifact_id": self.calibration_artifact_id,
            "policy_artifact_id": self.policy_artifact_id,
            "nearest_row_id": self.nearest_row_id,
            "nearest_score": self.nearest_score,
            "second_row_id": self.second_row_id,
            "second_score": self.second_score,
            "ambiguity_measure": self.ambiguity_measure,
            "local_evidence_score": self.local_evidence_score,
            "visible_evidence_fraction": self.visible_evidence_fraction,
            "critical_evidence_present": self.critical_evidence_present,
            "matched_row_id": self.matched_row_id,
            "exact_match": self.exact_match,
            "accepted_by": list(self.accepted_by),
            "trace": _thaw(self.trace),
        }
