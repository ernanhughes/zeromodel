"""Identity-bearing prototype-to-policy bindings over MatrixBlob vectors."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Dict, Mapping, Optional, Tuple

from .artifact import VPMValidationError
from .visual_address import (
    VISUAL_ADDRESS_MANIFEST_VERSION,
    _SCORE_SEMANTICS,
    _freeze,
    _json_bytes,
    _nonempty,
    _thaw,
)


@dataclass(frozen=True)
class PrototypeBinding:
    prototype_id: str
    vector_index: int
    policy_row_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _nonempty("prototype_id", self.prototype_id)
        _nonempty("policy_row_id", self.policy_row_id)
        if self.vector_index < 0:
            raise VPMValidationError("prototype vector_index cannot be negative")
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        _json_bytes(self.metadata)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "prototype_id": self.prototype_id,
            "vector_index": self.vector_index,
            "policy_row_id": self.policy_row_id,
            "metadata": _thaw(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrototypeBinding":
        return cls(**data)


@dataclass(frozen=True)
class VisualAddressManifest:
    """Thin row-binding manifest over a separate content-addressed matrix blob."""

    address_kind: str
    policy_artifact_id: str
    matrix_blob_id: str
    matrix_row_count: int
    representation_spec_digest: str
    calibration_artifact_id: str
    score_semantics: str
    source_scope: str
    prototype_bindings: Tuple[PrototypeBinding, ...]
    encoder_manifest_id: Optional[str] = None
    deployment_status: str = "research"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VISUAL_ADDRESS_MANIFEST_VERSION
    manifest_id: Optional[str] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "prototype_bindings", tuple(self.prototype_bindings))
        object.__setattr__(self, "metadata", _freeze(self.metadata))
        if self.version != VISUAL_ADDRESS_MANIFEST_VERSION:
            raise VPMValidationError("unsupported visual address manifest version")
        for name in (
            "address_kind", "policy_artifact_id", "matrix_blob_id",
            "representation_spec_digest", "calibration_artifact_id", "source_scope",
        ):
            _nonempty(name, str(getattr(self, name)))
        if self.score_semantics not in _SCORE_SEMANTICS:
            raise VPMValidationError("score_semantics must be distance or similarity")
        if self.deployment_status not in {"research", "validated"}:
            raise VPMValidationError("deployment_status must be research or validated")
        if self.matrix_row_count <= 0:
            raise VPMValidationError("matrix_row_count must be positive")
        if len(self.prototype_bindings) != self.matrix_row_count:
            raise VPMValidationError("prototype bindings must cover every matrix row exactly")
        ids = [item.prototype_id for item in self.prototype_bindings]
        indices = [item.vector_index for item in self.prototype_bindings]
        if len(set(ids)) != len(ids):
            raise VPMValidationError("prototype ids must be unique")
        if sorted(indices) != list(range(self.matrix_row_count)):
            raise VPMValidationError("prototype indices must cover matrix rows exactly")
        _json_bytes(self.metadata)
        expected = self.compute_manifest_id()
        if self.manifest_id is None:
            object.__setattr__(self, "manifest_id", expected)
        elif self.manifest_id != expected:
            raise VPMValidationError("visual address manifest id mismatch")

    @property
    def deployment_permitted(self) -> bool:
        return self.deployment_status == "validated"

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "address_kind": self.address_kind,
            "policy_artifact_id": self.policy_artifact_id,
            "matrix_blob_id": self.matrix_blob_id,
            "matrix_row_count": self.matrix_row_count,
            "representation_spec_digest": self.representation_spec_digest,
            "calibration_artifact_id": self.calibration_artifact_id,
            "score_semantics": self.score_semantics,
            "source_scope": self.source_scope,
            "prototype_bindings": [item.to_dict() for item in self.prototype_bindings],
            "encoder_manifest_id": self.encoder_manifest_id,
            "deployment_status": self.deployment_status,
            "metadata": _thaw(self.metadata),
        }

    def compute_manifest_id(self) -> str:
        return hashlib.sha256(
            b"zeromodel.visual-address-manifest.identity.v1\0"
            + _json_bytes(self.identity_payload())
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        result = self.identity_payload()
        result.update({
            "manifest_id": self.manifest_id,
            "deployment_permitted": self.deployment_permitted,
        })
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "VisualAddressManifest":
        payload = dict(data)
        payload.pop("deployment_permitted", None)
        payload["prototype_bindings"] = tuple(
            PrototypeBinding.from_dict(item) for item in payload["prototype_bindings"]
        )
        return cls(**payload)
