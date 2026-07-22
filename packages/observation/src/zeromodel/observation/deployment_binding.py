"""Approved binding between a visual address contract and a VPM policy."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Dict, Mapping, Optional

from zeromodel.core.artifact import VPMValidationError
from zeromodel.observation.visual_address import VisualAddressContract


DEPLOYMENT_BINDING_VERSION = "zeromodel-deployment-binding/v1"
_DEPLOYMENT_STATUSES = {"validated", "research"}


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError(
            "deployment binding must be JSON-serializable"
        ) from exc


@dataclass(frozen=True)
class DeploymentBinding:
    """Identity-bearing approval of one perception-policy pairing.

    A binding is not a cryptographic signature.  It records exactly which
    address, calibration, encoder, policy, and source scope were approved
    together so the runtime can reject accidental cross-pairing.
    """

    policy_artifact_id: str
    address_artifact_id: str
    calibration_artifact_id: str
    source_scope: str
    encoder_manifest_id: Optional[str] = None
    deployment_status: str = "validated"
    version: str = DEPLOYMENT_BINDING_VERSION
    binding_id: str = ""

    def __init__(
        self,
        *,
        policy_artifact_id: str,
        address_artifact_id: str,
        calibration_artifact_id: str,
        source_scope: str,
        encoder_manifest_id: Optional[str] = None,
        deployment_status: str = "validated",
        version: str = DEPLOYMENT_BINDING_VERSION,
        binding_id: Optional[str] = None,
    ) -> None:
        object.__setattr__(self, "policy_artifact_id", str(policy_artifact_id))
        object.__setattr__(self, "address_artifact_id", str(address_artifact_id))
        object.__setattr__(
            self,
            "calibration_artifact_id",
            str(calibration_artifact_id),
        )
        object.__setattr__(self, "source_scope", str(source_scope))
        object.__setattr__(
            self,
            "encoder_manifest_id",
            None if encoder_manifest_id is None else str(encoder_manifest_id),
        )
        object.__setattr__(self, "deployment_status", str(deployment_status))
        object.__setattr__(self, "version", str(version))
        computed = self.compute_binding_id()
        object.__setattr__(self, "binding_id", str(binding_id or computed))
        self.validate()

    @property
    def deployment_permitted(self) -> bool:
        return self.deployment_status == "validated"

    def validate(self) -> None:
        if self.version != DEPLOYMENT_BINDING_VERSION:
            raise VPMValidationError(
                "unsupported deployment binding version: %r" % self.version
            )
        for name, value in (
            ("policy_artifact_id", self.policy_artifact_id),
            ("address_artifact_id", self.address_artifact_id),
            ("calibration_artifact_id", self.calibration_artifact_id),
            ("source_scope", self.source_scope),
        ):
            if not value:
                raise VPMValidationError("%s cannot be empty" % name)
        if self.deployment_status not in _DEPLOYMENT_STATUSES:
            raise VPMValidationError(
                "deployment_status must be 'validated' or 'research'"
            )
        expected = self.compute_binding_id()
        if self.binding_id != expected:
            raise VPMValidationError(
                "deployment binding id mismatch: expected %s, got %s"
                % (expected, self.binding_id)
            )

    def identity_payload(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "policy_artifact_id": self.policy_artifact_id,
            "address_artifact_id": self.address_artifact_id,
            "calibration_artifact_id": self.calibration_artifact_id,
            "encoder_manifest_id": self.encoder_manifest_id,
            "source_scope": self.source_scope,
            "deployment_status": self.deployment_status,
        }

    def compute_binding_id(self) -> str:
        return hashlib.sha256(
            b"zeromodel.deployment-binding.identity.v1\0"
            + _canonical_json_bytes(self.identity_payload())
        ).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        payload = self.identity_payload()
        payload["binding_id"] = self.binding_id
        payload["deployment_permitted"] = self.deployment_permitted
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DeploymentBinding":
        return cls(
            policy_artifact_id=str(data["policy_artifact_id"]),
            address_artifact_id=str(data["address_artifact_id"]),
            calibration_artifact_id=str(data["calibration_artifact_id"]),
            encoder_manifest_id=data.get("encoder_manifest_id"),
            source_scope=str(data["source_scope"]),
            deployment_status=str(data.get("deployment_status", "validated")),
            version=str(data.get("version", DEPLOYMENT_BINDING_VERSION)),
            binding_id=data.get("binding_id"),
        )

    @classmethod
    def from_contract(
        cls,
        contract: VisualAddressContract,
        *,
        deployment_status: str = "validated",
        encoder_manifest_id: Optional[str] = None,
    ) -> "DeploymentBinding":
        if contract.source_scope is None or not contract.source_scope:
            raise VPMValidationError(
                "deployment binding requires contract source_scope"
            )
        return cls(
            policy_artifact_id=contract.policy_artifact_id,
            address_artifact_id=contract.address_artifact_id,
            calibration_artifact_id=contract.calibration_artifact_id,
            encoder_manifest_id=encoder_manifest_id,
            source_scope=contract.source_scope,
            deployment_status=deployment_status,
        )

    def verify_contract(
        self,
        contract: VisualAddressContract,
        *,
        allow_research: bool = False,
    ) -> None:
        contract.validate()
        mismatches = []
        for name, expected, actual in (
            (
                "policy_artifact_id",
                self.policy_artifact_id,
                contract.policy_artifact_id,
            ),
            (
                "address_artifact_id",
                self.address_artifact_id,
                contract.address_artifact_id,
            ),
            (
                "calibration_artifact_id",
                self.calibration_artifact_id,
                contract.calibration_artifact_id,
            ),
            ("source_scope", self.source_scope, contract.source_scope),
        ):
            if expected != actual:
                mismatches.append("%s=%r (expected %r)" % (name, actual, expected))
        if mismatches:
            raise VPMValidationError(
                "deployment binding does not match visual address contract: %s"
                % "; ".join(mismatches)
            )
        if not self.deployment_permitted and not allow_research:
            raise VPMValidationError(
                "research deployment binding is not permitted in validated runtime"
            )
