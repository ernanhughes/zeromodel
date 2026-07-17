"""Runtime composition of visual addressing and VPM policy lookup."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .artifact import VPMValidationError
from .deployment_binding import DeploymentBinding
from .policy_lookup import PolicyLookupDecision, VPMPolicyLookup
from .visual import VISUAL_READER_VERSION, VisualDecision, VisualSignReader
from .visual_address import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
    VisualAddressProvider,
)


VISUAL_POLICY_DECISION_VERSION = "zeromodel-visual-policy-decision/v1"


class DeterministicVisualAddressProvider:
    """Compatibility adapter around the implemented deterministic reader."""

    def __init__(
        self,
        reader: VisualSignReader,
        *,
        source_scope: Optional[str] = None,
    ) -> None:
        self.reader = reader
        self._contract = VisualAddressContract(
            provider_kind="deterministic_codebook",
            provider_version=VISUAL_READER_VERSION,
            score_semantics="distance",
            observation_spec_digest=reader.feature_spec.digest,
            representation_spec_digest=reader.feature_spec.digest,
            address_artifact_id=reader.visual_index_artifact.artifact_id,
            calibration_artifact_id=reader.calibration.digest,
            policy_artifact_id=reader.policy_artifact.artifact_id,
            source_scope=source_scope,
            replay_contract="exact_bytes",
            metadata={"adapter": "VisualSignReader"},
        )

    def contract(self) -> VisualAddressContract:
        return self._contract

    def read(
        self,
        observation: ImageObservation,
    ) -> VisualAddressDecision:
        if not isinstance(observation, ImageObservation):
            raise VPMValidationError(
                "provider requires ImageObservation"
            )
        return self._map(
            observation,
            self.reader.read(observation.pixels),
        )

    def _map(
        self,
        observation: ImageObservation,
        decision: VisualDecision,
    ) -> VisualAddressDecision:
        checks = []
        if (
            decision.nearest_distance
            <= decision.acceptance_threshold + 1e-12
        ):
            checks.append("distance_threshold")
        if (
            decision.distance_margin + 1e-12
            >= decision.required_margin
        ):
            checks.append("absolute_gap")
        if decision.exact_feature_match:
            checks.append("exact_feature_codeword")
        return VisualAddressDecision(
            accepted=decision.accepted,
            reason=decision.reason,
            observation_digest=decision.input_digest,
            representation_digest=decision.feature_digest,
            provider_kind=self._contract.provider_kind,
            provider_version=self._contract.provider_version,
            score_semantics="distance",
            address_artifact_id=self._contract.address_artifact_id,
            calibration_artifact_id=(
                self._contract.calibration_artifact_id
            ),
            policy_artifact_id=self._contract.policy_artifact_id,
            nearest_row_id=decision.nearest_row_id,
            nearest_score=decision.nearest_distance,
            second_row_id=decision.second_nearest_row_id,
            second_score=decision.second_nearest_distance,
            ambiguity_measure=decision.distance_margin,
            matched_row_id=decision.matched_row_id,
            exact_match=decision.exact_feature_match,
            accepted_by=tuple(checks),
            trace={
                "observation": observation.to_descriptor(),
                "legacy_visual_decision": decision.to_dict(),
            },
        )


@dataclass(frozen=True)
class VisualPolicyDecision:
    """Combined address and policy evidence without conflating the two."""

    accepted: bool
    reason: str
    address: VisualAddressDecision
    policy: Optional[PolicyLookupDecision] = None
    version: str = VISUAL_POLICY_DECISION_VERSION

    def __post_init__(self) -> None:
        if self.version != VISUAL_POLICY_DECISION_VERSION:
            raise VPMValidationError(
                "unsupported visual policy decision version"
            )
        if self.accepted:
            if self.policy is None or not self.address.accepted:
                raise VPMValidationError(
                    "accepted visual policy decision needs both traces"
                )
            if self.policy.row_id != self.address.matched_row_id:
                raise VPMValidationError(
                    "address and policy rows do not match"
                )
        elif self.policy is not None:
            raise VPMValidationError(
                "rejected decision cannot include policy evidence"
            )

    @property
    def action(self) -> Optional[str]:
        return None if self.policy is None else self.policy.action

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "accepted": self.accepted,
            "reason": self.reason,
            "address": self.address.to_dict(),
            "policy": (
                None if self.policy is None else self.policy.to_dict()
            ),
            "action": self.action,
        }


class VisualPolicyReader:
    """Apply one declared visual-address provider to one exact policy."""

    def __init__(
        self,
        provider: VisualAddressProvider,
        policy_lookup: VPMPolicyLookup,
        *,
        deployment_binding: Optional[DeploymentBinding] = None,
        allow_research_binding: bool = False,
    ) -> None:
        contract = provider.contract()
        contract.validate()
        if contract.policy_artifact_id != policy_lookup.artifact.artifact_id:
            raise VPMValidationError(
                "visual provider targets policy %s, not %s"
                % (
                    contract.policy_artifact_id,
                    policy_lookup.artifact.artifact_id,
                )
            )
        if deployment_binding is not None:
            deployment_binding.verify_contract(
                contract,
                allow_research=allow_research_binding,
            )
        self.provider = provider
        self.policy_lookup = policy_lookup
        self.deployment_binding = deployment_binding
        self._contract = contract

    def contract(self) -> VisualAddressContract:
        return self._contract

    def read(
        self,
        observation: ImageObservation,
    ) -> VisualPolicyDecision:
        address = self.provider.read(observation)
        observed = (
            address.provider_kind,
            address.provider_version,
            address.score_semantics,
            address.address_artifact_id,
            address.calibration_artifact_id,
            address.policy_artifact_id,
        )
        expected = (
            self._contract.provider_kind,
            self._contract.provider_version,
            self._contract.score_semantics,
            self._contract.address_artifact_id,
            self._contract.calibration_artifact_id,
            self._contract.policy_artifact_id,
        )
        if observed != expected:
            raise VPMValidationError(
                "visual address decision violates provider contract"
            )
        if not address.accepted:
            return VisualPolicyDecision(
                accepted=False,
                reason=address.reason,
                address=address,
            )
        policy = self.policy_lookup.read(str(address.matched_row_id))
        return VisualPolicyDecision(
            accepted=True,
            reason="accepted",
            address=address,
            policy=policy,
        )
