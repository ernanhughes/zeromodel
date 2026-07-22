from __future__ import annotations

import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.observation.deployment_binding import DeploymentBinding
from zeromodel.observation.visual_address import VisualAddressContract


def _contract() -> VisualAddressContract:
    return VisualAddressContract(
        provider_kind="frozen_embedding",
        provider_version="v1",
        score_semantics="similarity",
        observation_spec_digest="obs-spec",
        representation_spec_digest="rep-spec",
        address_artifact_id="address-1",
        calibration_artifact_id="calibration-1",
        policy_artifact_id="policy-1",
        source_scope="camera:bridge-east",
        replay_contract="exact_decision",
    )


def test_binding_round_trip_and_contract_verification() -> None:
    contract = _contract()
    binding = DeploymentBinding.from_contract(
        contract,
        encoder_manifest_id="encoder-1",
    )
    loaded = DeploymentBinding.from_dict(binding.to_dict())

    assert loaded.binding_id == binding.binding_id
    assert loaded.deployment_permitted
    loaded.verify_contract(contract)


def test_binding_rejects_mismatched_contract() -> None:
    binding = DeploymentBinding.from_contract(_contract())
    other = VisualAddressContract(
        provider_kind="frozen_embedding",
        provider_version="v1",
        score_semantics="similarity",
        observation_spec_digest="obs-spec",
        representation_spec_digest="rep-spec",
        address_artifact_id="address-2",
        calibration_artifact_id="calibration-1",
        policy_artifact_id="policy-1",
        source_scope="camera:bridge-east",
    )

    with pytest.raises(VPMValidationError, match="does not match"):
        binding.verify_contract(other)


def test_research_binding_is_blocked_by_default() -> None:
    binding = DeploymentBinding.from_contract(
        _contract(),
        deployment_status="research",
    )

    assert not binding.deployment_permitted
    with pytest.raises(VPMValidationError, match="research deployment"):
        binding.verify_contract(_contract())

    binding.verify_contract(_contract(), allow_research=True)


def test_binding_requires_source_scope() -> None:
    contract = VisualAddressContract(
        provider_kind="test",
        provider_version="v1",
        score_semantics="distance",
        observation_spec_digest="obs",
        representation_spec_digest="rep",
        address_artifact_id="address",
        calibration_artifact_id="cal",
        policy_artifact_id="policy",
        source_scope=None,
    )

    with pytest.raises(
        VPMValidationError,
        match="requires contract source_scope",
    ):
        DeploymentBinding.from_contract(contract)
