from __future__ import annotations

import pytest

from zeromodel.artifacts import ArtifactRef, sha256_digest
from zeromodel.trust import (
    ArtifactAuthorizationDTO,
    DeploymentScopeDTO,
    SignatureEnvelopeDTO,
    SignerIdentityDTO,
    TrustedSignerDTO,
    TrustPolicyDTO,
    TrustPolicyRuleDTO,
    compute_authorization_id,
    generate_signing_key,
    sign_digest,
)

VALID_FROM = "2026-01-01T00:00:00+00:00"
VALID_UNTIL = "2027-01-01T00:00:00+00:00"
EVAL_TIME_OK = "2026-06-01T00:00:00+00:00"
EVAL_TIME_TOO_EARLY = "2025-06-01T00:00:00+00:00"
EVAL_TIME_TOO_LATE = "2027-06-01T00:00:00+00:00"


@pytest.fixture
def artifact_bytes() -> bytes:
    return b"canonical-vpm-artifact-bytes-v1"


@pytest.fixture
def artifact_ref(artifact_bytes: bytes) -> ArtifactRef:
    return ArtifactRef(
        artifact_kind="traffic-policy", artifact_id=sha256_digest(artifact_bytes)
    )


@pytest.fixture
def signing_key():
    return generate_signing_key()


@pytest.fixture
def signer_identity(signing_key):
    return SignerIdentityDTO(
        signer_id="signer-a", public_key_hex=signing_key.public_key_hex
    )


@pytest.fixture
def trust_policy(signer_identity: SignerIdentityDTO) -> TrustPolicyDTO:
    scope_pattern = DeploymentScopeDTO(organization="acme")
    rule = TrustPolicyRuleDTO(
        rule_id="rule-1",
        signer_id="signer-a",
        allowed_artifact_kinds=("traffic-policy",),
        scope_pattern=scope_pattern,
    )
    trusted_signer = TrustedSignerDTO(
        signer=signer_identity, trusted_since="2026-01-01T00:00:00+00:00"
    )
    return TrustPolicyDTO(
        policy_id="policy-1",
        policy_epoch=5,
        trusted_signers=(trusted_signer,),
        rules=(rule,),
    )


@pytest.fixture
def authorization_scope() -> DeploymentScopeDTO:
    # Broad on purpose: only organization is pinned, so any concrete scope
    # under "acme" is covered (see DeploymentScopeDTO.matches_pattern).
    return DeploymentScopeDTO(organization="acme")


@pytest.fixture
def requested_scope() -> DeploymentScopeDTO:
    return DeploymentScopeDTO(
        organization="acme",
        application="traffic-control",
        environment="prod",
        device_group="fleet-1",
        location="site-1",
    )


@pytest.fixture
def authorization(
    artifact_ref: ArtifactRef, authorization_scope: DeploymentScopeDTO
) -> ArtifactAuthorizationDTO:
    authorization_id = compute_authorization_id(
        artifact_digest=artifact_ref.artifact_id,
        artifact_kind=artifact_ref.artifact_kind,
        deployment_scope=authorization_scope,
        policy_epoch=5,
        valid_from=VALID_FROM,
        valid_until=VALID_UNTIL,
        issuer_signer_id="signer-a",
    )
    return ArtifactAuthorizationDTO(
        artifact_digest=artifact_ref.artifact_id,
        artifact_kind=artifact_ref.artifact_kind,
        deployment_scope=authorization_scope,
        policy_epoch=5,
        valid_from=VALID_FROM,
        valid_until=VALID_UNTIL,
        issuer_signer_id="signer-a",
        authorization_id=authorization_id,
    )


@pytest.fixture
def signature_envelope(
    signing_key, authorization: ArtifactAuthorizationDTO
) -> SignatureEnvelopeDTO:
    signature_hex = sign_digest(signing_key.private_key, authorization.authorization_id)
    return SignatureEnvelopeDTO(
        authorization_id=authorization.authorization_id,
        signer_id="signer-a",
        signature_hex=signature_hex,
    )
