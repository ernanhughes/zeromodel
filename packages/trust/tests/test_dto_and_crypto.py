from __future__ import annotations

import pathlib

import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.trust import (
    ArtifactAuthorizationDTO,
    DeploymentScopeDTO,
    RevocationRecordDTO,
    SignerIdentityDTO,
    TrustDecisionDTO,
    TrustPolicyRuleDTO,
    compute_authorization_id,
    generate_signing_key,
    sign_digest,
    verify_signature,
)

VALID_FROM = "2026-01-01T00:00:00+00:00"
VALID_UNTIL = "2027-01-01T00:00:00+00:00"


def test_authorization_id_must_match_own_canonical_content():
    scope = DeploymentScopeDTO(organization="acme")
    real_digest = "sha256:" + "a" * 64
    correct_id = compute_authorization_id(
        artifact_digest=real_digest,
        artifact_kind="traffic-policy",
        deployment_scope=scope,
        policy_epoch=1,
        valid_from=VALID_FROM,
        valid_until=VALID_UNTIL,
        issuer_signer_id="signer-a",
    )
    # Constructing with the correct, self-consistent id succeeds.
    ArtifactAuthorizationDTO(
        artifact_digest=real_digest,
        artifact_kind="traffic-policy",
        deployment_scope=scope,
        policy_epoch=1,
        valid_from=VALID_FROM,
        valid_until=VALID_UNTIL,
        issuer_signer_id="signer-a",
        authorization_id=correct_id,
    )
    # A fabricated id (not the digest of the actual content) is rejected.
    with pytest.raises(VPMValidationError):
        ArtifactAuthorizationDTO(
            artifact_digest=real_digest,
            artifact_kind="traffic-policy",
            deployment_scope=scope,
            policy_epoch=1,
            valid_from=VALID_FROM,
            valid_until=VALID_UNTIL,
            issuer_signer_id="signer-a",
            authorization_id="sha256:" + "b" * 64,
        )


def test_signer_identity_rejects_non_ed25519_algorithm():
    with pytest.raises(VPMValidationError):
        SignerIdentityDTO(
            signer_id="signer-a", public_key_hex="ab" * 32, key_algorithm="rsa"
        )


def test_trust_policy_rule_rejects_empty_allowed_kinds():
    with pytest.raises(VPMValidationError):
        TrustPolicyRuleDTO(
            rule_id="rule-1",
            signer_id="signer-a",
            allowed_artifact_kinds=(),
            scope_pattern=DeploymentScopeDTO(),
        )


def test_revocation_record_rejects_unknown_target_kind():
    with pytest.raises(VPMValidationError):
        RevocationRecordDTO(
            revocation_id="rev-1",
            target_kind="not-a-real-target-kind",
            target_id="signer-a",
            revoked_at=VALID_FROM,
        )


def test_trust_decision_rejects_unknown_decision_value():
    with pytest.raises(VPMValidationError):
        TrustDecisionDTO(
            integrity_valid=True,
            signature_valid=True,
            signer_known=True,
            signer_trusted=True,
            artifact_kind_allowed=True,
            scope_authorized=True,
            time_valid=True,
            epoch_valid=True,
            not_revoked=True,
            decision="maybe",
        )


def test_deployment_scope_pattern_wildcards_unset_fields():
    pattern = DeploymentScopeDTO(organization="acme")
    matching = DeploymentScopeDTO(
        organization="acme", application="anything", environment="prod"
    )
    non_matching = DeploymentScopeDTO(organization="other-org")
    assert matching.matches_pattern(pattern)
    assert not non_matching.matches_pattern(pattern)


def test_crypto_sign_and_verify_round_trip():
    key = generate_signing_key()
    digest = "sha256:" + "c" * 64
    signature_hex = sign_digest(key.private_key, digest)
    assert verify_signature(
        public_key_hex=key.public_key_hex, digest=digest, signature_hex=signature_hex
    )


def test_crypto_verify_signature_never_raises_on_malformed_input():
    assert (
        verify_signature(
            public_key_hex="not-hex",
            digest="sha256:" + "d" * 64,
            signature_hex="also-not-hex",
        )
        is False
    )
    assert verify_signature(public_key_hex="", digest="", signature_hex="") is False


def test_no_private_key_material_committed_in_trust_package_source():
    """Private-key generation/signing helpers exist for tests/authoring only;
    nothing under the package source tree may persist actual key material."""
    package_root = pathlib.Path(__file__).resolve().parents[1]
    this_file = pathlib.Path(__file__).resolve()
    suspicious_markers = (
        "-----BEGIN PRIVATE KEY-----",
        "-----BEGIN EC PRIVATE KEY-----",
        "-----BEGIN OPENSSH PRIVATE KEY-----",
        "-----BEGIN RSA PRIVATE KEY-----",
    )
    offending_files = []
    for path in package_root.rglob("*"):
        if path == this_file:
            continue
        if not path.is_file() or path.suffix not in {
            ".py",
            ".md",
            ".toml",
            ".json",
            ".txt",
        }:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(marker in text for marker in suspicious_markers):
            offending_files.append(str(path))
    assert offending_files == []
