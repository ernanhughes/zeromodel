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
    TrustedSignerDTO,
    TrustPolicyDTO,
    TrustPolicyRuleDTO,
    compute_authorization_id,
    compute_signature_envelope_id,
    compute_trust_policy_id,
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


def _make_trust_decision(**overrides) -> TrustDecisionDTO:
    fields = dict(
        integrity_valid=True,
        signature_valid=True,
        signer_known=True,
        signer_trusted=True,
        artifact_kind_allowed=True,
        scope_authorized=True,
        time_valid=True,
        epoch_valid=True,
        not_revoked=True,
        decision="authorized",
        trust_policy_id="sha256:" + "a" * 64,
        authorization_id="sha256:" + "b" * 64,
        artifact_digest="sha256:" + "c" * 64,
        signer_id="signer-a",
        deployment_scope_id="sha256:" + "d" * 64,
    )
    fields.update(overrides)
    return TrustDecisionDTO(**fields)


def test_trust_decision_rejects_unknown_decision_value():
    with pytest.raises(VPMValidationError):
        _make_trust_decision(decision="maybe")


def test_trust_decision_requires_sha256_evidence_identities():
    with pytest.raises(VPMValidationError):
        _make_trust_decision(trust_policy_id="not-a-digest")
    with pytest.raises(VPMValidationError):
        _make_trust_decision(authorization_id="not-a-digest")
    with pytest.raises(VPMValidationError):
        _make_trust_decision(artifact_digest="not-a-digest")
    with pytest.raises(VPMValidationError):
        _make_trust_decision(deployment_scope_id="not-a-digest")
    with pytest.raises(VPMValidationError):
        _make_trust_decision(signer_id="")


def test_trust_decision_requires_sha256_signature_envelope_id_when_present():
    """Regression: `signature_envelope_id` must be a content-derived digest,
    never the raw signature hex stored under an identity-shaped name."""
    with pytest.raises(VPMValidationError):
        _make_trust_decision(signature_envelope_id="deadbeef")
    with pytest.raises(VPMValidationError):
        _make_trust_decision(
            signature_envelope_id="aa" * 64
        )  # raw hex, no sha256: prefix


def test_trust_decision_carries_a_complete_audit_receipt():
    envelope_id = compute_signature_envelope_id(
        authorization_id="sha256:" + "b" * 64,
        signer_id="signer-a",
        signature_hex="aa" * 64,
    )
    decision = _make_trust_decision(signature_envelope_id=envelope_id)
    assert decision.trust_policy_id == "sha256:" + "a" * 64
    assert decision.authorization_id == "sha256:" + "b" * 64
    assert decision.artifact_digest == "sha256:" + "c" * 64
    assert decision.signer_id == "signer-a"
    assert decision.deployment_scope_id == "sha256:" + "d" * 64
    assert decision.signature_envelope_id == envelope_id


def _make_signer(signer_id: str, public_key_hex: str) -> TrustedSignerDTO:
    return TrustedSignerDTO(
        signer=SignerIdentityDTO(signer_id=signer_id, public_key_hex=public_key_hex),
        trusted_since=VALID_FROM,
    )


def test_trust_policy_id_must_match_own_canonical_content():
    signer = _make_signer("signer-a", "aa" * 32)
    rule = TrustPolicyRuleDTO(
        rule_id="rule-1",
        signer_id="signer-a",
        allowed_artifact_kinds=("traffic-policy",),
        scope_pattern=DeploymentScopeDTO(organization="acme"),
    )
    correct_id = compute_trust_policy_id(
        policy_epoch=1, trusted_signers=(signer,), rules=(rule,)
    )

    # Constructing with the correct, self-consistent id succeeds.
    TrustPolicyDTO(
        policy_id=correct_id, policy_epoch=1, trusted_signers=(signer,), rules=(rule,)
    )

    # A fabricated id (not the digest of the actual content) is rejected.
    with pytest.raises(VPMValidationError):
        TrustPolicyDTO(
            policy_id="sha256:" + "b" * 64,
            policy_epoch=1,
            trusted_signers=(signer,),
            rules=(rule,),
        )


def test_trust_policy_rejects_duplicate_signer_id():
    signer_a = _make_signer("signer-a", "aa" * 32)
    signer_a_again = _make_signer("signer-a", "bb" * 32)
    policy_id = compute_trust_policy_id(
        policy_epoch=1, trusted_signers=(signer_a, signer_a_again), rules=()
    )
    with pytest.raises(VPMValidationError, match="duplicate trusted signer_id"):
        TrustPolicyDTO(
            policy_id=policy_id,
            policy_epoch=1,
            trusted_signers=(signer_a, signer_a_again),
            rules=(),
        )


def test_trust_policy_rejects_two_signers_sharing_a_public_key():
    signer_a = _make_signer("signer-a", "aa" * 32)
    signer_b = _make_signer(
        "signer-b", "aa" * 32
    )  # same key, different declared identity
    policy_id = compute_trust_policy_id(
        policy_epoch=1, trusted_signers=(signer_a, signer_b), rules=()
    )
    with pytest.raises(VPMValidationError, match="sharing public_key_hex"):
        TrustPolicyDTO(
            policy_id=policy_id,
            policy_epoch=1,
            trusted_signers=(signer_a, signer_b),
            rules=(),
        )


def test_trust_policy_rejects_duplicate_rule_id():
    signer = _make_signer("signer-a", "aa" * 32)
    rule_1 = TrustPolicyRuleDTO(
        rule_id="rule-1",
        signer_id="signer-a",
        allowed_artifact_kinds=("kind-a",),
        scope_pattern=DeploymentScopeDTO(),
    )
    rule_1_again = TrustPolicyRuleDTO(
        rule_id="rule-1",
        signer_id="signer-a",
        allowed_artifact_kinds=("kind-b",),
        scope_pattern=DeploymentScopeDTO(),
    )
    policy_id = compute_trust_policy_id(
        policy_epoch=1, trusted_signers=(signer,), rules=(rule_1, rule_1_again)
    )
    with pytest.raises(VPMValidationError, match="duplicate rule_id"):
        TrustPolicyDTO(
            policy_id=policy_id,
            policy_epoch=1,
            trusted_signers=(signer,),
            rules=(rule_1, rule_1_again),
        )


def test_trust_policy_rejects_a_rule_referencing_an_unknown_signer():
    signer = _make_signer("signer-a", "aa" * 32)
    rule_for_stranger = TrustPolicyRuleDTO(
        rule_id="rule-1",
        signer_id="signer-does-not-exist",
        allowed_artifact_kinds=("kind-a",),
        scope_pattern=DeploymentScopeDTO(),
    )
    policy_id = compute_trust_policy_id(
        policy_epoch=1, trusted_signers=(signer,), rules=(rule_for_stranger,)
    )
    with pytest.raises(VPMValidationError, match="unknown"):
        TrustPolicyDTO(
            policy_id=policy_id,
            policy_epoch=1,
            trusted_signers=(signer,),
            rules=(rule_for_stranger,),
        )


def test_authorization_rejects_valid_from_after_valid_until():
    scope = DeploymentScopeDTO(organization="acme")
    real_digest = "sha256:" + "e" * 64
    inverted_id = compute_authorization_id(
        artifact_digest=real_digest,
        artifact_kind="traffic-policy",
        deployment_scope=scope,
        policy_epoch=1,
        valid_from="2027-01-01T00:00:00+00:00",
        valid_until="2026-01-01T00:00:00+00:00",
        issuer_signer_id="signer-a",
    )
    with pytest.raises(
        VPMValidationError, match="valid_from must not be after valid_until"
    ):
        ArtifactAuthorizationDTO(
            artifact_digest=real_digest,
            artifact_kind="traffic-policy",
            deployment_scope=scope,
            policy_epoch=1,
            valid_from="2027-01-01T00:00:00+00:00",
            valid_until="2026-01-01T00:00:00+00:00",
            issuer_signer_id="signer-a",
            authorization_id=inverted_id,
        )


def test_authorization_rejects_a_timezone_naive_validity_timestamp():
    scope = DeploymentScopeDTO(organization="acme")
    real_digest = "sha256:" + "f" * 64
    naive_id = compute_authorization_id(
        artifact_digest=real_digest,
        artifact_kind="traffic-policy",
        deployment_scope=scope,
        policy_epoch=1,
        valid_from="2026-01-01T00:00:00",  # no UTC offset
        valid_until=VALID_UNTIL,
        issuer_signer_id="signer-a",
    )
    with pytest.raises(VPMValidationError, match="timezone-aware"):
        ArtifactAuthorizationDTO(
            artifact_digest=real_digest,
            artifact_kind="traffic-policy",
            deployment_scope=scope,
            policy_epoch=1,
            valid_from="2026-01-01T00:00:00",
            valid_until=VALID_UNTIL,
            issuer_signer_id="signer-a",
            authorization_id=naive_id,
        )


def test_deployment_scope_pattern_wildcards_unset_fields():
    pattern = DeploymentScopeDTO(organization="acme")
    matching = DeploymentScopeDTO(
        organization="acme", application="anything", environment="prod"
    )
    non_matching = DeploymentScopeDTO(organization="other-org")
    assert matching.matches_pattern(pattern)
    assert not non_matching.matches_pattern(pattern)


def test_signature_envelope_id_is_deterministic():
    kwargs = dict(
        authorization_id="sha256:" + "b" * 64,
        signer_id="signer-a",
        signature_hex="aa" * 64,
    )
    assert compute_signature_envelope_id(**kwargs) == compute_signature_envelope_id(
        **kwargs
    )


def test_signature_envelope_id_changes_with_any_component():
    """Regression for Stage C section 12: the envelope identity must bind
    authorization_id, signer_id, signature_hex, and key_algorithm - changing
    any one of them must change the identity."""
    base = compute_signature_envelope_id(
        authorization_id="sha256:" + "b" * 64,
        signer_id="signer-a",
        signature_hex="aa" * 64,
        key_algorithm="ed25519",
    )
    different_authorization = compute_signature_envelope_id(
        authorization_id="sha256:" + "c" * 64,
        signer_id="signer-a",
        signature_hex="aa" * 64,
        key_algorithm="ed25519",
    )
    different_signer = compute_signature_envelope_id(
        authorization_id="sha256:" + "b" * 64,
        signer_id="signer-b",
        signature_hex="aa" * 64,
        key_algorithm="ed25519",
    )
    different_signature = compute_signature_envelope_id(
        authorization_id="sha256:" + "b" * 64,
        signer_id="signer-a",
        signature_hex="bb" * 64,
        key_algorithm="ed25519",
    )
    different_algorithm = compute_signature_envelope_id(
        authorization_id="sha256:" + "b" * 64,
        signer_id="signer-a",
        signature_hex="aa" * 64,
        key_algorithm="ed448",
    )
    identities = {
        base,
        different_authorization,
        different_signer,
        different_signature,
        different_algorithm,
    }
    assert len(identities) == 5


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
