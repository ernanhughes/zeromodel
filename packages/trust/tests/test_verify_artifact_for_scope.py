from __future__ import annotations

import pytest

from zeromodel.artifacts import ArtifactRef, sha256_digest
from zeromodel.trust import (
    ArtifactNotAuthorized,
    DeploymentScopeDTO,
    InMemoryRevocationResolver,
    IndeterminateRevocationResolver,
    RevocationRecordDTO,
    SignatureEnvelopeDTO,
    TrustFailureCode,
    compute_authorization_id,
    require_authorized,
    sign_digest,
    verify_artifact_for_scope,
)
from zeromodel.trust.dto import ArtifactAuthorizationDTO

VALID_FROM = "2026-01-01T00:00:00+00:00"
VALID_UNTIL = "2027-01-01T00:00:00+00:00"
EVAL_TIME_OK = "2026-06-01T00:00:00+00:00"
EVAL_TIME_TOO_EARLY = "2025-06-01T00:00:00+00:00"
EVAL_TIME_TOO_LATE = "2027-06-01T00:00:00+00:00"


def _verify(
    *,
    artifact_ref,
    canonical_artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    deployment_scope,
    revocations=None,
    minimum_epoch=0,
    evaluation_time=EVAL_TIME_OK,
    require_signature=True,
):
    return verify_artifact_for_scope(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=canonical_artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=deployment_scope,
        revocations=revocations
        if revocations is not None
        else InMemoryRevocationResolver(),
        minimum_epoch=minimum_epoch,
        evaluation_time=evaluation_time,
        require_signature=require_signature,
    )


def test_valid_artifact_is_authorized(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert decision.decision == "authorized"
    assert not decision.blocks_execution
    assert decision.failure_codes == ()
    assert all(
        [
            decision.integrity_valid,
            decision.signature_valid,
            decision.signer_known,
            decision.signer_trusted,
            decision.artifact_kind_allowed,
            decision.scope_authorized,
            decision.time_valid,
            decision.epoch_valid,
            decision.not_revoked,
        ]
    )


def test_deterministic_repeated_verification(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    first = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    second = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert first == second


def test_changed_artifact_bytes_fail_integrity(
    artifact_ref, authorization, signature_envelope, trust_policy, requested_scope
):
    tampered_bytes = b"a-completely-different-payload"
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=tampered_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert decision.integrity_valid is False
    assert decision.decision == "rejected"
    assert decision.blocks_execution
    assert TrustFailureCode.INTEGRITY_MISMATCH.value in decision.failure_codes


def test_png_vs_canonical_bytes_distinction(
    artifact_ref, authorization, signature_envelope, trust_policy, requested_scope
):
    """A different byte encoding of "the same" content must not satisfy
    integrity - the signature covers canonical bytes, never a rendering."""
    rendered_lookalike = b"\x89PNG\r\n\x1a\nnot-the-canonical-bytes-at-all"
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=rendered_lookalike,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert decision.integrity_valid is False
    assert decision.decision == "rejected"


def test_changed_signature_fails_authenticity(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    tampered_signature_hex = (
        "00" if signature_envelope.signature_hex[:2] != "00" else "ff"
    ) + signature_envelope.signature_hex[2:]
    tampered_envelope = SignatureEnvelopeDTO(
        authorization_id=signature_envelope.authorization_id,
        signer_id=signature_envelope.signer_id,
        signature_hex=tampered_signature_hex,
    )
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=tampered_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert decision.signature_valid is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.SIGNATURE_INVALID.value in decision.failure_codes


def test_changed_authorization_field_reusing_old_signature_is_malformed(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    """Simulates an attacker changing a signed field (e.g. artifact_kind)
    while replaying an old signature - the envelope no longer matches the
    presented authorization's own id."""
    retargeted_id = compute_authorization_id(
        artifact_digest=authorization.artifact_digest,
        artifact_kind="different-kind",
        deployment_scope=authorization.deployment_scope,
        policy_epoch=authorization.policy_epoch,
        valid_from=authorization.valid_from,
        valid_until=authorization.valid_until,
        issuer_signer_id=authorization.issuer_signer_id,
    )
    retargeted_authorization = ArtifactAuthorizationDTO(
        artifact_digest=authorization.artifact_digest,
        artifact_kind="different-kind",
        deployment_scope=authorization.deployment_scope,
        policy_epoch=authorization.policy_epoch,
        valid_from=authorization.valid_from,
        valid_until=authorization.valid_until,
        issuer_signer_id=authorization.issuer_signer_id,
        authorization_id=retargeted_id,
    )
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=retargeted_authorization,
        signature_envelope=signature_envelope,  # still points at the old authorization_id
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert decision.signature_valid is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.MALFORMED_ENVELOPE.value in decision.failure_codes


def test_missing_signature_is_rejected(
    artifact_ref, artifact_bytes, authorization, trust_policy, requested_scope
):
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=None,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert decision.decision == "rejected"
    assert TrustFailureCode.MISSING_SIGNATURE.value in decision.failure_codes


def test_unknown_signer_is_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    trust_policy,
    requested_scope,
    signing_key,
):
    other_key = signing_key  # reuse key material, different declared identity
    signature_hex = sign_digest(other_key.private_key, authorization.authorization_id)
    envelope_from_stranger = SignatureEnvelopeDTO(
        authorization_id=authorization.authorization_id,
        signer_id="signer-unknown",
        signature_hex=signature_hex,
    )
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=envelope_from_stranger,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert decision.signer_known is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.SIGNER_UNKNOWN.value in decision.failure_codes


def test_untrusted_revoked_signer_is_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    revocations = InMemoryRevocationResolver(
        [
            RevocationRecordDTO(
                revocation_id="rev-1",
                target_kind="signer",
                target_id="signer-a",
                revoked_at=EVAL_TIME_OK,
            )
        ]
    )
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        revocations=revocations,
        minimum_epoch=5,
    )
    assert decision.signer_trusted is False
    assert decision.not_revoked is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.REVOKED_SIGNER.value in decision.failure_codes


def test_wrong_artifact_kind_is_rejected(
    artifact_bytes, authorization_scope, trust_policy, requested_scope, signing_key
):
    wrong_ref = ArtifactRef(
        artifact_kind="not-a-traffic-policy", artifact_id=sha256_digest(artifact_bytes)
    )
    authorization_id = compute_authorization_id(
        artifact_digest=wrong_ref.artifact_id,
        artifact_kind=wrong_ref.artifact_kind,
        deployment_scope=authorization_scope,
        policy_epoch=5,
        valid_from=VALID_FROM,
        valid_until=VALID_UNTIL,
        issuer_signer_id="signer-a",
    )
    wrong_authorization = ArtifactAuthorizationDTO(
        artifact_digest=wrong_ref.artifact_id,
        artifact_kind=wrong_ref.artifact_kind,
        deployment_scope=authorization_scope,
        policy_epoch=5,
        valid_from=VALID_FROM,
        valid_until=VALID_UNTIL,
        issuer_signer_id="signer-a",
        authorization_id=authorization_id,
    )
    signature_hex = sign_digest(signing_key.private_key, authorization_id)
    envelope = SignatureEnvelopeDTO(
        authorization_id=authorization_id,
        signer_id="signer-a",
        signature_hex=signature_hex,
    )

    decision = _verify(
        artifact_ref=wrong_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=wrong_authorization,
        signature_envelope=envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert decision.artifact_kind_allowed is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.ARTIFACT_KIND_NOT_ALLOWED.value in decision.failure_codes


def test_wrong_scope_is_rejected(
    artifact_ref, artifact_bytes, authorization, signature_envelope, trust_policy
):
    outside_scope = DeploymentScopeDTO(
        organization="other-org", application="traffic-control"
    )
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=outside_scope,
        minimum_epoch=5,
    )
    assert decision.scope_authorized is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.SCOPE_NOT_AUTHORIZED.value in decision.failure_codes


def test_expired_authorization_is_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
        evaluation_time=EVAL_TIME_TOO_LATE,
    )
    assert decision.time_valid is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.EXPIRED.value in decision.failure_codes


def test_not_yet_valid_authorization_is_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
        evaluation_time=EVAL_TIME_TOO_EARLY,
    )
    assert decision.time_valid is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.NOT_YET_VALID.value in decision.failure_codes


def test_old_policy_epoch_is_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=6,  # authorization.policy_epoch is 5
    )
    assert decision.epoch_valid is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.EPOCH_TOO_OLD.value in decision.failure_codes


def test_revoked_artifact_authorization_is_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    revocations = InMemoryRevocationResolver(
        [
            RevocationRecordDTO(
                revocation_id="rev-2",
                target_kind="artifact_authorization",
                target_id=authorization.authorization_id,
                revoked_at=EVAL_TIME_OK,
            )
        ]
    )
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        revocations=revocations,
        minimum_epoch=5,
    )
    assert decision.not_revoked is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.REVOKED_AUTHORIZATION.value in decision.failure_codes


def test_revoked_artifact_digest_is_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    revocations = InMemoryRevocationResolver(
        [
            RevocationRecordDTO(
                revocation_id="rev-3",
                target_kind="artifact_digest",
                target_id=authorization.artifact_digest,
                revoked_at=EVAL_TIME_OK,
            )
        ]
    )
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        revocations=revocations,
        minimum_epoch=5,
    )
    assert decision.not_revoked is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.REVOKED_ARTIFACT.value in decision.failure_codes


def test_indeterminate_revocation_blocks_but_is_distinct_from_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        revocations=IndeterminateRevocationResolver(),
        minimum_epoch=5,
    )
    assert decision.not_revoked is False
    assert decision.decision == "indeterminate"
    assert decision.blocks_execution
    assert TrustFailureCode.REVOCATION_INDETERMINATE.value in decision.failure_codes
    # every other property still held - only revocation could not be resolved
    assert decision.integrity_valid
    assert decision.signature_valid
    assert decision.signer_known
    assert decision.signer_trusted
    assert decision.artifact_kind_allowed
    assert decision.scope_authorized
    assert decision.time_valid
    assert decision.epoch_valid


def test_fail_closed_loading_raises_when_not_authorized(
    artifact_ref, authorization, signature_envelope, trust_policy, requested_scope
):
    tampered_bytes = b"not-the-real-artifact"
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=tampered_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    with pytest.raises(ArtifactNotAuthorized):
        require_authorized(decision)


def test_fail_closed_loading_passes_through_when_authorized(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    decision = _verify(
        artifact_ref=artifact_ref,
        canonical_artifact_bytes=artifact_bytes,
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        deployment_scope=requested_scope,
        minimum_epoch=5,
    )
    assert require_authorized(decision) is decision
