"""Regression tests for Stage C section 12.2: signature-envelope revocation
must target the content-derived envelope identity (`compute_signature_envelope_id`),
not the raw signature hex. Split out of `test_verify_artifact_for_scope.py`
to keep that module under the repository's per-module line limit; shares
the same `packages/trust/tests/conftest.py` fixtures.
"""

from __future__ import annotations

from zeromodel.trust import (
    InMemoryRevocationResolver,
    RevocationRecordDTO,
    TrustFailureCode,
    compute_signature_envelope_id,
    verify_artifact_for_scope,
)

EVAL_TIME_OK = "2026-06-01T00:00:00+00:00"


def _verify(
    *,
    artifact_ref,
    canonical_artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    deployment_scope,
    revocations,
    minimum_epoch=5,
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
        revocations=revocations,
        minimum_epoch=minimum_epoch,
        evaluation_time=evaluation_time,
        require_signature=require_signature,
    )


def test_revoked_signature_envelope_is_rejected(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    """Regression for Stage C section 12.2: revocation must target the
    content-derived envelope identity, not the raw signature hex."""
    envelope_id = compute_signature_envelope_id(
        authorization_id=signature_envelope.authorization_id,
        signer_id=signature_envelope.signer_id,
        signature_hex=signature_envelope.signature_hex,
        key_algorithm=signature_envelope.key_algorithm,
    )
    revocations = InMemoryRevocationResolver(
        [
            RevocationRecordDTO(
                revocation_id="rev-4",
                target_kind="signature_envelope",
                target_id=envelope_id,
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
    )
    assert decision.not_revoked is False
    assert decision.decision == "rejected"
    assert TrustFailureCode.REVOKED_ENVELOPE.value in decision.failure_codes


def test_revoking_raw_signature_hex_does_not_revoke_the_envelope(
    artifact_ref,
    artifact_bytes,
    authorization,
    signature_envelope,
    trust_policy,
    requested_scope,
):
    """Regression: a revocation record targeting the old, wrong identity
    (raw signature_hex) must not match the content-derived envelope id -
    this proves the migration actually changed the target, not just added
    a second accepted meaning for the same target kind."""
    revocations = InMemoryRevocationResolver(
        [
            RevocationRecordDTO(
                revocation_id="rev-5",
                target_kind="signature_envelope",
                target_id=signature_envelope.signature_hex,
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
    )
    assert decision.not_revoked is True
    assert decision.decision == "authorized"
