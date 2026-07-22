"""The trust verification service: `verify_artifact_for_scope`.

Every security property is computed as its own, independently-preserved
boolean on the returned `TrustDecisionDTO` - integrity, authenticity, trust,
authorization, freshness/rollback, and revocation are never collapsed into a
single opaque pass/fail. Each concern below is deliberately factored into
its own helper so no single function mixes more than one decision.

`evaluation_time` is always an explicit parameter. This module never reads
the wall clock itself.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from zeromodel.artifacts import ArtifactRef, sha256_digest
from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.content_identity import canonical_json_bytes

from zeromodel.trust import crypto
from zeromodel.trust.dto import (
    ArtifactAuthorizationDTO,
    DeploymentScopeDTO,
    SignatureEnvelopeDTO,
    TrustDecisionDTO,
    TrustFailureCode,
    TrustPolicyDTO,
    authorization_signing_payload,
)
from zeromodel.trust.revocation import RevocationResolver, RevocationStatus


def _parse_iso8601(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise VPMValidationError(f"not a valid ISO 8601 timestamp: {value!r}") from exc


def _check_integrity(
    *,
    artifact_ref: ArtifactRef,
    authorization: ArtifactAuthorizationDTO,
    canonical_artifact_bytes: bytes,
) -> bool:
    actual_digest = sha256_digest(canonical_artifact_bytes)
    return actual_digest == artifact_ref.artifact_id == authorization.artifact_digest


def _check_signature(
    *,
    authorization: ArtifactAuthorizationDTO,
    signature_envelope: Optional[SignatureEnvelopeDTO],
    trust_policy: TrustPolicyDTO,
    require_signature: bool,
    failure_codes: list[str],
) -> bool:
    if signature_envelope is None:
        if require_signature:
            failure_codes.append(TrustFailureCode.MISSING_SIGNATURE.value)
            return False
        # Nothing to authenticate, and the caller explicitly said that's
        # acceptable - this is not itself a signature failure.
        return True

    if signature_envelope.authorization_id != authorization.authorization_id:
        failure_codes.append(TrustFailureCode.MALFORMED_ENVELOPE.value)
        return False

    trusted_entry = trust_policy.signer(signature_envelope.signer_id)
    if trusted_entry is None:
        failure_codes.append(TrustFailureCode.SIGNATURE_INVALID.value)
        return False

    manifest_digest = sha256_digest(
        canonical_json_bytes(authorization_signing_payload(authorization))
    )
    signature_valid = crypto.verify_signature(
        public_key_hex=trusted_entry.signer.public_key_hex,
        digest=manifest_digest,
        signature_hex=signature_envelope.signature_hex,
    )
    if not signature_valid:
        failure_codes.append(TrustFailureCode.SIGNATURE_INVALID.value)
    return signature_valid


def _resolve_signer_id(
    authorization: ArtifactAuthorizationDTO,
    signature_envelope: Optional[SignatureEnvelopeDTO],
) -> str:
    return (
        signature_envelope.signer_id
        if signature_envelope
        else authorization.issuer_signer_id
    )


def _check_signer_trust(
    *,
    signer_id: str,
    trust_policy: TrustPolicyDTO,
    revocations: RevocationResolver,
    failure_codes: list[str],
) -> tuple[bool, bool]:
    """Returns (signer_known, signer_trusted).

    Deliberately distinct from `not_revoked` (computed elsewhere): a signer
    whose revocation status could not be resolved (INDETERMINATE) is not
    untrusted by policy - only a definite REVOKED makes it untrusted.
    """
    trusted_entry = trust_policy.signer(signer_id)
    if trusted_entry is None:
        failure_codes.append(TrustFailureCode.SIGNER_UNKNOWN.value)
        return False, False

    if revocations.status("signer", signer_id) == RevocationStatus.REVOKED:
        failure_codes.append(TrustFailureCode.SIGNER_UNTRUSTED.value)
        return True, False
    return True, True


def _check_authorization_scope(
    *,
    signer_id: str,
    signer_known: bool,
    authorization: ArtifactAuthorizationDTO,
    deployment_scope: DeploymentScopeDTO,
    trust_policy: TrustPolicyDTO,
    failure_codes: list[str],
) -> tuple[bool, bool]:
    """Returns (artifact_kind_allowed, scope_authorized)."""
    rules = trust_policy.rules_for_signer(signer_id) if signer_known else ()

    artifact_kind_allowed = any(
        authorization.artifact_kind in rule.allowed_artifact_kinds for rule in rules
    )
    if not artifact_kind_allowed:
        failure_codes.append(TrustFailureCode.ARTIFACT_KIND_NOT_ALLOWED.value)

    # The policy must permit this signer to authorize this artifact kind for
    # the scope the authorization was actually issued under, AND the scope
    # this call is evaluating for must fall within that issued scope.
    policy_permits_authorization = any(
        rule.allows(authorization.artifact_kind, authorization.deployment_scope)
        for rule in rules
    )
    requested_scope_covered = deployment_scope.matches_pattern(
        authorization.deployment_scope
    )
    scope_authorized = policy_permits_authorization and requested_scope_covered
    if not scope_authorized:
        failure_codes.append(TrustFailureCode.SCOPE_NOT_AUTHORIZED.value)

    return artifact_kind_allowed, scope_authorized


def _check_freshness(
    *,
    authorization: ArtifactAuthorizationDTO,
    minimum_epoch: int,
    evaluation_time: str,
    failure_codes: list[str],
) -> tuple[bool, bool]:
    """Returns (time_valid, epoch_valid)."""
    evaluated_at = _parse_iso8601(evaluation_time)
    valid_from = _parse_iso8601(authorization.valid_from)
    valid_until = _parse_iso8601(authorization.valid_until)
    time_valid = valid_from <= evaluated_at <= valid_until
    if evaluated_at < valid_from:
        failure_codes.append(TrustFailureCode.NOT_YET_VALID.value)
    elif evaluated_at > valid_until:
        failure_codes.append(TrustFailureCode.EXPIRED.value)

    epoch_valid = authorization.policy_epoch >= minimum_epoch
    if not epoch_valid:
        failure_codes.append(TrustFailureCode.EPOCH_TOO_OLD.value)

    return time_valid, epoch_valid


def _check_revocations(
    *,
    signer_id: str,
    authorization: ArtifactAuthorizationDTO,
    signature_envelope: Optional[SignatureEnvelopeDTO],
    revocations: RevocationResolver,
    failure_codes: list[str],
) -> tuple[bool, bool]:
    """Returns (any_revoked, any_indeterminate) across signer, signature
    envelope, artifact authorization, and artifact digest."""
    checks = [("signer", signer_id, TrustFailureCode.REVOKED_SIGNER)]
    if signature_envelope is not None:
        checks.append(
            (
                "signature_envelope",
                signature_envelope.signature_hex,
                TrustFailureCode.REVOKED_ENVELOPE,
            )
        )
    checks.append(
        (
            "artifact_authorization",
            authorization.authorization_id,
            TrustFailureCode.REVOKED_AUTHORIZATION,
        )
    )
    checks.append(
        (
            "artifact_digest",
            authorization.artifact_digest,
            TrustFailureCode.REVOKED_ARTIFACT,
        )
    )

    any_revoked = False
    any_indeterminate = False
    for target_kind, target_id, code in checks:
        status = revocations.status(target_kind, target_id)
        if status == RevocationStatus.REVOKED:
            any_revoked = True
            failure_codes.append(code.value)
        elif status == RevocationStatus.INDETERMINATE:
            any_indeterminate = True
    if any_indeterminate:
        failure_codes.append(TrustFailureCode.REVOCATION_INDETERMINATE.value)

    return any_revoked, any_indeterminate


def verify_artifact_for_scope(
    *,
    artifact_ref: ArtifactRef,
    canonical_artifact_bytes: bytes,
    authorization: ArtifactAuthorizationDTO,
    signature_envelope: Optional[SignatureEnvelopeDTO],
    trust_policy: TrustPolicyDTO,
    deployment_scope: DeploymentScopeDTO,
    revocations: RevocationResolver,
    minimum_epoch: int,
    evaluation_time: str,
    require_signature: bool = True,
) -> TrustDecisionDTO:
    """Verify integrity, authenticity, trust, authorization, and freshness.

    Parameters mirror the request concept `TrustVerificationRequestDTO`
    captures for auditing; this function takes the resolved objects
    directly so it has no hidden dependency on any particular
    `ArtifactResolver`/`ArtifactStore` implementation.
    """
    failure_codes: list[str] = []

    integrity_valid = _check_integrity(
        artifact_ref=artifact_ref,
        authorization=authorization,
        canonical_artifact_bytes=canonical_artifact_bytes,
    )
    if not integrity_valid:
        failure_codes.append(TrustFailureCode.INTEGRITY_MISMATCH.value)

    signature_valid = _check_signature(
        authorization=authorization,
        signature_envelope=signature_envelope,
        trust_policy=trust_policy,
        require_signature=require_signature,
        failure_codes=failure_codes,
    )
    signer_id = _resolve_signer_id(authorization, signature_envelope)
    signer_known, signer_trusted = _check_signer_trust(
        signer_id=signer_id,
        trust_policy=trust_policy,
        revocations=revocations,
        failure_codes=failure_codes,
    )
    artifact_kind_allowed, scope_authorized = _check_authorization_scope(
        signer_id=signer_id,
        signer_known=signer_known,
        authorization=authorization,
        deployment_scope=deployment_scope,
        trust_policy=trust_policy,
        failure_codes=failure_codes,
    )
    time_valid, epoch_valid = _check_freshness(
        authorization=authorization,
        minimum_epoch=minimum_epoch,
        evaluation_time=evaluation_time,
        failure_codes=failure_codes,
    )
    any_revoked, any_indeterminate = _check_revocations(
        signer_id=signer_id,
        authorization=authorization,
        signature_envelope=signature_envelope,
        revocations=revocations,
        failure_codes=failure_codes,
    )
    not_revoked = not any_revoked and not any_indeterminate

    core_checks_pass = (
        integrity_valid
        and signature_valid
        and signer_known
        and signer_trusted
        and artifact_kind_allowed
        and scope_authorized
        and time_valid
        and epoch_valid
    )
    if core_checks_pass and not_revoked:
        decision = "authorized"
    elif core_checks_pass and any_indeterminate and not any_revoked:
        # Every other property held; only the revocation check could not
        # return a confident answer. Distinct from a definite "rejected".
        decision = "indeterminate"
    else:
        decision = "rejected"

    return TrustDecisionDTO(
        integrity_valid=integrity_valid,
        signature_valid=signature_valid,
        signer_known=signer_known,
        signer_trusted=signer_trusted,
        artifact_kind_allowed=artifact_kind_allowed,
        scope_authorized=scope_authorized,
        time_valid=time_valid,
        epoch_valid=epoch_valid,
        not_revoked=not_revoked,
        decision=decision,
        failure_codes=tuple(dict.fromkeys(failure_codes)),
        evaluated_at=evaluation_time,
    )
