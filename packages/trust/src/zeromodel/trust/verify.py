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

from typing import Optional

from zeromodel.artifacts import ArtifactRef, sha256_digest
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
    compute_deployment_scope_id,
    compute_signature_envelope_id,
    parse_iso8601_utc,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.trust.revocation import RevocationResolver, RevocationStatus


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

    # The envelope's signer must be the same identity the (signed)
    # authorization claims issued it. Without this check, a trusted
    # signer could produce a validly-signed authorization whose signed
    # content falsely names a different signer as issuer - the signature
    # would still verify (it covers the authorization's real content
    # including that false issuer_signer_id), so this must be a separate
    # check, not something signature verification alone catches.
    if signature_envelope.signer_id != authorization.issuer_signer_id:
        failure_codes.append(TrustFailureCode.SIGNER_ISSUER_MISMATCH.value)
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
    trust_policy: TrustPolicyDTO,
    minimum_epoch: int,
    evaluation_time: str,
    failure_codes: list[str],
) -> tuple[bool, bool]:
    """Returns (time_valid, epoch_valid).

    `evaluation_time` is untrusted caller input (unlike
    `authorization.valid_from`/`valid_until`, which are already guaranteed
    parseable and UTC-aware by `ArtifactAuthorizationDTO.__post_init__`),
    so a malformed value fails closed with a declared failure code rather
    than raising an uncaught exception.

    `epoch_valid` is a rollback guard against *two* independent floors:
    the caller-supplied `minimum_epoch`, and the active `trust_policy`'s
    own `policy_epoch` - an authorization issued under an epoch older than
    the currently active policy must not be honored even if a caller
    passes a stale `minimum_epoch`.
    """
    try:
        evaluated_at = parse_iso8601_utc(evaluation_time, "evaluation_time")
    except VPMValidationError:
        failure_codes.append(TrustFailureCode.MALFORMED_EVALUATION_TIME.value)
        time_valid = False
    else:
        valid_from = parse_iso8601_utc(
            authorization.valid_from, "authorization.valid_from"
        )
        valid_until = parse_iso8601_utc(
            authorization.valid_until, "authorization.valid_until"
        )
        time_valid = valid_from <= evaluated_at <= valid_until
        if evaluated_at < valid_from:
            failure_codes.append(TrustFailureCode.NOT_YET_VALID.value)
        elif evaluated_at > valid_until:
            failure_codes.append(TrustFailureCode.EXPIRED.value)

    epoch_valid = (
        authorization.policy_epoch >= minimum_epoch
        and authorization.policy_epoch >= trust_policy.policy_epoch
    )
    if not epoch_valid:
        failure_codes.append(TrustFailureCode.EPOCH_TOO_OLD.value)

    return time_valid, epoch_valid


def _check_revocations(
    *,
    signer_id: str,
    authorization: ArtifactAuthorizationDTO,
    signature_envelope_id: Optional[str],
    revocations: RevocationResolver,
    failure_codes: list[str],
) -> tuple[bool, bool]:
    """Returns (any_revoked, any_indeterminate) across signer, signature
    envelope, artifact authorization, and artifact digest.

    `signature_envelope_id` is the content-derived identity computed by
    `compute_signature_envelope_id`, never the raw signature hex - a
    revocation targeting the old raw-hex value would silently fail to
    match after this migration, which is the intended, documented
    behavior (see the Stage C implementation report), not a bug to work
    around with a second target kind.
    """
    checks = [("signer", signer_id, TrustFailureCode.REVOKED_SIGNER)]
    if signature_envelope_id is not None:
        checks.append(
            (
                "signature_envelope",
                signature_envelope_id,
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


def _determine_decision(
    *,
    integrity_valid: bool,
    signature_valid: bool,
    signer_known: bool,
    signer_trusted: bool,
    artifact_kind_allowed: bool,
    scope_authorized: bool,
    time_valid: bool,
    epoch_valid: bool,
    any_revoked: bool,
    any_indeterminate: bool,
) -> str:
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
    if core_checks_pass and not any_revoked and not any_indeterminate:
        return "authorized"
    if core_checks_pass and any_indeterminate and not any_revoked:
        # Every other property held; only the revocation check could not
        # return a confident answer. Distinct from a definite "rejected".
        return "indeterminate"
    return "rejected"


def _resolve_signature_envelope_id(
    signature_envelope: Optional[SignatureEnvelopeDTO],
) -> Optional[str]:
    """Content-derived envelope identity, never the raw `signature_hex`.

    Feeds both `TrustDecisionDTO.signature_envelope_id` (the audit
    receipt) and `_check_revocations`'s "signature_envelope" target - the
    same identity must be used for both so a revocation actually targets
    what the decision records.
    """
    if signature_envelope is None:
        return None
    return compute_signature_envelope_id(
        authorization_id=signature_envelope.authorization_id,
        signer_id=signature_envelope.signer_id,
        signature_hex=signature_envelope.signature_hex,
        key_algorithm=signature_envelope.key_algorithm,
        spec_version=signature_envelope.spec_version,
    )


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
        trust_policy=trust_policy,
        minimum_epoch=minimum_epoch,
        evaluation_time=evaluation_time,
        failure_codes=failure_codes,
    )
    any_revoked, any_indeterminate = _check_revocations(
        signer_id=signer_id,
        authorization=authorization,
        signature_envelope_id=_resolve_signature_envelope_id(signature_envelope),
        revocations=revocations,
        failure_codes=failure_codes,
    )
    not_revoked = not any_revoked and not any_indeterminate
    decision = _determine_decision(
        integrity_valid=integrity_valid,
        signature_valid=signature_valid,
        signer_known=signer_known,
        signer_trusted=signer_trusted,
        artifact_kind_allowed=artifact_kind_allowed,
        scope_authorized=scope_authorized,
        time_valid=time_valid,
        epoch_valid=epoch_valid,
        any_revoked=any_revoked,
        any_indeterminate=any_indeterminate,
    )

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
        trust_policy_id=trust_policy.policy_id,
        authorization_id=authorization.authorization_id,
        artifact_digest=artifact_ref.artifact_id,
        signer_id=signer_id,
        deployment_scope_id=compute_deployment_scope_id(deployment_scope),
        signature_envelope_id=_resolve_signature_envelope_id(signature_envelope),
        failure_codes=tuple(dict.fromkeys(failure_codes)),
        evaluated_at=evaluation_time,
    )
