from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple

from zeromodel.artifacts import canonical_json_bytes, is_sha256_digest, sha256_digest
from zeromodel.core.artifact import VPMValidationError

SPEC_VERSION = "zeromodel-trust/v1"


class TrustFailureCode(str, Enum):
    """Every reason a TrustDecisionDTO can fail to be "authorized"."""

    INTEGRITY_MISMATCH = "integrity_mismatch"
    MALFORMED_ENVELOPE = "malformed_envelope"
    MISSING_SIGNATURE = "missing_signature"
    SIGNATURE_INVALID = "signature_invalid"
    SIGNER_ISSUER_MISMATCH = "signer_issuer_mismatch"
    SIGNER_UNKNOWN = "signer_unknown"
    SIGNER_UNTRUSTED = "signer_untrusted"
    ARTIFACT_KIND_NOT_ALLOWED = "artifact_kind_not_allowed"
    SCOPE_NOT_AUTHORIZED = "scope_not_authorized"
    NOT_YET_VALID = "not_yet_valid"
    EXPIRED = "expired"
    EPOCH_TOO_OLD = "epoch_too_old"
    REVOKED_SIGNER = "revoked_signer"
    REVOKED_ENVELOPE = "revoked_envelope"
    REVOKED_AUTHORIZATION = "revoked_authorization"
    REVOKED_ARTIFACT = "revoked_artifact"
    REVOCATION_INDETERMINATE = "revocation_indeterminate"
    MALFORMED_EVALUATION_TIME = "malformed_evaluation_time"


def _require_sha256(value: str, message: str) -> None:
    if not is_sha256_digest(value):
        raise VPMValidationError(message)


def _require_nonempty_str(value: object, message: str) -> None:
    if not isinstance(value, str) or not value:
        raise VPMValidationError(message)


def parse_iso8601_utc(value: str, context: str) -> datetime:
    """Parse an ISO 8601 timestamp, requiring an explicit UTC/timezone
    offset. A naive (offset-less) timestamp is ambiguous about which
    instant it names and is rejected rather than silently assumed to be
    UTC or local time."""
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise VPMValidationError(
            f"{context} is not a valid ISO 8601 timestamp: {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise VPMValidationError(
            f"{context} must be a timezone-aware timestamp with an explicit UTC offset: {value!r}"
        )
    return parsed


@dataclass(frozen=True, slots=True)
class DeploymentScopeDTO:
    """A declared (or requested) deployment scope.

    Used both as the concrete scope an authorization was issued for, and as
    a wildcard-capable pattern on a `TrustPolicyRuleDTO` (a `None` field on
    the *pattern* side matches anything; a concrete scope should generally
    have every field it cares about set).
    """

    organization: Optional[str] = None
    application: Optional[str] = None
    environment: Optional[str] = None
    device_group: Optional[str] = None
    location: Optional[str] = None
    artifact_kind: Optional[str] = None
    corpus: Optional[str] = None
    policy_family: Optional[str] = None
    spec_version: str = SPEC_VERSION

    def matches_pattern(self, pattern: "DeploymentScopeDTO") -> bool:
        """True if every field the pattern specifies matches this scope."""
        for attr in (
            "organization",
            "application",
            "environment",
            "device_group",
            "location",
            "artifact_kind",
            "corpus",
            "policy_family",
        ):
            pattern_value = getattr(pattern, attr)
            if pattern_value is not None and getattr(self, attr) != pattern_value:
                return False
        return True


@dataclass(frozen=True, slots=True)
class SignerIdentityDTO:
    """A stable signer identity - never a display name alone."""

    signer_id: str
    public_key_hex: str
    key_algorithm: str = "ed25519"
    display_name: Optional[str] = None
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.signer_id, "SignerIdentityDTO.signer_id must be non-empty"
        )
        _require_nonempty_str(
            self.public_key_hex, "SignerIdentityDTO.public_key_hex must be non-empty"
        )
        if self.key_algorithm != "ed25519":
            raise VPMValidationError(
                f"unsupported key_algorithm: {self.key_algorithm!r} (only 'ed25519' is implemented)"
            )


@dataclass(frozen=True, slots=True)
class TrustedSignerDTO:
    """One entry in a TrustPolicyDTO's accepted-signer roster."""

    signer: SignerIdentityDTO
    trusted_since: str
    spec_version: str = SPEC_VERSION


@dataclass(frozen=True, slots=True)
class TrustPolicyRuleDTO:
    """ "Signer X may authorize artifact kinds Y for scope pattern Z."""

    rule_id: str
    signer_id: str
    allowed_artifact_kinds: Tuple[str, ...]
    scope_pattern: DeploymentScopeDTO
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.rule_id, "TrustPolicyRuleDTO.rule_id must be non-empty"
        )
        _require_nonempty_str(
            self.signer_id, "TrustPolicyRuleDTO.signer_id must be non-empty"
        )
        if not self.allowed_artifact_kinds:
            raise VPMValidationError(
                "TrustPolicyRuleDTO.allowed_artifact_kinds must not be empty"
            )

    def allows(self, artifact_kind: str, scope: DeploymentScopeDTO) -> bool:
        return artifact_kind in self.allowed_artifact_kinds and scope.matches_pattern(
            self.scope_pattern
        )


@dataclass(frozen=True, slots=True)
class TrustPolicyDTO:
    """An identified, bounded trust policy - never one universal signer.

    `policy_id` is this policy's own content digest (see
    `trust_policy_identity_payload`), the same self-validating pattern used
    by `ArtifactAuthorizationDTO.authorization_id`: two policies with the
    same `policy_id` are guaranteed to have identical trusted signers,
    rules, and epoch. A `TrustDecisionDTO` can therefore record exactly
    which policy authorized it without a separate binding mechanism.
    """

    policy_id: str
    policy_epoch: int
    trusted_signers: Tuple[TrustedSignerDTO, ...]
    rules: Tuple[TrustPolicyRuleDTO, ...]
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.policy_id, "TrustPolicyDTO.policy_id must be a sha256: digest"
        )
        if self.policy_epoch < 0:
            raise VPMValidationError("TrustPolicyDTO.policy_epoch must be >= 0")

        seen_signer_ids: set = set()
        seen_public_keys: set = set()
        for entry in self.trusted_signers:
            if entry.signer.signer_id in seen_signer_ids:
                raise VPMValidationError(
                    f"TrustPolicyDTO has a duplicate trusted signer_id: {entry.signer.signer_id!r}"
                )
            seen_signer_ids.add(entry.signer.signer_id)
            if entry.signer.public_key_hex in seen_public_keys:
                raise VPMValidationError(
                    "TrustPolicyDTO has two different trusted signers sharing "
                    f"public_key_hex {entry.signer.public_key_hex!r}"
                )
            seen_public_keys.add(entry.signer.public_key_hex)

        seen_rule_ids: set = set()
        for rule in self.rules:
            if rule.rule_id in seen_rule_ids:
                raise VPMValidationError(
                    f"TrustPolicyDTO has a duplicate rule_id: {rule.rule_id!r}"
                )
            seen_rule_ids.add(rule.rule_id)
            if rule.signer_id not in seen_signer_ids:
                raise VPMValidationError(
                    f"TrustPolicyDTO rule {rule.rule_id!r} references unknown "
                    f"signer_id {rule.signer_id!r} (not in trusted_signers)"
                )

        expected_id = sha256_digest(
            canonical_json_bytes(trust_policy_identity_payload(self))
        )
        if self.policy_id != expected_id:
            raise VPMValidationError(
                "TrustPolicyDTO.policy_id does not match its own canonical content"
            )

    def signer(self, signer_id: str) -> Optional[TrustedSignerDTO]:
        for entry in self.trusted_signers:
            if entry.signer.signer_id == signer_id:
                return entry
        return None

    def rules_for_signer(self, signer_id: str) -> Tuple[TrustPolicyRuleDTO, ...]:
        return tuple(rule for rule in self.rules if rule.signer_id == signer_id)


def _deployment_scope_payload(scope: "DeploymentScopeDTO") -> dict:
    return {
        "organization": scope.organization,
        "application": scope.application,
        "environment": scope.environment,
        "device_group": scope.device_group,
        "location": scope.location,
        "artifact_kind": scope.artifact_kind,
        "corpus": scope.corpus,
        "policy_family": scope.policy_family,
    }


def compute_deployment_scope_id(scope: "DeploymentScopeDTO") -> str:
    """Content digest identifying one concrete deployment scope.

    Lets a `TrustDecisionDTO` record exactly which scope it was evaluated
    for without embedding the full `DeploymentScopeDTO` - the same
    reference-by-digest pattern used for `TrustPolicyDTO.policy_id` and
    `ArtifactAuthorizationDTO.authorization_id`.
    """
    return sha256_digest(canonical_json_bytes(_deployment_scope_payload(scope)))


def _trust_policy_identity_payload_fields(
    *,
    policy_epoch: int,
    trusted_signers: Tuple[TrustedSignerDTO, ...],
    rules: Tuple[TrustPolicyRuleDTO, ...],
    spec_version: str,
) -> dict:
    return {
        "spec_version": spec_version,
        "policy_epoch": policy_epoch,
        "trusted_signers": [
            {
                "signer_id": entry.signer.signer_id,
                "public_key_hex": entry.signer.public_key_hex,
                "key_algorithm": entry.signer.key_algorithm,
                "trusted_since": entry.trusted_since,
            }
            for entry in trusted_signers
        ],
        "rules": [
            {
                "rule_id": rule.rule_id,
                "signer_id": rule.signer_id,
                "allowed_artifact_kinds": list(rule.allowed_artifact_kinds),
                "scope_pattern": _deployment_scope_payload(rule.scope_pattern),
            }
            for rule in rules
        ],
    }


def trust_policy_identity_payload(policy: TrustPolicyDTO) -> dict:
    """The exact canonical payload `policy_id` covers."""
    return _trust_policy_identity_payload_fields(
        policy_epoch=policy.policy_epoch,
        trusted_signers=policy.trusted_signers,
        rules=policy.rules,
        spec_version=policy.spec_version,
    )


def compute_trust_policy_id(
    *,
    policy_epoch: int,
    trusted_signers: Tuple[TrustedSignerDTO, ...],
    rules: Tuple[TrustPolicyRuleDTO, ...],
    spec_version: str = SPEC_VERSION,
) -> str:
    """Compute the policy_id for a not-yet-constructed TrustPolicyDTO.

    Used by authoring workflows to build a valid TrustPolicyDTO (which
    validates its own id in `__post_init__`) without duplicating the
    canonicalization logic.
    """
    payload = _trust_policy_identity_payload_fields(
        policy_epoch=policy_epoch,
        trusted_signers=trusted_signers,
        rules=rules,
        spec_version=spec_version,
    )
    return sha256_digest(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class ArtifactAuthorizationDTO:
    """The signed trust manifest for one artifact.

    `authorization_id` is this manifest's own content digest - the identity
    the signature actually covers, computed over every field below except
    itself (see `authorization_signing_payload`).
    """

    artifact_digest: str
    artifact_kind: str
    deployment_scope: DeploymentScopeDTO
    policy_epoch: int
    valid_from: str
    valid_until: str
    issuer_signer_id: str
    authorization_id: str
    adapter_contract: Optional[str] = None
    consumer_contract: Optional[str] = None
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.artifact_digest,
            "ArtifactAuthorizationDTO.artifact_digest must be a sha256: digest",
        )
        _require_nonempty_str(
            self.artifact_kind,
            "ArtifactAuthorizationDTO.artifact_kind must be non-empty",
        )
        _require_nonempty_str(
            self.issuer_signer_id,
            "ArtifactAuthorizationDTO.issuer_signer_id must be non-empty",
        )
        _require_sha256(
            self.authorization_id,
            "ArtifactAuthorizationDTO.authorization_id must be a sha256: digest",
        )
        if self.policy_epoch < 0:
            raise VPMValidationError(
                "ArtifactAuthorizationDTO.policy_epoch must be >= 0"
            )
        valid_from_dt = parse_iso8601_utc(
            self.valid_from, "ArtifactAuthorizationDTO.valid_from"
        )
        valid_until_dt = parse_iso8601_utc(
            self.valid_until, "ArtifactAuthorizationDTO.valid_until"
        )
        if valid_from_dt > valid_until_dt:
            raise VPMValidationError(
                "ArtifactAuthorizationDTO.valid_from must not be after valid_until"
            )
        expected_id = sha256_digest(
            canonical_json_bytes(authorization_signing_payload(self))
        )
        if self.authorization_id != expected_id:
            raise VPMValidationError(
                "ArtifactAuthorizationDTO.authorization_id does not match its own canonical content"
            )


def _authorization_signing_payload_fields(
    *,
    artifact_digest: str,
    artifact_kind: str,
    deployment_scope: DeploymentScopeDTO,
    policy_epoch: int,
    valid_from: str,
    valid_until: str,
    issuer_signer_id: str,
    adapter_contract: Optional[str],
    consumer_contract: Optional[str],
    spec_version: str,
) -> dict:
    scope = deployment_scope
    return {
        "spec_version": spec_version,
        "artifact_digest": artifact_digest,
        "artifact_kind": artifact_kind,
        "adapter_contract": adapter_contract,
        "consumer_contract": consumer_contract,
        "deployment_scope": {
            "organization": scope.organization,
            "application": scope.application,
            "environment": scope.environment,
            "device_group": scope.device_group,
            "location": scope.location,
            "artifact_kind": scope.artifact_kind,
            "corpus": scope.corpus,
            "policy_family": scope.policy_family,
        },
        "policy_epoch": policy_epoch,
        "valid_from": valid_from,
        "valid_until": valid_until,
        "issuer_signer_id": issuer_signer_id,
    }


def authorization_signing_payload(authorization: ArtifactAuthorizationDTO) -> dict:
    """The exact canonical payload a signature covers (via its digest).

    Excludes `authorization_id` itself (self-referential) and `spec_version`
    is included so a future spec revision cannot silently reinterpret an old
    signature.
    """
    return _authorization_signing_payload_fields(
        artifact_digest=authorization.artifact_digest,
        artifact_kind=authorization.artifact_kind,
        deployment_scope=authorization.deployment_scope,
        policy_epoch=authorization.policy_epoch,
        valid_from=authorization.valid_from,
        valid_until=authorization.valid_until,
        issuer_signer_id=authorization.issuer_signer_id,
        adapter_contract=authorization.adapter_contract,
        consumer_contract=authorization.consumer_contract,
        spec_version=authorization.spec_version,
    )


def compute_authorization_id(
    *,
    artifact_digest: str,
    artifact_kind: str,
    deployment_scope: DeploymentScopeDTO,
    policy_epoch: int,
    valid_from: str,
    valid_until: str,
    issuer_signer_id: str,
    adapter_contract: Optional[str] = None,
    consumer_contract: Optional[str] = None,
    spec_version: str = SPEC_VERSION,
) -> str:
    """Compute the authorization_id for a not-yet-constructed authorization.

    Used by authoring workflows to build a valid ArtifactAuthorizationDTO
    (which validates its own id in `__post_init__`) without duplicating the
    canonicalization logic.
    """
    payload = _authorization_signing_payload_fields(
        artifact_digest=artifact_digest,
        artifact_kind=artifact_kind,
        deployment_scope=deployment_scope,
        policy_epoch=policy_epoch,
        valid_from=valid_from,
        valid_until=valid_until,
        issuer_signer_id=issuer_signer_id,
        adapter_contract=adapter_contract,
        consumer_contract=consumer_contract,
        spec_version=spec_version,
    )
    return sha256_digest(canonical_json_bytes(payload))


@dataclass(frozen=True, slots=True)
class SignatureEnvelopeDTO:
    """A signature over one ArtifactAuthorizationDTO's authorization_id."""

    authorization_id: str
    signer_id: str
    signature_hex: str
    key_algorithm: str = "ed25519"
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.authorization_id,
            "SignatureEnvelopeDTO.authorization_id must be a sha256: digest",
        )
        _require_nonempty_str(
            self.signer_id, "SignatureEnvelopeDTO.signer_id must be non-empty"
        )
        _require_nonempty_str(
            self.signature_hex, "SignatureEnvelopeDTO.signature_hex must be non-empty"
        )
        if self.key_algorithm != "ed25519":
            raise VPMValidationError(
                f"unsupported key_algorithm: {self.key_algorithm!r} (only 'ed25519' is implemented)"
            )


@dataclass(frozen=True, slots=True)
class RevocationRecordDTO:
    """One auditable revocation entry."""

    revocation_id: str
    target_kind: str  # "signer" | "signature_envelope" | "artifact_authorization" | "artifact_digest"
    target_id: str
    revoked_at: str
    reason: Optional[str] = None
    spec_version: str = SPEC_VERSION

    _VALID_TARGET_KINDS = (
        "signer",
        "signature_envelope",
        "artifact_authorization",
        "artifact_digest",
    )

    def __post_init__(self) -> None:
        _require_nonempty_str(
            self.revocation_id, "RevocationRecordDTO.revocation_id must be non-empty"
        )
        if self.target_kind not in self._VALID_TARGET_KINDS:
            raise VPMValidationError(
                f"RevocationRecordDTO.target_kind must be one of {self._VALID_TARGET_KINDS}"
            )
        _require_nonempty_str(
            self.target_id, "RevocationRecordDTO.target_id must be non-empty"
        )


@dataclass(frozen=True, slots=True)
class TrustVerificationRequestDTO:
    """Everything `verify_artifact_for_scope` needs, as an auditable record."""

    artifact_digest: str
    artifact_kind: str
    deployment_scope: DeploymentScopeDTO
    minimum_epoch: int
    evaluation_time: str
    require_signature: bool = True
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        _require_sha256(
            self.artifact_digest,
            "TrustVerificationRequestDTO.artifact_digest must be a sha256: digest",
        )
        _require_nonempty_str(
            self.artifact_kind,
            "TrustVerificationRequestDTO.artifact_kind must be non-empty",
        )
        _require_nonempty_str(
            self.evaluation_time,
            "TrustVerificationRequestDTO.evaluation_time must be an explicit, non-empty timestamp",
        )
        if self.minimum_epoch < 0:
            raise VPMValidationError(
                "TrustVerificationRequestDTO.minimum_epoch must be >= 0"
            )


@dataclass(frozen=True, slots=True)
class TrustDecisionDTO:
    """Every component outcome, preserved - never collapsed to one boolean.

    Also a complete audit receipt: `trust_policy_id`, `authorization_id`,
    `artifact_digest`, `signer_id`, `deployment_scope_id`, and (when a
    signature was presented) `signature_envelope_id` identify exactly which
    policy, authorization, artifact, signer, scope, and signature this
    decision was computed from - without these, a caller could see that
    *some* artifact was authorized but not reconstruct, from the decision
    alone, which policy or authorization actually granted it.
    """

    integrity_valid: bool
    signature_valid: bool
    signer_known: bool
    signer_trusted: bool
    artifact_kind_allowed: bool
    scope_authorized: bool
    time_valid: bool
    epoch_valid: bool
    not_revoked: bool
    decision: str  # "authorized" | "rejected" | "indeterminate"
    trust_policy_id: str
    authorization_id: str
    artifact_digest: str
    signer_id: str
    deployment_scope_id: str
    signature_envelope_id: Optional[str] = None
    failure_codes: Tuple[str, ...] = field(default_factory=tuple)
    evaluated_at: str = ""
    spec_version: str = SPEC_VERSION

    _VALID_DECISIONS = ("authorized", "rejected", "indeterminate")

    def __post_init__(self) -> None:
        if self.decision not in self._VALID_DECISIONS:
            raise VPMValidationError(
                f"TrustDecisionDTO.decision must be one of {self._VALID_DECISIONS}"
            )
        _require_sha256(
            self.trust_policy_id,
            "TrustDecisionDTO.trust_policy_id must be a sha256: digest",
        )
        _require_sha256(
            self.authorization_id,
            "TrustDecisionDTO.authorization_id must be a sha256: digest",
        )
        _require_sha256(
            self.artifact_digest,
            "TrustDecisionDTO.artifact_digest must be a sha256: digest",
        )
        _require_nonempty_str(
            self.signer_id, "TrustDecisionDTO.signer_id must be non-empty"
        )
        _require_sha256(
            self.deployment_scope_id,
            "TrustDecisionDTO.deployment_scope_id must be a sha256: digest",
        )
        if self.signature_envelope_id is not None:
            _require_nonempty_str(
                self.signature_envelope_id,
                "TrustDecisionDTO.signature_envelope_id must be non-empty when present",
            )

    @property
    def blocks_execution(self) -> bool:
        """True for both 'rejected' and 'indeterminate' (fail-closed)."""
        return self.decision != "authorized"
