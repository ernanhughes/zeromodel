"""zeromodel-trust: artifact trust and deployment-authorization kernel.

Depends only on `zeromodel` (core) and `zeromodel-artifacts`. Nothing in
`zeromodel.core` or `zeromodel.artifacts` depends back on this package.
"""

from zeromodel.trust.crypto import (
    GeneratedSigningKey,
    generate_signing_key,
    sign_digest,
    verify_signature,
)
from zeromodel.trust.loading import ArtifactNotAuthorized, require_authorized
from zeromodel.trust.dto import (
    SPEC_VERSION,
    ArtifactAuthorizationDTO,
    DeploymentScopeDTO,
    RevocationRecordDTO,
    SignatureEnvelopeDTO,
    SignerIdentityDTO,
    TrustDecisionDTO,
    TrustedSignerDTO,
    TrustFailureCode,
    TrustPolicyDTO,
    TrustPolicyRuleDTO,
    TrustVerificationRequestDTO,
    authorization_signing_payload,
    compute_authorization_id,
    compute_deployment_scope_id,
    compute_trust_policy_id,
    trust_policy_identity_payload,
)
from zeromodel.trust.revocation import (
    IndeterminateRevocationResolver,
    InMemoryRevocationResolver,
    RevocationResolver,
    RevocationStatus,
)
from zeromodel.trust.verify import verify_artifact_for_scope

__all__ = [
    "SPEC_VERSION",
    "ArtifactAuthorizationDTO",
    "ArtifactNotAuthorized",
    "DeploymentScopeDTO",
    "GeneratedSigningKey",
    "IndeterminateRevocationResolver",
    "InMemoryRevocationResolver",
    "RevocationRecordDTO",
    "RevocationResolver",
    "RevocationStatus",
    "SignatureEnvelopeDTO",
    "SignerIdentityDTO",
    "TrustDecisionDTO",
    "TrustedSignerDTO",
    "TrustFailureCode",
    "TrustPolicyDTO",
    "TrustPolicyRuleDTO",
    "TrustVerificationRequestDTO",
    "authorization_signing_payload",
    "compute_authorization_id",
    "compute_deployment_scope_id",
    "compute_trust_policy_id",
    "generate_signing_key",
    "require_authorized",
    "sign_digest",
    "trust_policy_identity_payload",
    "verify_artifact_for_scope",
    "verify_signature",
]
