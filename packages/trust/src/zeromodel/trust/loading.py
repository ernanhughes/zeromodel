"""Fail-closed loading example.

This is deliberately small: it demonstrates the pattern a consumer should
follow when loading a trust-verified artifact. It does not execute any real
controller and makes no safety-certification claim.
"""

from __future__ import annotations

from zeromodel.trust.dto import TrustDecisionDTO


class ArtifactNotAuthorized(Exception):
    """Raised when a `TrustDecisionDTO` does not authorize artifact use.

    Both "rejected" and "indeterminate" decisions raise this - only
    "authorized" is safe to act on.
    """

    def __init__(self, decision: TrustDecisionDTO) -> None:
        self.decision = decision
        super().__init__(
            f"artifact authorization is {decision.decision!r} "
            f"(failure_codes={decision.failure_codes})"
        )


def require_authorized(decision: TrustDecisionDTO) -> TrustDecisionDTO:
    """Fail closed: return `decision` unchanged if authorized, else raise.

    Example::

        decision = verify_artifact_for_scope(...)
        require_authorized(decision)
        # only now is it safe to load/use the artifact
    """
    if decision.decision != "authorized":
        raise ArtifactNotAuthorized(decision)
    return decision
