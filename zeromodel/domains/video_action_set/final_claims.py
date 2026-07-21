from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...artifact import VPMValidationError
from .canonical_json import canonical_sha256
from .final_access_dto import FinalExecutionReceiptDTO


FINAL_CLAIM_REGISTRY_VERSION = "zeromodel-video-final-claim-registry/v1"


def build_final_claim_registry(
    *,
    receipt: FinalExecutionReceiptDTO,
    evaluation_result: Mapping[str, object],
    claim_rules: Mapping[str, object],
) -> dict[str, Any]:
    decision = evaluation_result.get("decision")
    if decision not in {"passed", "failed", "indeterminate"}:
        raise VPMValidationError("final evaluation result mismatch")
    requested_claims = claim_rules.get("claims", ())
    if not isinstance(requested_claims, list):
        raise VPMValidationError("final claim rules mismatch")
    status = "eligible" if decision == "passed" else "blocked"
    claims = [
        {
            "claim_id": str(claim.get("claim_id")),
            "status": status,
            "blocking_reason": None
            if status == "eligible"
            else f"final evaluation was {decision}",
        }
        for claim in requested_claims
        if isinstance(claim, Mapping)
    ]
    payload = {
        "version": FINAL_CLAIM_REGISTRY_VERSION,
        "receipt_digest": receipt.receipt_digest,
        "evaluation_digest": evaluation_result.get("evaluation_digest"),
        "decision": decision,
        "claims": claims,
    }
    return payload | {"claim_registry_digest": canonical_sha256(payload)}


__all__ = [
    "FINAL_CLAIM_REGISTRY_VERSION",
    "build_final_claim_registry",
]
