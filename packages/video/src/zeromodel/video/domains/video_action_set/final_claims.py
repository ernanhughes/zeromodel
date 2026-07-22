from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.final_access_dto import FinalEvaluationResultDTO, FinalExecutionReceiptDTO


FINAL_CLAIM_REGISTRY_VERSION = "zeromodel-video-final-claim-registry/v1"


def build_final_claim_registry(
    *,
    receipt: FinalExecutionReceiptDTO,
    evaluation_result: FinalEvaluationResultDTO,
    claim_rules: Mapping[str, object],
) -> dict[str, Any]:
    if not isinstance(receipt, FinalExecutionReceiptDTO):
        raise VPMValidationError("final execution receipt mismatch")
    if not isinstance(evaluation_result, FinalEvaluationResultDTO):
        raise VPMValidationError("final evaluation result mismatch")
    validate_receipt_evaluation_binding(receipt, evaluation_result)
    decision = evaluation_result.decision
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
        "evaluation_digest": evaluation_result.evaluation_digest,
        "decision": decision,
        "claims": claims,
    }
    return payload | {"claim_registry_digest": canonical_sha256(payload)}


def validate_receipt_evaluation_binding(
    receipt: FinalExecutionReceiptDTO,
    evaluation_result: FinalEvaluationResultDTO,
) -> None:
    if (
        evaluation_result.evaluation_digest != receipt.evaluation_digest
        or evaluation_result.evidence_digest != receipt.evidence_digest
        or evaluation_result.protocol_digest != receipt.protocol_digest
        or evaluation_result.decision != receipt.decision
    ):
        raise VPMValidationError("final receipt evaluation binding mismatch")


__all__ = [
    "FINAL_CLAIM_REGISTRY_VERSION",
    "build_final_claim_registry",
    "validate_receipt_evaluation_binding",
]
