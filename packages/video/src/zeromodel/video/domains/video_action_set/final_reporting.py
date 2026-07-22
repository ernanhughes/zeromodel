from __future__ import annotations

from collections.abc import Mapping

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.final_access_dto import FinalEvaluationResultDTO, FinalExecutionReceiptDTO
from zeromodel.video.domains.video_action_set.final_claims import validate_receipt_evaluation_binding


FINAL_REPORT_VERSION = "zeromodel-video-final-report/v1"


def generate_final_report(
    *,
    receipt: FinalExecutionReceiptDTO,
    evaluation_result: FinalEvaluationResultDTO,
    claim_registry: Mapping[str, object],
) -> str:
    if not isinstance(receipt, FinalExecutionReceiptDTO):
        raise VPMValidationError("final execution receipt mismatch")
    if receipt.state != "completed":
        raise VPMValidationError("final report requires a completed receipt")
    if not isinstance(evaluation_result, FinalEvaluationResultDTO):
        raise VPMValidationError("final evaluation result mismatch")
    validate_receipt_evaluation_binding(receipt, evaluation_result)
    decision = evaluation_result.decision
    claim_digest = claim_registry.get("claim_registry_digest")
    if not isinstance(claim_digest, str):
        raise VPMValidationError("final claim registry mismatch")
    claim_payload = {
        key: value
        for key, value in claim_registry.items()
        if key != "claim_registry_digest"
    }
    if (
        canonical_sha256(claim_payload) != claim_digest
        or claim_registry.get("receipt_digest") != receipt.receipt_digest
        or claim_registry.get("evaluation_digest")
        != evaluation_result.evaluation_digest
        or claim_registry.get("decision") != decision
    ):
        raise VPMValidationError("final claim registry binding mismatch")
    lines = [
        f"# Video Action-Set Final Report ({FINAL_REPORT_VERSION})",
        "",
        f"- access_id: `{receipt.access_id}`",
        f"- authorization_id: `{receipt.authorization_id}`",
        f"- receipt_digest: `{receipt.receipt_digest}`",
        f"- protocol_digest: `{receipt.protocol_digest}`",
        f"- sealed_plan_digest: `{receipt.sealed_plan_digest}`",
        f"- event_chain_digest: `{receipt.event_chain_digest}`",
        f"- final_decision: `{decision}`",
        f"- claim_registry_digest: `{claim_digest}`",
        "",
        "This report is a reconstruction artifact. It is not a rerun of final access.",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "FINAL_REPORT_VERSION",
    "generate_final_report",
]
