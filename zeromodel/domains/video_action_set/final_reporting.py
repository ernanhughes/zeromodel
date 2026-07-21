from __future__ import annotations

from collections.abc import Mapping

from ...artifact import VPMValidationError
from .final_access_dto import FinalExecutionReceiptDTO


FINAL_REPORT_VERSION = "zeromodel-video-final-report/v1"


def generate_final_report(
    *,
    receipt: FinalExecutionReceiptDTO,
    evaluation_result: Mapping[str, object],
    claim_registry: Mapping[str, object],
) -> str:
    if receipt.state != "completed":
        raise VPMValidationError("final report requires a completed receipt")
    decision = evaluation_result.get("decision")
    if decision not in {"passed", "failed", "indeterminate"}:
        raise VPMValidationError("final evaluation result mismatch")
    claim_digest = claim_registry.get("claim_registry_digest")
    if not isinstance(claim_digest, str):
        raise VPMValidationError("final claim registry mismatch")
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
