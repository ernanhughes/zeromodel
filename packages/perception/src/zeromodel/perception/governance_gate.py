"""Audit-gated governed rollback execution for Stage P17G.

P17F can prove whether lifecycle, governance, and execution-journal history agree. P17G
uses that report as an explicit execution precondition and verifies the resulting chain again
after execution. The only non-valid pre-state accepted is one recoverable prepared-only
warning for the exact deterministic attempt being resumed.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .compatibility import ModelCompatibilityContractDTO
from .disposition import OperationalRecommendationDispositionDTO
from .execution_journal import (
    GovernedExecutionAttemptDTO,
    SqliteGovernedExecutionAttemptStore,
    build_governed_execution_attempt,
    execute_journaled_approved_rollback,
)
from .governance_audit import (
    GovernanceIntegrityAuditReportDTO,
    audit_governance_integrity,
)
from .lifecycle import PerceptionModelLifecycleStore
from .recommendation import OperationalRecommendationDTO
from .sql_governance import (
    GovernanceExecutionReceiptDTO,
    SqlitePerceptionGovernanceLedgerStore,
)

GOVERNANCE_EXECUTION_GATE_VERSION: Final = "perception-governance-execution-gate/1"
GOVERNANCE_EXECUTION_GATE_SEMANTICS: Final = (
    "pre_and_post_integrity_audit_gated_governed_rollback_execution"
)
GOVERNANCE_EXECUTION_GATE_MODES: Final = {"clean", "recover_prepared_attempt"}


class PerceptionGovernanceExecutionGateError(ValueError):
    """Raised when cross-store integrity does not authorize governed execution."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


@dataclass(frozen=True)
class GovernanceExecutionGateDTO:
    gate_id: str
    recommendation_id: str
    disposition_id: str
    attempt_id: str
    receipt_id: str
    pre_audit_report_id: str
    post_audit_report_id: str
    authorization_mode: str
    resulting_pointer_revision: int
    semantics: str = GOVERNANCE_EXECUTION_GATE_SEMANTICS
    version: str = GOVERNANCE_EXECUTION_GATE_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.gate_id,
                self.recommendation_id,
                self.disposition_id,
                self.attempt_id,
                self.receipt_id,
                self.pre_audit_report_id,
                self.post_audit_report_id,
            )
        ):
            raise PerceptionGovernanceExecutionGateError(
                "governance execution gate identities must be non-empty"
            )
        if self.authorization_mode not in GOVERNANCE_EXECUTION_GATE_MODES:
            raise PerceptionGovernanceExecutionGateError(
                "unsupported governance execution authorization mode"
            )
        if self.resulting_pointer_revision <= 0:
            raise PerceptionGovernanceExecutionGateError(
                "governance execution gate requires a positive resulting pointer revision"
            )
        if self.semantics != GOVERNANCE_EXECUTION_GATE_SEMANTICS:
            raise PerceptionGovernanceExecutionGateError(
                "unsupported governance execution gate semantics"
            )
        if self.version != GOVERNANCE_EXECUTION_GATE_VERSION:
            raise PerceptionGovernanceExecutionGateError(
                "unsupported governance execution gate version"
            )


def authorize_governance_execution(
    report: GovernanceIntegrityAuditReportDTO,
    attempt: GovernedExecutionAttemptDTO,
) -> str:
    """Return the permitted gate mode or reject the audited pre-state.

    A valid report authorizes a clean execution. An attention-required report authorizes
    only recovery of one prepared-only warning belonging to this exact attempt. Informational
    findings may coexist with either accepted state because they do not downgrade integrity.
    """

    if report.status == "valid":
        return "clean"
    if report.status == "invalid":
        raise PerceptionGovernanceExecutionGateError(
            "governance integrity audit is invalid; execution is blocked"
        )

    non_info = tuple(item for item in report.findings if item.severity != "info")
    exact_recovery = (
        len(non_info) == 1
        and non_info[0].severity == "warning"
        and non_info[0].code == "attempt_prepared_incomplete"
        and non_info[0].subject_kind == "attempt"
        and non_info[0].subject_id == attempt.attempt_id
    )
    if exact_recovery:
        return "recover_prepared_attempt"
    raise PerceptionGovernanceExecutionGateError(
        "governance integrity requires attention unrelated to this recoverable attempt"
    )


def _build_gate(
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    attempt: GovernedExecutionAttemptDTO,
    receipt: GovernanceExecutionReceiptDTO,
    pre_audit: GovernanceIntegrityAuditReportDTO,
    post_audit: GovernanceIntegrityAuditReportDTO,
    *,
    authorization_mode: str,
) -> GovernanceExecutionGateDTO:
    payload: Mapping[str, object] = {
        "attempt_id": attempt.attempt_id,
        "authorization_mode": authorization_mode,
        "disposition_id": disposition.disposition_id,
        "post_audit_report_id": post_audit.report_id,
        "pre_audit_report_id": pre_audit.report_id,
        "receipt_id": receipt.receipt_id,
        "recommendation_id": recommendation.recommendation_id,
        "resulting_pointer_revision": receipt.pointer_revision,
        "semantics": GOVERNANCE_EXECUTION_GATE_SEMANTICS,
        "version": GOVERNANCE_EXECUTION_GATE_VERSION,
    }
    return GovernanceExecutionGateDTO(gate_id=_digest(payload), **payload)


def execute_audit_gated_approved_rollback(
    lifecycle_store: PerceptionModelLifecycleStore,
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    attempt_store: SqliteGovernedExecutionAttemptStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
) -> tuple[
    GovernanceExecutionGateDTO,
    GovernedExecutionAttemptDTO,
    GovernanceExecutionReceiptDTO,
    GovernanceIntegrityAuditReportDTO,
]:
    """Audit, execute or recover once, and prove the resulting governance chain is valid."""

    attempt = build_governed_execution_attempt(
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    pre_audit = audit_governance_integrity(
        lifecycle_store,
        governance_store,
        attempt_store,
    )
    authorization_mode = authorize_governance_execution(pre_audit, attempt)

    executed_attempt, receipt = execute_journaled_approved_rollback(
        lifecycle_store,
        governance_store,
        attempt_store,
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    if executed_attempt.attempt_id != attempt.attempt_id:
        raise PerceptionGovernanceExecutionGateError(
            "executed attempt identity differs from audited attempt"
        )

    post_audit = audit_governance_integrity(
        lifecycle_store,
        governance_store,
        attempt_store,
    )
    if post_audit.status != "valid":
        raise PerceptionGovernanceExecutionGateError(
            "governed execution completed but the resulting integrity audit is not valid"
        )

    gate = _build_gate(
        recommendation,
        disposition,
        attempt,
        receipt,
        pre_audit,
        post_audit,
        authorization_mode=authorization_mode,
    )
    return gate, attempt, receipt, post_audit
