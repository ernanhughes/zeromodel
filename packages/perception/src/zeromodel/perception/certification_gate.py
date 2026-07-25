"""Certification-aware governed rollback execution for Stage P17J."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .certification_audit import (
    CertificationIntegrityAuditReportDTO,
    audit_certification_integrity,
)
from .compatibility import ModelCompatibilityContractDTO
from .disposition import OperationalRecommendationDispositionDTO
from .execution_journal import (
    GovernedExecutionAttemptDTO,
    SqliteGovernedExecutionAttemptStore,
)
from .lifecycle import PerceptionModelLifecycleStore
from .recommendation import OperationalRecommendationDTO
from .sql_certification import (
    GovernanceExecutionCertificationBundleDTO,
    SqliteGovernanceCertificationStore,
    execute_and_certify_audit_gated_rollback,
)
from .sql_governance import (
    GovernanceExecutionReceiptDTO,
    SqlitePerceptionGovernanceLedgerStore,
)

CERTIFICATION_EXECUTION_GATE_VERSION: Final = (
    "perception-certification-execution-gate/1"
)
CERTIFICATION_EXECUTION_GATE_SEMANTICS: Final = (
    "four_store_preflight_and_postflight_gated_governed_execution"
)


class PerceptionCertificationExecutionGateError(ValueError):
    """Raised when certification integrity does not authorize or certify execution."""


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class CertificationExecutionGateDTO:
    gate_id: str
    certification_id: str
    attempt_id: str
    receipt_id: str
    preflight_report_id: str
    postflight_report_id: str
    resulting_pointer_revision: int
    semantics: str = CERTIFICATION_EXECUTION_GATE_SEMANTICS
    version: str = CERTIFICATION_EXECUTION_GATE_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.gate_id,
                self.certification_id,
                self.attempt_id,
                self.receipt_id,
                self.preflight_report_id,
                self.postflight_report_id,
            )
        ):
            raise PerceptionCertificationExecutionGateError(
                "certification execution gate identities must be non-empty"
            )
        if self.resulting_pointer_revision <= 0:
            raise PerceptionCertificationExecutionGateError(
                "certification execution gate requires a positive pointer revision"
            )
        if self.semantics != CERTIFICATION_EXECUTION_GATE_SEMANTICS:
            raise PerceptionCertificationExecutionGateError(
                "unsupported gate semantics"
            )
        if self.version != CERTIFICATION_EXECUTION_GATE_VERSION:
            raise PerceptionCertificationExecutionGateError("unsupported gate version")


def authorize_certification_execution(
    report: CertificationIntegrityAuditReportDTO,
) -> None:
    """Require a completely valid four-store state before starting fresh execution."""

    if report.status != "valid":
        raise PerceptionCertificationExecutionGateError(
            "certification integrity is not valid; fresh execution is blocked"
        )


def execute_certification_gated_approved_rollback(
    lifecycle_store: PerceptionModelLifecycleStore,
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    attempt_store: SqliteGovernedExecutionAttemptStore,
    certification_store: SqliteGovernanceCertificationStore,
    recommendation: OperationalRecommendationDTO,
    disposition: OperationalRecommendationDispositionDTO,
    *,
    current_contract: ModelCompatibilityContractDTO,
    target_contract: ModelCompatibilityContractDTO,
) -> tuple[
    CertificationExecutionGateDTO,
    GovernanceExecutionCertificationBundleDTO,
    GovernedExecutionAttemptDTO,
    GovernanceExecutionReceiptDTO,
    CertificationIntegrityAuditReportDTO,
]:
    """Preflight four stores, execute and certify, then prove all four stores are valid."""

    preflight = audit_certification_integrity(
        lifecycle_store,
        governance_store,
        attempt_store,
        certification_store,
    )
    authorize_certification_execution(preflight)

    bundle, attempt, receipt = execute_and_certify_audit_gated_rollback(
        lifecycle_store,
        governance_store,
        attempt_store,
        certification_store,
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )
    restored = certification_store.get_certification(
        bundle.certification.certification_id
    )
    if restored != bundle:
        raise PerceptionCertificationExecutionGateError(
            "persisted certification differs from completed execution bundle"
        )

    postflight = audit_certification_integrity(
        lifecycle_store,
        governance_store,
        attempt_store,
        certification_store,
    )
    if postflight.status != "valid":
        raise PerceptionCertificationExecutionGateError(
            "certified execution completed but four-store integrity is not valid"
        )

    payload: Mapping[str, object] = {
        "attempt_id": attempt.attempt_id,
        "certification_id": bundle.certification.certification_id,
        "postflight_report_id": postflight.report_id,
        "preflight_report_id": preflight.report_id,
        "receipt_id": receipt.receipt_id,
        "resulting_pointer_revision": receipt.pointer_revision,
        "semantics": CERTIFICATION_EXECUTION_GATE_SEMANTICS,
        "version": CERTIFICATION_EXECUTION_GATE_VERSION,
    }
    gate = CertificationExecutionGateDTO(gate_id=_digest(payload), **payload)
    return gate, bundle, attempt, receipt, postflight
