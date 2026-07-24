"""Read-only four-store certification integrity audit for Stage P17I."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .execution_journal import SqliteGovernedExecutionAttemptStore
from .governance_audit import audit_governance_integrity
from .lifecycle import PerceptionModelLifecycleStore
from .sql_certification import SqliteGovernanceCertificationStore
from .sql_governance import SqlitePerceptionGovernanceLedgerStore

CERTIFICATION_AUDIT_VERSION: Final = "perception-certification-integrity-audit/1"
CERTIFICATION_AUDIT_FINDING_VERSION: Final = "perception-certification-integrity-finding/1"
CERTIFICATION_AUDIT_SEMANTICS: Final = (
    "read_only_reconciliation_of_lifecycle_governance_attempt_and_certification_history"
)
CERTIFICATION_AUDIT_STATUSES: Final = {"valid", "attention_required", "invalid"}
CERTIFICATION_AUDIT_SEVERITIES: Final = {"info", "warning", "error"}


class PerceptionCertificationAuditError(ValueError):
    """Raised when certification audit contracts are violated."""


def _digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


@dataclass(frozen=True)
class CertificationIntegrityFindingDTO:
    finding_id: str
    severity: str
    code: str
    subject_kind: str
    subject_id: str
    related_ids: tuple[str, ...]
    message: str
    version: str = CERTIFICATION_AUDIT_FINDING_VERSION

    def __post_init__(self) -> None:
        if not all((self.finding_id, self.severity, self.code, self.subject_kind, self.subject_id, self.message)):
            raise PerceptionCertificationAuditError("finding fields must be non-empty")
        if self.severity not in CERTIFICATION_AUDIT_SEVERITIES:
            raise PerceptionCertificationAuditError("unsupported finding severity")
        if self.related_ids != tuple(sorted(set(self.related_ids))):
            raise PerceptionCertificationAuditError("related identities must be sorted and unique")
        if self.version != CERTIFICATION_AUDIT_FINDING_VERSION:
            raise PerceptionCertificationAuditError("unsupported finding version")


@dataclass(frozen=True)
class CertificationIntegrityAuditReportDTO:
    report_id: str
    status: str
    governance_audit_report_id: str
    governance_audit_status: str
    certification_count: int
    successful_attempt_count: int
    finding_count: int
    findings: tuple[CertificationIntegrityFindingDTO, ...]
    semantics: str = CERTIFICATION_AUDIT_SEMANTICS
    version: str = CERTIFICATION_AUDIT_VERSION

    def __post_init__(self) -> None:
        if not self.report_id or not self.governance_audit_report_id:
            raise PerceptionCertificationAuditError("report identities must be non-empty")
        if self.status not in CERTIFICATION_AUDIT_STATUSES:
            raise PerceptionCertificationAuditError("unsupported report status")
        if min(self.certification_count, self.successful_attempt_count, self.finding_count) < 0:
            raise PerceptionCertificationAuditError("report counts cannot be negative")
        if self.finding_count != len(self.findings):
            raise PerceptionCertificationAuditError("finding count mismatch")
        expected = tuple(sorted(self.findings, key=lambda item: (item.severity, item.code, item.subject_kind, item.subject_id, item.finding_id)))
        if self.findings != expected:
            raise PerceptionCertificationAuditError("findings must be canonically sorted")
        if self.semantics != CERTIFICATION_AUDIT_SEMANTICS or self.version != CERTIFICATION_AUDIT_VERSION:
            raise PerceptionCertificationAuditError("unsupported report contract")


def _finding(severity: str, code: str, subject_kind: str, subject_id: str, message: str, *related_ids: str) -> CertificationIntegrityFindingDTO:
    related = tuple(sorted(set(related_ids)))
    payload: Mapping[str, object] = {
        "severity": severity,
        "code": code,
        "subject_kind": subject_kind,
        "subject_id": subject_id,
        "related_ids": related,
        "message": message,
        "version": CERTIFICATION_AUDIT_FINDING_VERSION,
    }
    return CertificationIntegrityFindingDTO(finding_id=_digest(payload), **payload)


def audit_certification_integrity(
    lifecycle_store: PerceptionModelLifecycleStore,
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    attempt_store: SqliteGovernedExecutionAttemptStore,
    certification_store: SqliteGovernanceCertificationStore,
) -> CertificationIntegrityAuditReportDTO:
    """Reconcile certifications with lifecycle, governance, and attempt history."""

    base = audit_governance_integrity(lifecycle_store, governance_store, attempt_store)
    recommendations = {item.recommendation_id: item for item in governance_store.list_recommendations()}
    dispositions = {item.disposition_id: item for item in governance_store.list_dispositions()}
    receipts = {item.receipt_id: item for item in governance_store.list_execution_receipts()}
    attempts = {item.attempt_id: item for item in attempt_store.list_attempts()}
    bundles = certification_store.list_certifications()
    certified_attempts = {item.certification.attempt_id for item in bundles}
    findings: list[CertificationIntegrityFindingDTO] = []

    if base.status != "valid":
        severity = "error" if base.status == "invalid" else "warning"
        findings.append(_finding(severity, f"underlying_governance_{base.status}", "governance_audit", base.report_id, "three-store governance integrity is not valid"))

    for bundle in bundles:
        item = bundle.certification
        attempt = attempts.get(item.attempt_id)
        disposition = dispositions.get(item.disposition_id)
        recommendation = recommendations.get(item.recommendation_id)
        receipt = receipts.get(item.receipt_id)
        for owned, code, related in (
            (attempt, "certification_missing_attempt", item.attempt_id),
            (disposition, "certification_missing_disposition", item.disposition_id),
            (recommendation, "certification_missing_recommendation", item.recommendation_id),
            (receipt, "certification_missing_receipt", item.receipt_id),
        ):
            if owned is None:
                findings.append(_finding("error", code, "certification", item.certification_id, "certification references a missing artifact", related))
        if attempt is not None and (attempt.disposition_id != item.disposition_id or attempt.recommendation_id != item.recommendation_id):
            findings.append(_finding("error", "certification_attempt_ownership_mismatch", "certification", item.certification_id, "attempt ownership differs from certification", attempt.attempt_id))
        if disposition is not None and disposition.recommendation_id != item.recommendation_id:
            findings.append(_finding("error", "certification_disposition_ownership_mismatch", "certification", item.certification_id, "disposition ownership differs from certification", disposition.disposition_id))
        if receipt is not None:
            if receipt.disposition_id != item.disposition_id or receipt.recommendation_id != item.recommendation_id:
                findings.append(_finding("error", "certification_receipt_ownership_mismatch", "certification", item.certification_id, "receipt ownership differs from certification", receipt.receipt_id))
            if receipt.pointer_revision != item.resulting_pointer_revision:
                findings.append(_finding("error", "certification_receipt_revision_mismatch", "certification", item.certification_id, "receipt revision differs from certification", receipt.receipt_id))

    successful = 0
    for attempt in attempts.values():
        events = attempt_store.list_events(attempt.attempt_id)
        terminal = events[-1] if events else None
        if terminal is None or terminal.event_kind not in {"completed", "reconciled", "idempotent"}:
            continue
        successful += 1
        if attempt.attempt_id not in certified_attempts:
            findings.append(_finding("warning", "successful_attempt_uncertified", "attempt", attempt.attempt_id, "successful execution has no durable certification", attempt.disposition_id))

    canonical = tuple(sorted(findings, key=lambda item: (item.severity, item.code, item.subject_kind, item.subject_id, item.finding_id)))
    status = "invalid" if any(item.severity == "error" for item in canonical) else "attention_required" if any(item.severity == "warning" for item in canonical) else "valid"
    payload: Mapping[str, object] = {
        "status": status,
        "governance_audit_report_id": base.report_id,
        "governance_audit_status": base.status,
        "certification_count": len(bundles),
        "successful_attempt_count": successful,
        "finding_ids": tuple(item.finding_id for item in canonical),
        "semantics": CERTIFICATION_AUDIT_SEMANTICS,
        "version": CERTIFICATION_AUDIT_VERSION,
    }
    return CertificationIntegrityAuditReportDTO(
        report_id=_digest(payload),
        status=status,
        governance_audit_report_id=base.report_id,
        governance_audit_status=base.status,
        certification_count=len(bundles),
        successful_attempt_count=successful,
        finding_count=len(canonical),
        findings=canonical,
    )
