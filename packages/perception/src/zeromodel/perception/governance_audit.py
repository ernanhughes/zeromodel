"""Read-only cross-store governance integrity audit for Stage P17F.

The lifecycle store remains authoritative for active-model history. The governance ledger owns
recommendations, dispositions, and receipts. The execution journal owns attempt intent and
terminal outcomes. This module reconciles those immutable records into one deterministic audit
report without mutating any store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .execution_journal import (
    EXECUTION_ATTEMPT_TERMINAL_KINDS,
    GovernedExecutionAttemptDTO,
    GovernedExecutionAttemptEventDTO,
    SqliteGovernedExecutionAttemptStore,
)
from .lifecycle import PerceptionModelLifecycleStore
from .sql_governance import SqlitePerceptionGovernanceLedgerStore

GOVERNANCE_AUDIT_VERSION: Final = "perception-governance-integrity-audit/1"
GOVERNANCE_AUDIT_FINDING_VERSION: Final = "perception-governance-integrity-finding/1"
GOVERNANCE_AUDIT_SEMANTICS: Final = (
    "read_only_cross_store_reconciliation_of_lifecycle_governance_and_execution_history"
)
GOVERNANCE_AUDIT_SEVERITIES: Final = {"info", "warning", "error"}
GOVERNANCE_AUDIT_STATUSES: Final = {"valid", "attention_required", "invalid"}


class PerceptionGovernanceAuditError(ValueError):
    """Raised when governance audit DTO contracts are violated."""


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
class GovernanceIntegrityFindingDTO:
    finding_id: str
    severity: str
    code: str
    subject_kind: str
    subject_id: str
    related_ids: tuple[str, ...]
    message: str
    version: str = GOVERNANCE_AUDIT_FINDING_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.finding_id,
                self.severity,
                self.code,
                self.subject_kind,
                self.subject_id,
                self.message,
            )
        ):
            raise PerceptionGovernanceAuditError("audit finding fields must be non-empty")
        if self.severity not in GOVERNANCE_AUDIT_SEVERITIES:
            raise PerceptionGovernanceAuditError("unsupported audit finding severity")
        if self.related_ids != tuple(sorted(set(self.related_ids))):
            raise PerceptionGovernanceAuditError("audit related identities must be sorted and unique")
        if self.version != GOVERNANCE_AUDIT_FINDING_VERSION:
            raise PerceptionGovernanceAuditError("unsupported audit finding version")


@dataclass(frozen=True)
class GovernanceIntegrityAuditReportDTO:
    report_id: str
    status: str
    active_pointer_id: str
    active_pointer_revision: int
    recommendation_count: int
    disposition_count: int
    receipt_count: int
    attempt_count: int
    finding_count: int
    findings: tuple[GovernanceIntegrityFindingDTO, ...]
    semantics: str = GOVERNANCE_AUDIT_SEMANTICS
    version: str = GOVERNANCE_AUDIT_VERSION

    def __post_init__(self) -> None:
        if not self.report_id or not self.active_pointer_id:
            raise PerceptionGovernanceAuditError("audit report identities must be non-empty")
        if self.status not in GOVERNANCE_AUDIT_STATUSES:
            raise PerceptionGovernanceAuditError("unsupported audit report status")
        if self.active_pointer_revision < 0:
            raise PerceptionGovernanceAuditError("audit pointer revision cannot be negative")
        counts = (
            self.recommendation_count,
            self.disposition_count,
            self.receipt_count,
            self.attempt_count,
            self.finding_count,
        )
        if any(value < 0 for value in counts):
            raise PerceptionGovernanceAuditError("audit counts cannot be negative")
        if self.finding_count != len(self.findings):
            raise PerceptionGovernanceAuditError("audit finding count does not match findings")
        expected = tuple(
            sorted(
                self.findings,
                key=lambda item: (
                    item.severity,
                    item.code,
                    item.subject_kind,
                    item.subject_id,
                    item.finding_id,
                ),
            )
        )
        if self.findings != expected:
            raise PerceptionGovernanceAuditError("audit findings must be canonically sorted")
        if self.semantics != GOVERNANCE_AUDIT_SEMANTICS:
            raise PerceptionGovernanceAuditError("unsupported audit semantics")
        if self.version != GOVERNANCE_AUDIT_VERSION:
            raise PerceptionGovernanceAuditError("unsupported audit version")


def _finding(
    *,
    severity: str,
    code: str,
    subject_kind: str,
    subject_id: str,
    message: str,
    related_ids: tuple[str, ...] = (),
) -> GovernanceIntegrityFindingDTO:
    related = tuple(sorted(set(related_ids)))
    payload: Mapping[str, object] = {
        "code": code,
        "message": message,
        "related_ids": related,
        "severity": severity,
        "subject_id": subject_id,
        "subject_kind": subject_kind,
        "version": GOVERNANCE_AUDIT_FINDING_VERSION,
    }
    return GovernanceIntegrityFindingDTO(finding_id=_digest(payload), **payload)


def _terminal_event(
    events: tuple[GovernedExecutionAttemptEventDTO, ...],
) -> GovernedExecutionAttemptEventDTO | None:
    if not events:
        return None
    terminal = events[-1]
    return terminal if terminal.event_kind in EXECUTION_ATTEMPT_TERMINAL_KINDS else None


def _audit_attempt(
    attempt: GovernedExecutionAttemptDTO,
    events: tuple[GovernedExecutionAttemptEventDTO, ...],
    *,
    recommendations_by_id: Mapping[str, object],
    dispositions_by_id: Mapping[str, object],
    receipts_by_disposition: Mapping[str, object],
) -> tuple[GovernanceIntegrityFindingDTO, ...]:
    findings: list[GovernanceIntegrityFindingDTO] = []
    recommendation = recommendations_by_id.get(attempt.recommendation_id)
    disposition = dispositions_by_id.get(attempt.disposition_id)
    receipt = receipts_by_disposition.get(attempt.disposition_id)

    if recommendation is None:
        findings.append(
            _finding(
                severity="error",
                code="attempt_missing_recommendation",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                related_ids=(attempt.recommendation_id,),
                message="execution attempt references a missing recommendation",
            )
        )
    if disposition is None:
        findings.append(
            _finding(
                severity="error",
                code="attempt_missing_disposition",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                related_ids=(attempt.disposition_id,),
                message="execution attempt references a missing disposition",
            )
        )
    else:
        if disposition.recommendation_id != attempt.recommendation_id:  # type: ignore[attr-defined]
            findings.append(
                _finding(
                    severity="error",
                    code="attempt_disposition_recommendation_mismatch",
                    subject_kind="attempt",
                    subject_id=attempt.attempt_id,
                    related_ids=(attempt.disposition_id, attempt.recommendation_id),
                    message="attempt and disposition do not reference the same recommendation",
                )
            )
        if disposition.status != "approved":  # type: ignore[attr-defined]
            findings.append(
                _finding(
                    severity="error",
                    code="attempt_without_approval",
                    subject_kind="attempt",
                    subject_id=attempt.attempt_id,
                    related_ids=(attempt.disposition_id,),
                    message="execution attempt does not belong to an approved disposition",
                )
            )

    if not events:
        findings.append(
            _finding(
                severity="error",
                code="attempt_without_prepared_event",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                message="persisted execution attempt has no prepared event",
            )
        )
        return tuple(findings)

    if events[0].event_kind != "prepared" or events[0].sequence_number != 1:
        findings.append(
            _finding(
                severity="error",
                code="attempt_invalid_first_event",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                related_ids=(events[0].event_id,),
                message="execution attempt does not begin with the canonical prepared event",
            )
        )

    terminal = _terminal_event(events)
    if terminal is None:
        findings.append(
            _finding(
                severity="warning",
                code="attempt_prepared_incomplete",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                related_ids=(events[0].event_id,),
                message="prepared execution attempt has no terminal event and may require recovery",
            )
        )
        return tuple(findings)

    if terminal.event_kind == "failed":
        if receipt is not None:
            findings.append(
                _finding(
                    severity="error",
                    code="failed_attempt_has_receipt",
                    subject_kind="attempt",
                    subject_id=attempt.attempt_id,
                    related_ids=(receipt.receipt_id, terminal.event_id),  # type: ignore[attr-defined]
                    message="terminally failed attempt also has a success receipt",
                )
            )
        return tuple(findings)

    if receipt is None:
        findings.append(
            _finding(
                severity="error",
                code="successful_attempt_missing_receipt",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                related_ids=(terminal.event_id,),
                message="successful terminal attempt does not resolve to a governance receipt",
            )
        )
        return tuple(findings)

    if terminal.receipt_id != receipt.receipt_id:  # type: ignore[attr-defined]
        findings.append(
            _finding(
                severity="error",
                code="attempt_receipt_identity_mismatch",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                related_ids=(terminal.event_id, receipt.receipt_id),  # type: ignore[attr-defined]
                message="terminal event references a different receipt identity",
            )
        )
    if terminal.pointer_revision != receipt.pointer_revision:  # type: ignore[attr-defined]
        findings.append(
            _finding(
                severity="error",
                code="attempt_receipt_revision_mismatch",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                related_ids=(terminal.event_id, receipt.receipt_id),  # type: ignore[attr-defined]
                message="terminal event and receipt disagree on resulting pointer revision",
            )
        )
    if receipt.resulting_promoted_model_id != attempt.target_promoted_model_id:  # type: ignore[attr-defined]
        findings.append(
            _finding(
                severity="error",
                code="attempt_receipt_target_mismatch",
                subject_kind="attempt",
                subject_id=attempt.attempt_id,
                related_ids=(receipt.receipt_id,),  # type: ignore[attr-defined]
                message="execution receipt does not activate the attempt target model",
            )
        )
    return tuple(findings)


def audit_governance_integrity(
    lifecycle_store: PerceptionModelLifecycleStore,
    governance_store: SqlitePerceptionGovernanceLedgerStore,
    attempt_store: SqliteGovernedExecutionAttemptStore,
) -> GovernanceIntegrityAuditReportDTO:
    """Reconcile all durable governance records without mutating any store."""

    pointer = lifecycle_store.get_active_pointer()
    transitions = lifecycle_store.list_transitions()
    transitions_by_id = {item.transition_id: item for item in transitions}
    recommendations = governance_store.list_recommendations()
    dispositions = governance_store.list_dispositions()
    receipts = governance_store.list_execution_receipts()
    attempts = attempt_store.list_attempts()

    recommendations_by_id = {item.recommendation_id: item for item in recommendations}
    dispositions_by_id = {item.disposition_id: item for item in dispositions}
    dispositions_by_recommendation = {item.recommendation_id: item for item in dispositions}
    receipts_by_disposition = {item.disposition_id: item for item in receipts}
    attempts_by_disposition = {item.disposition_id: item for item in attempts}

    findings: list[GovernanceIntegrityFindingDTO] = []

    for disposition in dispositions:
        recommendation = recommendations_by_id.get(disposition.recommendation_id)
        if recommendation is None:
            findings.append(
                _finding(
                    severity="error",
                    code="disposition_missing_recommendation",
                    subject_kind="disposition",
                    subject_id=disposition.disposition_id,
                    related_ids=(disposition.recommendation_id,),
                    message="operator disposition references a missing recommendation",
                )
            )
        elif disposition.recommendation_status != recommendation.status:
            findings.append(
                _finding(
                    severity="error",
                    code="disposition_recommendation_status_mismatch",
                    subject_kind="disposition",
                    subject_id=disposition.disposition_id,
                    related_ids=(recommendation.recommendation_id,),
                    message="disposition records a different recommendation status",
                )
            )

    for recommendation in recommendations:
        if recommendation.recommendation_id not in dispositions_by_recommendation:
            findings.append(
                _finding(
                    severity="info",
                    code="recommendation_awaiting_disposition",
                    subject_kind="recommendation",
                    subject_id=recommendation.recommendation_id,
                    message="recommendation has not received a final operator disposition",
                )
            )

    for receipt in receipts:
        disposition = dispositions_by_id.get(receipt.disposition_id)
        if disposition is None:
            findings.append(
                _finding(
                    severity="error",
                    code="receipt_missing_disposition",
                    subject_kind="receipt",
                    subject_id=receipt.receipt_id,
                    related_ids=(receipt.disposition_id,),
                    message="execution receipt references a missing disposition",
                )
            )
        else:
            if disposition.status != "approved":
                findings.append(
                    _finding(
                        severity="error",
                        code="receipt_without_approval",
                        subject_kind="receipt",
                        subject_id=receipt.receipt_id,
                        related_ids=(disposition.disposition_id,),
                        message="execution receipt belongs to a non-approved disposition",
                    )
                )
            if disposition.recommendation_id != receipt.recommendation_id:
                findings.append(
                    _finding(
                        severity="error",
                        code="receipt_recommendation_mismatch",
                        subject_kind="receipt",
                        subject_id=receipt.receipt_id,
                        related_ids=(disposition.disposition_id, receipt.recommendation_id),
                        message="receipt and disposition reference different recommendations",
                    )
                )
        transition = transitions_by_id.get(receipt.transition_id)
        if transition is None:
            findings.append(
                _finding(
                    severity="error",
                    code="receipt_missing_transition",
                    subject_kind="receipt",
                    subject_id=receipt.receipt_id,
                    related_ids=(receipt.transition_id,),
                    message="execution receipt references a missing lifecycle transition",
                )
            )
        else:
            if transition.transition_kind != "rollback":
                findings.append(
                    _finding(
                        severity="error",
                        code="receipt_transition_not_rollback",
                        subject_kind="receipt",
                        subject_id=receipt.receipt_id,
                        related_ids=(transition.transition_id,),
                        message="execution receipt references a non-rollback transition",
                    )
                )
            if transition.sequence_number != receipt.pointer_revision:
                findings.append(
                    _finding(
                        severity="error",
                        code="receipt_transition_revision_mismatch",
                        subject_kind="receipt",
                        subject_id=receipt.receipt_id,
                        related_ids=(transition.transition_id,),
                        message="receipt pointer revision differs from transition sequence",
                    )
                )
            if transition.next_promoted_model_id != receipt.resulting_promoted_model_id:
                findings.append(
                    _finding(
                        severity="error",
                        code="receipt_transition_target_mismatch",
                        subject_kind="receipt",
                        subject_id=receipt.receipt_id,
                        related_ids=(transition.transition_id,),
                        message="receipt result differs from lifecycle rollback target",
                    )
                )
        if receipt.disposition_id not in attempts_by_disposition:
            findings.append(
                _finding(
                    severity="warning",
                    code="legacy_unjournaled_receipt",
                    subject_kind="receipt",
                    subject_id=receipt.receipt_id,
                    related_ids=(receipt.disposition_id,),
                    message="receipt has no P17E attempt record and may predate execution journaling",
                )
            )

    for attempt in attempts:
        findings.extend(
            _audit_attempt(
                attempt,
                attempt_store.list_events(attempt.attempt_id),
                recommendations_by_id=recommendations_by_id,
                dispositions_by_id=dispositions_by_id,
                receipts_by_disposition=receipts_by_disposition,
            )
        )

    canonical_findings = tuple(
        sorted(
            findings,
            key=lambda item: (
                item.severity,
                item.code,
                item.subject_kind,
                item.subject_id,
                item.finding_id,
            ),
        )
    )
    if any(item.severity == "error" for item in canonical_findings):
        status = "invalid"
    elif any(item.severity == "warning" for item in canonical_findings):
        status = "attention_required"
    else:
        status = "valid"

    payload: Mapping[str, object] = {
        "active_pointer_id": pointer.pointer_id,
        "active_pointer_revision": pointer.revision,
        "attempt_count": len(attempts),
        "disposition_count": len(dispositions),
        "finding_count": len(canonical_findings),
        "finding_ids": tuple(item.finding_id for item in canonical_findings),
        "receipt_count": len(receipts),
        "recommendation_count": len(recommendations),
        "semantics": GOVERNANCE_AUDIT_SEMANTICS,
        "status": status,
        "version": GOVERNANCE_AUDIT_VERSION,
    }
    return GovernanceIntegrityAuditReportDTO(
        report_id=_digest(payload),
        status=status,
        active_pointer_id=pointer.pointer_id,
        active_pointer_revision=pointer.revision,
        recommendation_count=len(recommendations),
        disposition_count=len(dispositions),
        receipt_count=len(receipts),
        attempt_count=len(attempts),
        finding_count=len(canonical_findings),
        findings=canonical_findings,
    )
