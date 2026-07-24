from __future__ import annotations

from types import SimpleNamespace

from zeromodel.perception.certification_audit import audit_certification_integrity
from zeromodel.perception.governance_audit import GovernanceIntegrityAuditReportDTO
from zeromodel.perception.governance_gate import GovernanceExecutionGateDTO
from zeromodel.perception.sql_certification import (
    GovernanceExecutionCertificationBundleDTO,
    GovernanceExecutionCertificationDTO,
)


class _Lifecycle:
    def get_active_pointer(self):
        return SimpleNamespace(pointer_id="sha256:pointer", revision=0)

    def list_transitions(self):
        return ()


class _Governance:
    def __init__(self, *, receipts=()):
        self._receipts = receipts

    def list_recommendations(self):
        return ()

    def list_dispositions(self):
        return ()

    def list_execution_receipts(self):
        return self._receipts


class _Attempts:
    def list_attempts(self):
        return ()

    def list_events(self, attempt_id):
        del attempt_id
        return ()


class _Certifications:
    def __init__(self, bundles=()):
        self._bundles = bundles

    def list_certifications(self):
        return self._bundles


def _audit(report_id: str, revision: int) -> GovernanceIntegrityAuditReportDTO:
    return GovernanceIntegrityAuditReportDTO(
        report_id=report_id,
        status="valid",
        active_pointer_id="sha256:pointer",
        active_pointer_revision=revision,
        recommendation_count=0,
        disposition_count=0,
        receipt_count=0,
        attempt_count=0,
        finding_count=0,
        findings=(),
    )


def _orphan_bundle() -> GovernanceExecutionCertificationBundleDTO:
    gate = GovernanceExecutionGateDTO(
        gate_id="sha256:gate",
        recommendation_id="sha256:recommendation",
        disposition_id="sha256:disposition",
        attempt_id="sha256:attempt",
        receipt_id="sha256:receipt",
        pre_audit_report_id="sha256:pre",
        post_audit_report_id="sha256:post",
        authorization_mode="clean",
        resulting_pointer_revision=1,
    )
    certification = GovernanceExecutionCertificationDTO(
        certification_id="sha256:certification",
        gate_id=gate.gate_id,
        recommendation_id=gate.recommendation_id,
        disposition_id=gate.disposition_id,
        attempt_id=gate.attempt_id,
        receipt_id=gate.receipt_id,
        pre_audit_report_id=gate.pre_audit_report_id,
        post_audit_report_id=gate.post_audit_report_id,
        resulting_pointer_revision=1,
    )
    return GovernanceExecutionCertificationBundleDTO(
        certification=certification,
        gate=gate,
        pre_audit=_audit("sha256:pre", 0),
        post_audit=_audit("sha256:post", 1),
    )


def test_empty_four_store_history_is_valid_and_deterministic() -> None:
    first = audit_certification_integrity(
        _Lifecycle(), _Governance(), _Attempts(), _Certifications()
    )
    second = audit_certification_integrity(
        _Lifecycle(), _Governance(), _Attempts(), _Certifications()
    )

    assert first == second
    assert first.status == "valid"
    assert first.certification_count == 0
    assert first.findings == ()


def test_orphan_certification_is_invalid() -> None:
    report = audit_certification_integrity(
        _Lifecycle(),
        _Governance(),
        _Attempts(),
        _Certifications((_orphan_bundle(),)),
    )

    assert report.status == "invalid"
    assert tuple(item.code for item in report.findings) == (
        "certification_missing_attempt",
        "certification_missing_disposition",
        "certification_missing_receipt",
        "certification_missing_recommendation",
    )


def test_invalid_three_store_history_propagates_to_certification_audit() -> None:
    orphan_receipt = SimpleNamespace(
        receipt_id="sha256:receipt",
        disposition_id="sha256:missing-disposition",
        recommendation_id="sha256:missing-recommendation",
        transition_id="sha256:missing-transition",
        pointer_revision=1,
        resulting_promoted_model_id="promoted-earlier",
    )
    report = audit_certification_integrity(
        _Lifecycle(),
        _Governance(receipts=(orphan_receipt,)),
        _Attempts(),
        _Certifications(),
    )

    assert report.status == "invalid"
    assert report.governance_audit_status == "invalid"
    assert "underlying_governance_invalid" in tuple(
        item.code for item in report.findings
    )
