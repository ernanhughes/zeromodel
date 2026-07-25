from __future__ import annotations

import zeromodel.perception as perception


def test_p17f_governance_audit_is_public() -> None:
    expected = {
        "GOVERNANCE_AUDIT_FINDING_VERSION",
        "GOVERNANCE_AUDIT_SEMANTICS",
        "GOVERNANCE_AUDIT_SEVERITIES",
        "GOVERNANCE_AUDIT_STATUSES",
        "GOVERNANCE_AUDIT_VERSION",
        "GovernanceIntegrityAuditReportDTO",
        "GovernanceIntegrityFindingDTO",
        "PerceptionGovernanceAuditError",
        "audit_governance_integrity",
    }

    assert expected <= set(perception.__all__)
    assert hasattr(perception.SqliteGovernedExecutionAttemptStore, "list_attempts")
