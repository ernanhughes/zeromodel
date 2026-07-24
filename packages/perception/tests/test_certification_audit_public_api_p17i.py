from zeromodel.perception.certification_audit import (
    CERTIFICATION_AUDIT_FINDING_VERSION,
    CERTIFICATION_AUDIT_SEMANTICS,
    CERTIFICATION_AUDIT_SEVERITIES,
    CERTIFICATION_AUDIT_STATUSES,
    CERTIFICATION_AUDIT_VERSION,
    CertificationIntegrityAuditReportDTO,
    CertificationIntegrityFindingDTO,
    PerceptionCertificationAuditError,
    audit_certification_integrity,
)


def test_p17i_certification_audit_public_contract() -> None:
    assert CERTIFICATION_AUDIT_VERSION == "perception-certification-integrity-audit/1"
    assert CERTIFICATION_AUDIT_FINDING_VERSION.endswith("/1")
    assert CERTIFICATION_AUDIT_SEMANTICS
    assert CERTIFICATION_AUDIT_STATUSES == {
        "valid",
        "attention_required",
        "invalid",
    }
    assert CERTIFICATION_AUDIT_SEVERITIES == {"info", "warning", "error"}
    assert CertificationIntegrityAuditReportDTO
    assert CertificationIntegrityFindingDTO
    assert PerceptionCertificationAuditError
    assert callable(audit_certification_integrity)
