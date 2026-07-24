from __future__ import annotations

from zeromodel.perception.sql_certification import (
    GOVERNANCE_CERTIFICATION_SEMANTICS,
    GOVERNANCE_CERTIFICATION_VERSION,
    SQL_CERTIFICATION_SCHEMA_VERSION,
    SQL_CERTIFICATION_STORE_VERSION,
    GovernanceExecutionCertificationBundleDTO,
    GovernanceExecutionCertificationDTO,
    PerceptionSqlCertificationError,
    SqliteGovernanceCertificationStore,
    build_governance_execution_certification,
    execute_and_certify_audit_gated_rollback,
)


def test_p17h_certification_module_is_public() -> None:
    assert GOVERNANCE_CERTIFICATION_VERSION.endswith("/1")
    assert SQL_CERTIFICATION_SCHEMA_VERSION.endswith("/1")
    assert SQL_CERTIFICATION_STORE_VERSION.endswith("/1")
    assert "durable_bundle" in GOVERNANCE_CERTIFICATION_SEMANTICS
    assert GovernanceExecutionCertificationDTO is not None
    assert GovernanceExecutionCertificationBundleDTO is not None
    assert PerceptionSqlCertificationError is not None
    assert SqliteGovernanceCertificationStore is not None
    assert callable(build_governance_execution_certification)
    assert callable(execute_and_certify_audit_gated_rollback)
