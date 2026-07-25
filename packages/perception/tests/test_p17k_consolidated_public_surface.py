from __future__ import annotations

import zeromodel.perception as perception


def test_p17k_contract_remains_exposed_from_package_root() -> None:
    assert perception.CERTIFICATION_EXECUTION_GATE_VERSION
    assert perception.CERTIFICATION_AUDIT_VERSION
    assert perception.GOVERNANCE_EXECUTION_GATE_VERSION
    assert perception.SQL_CERTIFICATION_SCHEMA_VERSION
    assert perception.SQL_ADMISSION_SCHEMA_VERSION
    assert callable(perception.audit_certification_integrity)
    assert callable(perception.execute_certification_gated_approved_rollback)
    assert callable(perception.execute_and_persist_certification_admission)
