from __future__ import annotations

from zeromodel.perception.governance_gate import (
    GOVERNANCE_EXECUTION_GATE_MODES,
    GOVERNANCE_EXECUTION_GATE_SEMANTICS,
    GOVERNANCE_EXECUTION_GATE_VERSION,
    GovernanceExecutionGateDTO,
    PerceptionGovernanceExecutionGateError,
    authorize_governance_execution,
    execute_audit_gated_approved_rollback,
)


def test_p17g_governance_gate_public_contract() -> None:
    assert GOVERNANCE_EXECUTION_GATE_VERSION.endswith("/1")
    assert "integrity_audit" in GOVERNANCE_EXECUTION_GATE_SEMANTICS
    assert GOVERNANCE_EXECUTION_GATE_MODES == {
        "clean",
        "recover_prepared_attempt",
    }
    assert GovernanceExecutionGateDTO.__name__ == "GovernanceExecutionGateDTO"
    assert issubclass(PerceptionGovernanceExecutionGateError, ValueError)
    assert callable(authorize_governance_execution)
    assert callable(execute_audit_gated_approved_rollback)
