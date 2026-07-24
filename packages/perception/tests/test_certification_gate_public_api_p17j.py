from zeromodel.perception.certification_gate import (
    CERTIFICATION_EXECUTION_GATE_SEMANTICS,
    CERTIFICATION_EXECUTION_GATE_VERSION,
    CertificationExecutionGateDTO,
    PerceptionCertificationExecutionGateError,
    authorize_certification_execution,
    execute_certification_gated_approved_rollback,
)


def test_p17j_certification_gate_public_contract() -> None:
    assert CERTIFICATION_EXECUTION_GATE_VERSION == (
        "perception-certification-execution-gate/1"
    )
    assert CERTIFICATION_EXECUTION_GATE_SEMANTICS
    assert CertificationExecutionGateDTO
    assert PerceptionCertificationExecutionGateError
    assert callable(authorize_certification_execution)
    assert callable(execute_certification_gated_approved_rollback)
