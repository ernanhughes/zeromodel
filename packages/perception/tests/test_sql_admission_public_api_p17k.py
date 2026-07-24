from zeromodel.perception.sql_admission import (
    SQL_ADMISSION_SCHEMA_VERSION,
    SQL_ADMISSION_STORE_VERSION,
    CertificationExecutionAdmissionBundleDTO,
    PerceptionSqlAdmissionError,
    SqliteCertificationExecutionAdmissionStore,
    execute_and_persist_certification_admission,
)


def test_p17k_public_module_contract() -> None:
    assert SQL_ADMISSION_SCHEMA_VERSION == "perception-sql-admission-schema/1"
    assert SQL_ADMISSION_STORE_VERSION == "perception-sql-admission-store/1"
    assert CertificationExecutionAdmissionBundleDTO is not None
    assert PerceptionSqlAdmissionError is not None
    assert SqliteCertificationExecutionAdmissionStore is not None
    assert callable(execute_and_persist_certification_admission)
