from zeromodel import perception


def test_public_api_exposes_p15_sql_production_contract() -> None:
    assert perception.PERCEPTION_STAGE == "P15"
    assert perception.SQL_PRODUCTION_SCHEMA_VERSION == "perception-sql-production-schema/1"
    assert perception.SQL_PRODUCTION_STORE_VERSION == "perception-sql-production-store/1"
    assert perception.SQL_PRODUCTION_SEMANTICS == (
        "durable_append_only_production_inference_and_outcome_ledger"
    )
    assert perception.SqlitePerceptionProductionLedgerStore is not None
    assert perception.PerceptionSqlProductionError is not None
