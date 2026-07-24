from __future__ import annotations

from zeromodel.perception import (
    SQL_LIFECYCLE_SCHEMA_VERSION,
    SQL_LIFECYCLE_SEMANTICS,
    SQL_LIFECYCLE_STORE_VERSION,
    SqlitePerceptionModelLifecycleStore,
)


def test_sql_lifecycle_public_contract() -> None:
    assert SQL_LIFECYCLE_SCHEMA_VERSION == "perception-sql-lifecycle-schema/1"
    assert SQL_LIFECYCLE_STORE_VERSION == "perception-sql-lifecycle-store/1"
    assert SQL_LIFECYCLE_SEMANTICS == (
        "sqlite_append_only_model_ledger_with_atomic_transition_pointer_commit"
    )
    assert SqlitePerceptionModelLifecycleStore.version == SQL_LIFECYCLE_STORE_VERSION
