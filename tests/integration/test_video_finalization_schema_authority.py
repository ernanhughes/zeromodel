from __future__ import annotations

import hashlib
from pathlib import Path
import sqlite3

import pytest
from sqlalchemy import inspect, text

from video_final_test_support import approved_protocol, authorization
from zeromodel.artifact import VPMValidationError
from zeromodel.db.runtime import (
    build_finalization_sqlite_runtime,
    build_sqlite_runtime,
)
from zeromodel.db.session import (
    FINALIZATION_AUTHORITY_ID,
    FINALIZATION_AUTHORITY_KIND,
    FINALIZATION_SCHEMA_VERSION,
)


pytestmark = pytest.mark.integration


def _database_url(path: Path) -> str:
    return path.resolve().as_uri().replace("file:///", "sqlite:///")


@pytest.mark.parametrize("precreate_zero_byte", [False, True])
def test_fresh_and_zero_byte_authorities_initialize_reopen_and_enable_fks(
    tmp_path: Path,
    precreate_zero_byte: bool,
) -> None:
    path = tmp_path / f"authority-{precreate_zero_byte}.sqlite3"
    if precreate_zero_byte:
        path.touch()
    runtime = build_finalization_sqlite_runtime(
        _database_url(path),
        initialize_authority=True,
    )
    store = runtime.video_action_set.engine.final_access_service.store
    store.assert_finalization_authority()
    engine = store._finalization_engine  # type: ignore[attr-defined]
    assert engine is not None
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
    assert {
        "video_action_set_finalization_schema",
        "video_action_set_final_evaluation_protocol",
        "video_action_set_final_access_authorization",
        "video_action_set_final_access_record",
        "video_action_set_final_access_event",
    }.issubset(inspect(engine).get_table_names())

    reopened = build_finalization_sqlite_runtime(_database_url(path))
    reopened.video_action_set.engine.final_access_service.store.assert_finalization_authority()


def test_historical_and_finalization_authorities_remain_distinct(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    contract = auth.authorization_payload.to_value()
    assert isinstance(contract, dict)
    historical = contract["historical_authority"]
    assert isinstance(historical, dict)
    historical_path = Path(str(historical["historical_database_path"]))
    historical_before = historical_path.read_bytes()
    historical_digest = hashlib.sha256(historical_before).hexdigest()

    final_path = Path(auth.database_path)
    build_finalization_sqlite_runtime(
        _database_url(final_path),
        initialize_authority=True,
    )

    assert final_path.resolve() != historical_path.resolve()
    assert historical_path.read_bytes() == historical_before
    assert hashlib.sha256(historical_path.read_bytes()).hexdigest() == historical_digest


def test_unrelated_and_stage8_shaped_databases_are_rejected(tmp_path: Path) -> None:
    unrelated = tmp_path / "unrelated.sqlite3"
    with sqlite3.connect(unrelated) as connection:
        connection.execute("CREATE TABLE unrelated (id TEXT PRIMARY KEY)")
    with pytest.raises(VPMValidationError, match="schema mismatch"):
        build_finalization_sqlite_runtime(_database_url(unrelated))

    stage8_shaped = tmp_path / "stage8-shaped.sqlite3"
    build_sqlite_runtime(_database_url(stage8_shaped), initialize_schema=True)
    with pytest.raises(VPMValidationError, match="authority marker"):
        build_finalization_sqlite_runtime(_database_url(stage8_shaped))


@pytest.mark.parametrize(
    "mutation",
    [
        "partial",
        "wrong_marker",
        "wrong_version",
        "missing_table",
        "missing_column",
    ],
)
def test_malformed_finalization_authorities_fail_closed(
    tmp_path: Path,
    mutation: str,
) -> None:
    path = tmp_path / f"{mutation}.sqlite3"
    if mutation == "partial":
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE video_action_set_finalization_schema "
                "(authority_id TEXT PRIMARY KEY, schema_version TEXT, "
                "authority_kind TEXT, created_utc TEXT)"
            )
    else:
        build_finalization_sqlite_runtime(
            _database_url(path),
            initialize_authority=True,
        )
        with sqlite3.connect(path) as connection:
            if mutation == "wrong_marker":
                connection.execute(
                    "UPDATE video_action_set_finalization_schema "
                    "SET authority_kind = ? WHERE authority_id = ?",
                    ("wrong-authority", FINALIZATION_AUTHORITY_ID),
                )
            elif mutation == "wrong_version":
                connection.execute(
                    "UPDATE video_action_set_finalization_schema "
                    "SET schema_version = ? WHERE authority_id = ?",
                    (FINALIZATION_SCHEMA_VERSION + "-wrong", FINALIZATION_AUTHORITY_ID),
                )
            elif mutation == "missing_table":
                connection.execute("PRAGMA foreign_keys=OFF")
                connection.execute("DROP TABLE video_action_set_final_access_event")
            else:
                connection.execute("PRAGMA foreign_keys=OFF")
                connection.execute(
                    "ALTER TABLE video_action_set_final_access_record "
                    "RENAME TO video_action_set_final_access_record_complete"
                )
                connection.execute(
                    "CREATE TABLE video_action_set_final_access_record "
                    "(access_id TEXT PRIMARY KEY)"
                )

    with pytest.raises(VPMValidationError, match="schema|authority"):
        build_finalization_sqlite_runtime(_database_url(path))


def test_expected_authority_marker_constants_are_stable() -> None:
    assert FINALIZATION_AUTHORITY_ID == "local-finalization-authority"
    assert FINALIZATION_AUTHORITY_KIND == "separate-finalization-authority"
    assert FINALIZATION_SCHEMA_VERSION == "zeromodel-video-finalization-schema/v1"
