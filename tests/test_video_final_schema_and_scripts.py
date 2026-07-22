from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys

import pytest

from video_final_test_support import (
    SyntheticFinalExecutor,
    approved_protocol,
    authorization,
    request,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.persistence.sqlalchemy.db.runtime import (
    build_finalization_sqlite_runtime,
    build_sqlite_runtime,
)
from zeromodel.persistence.sqlalchemy.db.session import (
    FINALIZATION_AUTHORITY_ID,
    FINALIZATION_AUTHORITY_KIND,
    FINALIZATION_SCHEMA_VERSION,
    sqlite_database_url,
)
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FinalExecutionReceiptDTO,
    validate_final_identifier,
)
from zeromodel.video.domains.video_action_set.final_access_service import FinalAccessService


REPO_ROOT = Path(__file__).resolve().parents[1]


def _source_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        str(REPO_ROOT / path)
        for path in (
            "packages/core/src",
            "packages/observation/src",
            "packages/video/src",
            "packages/sqlalchemy/src",
        )
    )
    return env


def _database_url(path: Path) -> str:
    return sqlite_database_url(path)

def test_fresh_finalization_authority_is_reopenable(tmp_path: Path) -> None:
    path = tmp_path / "fresh-finalization.sqlite3"
    runtime = build_finalization_sqlite_runtime(
        _database_url(path),
        initialize_authority=True,
    )
    runtime.video_action_set.engine.final_access_service.store.assert_finalization_authority()
    reopened = build_finalization_sqlite_runtime(_database_url(path))
    reopened.video_action_set.engine.final_access_service.store.assert_finalization_authority()


def test_synthetic_completion_uses_dedicated_sqlite_authority(
    tmp_path: Path,
) -> None:
    protocol = approved_protocol()
    auth = authorization(tmp_path, protocol)
    runtime = build_finalization_sqlite_runtime(
        _database_url(Path(auth.database_path)),
        initialize_authority=True,
    )
    service = FinalAccessService(
        store=runtime.video_action_set.engine.final_access_service.store,
        final_executor=SyntheticFinalExecutor(),
    )
    receipt = FinalExecutionReceiptDTO.from_dict(
        service.execute_final_once(request(tmp_path, auth))
    )
    assert receipt.state == "completed"
    observed = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.persistence.sqlalchemy.video_action_set_final_admin_cli",
            "observe",
            "--database-path",
            auth.database_path,
            "--access-id",
            receipt.access_id,
        ],
        cwd=REPO_ROOT,
        env=_source_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert observed.returncode == 0, observed.stderr
    payload = json.loads(observed.stdout)
    assert payload["publication_status"] == "completed_receipt_valid"
    assert payload["publishable_success"] is True


def test_prefinal_stage8_database_is_not_adopted(tmp_path: Path) -> None:
    path = tmp_path / "stage8.sqlite3"
    build_sqlite_runtime(_database_url(path), initialize_schema=True)
    with pytest.raises(VPMValidationError, match="authority marker"):
        build_finalization_sqlite_runtime(_database_url(path))


def test_database_without_final_tables_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "historical.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE historical_evidence (id TEXT PRIMARY KEY)")
    with pytest.raises(VPMValidationError, match="schema mismatch"):
        build_finalization_sqlite_runtime(_database_url(path))


def test_final_tables_without_observation_owner_column_fail_closed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "missing-observation-owner.sqlite3"
    statements = (
        "CREATE TABLE video_action_set_finalization_schema "
        "(authority_id TEXT PRIMARY KEY, schema_version TEXT, authority_kind TEXT, created_utc TEXT)",
        "CREATE TABLE video_action_set_final_evaluation_protocol "
        "(protocol_digest TEXT, protocol_id TEXT, protocol_status TEXT, benchmark_seed_digest TEXT, sealed_plan_digest TEXT, payload_json TEXT)",
        "CREATE TABLE video_action_set_final_access_authorization "
        "(authorization_id TEXT, authorization_status TEXT, authorization_digest TEXT, protocol_digest TEXT, benchmark_seed_digest TEXT, sealed_plan_digest TEXT, created_utc TEXT, payload_json TEXT)",
        "CREATE TABLE video_action_set_final_access_record "
        "(access_id TEXT, authorization_id TEXT, state TEXT, current_event_ordinal INTEGER, last_event_digest TEXT, record_digest TEXT, payload_json TEXT)",
        "CREATE TABLE video_action_set_final_access_event "
        "(event_digest TEXT, access_id TEXT, ordinal INTEGER, previous_state TEXT, new_state TEXT, previous_event_digest TEXT, payload_json TEXT)",
        "CREATE TABLE video_action_set_observation (frame_id TEXT PRIMARY KEY)",
    )
    with sqlite3.connect(path) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.execute(
            "INSERT INTO video_action_set_finalization_schema VALUES (?, ?, ?, ?)",
            (
                FINALIZATION_AUTHORITY_ID,
                FINALIZATION_SCHEMA_VERSION,
                FINALIZATION_AUTHORITY_KIND,
                "2026-07-21T00:00:00Z",
            ),
        )
    with pytest.raises(VPMValidationError, match="schema mismatch"):
        build_finalization_sqlite_runtime(_database_url(path))


def test_partially_upgraded_and_wrong_version_databases_fail_closed(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "partial.sqlite3"
    with sqlite3.connect(partial) as connection:
        connection.execute(
            "CREATE TABLE video_action_set_finalization_schema "
            "(authority_id TEXT PRIMARY KEY, schema_version TEXT, authority_kind TEXT, created_utc TEXT)"
        )
    with pytest.raises(VPMValidationError, match="schema mismatch"):
        build_finalization_sqlite_runtime(_database_url(partial))

    wrong = tmp_path / "wrong-version.sqlite3"
    build_finalization_sqlite_runtime(
        _database_url(wrong),
        initialize_authority=True,
    )
    with sqlite3.connect(wrong) as connection:
        connection.execute(
            "UPDATE video_action_set_finalization_schema SET schema_version = ?",
            ("zeromodel-video-finalization-schema/v999",),
        )
    with pytest.raises(VPMValidationError, match="schema version"):
        build_finalization_sqlite_runtime(_database_url(wrong))


def test_admin_cli_accepts_unicode_and_spaced_database_paths(tmp_path: Path) -> None:
    path = tmp_path / "final authority \u03a9.sqlite3"
    build_finalization_sqlite_runtime(
        _database_url(path),
        initialize_authority=True,
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.persistence.sqlalchemy.video_action_set_final_admin_cli",
            "observe",
            "--database-path",
            str(path),
            "--access-id",
            "final-access:missing",
        ],
        cwd=REPO_ROOT,
        env=_source_env(),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "final access record is missing" in result.stderr
    assert "finalization database schema" not in result.stderr


def test_operator_scripts_pass_hostile_access_ids_only_as_data(
    tmp_path: Path,
) -> None:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    path = tmp_path / "final authority.sqlite3"
    build_finalization_sqlite_runtime(
        _database_url(path),
        initialize_authority=True,
    )
    sentinel = tmp_path / "must-not-exist.txt"
    python_payload = (
        "''');__import__('pathlib').Path("
        + repr(str(sentinel))
        + ").write_text('owned');#"
    )
    hostile_values = (
        "'''",
        '"',
        "`",
        "$()",
        ";&|",
        "line1\nline2",
        "unicode-\u03a9",
        "contains spaces",
        python_payload,
    )
    for hostile in hostile_values:
        with pytest.raises(VPMValidationError, match="final access id"):
            validate_final_identifier(hostile, "final access id mismatch")
    for script_name in ("video-final-observe.ps1", "video-final-reconstruct.ps1"):
        script = REPO_ROOT / "scripts" / script_name
        text = script.read_text(encoding="utf-8")
        assert " -c " not in text
        assert '$script = @"' not in text
        result = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-File",
                str(script),
                "-DatabasePath",
                str(path),
                "-AccessId",
                python_payload,
                "-Python",
                sys.executable,
            ],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert not sentinel.exists()
