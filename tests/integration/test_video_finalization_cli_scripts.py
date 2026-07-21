from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from video_final_test_support import (
    SyntheticFinalExecutor,
    approved_protocol,
    authorization,
    request,
)
from zeromodel.db.runtime import build_finalization_sqlite_runtime
from zeromodel.db.session import sqlite_database_url
from zeromodel.domains.video_action_set.final_access_dto import (
    FinalExecutionReceiptDTO,
)
from zeromodel.domains.video_action_set.final_access_service import FinalAccessService


pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digests(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _database_url(path: Path) -> str:
    return sqlite_database_url(path)

def _completed(tmp_path: Path) -> tuple[object, FinalExecutionReceiptDTO]:
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
    return auth, receipt


def test_preflight_only_is_read_only_with_spaced_unicode_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "synthetic final authority \u03a9"
    root.mkdir()
    protocol = approved_protocol()
    auth = authorization(root, protocol)
    req = request(root, auth, preflight_only=True)
    before = _tree_digests(root)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.video_action_set_final_cli",
            "--output-dir",
            auth.output_dir,
            "--authorization-file",
            req.authorization_file,
            "--expected-authorization-digest",
            auth.authorization_digest,
            "--expected-sealed-plan-digest",
            auth.expected_sealed_plan_digest,
            "--database-path",
            auth.database_path,
            "--preflight-only",
            "--non-interactive",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["read_only"] is True
    assert _tree_digests(root) == before
    assert not Path(auth.database_path).exists()
    assert not Path(auth.output_dir).exists()


@pytest.mark.parametrize("command", ["observe", "reconstruct"])
def test_admin_cli_reads_without_mutating_authority_or_artifacts(
    tmp_path: Path,
    command: str,
) -> None:
    auth, receipt = _completed(tmp_path)
    database = Path(auth.database_path)  # type: ignore[attr-defined]
    output = Path(auth.output_dir)  # type: ignore[attr-defined]
    before_database = _sha256(database)
    before_output = _tree_digests(output)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.video_action_set_final_admin_cli",
            command,
            "--database-path",
            str(database),
            "--access-id",
            receipt.access_id,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert json.loads(result.stdout)["publication_status"] == (
        "completed_receipt_valid"
    )
    assert _sha256(database) == before_database
    assert _tree_digests(output) == before_output


@pytest.mark.parametrize("option", ["--force", "--resume", "--overwrite"])
def test_final_cli_has_no_force_resume_or_overwrite(
    tmp_path: Path,
    option: str,
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.video_action_set_final_cli",
            "--output-dir",
            str(tmp_path / "output"),
            "--authorization-file",
            str(tmp_path / "authorization.json"),
            "--expected-authorization-digest",
            "sha256:" + "a" * 64,
            "--expected-sealed-plan-digest",
            "sha256:" + "b" * 64,
            "--database-path",
            str(tmp_path / "final.sqlite3"),
            option,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr
    assert result.stdout == ""


def test_final_cli_requires_all_paths_explicitly() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "zeromodel.video_action_set_final_cli"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 2
    for option in ("--output-dir", "--authorization-file", "--database-path"):
        assert option in result.stderr


def test_hostile_access_id_is_data_and_cannot_execute(tmp_path: Path) -> None:
    database = tmp_path / "hostile authority.sqlite3"
    build_finalization_sqlite_runtime(
        _database_url(database),
        initialize_authority=True,
    )
    sentinel = tmp_path / "must-not-exist.txt"
    hostile = f"x; Set-Content -LiteralPath '{sentinel}' -Value owned"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "zeromodel.video_action_set_final_admin_cli",
            "observe",
            "--database-path",
            str(database),
            "--access-id",
            hostile,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    assert "final access id mismatch" in result.stderr
    assert not sentinel.exists()


@pytest.mark.parametrize(
    ("script_name", "expected_key"),
    [
        ("video-final-observe.ps1", "state"),
        ("video-final-reconstruct.ps1", "events"),
    ],
)
def test_powershell_admin_wrapper_propagates_json_and_exit_code(
    tmp_path: Path,
    script_name: str,
    expected_key: str,
) -> None:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("Windows PowerShell is unavailable")
    root = tmp_path / "PowerShell synthetic \u03a9"
    root.mkdir()
    auth, receipt = _completed(root)
    result = subprocess.run(
        [
            powershell,
            "-NoProfile",
            "-File",
            str(REPO_ROOT / "scripts" / script_name),
            "-DatabasePath",
            auth.database_path,  # type: ignore[attr-defined]
            "-AccessId",
            receipt.access_id,
            "-Python",
            sys.executable,
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    assert result.stderr == ""
    assert expected_key in json.loads(result.stdout)
