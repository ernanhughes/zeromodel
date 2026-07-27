from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/validate_release_candidate.py")
SPEC = importlib.util.spec_from_file_location("validate_release_candidate_gates", SCRIPT)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_required_missing_gate_is_explicitly_missing(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    result = validator.run_pytest_gate(
        validator.ReleaseGate("missing", ("does-not-exist",))
    )
    assert result["status"] == "missing"
    assert result["missing_paths"] == ["does-not-exist"]


def test_optional_missing_gate_is_skipped_optional(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    result = validator.run_pytest_gate(
        validator.ReleaseGate("optional", ("does-not-exist",), required=False)
    )
    assert result["status"] == "skipped_optional"


def test_zero_test_gate_is_explicitly_zero_tests(tmp_path, monkeypatch) -> None:
    (tmp_path / "empty").mkdir()
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    result = validator.run_pytest_gate(validator.ReleaseGate("empty", ("empty",)))
    assert result["status"] == "zero_tests"
    assert result["collected"] == 0


def test_passing_gate_reports_passed(tmp_path, monkeypatch) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_ok.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8"
    )
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    result = validator.run_pytest_gate(validator.ReleaseGate("ok", ("tests",)))
    assert result["status"] == "passed"
    assert result["passed"] == 1


def test_failing_gate_reports_failed(tmp_path, monkeypatch) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_fail.py").write_text(
        "def test_fail():\n    assert False\n", encoding="utf-8"
    )
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    result = validator.run_pytest_gate(validator.ReleaseGate("fail", ("tests",)))
    assert result["status"] == "failed"
    assert result["failed"] == 1


def test_timed_out_gate_reports_timed_out(tmp_path, monkeypatch) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_sleep.py").write_text(
        "import time\n\ndef test_sleep():\n    time.sleep(3)\n", encoding="utf-8"
    )
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    result = validator.run_pytest_gate(
        validator.ReleaseGate("timeout", ("tests",), timeout_seconds=1)
    )
    assert result["status"] == "timed_out"
    assert result["returncode"] == 124


def test_malformed_collection_failure_reports_setup_failed(
    tmp_path, monkeypatch
) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_bad.py").write_text("def broken(:\n", encoding="utf-8")
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)
    result = validator.run_pytest_gate(validator.ReleaseGate("bad", ("tests",)))
    assert result["status"] == "setup_failed"
