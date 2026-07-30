from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "validate_release_candidate.py"

SPEC = importlib.util.spec_from_file_location("validate_release_candidate", SCRIPT)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_pytest_count_parses_a_passed_summary_line() -> None:
    counts = validator._pytest_count(["packages/core/tests"])
    assert counts["passed"] > 0
    assert counts["failed"] == 0
    assert counts["returncode"] == 0
