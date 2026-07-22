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


def test_release_test_layer_report_distinguishes_every_required_layer(
    tmp_path, monkeypatch
) -> None:
    # Exercise the report's structure/labelling without paying for a full
    # six-package build+install+probe cycle: monkeypatch the two expensive
    # inputs (WHEEL_SMOKE_RESULTS, and the pytest-invoking helper) and prove
    # the report still asks for and produces every required layer.
    monkeypatch.setattr(
        validator,
        "WHEEL_SMOKE_RESULTS",
        {"core": {"imported_from_installed_wheel": True}},
    )

    calls: list[list[str]] = []

    def fake_pytest_count(args, *, timeout: int = 180):
        calls.append(args)
        return {
            "passed": 1,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
            "returncode": 0,
            "summary_line": "1 passed",
        }

    monkeypatch.setattr(validator, "_pytest_count", fake_pytest_count)
    # Redirect REPO_ROOT so the report is written under tmp_path instead of
    # overwriting the real, data-driven docs/architecture report with this
    # test's fake counts.
    (tmp_path / "docs" / "architecture").mkdir(parents=True)
    monkeypatch.setattr(validator, "REPO_ROOT", tmp_path)

    report = validator.release_test_layer_report()

    written_path = (
        tmp_path / "docs" / "architecture" / "package-release-test-layers-1.0.13.json"
    )
    assert written_path.exists()

    assert set(report) == {
        "source_tree_fast_production_tests",
        "package_local_source_tests_by_package",
        "cross_package_integration_tests",
        "installed_wheel_smoke_result_by_package",
        "research",
    }
    assert set(report["package_local_source_tests_by_package"]) == set(
        validator.PACKAGES
    )
    assert report["research"]["status"] == "excluded_by_policy"
    assert report["installed_wheel_smoke_result_by_package"] == {
        "core": {"imported_from_installed_wheel": True}
    }
    # Every pytest invocation this function makes must exclude research from
    # anything claiming to be a production test count.
    for args in calls:
        if "-m" in args:
            expr = args[args.index("-m") + 1]
            if expr != "integration":
                assert "research" in expr
