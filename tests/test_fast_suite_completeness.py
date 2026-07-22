from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "run_fast_tests.py"

SPEC = importlib.util.spec_from_file_location("run_fast_tests", SCRIPT)
assert SPEC is not None
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


NINE_PACKAGE_TEST_ROOTS = [
    "packages/core/tests",
    "packages/analysis/tests",
    "packages/observation/tests",
    "packages/vision/tests",
    "packages/video/tests",
    "packages/sqlalchemy/tests",
    "packages/artifacts/tests",
    "packages/trust/tests",
    "packages/navigation/tests",
]


def test_every_nine_package_test_root_is_passed_to_pytest() -> None:
    for root in NINE_PACKAGE_TEST_ROOTS:
        assert root in runner.TEST_ROOTS, f"{root} is missing from the canonical fast suite"


def test_repository_wide_roots_are_also_included() -> None:
    assert "tests" in runner.TEST_ROOTS
    assert "integration_tests" in runner.TEST_ROOTS


def test_package_local_suite_cannot_silently_shrink() -> None:
    # A reviewer diffing this exact list is the guard against a package
    # quietly disappearing from the fast suite.
    assert runner.TEST_ROOTS == [
        "tests",
        "integration_tests",
        "packages/core/tests",
        "packages/analysis/tests",
        "packages/observation/tests",
        "packages/vision/tests",
        "packages/video/tests",
        "packages/sqlalchemy/tests",
        "packages/artifacts/tests",
        "packages/trust/tests",
        "packages/navigation/tests",
    ]


def test_research_is_excluded_by_marker_expression() -> None:
    assert "not research" in runner.MARKER_EXPRESSION


def test_slow_and_external_are_excluded_by_marker_expression() -> None:
    assert "not slow" in runner.MARKER_EXPRESSION
    assert "not external" in runner.MARKER_EXPRESSION


def test_integration_is_not_blanket_excluded_by_marker_expression() -> None:
    # Directory location alone must not define whether an integration test
    # runs in the fast suite - only slow/external/research do.
    assert "integration" not in runner.MARKER_EXPRESSION


def test_collection_error_fails_the_command() -> None:
    exit_code, message = runner.evaluate_summary(
        {"collected": 10, "collection_errors": 2, "deselected": 0}, 0
    )
    assert exit_code == 1
    assert "collection error" in message.lower()


def test_zero_collected_production_tests_fails_the_command() -> None:
    exit_code, message = runner.evaluate_summary(
        {"collected": 0, "collection_errors": 0, "deselected": 0}, 0
    )
    assert exit_code == 1
    assert "zero" in message.lower()


def test_missing_summary_report_is_treated_as_a_failure() -> None:
    exit_code, message = runner.evaluate_summary(None, 0)
    assert exit_code == 1
    assert message


def test_clean_run_propagates_pytests_return_code() -> None:
    exit_code, message = runner.evaluate_summary(
        {"collected": 5, "collection_errors": 0, "deselected": 1}, 0
    )
    assert exit_code == 0
    assert message == ""

    exit_code, message = runner.evaluate_summary(
        {"collected": 5, "collection_errors": 0, "deselected": 1}, 1
    )
    assert exit_code == 1
