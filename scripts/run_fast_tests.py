from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


FAST_SUITE_BUDGET_SECONDS = 120
FORBIDDEN_INTEGRATION_FLAGS = {"--run-integration", "--run-slow"}

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent

# Bounded, deterministic production behavior only. `slow`, `external`, and
# `research` tests are excluded by marker, not by omitting directories, so
# that a bounded integration test is not silently dropped just because it
# lives next to expensive ones.
MARKER_EXPRESSION = "not slow and not external and not research"

# The canonical production fast suite: the two repository-wide test roots
# plus every one of the six package-local suites. A package cannot silently
# disappear from this list without a reviewer noticing the diff.
TEST_ROOTS = [
    "tests",
    "integration_tests",
    "packages/core/tests",
    "packages/analysis/tests",
    "packages/observation/tests",
    "packages/vision/tests",
    "packages/video/tests",
    "packages/sqlalchemy/tests",
]

REPORT_PATH = REPO_ROOT / "build" / "reports" / "fast-test-summary.json"


def _read_summary() -> dict | None:
    if not REPORT_PATH.exists():
        return None
    try:
        return json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def evaluate_summary(summary: dict | None, subprocess_returncode: int) -> tuple[int, str]:
    """Decide the fast suite's final exit code from its structured summary.

    Kept separate from subprocess/IO handling so this decision (fail on
    collection errors, fail on zero collected tests, otherwise propagate
    pytest's own return code) is directly unit-testable.
    """
    if summary is None:
        return (subprocess_returncode or 1, "No fast-suite summary report was produced - treating as a failure.")
    if summary.get("collection_errors"):
        return (1, "Collection errors were detected during collection - failing.")
    if summary.get("collected", 0) == 0:
        return (1, "Zero production tests were collected - failing.")
    return (subprocess_returncode, "")


def main() -> int:
    forbidden = [
        argument for argument in sys.argv[1:] if argument in FORBIDDEN_INTEGRATION_FLAGS
    ]
    if forbidden:
        print(
            "The fast-test runner does not permit integration opt-in flags: "
            + ", ".join(forbidden),
            file=sys.stderr,
        )
        print(
            "Run integration or slow tests explicitly with pytest instead.",
            file=sys.stderr,
        )
        return 2

    print("Fast-suite test roots:")
    for root in TEST_ROOTS:
        print(f"  - {root}")
    print(f"Fast-suite marker expression: {MARKER_EXPRESSION}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if REPORT_PATH.exists():
        REPORT_PATH.unlink()

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(SCRIPTS_DIR), env.get("PYTHONPATH", "")) if part
    )
    env["ZEROMODEL_FAST_SUITE_REPORT_PATH"] = str(REPORT_PATH)
    env["ZEROMODEL_FAST_SUITE_TEST_ROOTS"] = os.pathsep.join(TEST_ROOTS)
    env["ZEROMODEL_FAST_SUITE_MARKER_EXPRESSION"] = MARKER_EXPRESSION

    # --run-integration defeats this repository's legacy opt-in-only
    # deselection of every `integration`-marked item (see tests/conftest.py);
    # the marker expression above is the actual, authoritative filter now.
    # --run-slow is deliberately NOT passed, so slow-marked items stay
    # excluded by that legacy mechanism too (redundant with `-m`, not
    # conflicting with it).
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--maxfail=1",
        "-p",
        "fast_suite_reporter",
        "--run-integration",
        "-m",
        MARKER_EXPRESSION,
        *TEST_ROOTS,
        *sys.argv[1:],
    ]
    started = time.monotonic()

    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=FAST_SUITE_BUDGET_SECONDS,
            cwd=REPO_ROOT,
            env=env,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        print(
            f"\nFAST TEST BUDGET EXCEEDED: {elapsed:.1f}s > "
            f"{FAST_SUITE_BUDGET_SECONDS}s",
            file=sys.stderr,
        )
        print(
            "Move expensive tests to the slow/external/research tier or reduce their fixture scope.",
            file=sys.stderr,
        )
        return 124

    elapsed = time.monotonic() - started
    print(f"\nFast-suite runtime: {elapsed:.2f}s (budget: {FAST_SUITE_BUDGET_SECONDS}s)")

    summary = _read_summary()
    if summary is not None:
        print(
            "Collected: {collected}, deselected: {deselected}, "
            "passed: {passed}, failed: {failed}, skipped: {skipped}".format(**summary)
        )

    exit_code, message = evaluate_summary(summary, completed.returncode)
    if message:
        print(message, file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
