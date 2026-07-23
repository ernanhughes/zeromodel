"""Regression tests for the conftest.py research-import isolation fix:
`research.benchmarks.video_action_set_benchmark` must never be imported by
production fast-suite collection unless the collected module actually
belongs to the small Stage 6 materialization set that needs it.

Each check runs pytest collection in a fresh subprocess so results are
never contaminated by another test in the *same* pytest session having
already triggered the Stage 6 fixture's lazy import.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_MARKER = "RESEARCH_BENCHMARK_MODULE_IMPORTED"

_PROBE_SCRIPT = """
import sys
import pytest
pytest.main(["--collect-only", "-q", *{args!r}])
print({marker!r}, "research.benchmarks.video_action_set_benchmark" in sys.modules)
"""


def _collect_and_check_research_import(*args: str) -> bool:
    script = _PROBE_SCRIPT.format(args=list(args), marker=_MARKER)
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for line in result.stdout.splitlines():
        if line.startswith(_MARKER):
            return line.split()[-1] == "True"
    raise AssertionError(f"marker line not found in output:\n{result.stdout}")


def test_unrelated_production_test_collection_does_not_import_research_benchmark() -> (
    None
):
    """Collecting a plain production test module (no Stage 6 involvement)
    must not import the research benchmark module at all - this is the
    exact defect the fix closes: unconditional module-level import in
    conftest.py could fail production fast-suite *collection* even for
    runs that never touch Stage 6."""
    imported = _collect_and_check_research_import(
        "tests/test_release_candidate_validation.py"
    )
    assert imported is False


def test_stage6_materialization_test_collection_can_still_import_research_benchmark() -> (
    None
):
    """Sanity check for the other direction: collecting a real Stage 6
    materialization test module still reaches the lazy import inside the
    fixture (proving the fix didn't simply delete the capability)."""
    imported = _collect_and_check_research_import(
        "tests/test_video_episode_materialization.py"
    )
    assert imported is True


def test_research_marked_tests_remain_excluded_from_the_fast_suite() -> None:
    """The `research` marker's default-deselect behavior (registered via
    conftest.py's pytest_collection_modifyitems) is unrelated to *how* the
    Stage 6 fixture imports its dependency and must remain unchanged: a
    file matching RESEARCH_TEST_PREFIXES still collects zero selected
    items under the production fast-suite marker expression."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-m",
            "not slow and not external and not research",
            "tests/test_video_action_set_benchmark.py",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 5, result.stdout + result.stderr
    assert "no tests collected" in result.stdout.lower()
    assert "deselected" in result.stdout
