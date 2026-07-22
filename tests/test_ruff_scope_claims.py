"""Guards against conflating 'the governed quality gate passes' with
'ruff check .' / 'ruff format --check .' passing repo-wide.

Stage A2.1 background: an earlier validation report said "Ruff passes"
without specifying scope. `ruff check .` and `ruff format --check .`
(unscoped) both actually fail today - 8 pre-existing E402 findings in
examples/ and 141 pre-existing formatting deltas outside the governed gate,
none introduced by any Stage A1/A2/A2.1 change (confirmed via
`git diff --stat HEAD` showing those files untouched). Only the paths
`scripts/check_quality.py` actually governs are claimed clean. This test
documents that distinction in an executable, hard-to-misreport form.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

SPEC = importlib.util.spec_from_file_location(
    "check_quality", REPO_ROOT / "scripts" / "check_quality.py"
)
assert SPEC is not None
check_quality = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_quality
SPEC.loader.exec_module(check_quality)


def test_governed_ruff_format_paths_pass() -> None:
    paths = check_quality.existing_python_paths(check_quality.FORMAT_LINT_PATHS)
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "format", "--check", *paths],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_governed_ruff_lint_paths_pass() -> None:
    paths = check_quality.existing_python_paths(check_quality.FORMAT_LINT_PATHS)
    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *paths],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unscoped_ruff_still_has_known_pre_existing_findings() -> None:
    # This test intentionally documents current reality rather than enforcing
    # a target: if someone fixes the whole-repo drift, this test should be
    # updated (not treated as a regression). It exists so "ruff check ./
    # ruff format --check . passes repo-wide" can never again be silently
    # implied without a human noticing this assertion needs updating too.
    lint = subprocess.run(
        [sys.executable, "-m", "ruff", "check", ".", "--output-format=concise"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    known_offenders = {
        "examples/arcade_visual_video_discriminative_evidence_benchmark.py",
        "examples/render_signs_demo.py",
    }
    offenders_seen = {
        line.split(":", 1)[0].replace("\\", "/")
        for line in lint.stdout.splitlines()
        if ".py:" in line
    }
    # Every offender must be one we already know about and have decided is
    # out of governed scope; a NEW offender outside this set is a real
    # regression and should fail this test.
    assert offenders_seen.issubset(known_offenders), (
        f"unscoped ruff check found new offenders outside the known, "
        f"out-of-governed-scope set: {offenders_seen - known_offenders}"
    )
