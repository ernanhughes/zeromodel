from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Stage A2.1: these two tests were the dominant cost in the fast suite
# (15.75s and ~3s x4 respectively, out of a then-122.84s total run) once
# `--run-integration` started including tests/integration/ by default. They
# are correctly expensive/environment-dependent, not badly-written fast
# tests, so they were reclassified rather than optimized:
#   - building a real wheel + venv is never a "fast, bounded" operation;
#   - the PowerShell wrapper test requires a pwsh/PowerShell executable on
#     PATH, an environment-specific dependency matching the `external` tier.
# This test guards against either marker silently regressing back off,
# which would push the fast suite back toward its 122.84s pre-fix runtime.


def test_wheel_building_finalization_test_is_marked_slow() -> None:
    source = (
        REPO_ROOT
        / "tests"
        / "integration"
        / "test_video_finalization_package_boundary.py"
    ).read_text(encoding="utf-8")
    assert "@pytest.mark.slow" in source


def test_powershell_wrapper_test_is_marked_external() -> None:
    source = (
        REPO_ROOT / "tests" / "integration" / "test_video_finalization_cli_scripts.py"
    ).read_text(encoding="utf-8")
    assert "@pytest.mark.external" in source
