from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "check_quality.py"

SPEC = importlib.util.spec_from_file_location("check_quality", SCRIPT)
assert SPEC is not None
check_quality = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = check_quality
SPEC.loader.exec_module(check_quality)

SIX_PACKAGE_SRC_ROOTS = [
    "packages/core/src",
    "packages/analysis/src",
    "packages/observation/src",
    "packages/video/src",
    "packages/sqlalchemy/src",
]
# vision/src is listed as three explicit files rather than the directory
# (its entire src tree), so it is checked separately below.
VISION_SRC_FILES = [
    "packages/vision/src/zeromodel/vision/visual.py",
    "packages/vision/src/zeromodel/vision/visual_policy.py",
    "packages/vision/src/zeromodel/vision/__init__.py",
]


def _paths(constant_name: str) -> list[str]:
    return [path.as_posix() for path in getattr(check_quality, constant_name)]


def test_sqlalchemy_is_covered_by_ruff_format_and_lint() -> None:
    paths = _paths("FORMAT_LINT_PATHS")
    assert "packages/sqlalchemy/src" in paths
    assert "packages/sqlalchemy/tests" in paths


def test_sqlalchemy_is_covered_by_mypy() -> None:
    assert "packages/sqlalchemy/src" in _paths("TYPING_PATHS")


def test_sqlalchemy_is_covered_by_code_quality_limits() -> None:
    paths = _paths("QUALITY_LIMIT_PATHS")
    assert "packages/sqlalchemy/src" in paths
    assert "packages/sqlalchemy/tests" in paths


def test_sqlalchemy_src_is_in_root_mypy_path() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "packages/sqlalchemy/src" in pyproject


def test_all_six_packages_are_covered_by_ruff_format_and_lint() -> None:
    paths = _paths("FORMAT_LINT_PATHS")
    for root in SIX_PACKAGE_SRC_ROOTS:
        assert root in paths, f"{root} missing from FORMAT_LINT_PATHS"
    for path in VISION_SRC_FILES:
        assert path in paths, f"{path} missing from FORMAT_LINT_PATHS"


def test_all_six_packages_are_covered_by_mypy() -> None:
    paths = _paths("TYPING_PATHS")
    for root in SIX_PACKAGE_SRC_ROOTS:
        assert root in paths, f"{root} missing from TYPING_PATHS"
    for path in VISION_SRC_FILES:
        assert path in paths, f"{path} missing from TYPING_PATHS"


def test_integration_tests_are_covered_by_ruff() -> None:
    assert "integration_tests" in _paths("FORMAT_LINT_PATHS")


def test_quality_gate_stages_run_in_the_required_order() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    stage_labels = ["Ruff format check", "Ruff lint check", "mypy", "Package boundaries", "Architecture rules", "Code-quality limits"]
    positions = [source.index(f'"{label}"') for label in stage_labels]
    assert positions == sorted(positions), "quality-gate stages are out of the required order"


def test_a_failing_stage_stops_the_gate() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "raise SystemExit(result.returncode)" in source
