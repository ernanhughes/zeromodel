from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

FORBIDDEN_ROOT_INSTALL_PATTERNS = [
    re.compile(r"pip install[^\n]*\s\.\[dev\]"),
    re.compile(r"pip install[^\n]*-e\s+\.\[dev\]"),
    re.compile(r"pip install[^\n]*-e\s+['\"]?\.\[dev\]"),
    re.compile(r"pip install[^\n]*-e\s+['\"]?\.\[release\]"),
    re.compile(r"pip install[^\n]*-e\s+['\"]?\.\[vision\]"),
    re.compile(r"^\s*python -m build\s*$", re.MULTILINE),
]


def _all_workflow_files() -> list[Path]:
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


def test_requirements_dev_installs_all_six_packages_editable() -> None:
    text = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    for package in ("core", "analysis", "observation", "vision", "video", "sqlalchemy"):
        assert f"-e ./packages/{package}" in text, (
            f"missing editable install for {package}"
        )


def test_requirements_dev_supplies_tomli_fallback_for_python_3_10() -> None:
    text = (REPO_ROOT / "requirements-dev.txt").read_text(encoding="utf-8")
    assert "tomli" in text
    assert 'python_version < "3.11"' in text


def test_no_active_workflow_invokes_root_editable_or_root_build() -> None:
    offenders: list[str] = []
    for workflow in _all_workflow_files():
        text = workflow.read_text(encoding="utf-8")
        for pattern in FORBIDDEN_ROOT_INSTALL_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{workflow.name}: {pattern.pattern}")
    assert offenders == []


def test_root_pyproject_has_no_project_table() -> None:
    # The repository root is deliberately not a Python distribution. This
    # guards against silently reintroducing a root [project] table, which
    # would change the meaning of "no active workflow builds the root".
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "[project]" not in text
    assert "[build-system]" not in text


def test_package_integration_workflow_installs_from_requirements_dev() -> None:
    text = (WORKFLOWS_DIR / "package-integration.yml").read_text(encoding="utf-8")
    assert "pip install -r requirements-dev.txt" in text


def test_python_workflow_trigger_paths_cover_packages_and_requirements_dev() -> None:
    text = (WORKFLOWS_DIR / "python.yml").read_text(encoding="utf-8")
    for required in ("packages/**", "integration_tests/**", "requirements-dev.txt"):
        assert required in text, f"python.yml trigger paths missing {required}"
    assert "zeromodel/**" not in text


def test_package_integration_workflow_trigger_paths_cover_tests_and_requirements_dev() -> (
    None
):
    text = (WORKFLOWS_DIR / "package-integration.yml").read_text(encoding="utf-8")
    for required in ("tests/**", "requirements-dev.txt", "pyproject.toml"):
        assert required in text, (
            f"package-integration.yml trigger paths missing {required}"
        )


def test_check_package_boundaries_has_a_python_3_10_toml_fallback() -> None:
    text = (REPO_ROOT / "scripts" / "check_package_boundaries.py").read_text(
        encoding="utf-8"
    )
    assert "import tomllib" in text
    assert "import tomli as tomllib" in text


def test_validate_release_candidate_has_a_python_3_10_toml_fallback() -> None:
    text = (REPO_ROOT / "scripts" / "validate_release_candidate.py").read_text(
        encoding="utf-8"
    )
    assert "import tomllib" in text
    assert "import tomli as tomllib" in text


def test_publish_testpypi_workflow_does_not_claim_to_publish() -> None:
    text = (WORKFLOWS_DIR / "publish-testpypi.yml").read_text(encoding="utf-8")
    assert "pypa/gh-action-pypi-publish" not in text
    assert "id-token: write" not in text
