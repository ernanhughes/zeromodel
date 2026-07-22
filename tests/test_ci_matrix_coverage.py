from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_INTEGRATION_WORKFLOW = (
    REPO_ROOT / ".github" / "workflows" / "package-integration.yml"
)


def test_package_integration_runs_the_full_python_version_matrix() -> None:
    source = PACKAGE_INTEGRATION_WORKFLOW.read_text(encoding="utf-8")
    for version in ('"3.10"', '"3.11"', '"3.12"'):
        assert version in source


def test_package_integration_demonstrates_workspace_install_fast_suite_quality_and_release() -> None:
    source = PACKAGE_INTEGRATION_WORKFLOW.read_text(encoding="utf-8")
    assert "pip install -r requirements-dev.txt" in source
    assert "scripts/run_fast_tests.py" in source
    assert "scripts/check_quality.py" in source
    assert "scripts/validate_release_candidate.py" in source


def test_no_workflow_publishes_or_tags_a_release_during_stage_a2() -> None:
    workflows_dir = REPO_ROOT / ".github" / "workflows"
    for workflow in sorted(workflows_dir.glob("*.yml")):
        text = workflow.read_text(encoding="utf-8")
        assert "pypa/gh-action-pypi-publish" not in text
