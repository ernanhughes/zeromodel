from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_fast_runner_blocks_opt_in_test_tiers() -> None:
    runner = (REPO_ROOT / "scripts" / "run_fast_tests.py").read_text(encoding="utf-8")

    assert 'FAST_SUITE_BUDGET_SECONDS = 120' in runner
    assert 'FORBIDDEN_INTEGRATION_FLAGS = {"--run-integration", "--run-slow"}' in runner
    assert "Run integration or slow tests explicitly with pytest instead." in runner


def test_runbook_documents_exact_test_tier_commands_and_agent_boundary() -> None:
    runbook = (REPO_ROOT / "docs" / "video-action-set-stage8-runbook.md").read_text(
        encoding="utf-8"
    )

    assert "python scripts/run_fast_tests.py" in runbook
    assert "python -m pytest -q --run-integration -m integration" in runbook
    assert "python -m pytest -q --run-slow -m slow" in runbook
    assert "Tests marked with both tiers require both opt-in flags." in runbook
    assert (
        "Coding agents must not execute integration tests, slow tests, "
        "scientific builds,"
    ) in runbook
    assert "create or\nupdate a script under `scripts/`" in runbook


def test_pytest_markers_keep_slow_distinct_from_integration() -> None:
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    conftest = (REPO_ROOT / "tests" / "conftest.py").read_text(encoding="utf-8")

    assert "slow: tests normally too expensive for the fast suite" in pyproject
    assert "deprecated compatibility alias" not in pyproject
    assert 'run_integration = bool(config.getoption("--run-integration"))' in conftest
    assert 'run_slow = bool(config.getoption("--run-slow"))' in conftest
    assert (
        "if (is_integration and not run_integration) or (is_slow and not run_slow):"
        in conftest
    )


def test_integration_workflow_supplies_both_opt_in_flags_for_combined_marker() -> None:
    workflow = (
        REPO_ROOT / ".github" / "workflows" / "integration.yml"
    ).read_text(encoding="utf-8")
    command = workflow.split("python -m pytest -q", 1)[1].split("--durations=50", 1)[0]

    assert '--run-integration' in command
    assert '--run-slow' in command
    assert '-m "integration or slow"' in command
