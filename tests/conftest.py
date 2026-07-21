from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import pytest

import zeromodel.video_action_set_benchmark as benchmark


INTEGRATION_TEST_PREFIXES = ("test_video_action_set_",)
INTEGRATION_TEST_FILES = {
    "test_arcade_visual_local_baseline_showdown.py",
    "test_arcade_visual_registered_calibration_v2.py",
    "test_installed_wheel_video_instrument.py",
    "test_video_episode_plan_sql_store.py",
    "test_video_observation_sql_store.py",
    "test_video_discriminative_evidence.py",
    "test_video_discriminative_measurement_audit.py",
    "test_video_discriminative_representation_audit.py",
    "test_video_discriminative_v2_benchmark.py",
    "test_video_discriminative_v2_integrity.py",
    "test_video_discriminative_v2_selection.py",
    "test_video_local_correlation.py",
    "test_video_prospective_providers.py",
    "test_video_prospective_runtime_equivalence.py",
    "test_visual_local_baseline_result_records.py",
    "test_visual_local_baselines.py",
    "test_visual_registered_calibration_v2.py",
    "test_visual_result_records.py",
}
_STAGE6_MATERIALIZATION_TEST_MODULES = {
    "test_video_episode_materialization",
    "test_video_materialization_kernels",
    "test_video_materialization_reachability",
    "test_video_materialization_validation",
    "test_video_provider_observation_boundary",
}
_STAGE6_PLAN_CACHE: dict[tuple[Any, ...], list[dict[str, Any]]] = {}


def _stage6_plan_cache_key(
    identity: Any,
    split: str,
    row_ids: Sequence[str],
    row_actions: Mapping[str, str],
) -> tuple[Any, ...]:
    return (
        identity.seed_digest,
        split,
        tuple(row_ids),
        tuple((row_id, row_actions[row_id]) for row_id in row_ids),
    )


@pytest.fixture(scope="module", autouse=True)
def cache_stage6_materialization_plans(request: pytest.FixtureRequest):
    """Reuse deterministic plans only in the Stage 6 materialization tests."""
    module_name = request.module.__name__.rsplit(".", 1)[-1]
    if module_name not in _STAGE6_MATERIALIZATION_TEST_MODULES:
        yield
        return

    original = benchmark._episode_plans_for_split

    def cached_episode_plans(
        identity: Any,
        split: str,
        row_ids: Sequence[str],
        row_actions: Mapping[str, str],
    ) -> list[dict[str, Any]]:
        key = _stage6_plan_cache_key(identity, split, row_ids, row_actions)
        if key not in _STAGE6_PLAN_CACHE:
            _STAGE6_PLAN_CACHE[key] = original(identity, split, row_ids, row_actions)
        return deepcopy(_STAGE6_PLAN_CACHE[key])

    benchmark._episode_plans_for_split = cached_episode_plans
    try:
        yield
    finally:
        benchmark._episode_plans_for_split = original


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("zeromodel test tiers")
    group.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration.",
    )
    group.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: end-to-end, persistence, materialization, or cross-component tests excluded from normal development",
    )
    config.addinivalue_line(
        "markers",
        "slow: tests normally too expensive for the fast suite, including exhaustive sweeps, profiling, full builds, and mutation audits",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    run_integration = bool(config.getoption("--run-integration"))
    run_slow = bool(config.getoption("--run-slow"))

    for item in items:
        filename = item.path.name
        if (
            "integration" in item.path.parts
            or filename in INTEGRATION_TEST_FILES
            or filename.startswith(INTEGRATION_TEST_PREFIXES)
        ):
            item.add_marker(pytest.mark.integration)

    if run_integration and run_slow:
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        is_integration = "integration" in item.keywords
        is_slow = "slow" in item.keywords
        if (is_integration and not run_integration) or (is_slow and not run_slow):
            deselected.append(item)
        else:
            selected.append(item)

    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
