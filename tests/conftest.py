from __future__ import annotations

import pytest


INTEGRATION_MARKERS = {"integration", "slow"}
INTEGRATION_TEST_PREFIXES = ("test_video_action_set_",)
INTEGRATION_TEST_FILES = {
    "test_arcade_visual_local_baseline_showdown.py",
    "test_arcade_visual_registered_calibration_v2.py",
    "test_installed_wheel_video_instrument.py",
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


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("zeromodel test tiers")
    group.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run tests marked as integration or slow.",
    )
    group.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Deprecated compatibility alias for --run-integration.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "integration: end-to-end, exhaustive, materialization, or long-running test excluded from normal development",
    )
    config.addinivalue_line(
        "markers",
        "slow: deprecated compatibility alias for integration",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    run_integration = bool(
        config.getoption("--run-integration")
        or config.getoption("--run-slow")
    )

    for item in items:
        filename = item.path.name
        if (
            "integration" in item.path.parts
            or filename.startswith(INTEGRATION_TEST_PREFIXES)
            or filename in INTEGRATION_TEST_FILES
        ):
            item.add_marker(pytest.mark.integration)

    if run_integration:
        return

    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []

    for item in items:
        if any(marker in item.keywords for marker in INTEGRATION_MARKERS):
            deselected.append(item)
        else:
            selected.append(item)

    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
