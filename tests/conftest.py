from __future__ import annotations

import pytest


INTEGRATION_MARKERS = {"integration", "slow"}


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
        if "integration" in item.path.parts:
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
