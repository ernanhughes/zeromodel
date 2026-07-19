from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run tests marked as slow.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: long-running or currently pathological test excluded from the default suite",
    )


def pytest_collection_modifyitems(
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    if config.getoption("--run-slow"):
        return

    skip_slow = pytest.mark.skip(
        reason="slow test; rerun with --run-slow",
    )

    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)