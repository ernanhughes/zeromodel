from __future__ import annotations

import pytest

from examples.hand_anatomy import pose_library as pl


def pytest_collection_modifyitems(items):
    for item in items:
        if "hand_anatomy_02" in item.path.parts:
            item.add_marker(pytest.mark.research)


@pytest.fixture(scope="session")
def small_lib():
    return pl.build_library(gestures=["open", "spread", "fist"], yaws=(0.0, 40.0), shapes=["standard"])
