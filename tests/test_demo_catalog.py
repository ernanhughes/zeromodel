from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_demos", ROOT / "scripts" / "build_demos.py"
)
assert SPEC is not None and SPEC.loader is not None
build_demos = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = build_demos
SPEC.loader.exec_module(build_demos)


def test_demo_catalogue_is_valid() -> None:
    demos = build_demos.load_catalog()
    build_demos.validate(demos)


def test_foundation_catalogue_contains_three_fast_demos() -> None:
    demos = build_demos.load_catalog()
    assert [demo.id for demo in demos] == [
        "vpm-artifact",
        "visual-sign-reader",
        "lua-edge-policy",
    ]
    assert all(demo.execution_profile == "fast" for demo in demos)
    assert all(not demo.network for demo in demos)
