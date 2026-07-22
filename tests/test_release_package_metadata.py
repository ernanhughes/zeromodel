from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/validate_release_candidate.py")
SPEC = importlib.util.spec_from_file_location("validate_release_candidate", SCRIPT)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_package_versions_and_dependency_graph_are_synchronized() -> None:
    validator.validate_versions()


def test_release_manifest_lists_all_nine_distributions() -> None:
    assert set(validator.PACKAGES) == {
        "core",
        "analysis",
        "observation",
        "vision",
        "video",
        "sqlalchemy",
        "artifacts",
        "trust",
        "navigation",
    }
    assert all(
        item["distribution"].startswith("zeromodel")
        for item in validator.PACKAGES.values()
    )
