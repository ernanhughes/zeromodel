from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path("scripts/check_package_boundaries.py")
SPEC = importlib.util.spec_from_file_location("check_package_boundaries", SCRIPT_PATH)
assert SPEC is not None
checker = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def test_boundary_manifest_defines_all_ten_packages() -> None:
    manifest = checker.load_manifest()

    assert set(manifest["packages"]) == {
        "core",
        "analysis",
        "observation",
        "vision",
        "perception",
        "video",
        "sqlalchemy",
        "artifacts",
        "trust",
        "navigation",
    }
    assert manifest["release_version"] == "1.0.13"


def test_production_modules_are_discovered_from_package_roots() -> None:
    modules = checker.discover_modules(checker.load_manifest())

    assert "zeromodel.core.artifact" in modules
    assert "zeromodel.perception" in modules
    assert "zeromodel.video.domains.video_action_set.dto" in modules
    assert "zeromodel.persistence.sqlalchemy.db.session" in modules
    assert all(not record.path.as_posix().startswith("zeromodel/") for record in modules.values())


def test_old_root_initializer_is_absent() -> None:
    assert not Path("zeromodel/__init__.py").exists()
