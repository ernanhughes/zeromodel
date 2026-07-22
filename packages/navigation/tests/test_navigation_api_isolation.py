from __future__ import annotations

import os
import subprocess
import sys

import zeromodel.navigation as navigation


def test_navigation_public_api_is_deliberate() -> None:
    assert hasattr(navigation, "__all__")
    assert set(navigation.__all__) == {
        "HierarchyCompilerSpecDTO",
        "HierarchyManifestDTO",
        "NavigationTileDTO",
        "TraversalRequestDTO",
        "TraversalResultDTO",
        "TraversalReceiptDTO",
        "TraversalRule",
        "TraversalStepDTO",
        "compile_hierarchy",
        "replay_traversal",
        "traverse",
        "validate_hierarchy",
    }
    assert "ScoreTable" not in navigation.__all__
    assert "VideoPolicyReader" not in navigation.__all__


def test_navigation_import_loads_core_and_artifacts_but_never_trust() -> None:
    script = r"""
import json
import sys

import zeromodel.navigation

forbidden = [
    "zeromodel.analysis",
    "zeromodel.observation",
    "zeromodel.vision",
    "zeromodel.video",
    "zeromodel.trust",
    "zeromodel.persistence",
    "sqlalchemy",
    "torch",
    "torchvision",
    "transformers",
    "PIL",
]

loaded_forbidden = [
    name
    for name in forbidden
    if name in sys.modules
    or any(module.startswith(name + ".") for module in sys.modules)
]

core_loaded = (
    "zeromodel.core" in sys.modules
    or any(module.startswith("zeromodel.core.") for module in sys.modules)
)
artifacts_loaded = (
    "zeromodel.artifacts" in sys.modules
    or any(module.startswith("zeromodel.artifacts.") for module in sys.modules)
)

print(json.dumps({
    "core_loaded": core_loaded,
    "artifacts_loaded": artifacts_loaded,
    "forbidden": loaded_forbidden,
}))
raise SystemExit(1 if loaded_forbidden or not core_loaded or not artifacts_loaded else 0)
"""
    args = [sys.executable, "-c", script]
    if os.environ.get("NAVIGATION_WHEEL_PATH"):
        args.insert(1, "-I")
    result = subprocess.run(args, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr


def test_navigation_wheel_contains_only_navigation_namespace_when_path_is_provided() -> (
    None
):
    wheel_path = os.environ.get("NAVIGATION_WHEEL_PATH")
    if not wheel_path:
        return
    import zipfile

    with zipfile.ZipFile(wheel_path) as archive:
        names = archive.namelist()
    for name in names:
        if (
            name.endswith(".dist-info/")
            or "/.dist-info/" in name
            or name.endswith(
                (
                    ".dist-info/METADATA",
                    ".dist-info/RECORD",
                    ".dist-info/WHEEL",
                    ".dist-info/top_level.txt",
                )
            )
        ):
            continue
        assert name.startswith("zeromodel/navigation/"), name
