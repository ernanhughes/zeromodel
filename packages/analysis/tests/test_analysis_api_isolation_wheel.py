from __future__ import annotations

import subprocess
import sys
import zipfile
import os
from pathlib import Path

import zeromodel.analysis as analysis


EXPECTED_MODULES = {
    "zeromodel/analysis/__init__.py",
    "zeromodel/analysis/adapters/__init__.py",
    "zeromodel/analysis/adapters/common.py",
    "zeromodel/analysis/adapters/jsonl.py",
    "zeromodel/analysis/adapters/tensorboard.py",
    "zeromodel/analysis/adapters/trackio.py",
    "zeromodel/analysis/adapters/wandb.py",
    "zeromodel/analysis/compare.py",
    "zeromodel/analysis/compose.py",
    "zeromodel/analysis/controller.py",
    "zeromodel/analysis/critic.py",
    "zeromodel/analysis/domains/__init__.py",
    "zeromodel/analysis/edge.py",
    "zeromodel/analysis/hierarchy.py",
    "zeromodel/analysis/learning.py",
    "zeromodel/analysis/manifold.py",
    "zeromodel/analysis/patterns.py",
    "zeromodel/analysis/phos.py",
    "zeromodel/analysis/policy_diagnostics.py",
    "zeromodel/analysis/policy_properties.py",
    "zeromodel/analysis/spatial.py",
    "zeromodel/analysis/training.py",
}


def test_analysis_public_api_is_deliberate_and_analysis_owned() -> None:
    assert hasattr(analysis, "__all__")
    assert "ScoreTable" not in analysis.__all__
    assert "VPMArtifact" not in analysis.__all__
    assert "SpatialOptimizer" in analysis.__all__
    assert "PolicyPropertyChecker" in analysis.__all__


def test_analysis_import_loads_core_but_no_forbidden_siblings() -> None:
    script = r"""
import json
import sys

import zeromodel.analysis

forbidden = [
    "zeromodel.observation",
    "zeromodel.vision",
    "zeromodel.video",
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
print(json.dumps({"core_loaded": core_loaded, "forbidden": loaded_forbidden}))
raise SystemExit(1 if loaded_forbidden or not core_loaded else 0)
"""
    args = [sys.executable, "-c", script]
    if os.environ.get("ANALYSIS_WHEEL_PATH"):
        args.insert(1, "-I")

    result = subprocess.run(
        args,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_analysis_wheel_contains_only_analysis_namespace_when_wheel_path_is_provided() -> (
    None
):
    wheel_path = os.environ.get("ANALYSIS_WHEEL_PATH")
    if not wheel_path:
        return

    with zipfile.ZipFile(Path(wheel_path)) as wheel:
        names = set(wheel.namelist())

    assert EXPECTED_MODULES <= names
    assert "zeromodel/__init__.py" not in names
    forbidden_prefixes = (
        "zeromodel/core/",
        "zeromodel/observation/",
        "zeromodel/vision/",
        "zeromodel/video/",
        "zeromodel/persistence/",
        "tests/",
        "research/",
        "examples/",
        "docs/",
        "scripts/",
    )
    assert not [
        name
        for name in names
        if any(name.startswith(prefix) for prefix in forbidden_prefixes)
    ]
