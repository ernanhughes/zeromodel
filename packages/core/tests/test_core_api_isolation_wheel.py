from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import zeromodel.core as core


EXPECTED_EXPORTS = {
    "LayoutRecipe",
    "ScoreTable",
    "SignReader",
    "VPMArtifact",
    "VPMCell",
    "VPMPolicyLookup",
    "VPMRegion",
    "VPMValidationError",
    "MatrixBlob",
    "PolicyLookupDecision",
    "build_vpm",
    "from_bundle",
    "to_bundle",
    "png_bytes",
    "svg_text",
    "compiled_plan_id",
    "lua_policy_source",
    "write_lua_policy",
}


def test_core_public_api_is_deliberate_and_core_owned() -> None:
    assert EXPECTED_EXPORTS <= set(core.__all__)
    assert getattr(__import__("zeromodel"), "__file__", None) is None
    location = Path(inspect.getfile(core)).as_posix()
    assert "zeromodel/core/__init__.py" in location


def test_core_import_does_not_load_optional_or_sibling_packages() -> None:
    script = r"""
import json
import sys
import zeromodel.core

forbidden = [
    "sqlalchemy",
    "torch",
    "torchvision",
    "transformers",
    "PIL",
    "zeromodel.analysis",
    "zeromodel.observation",
    "zeromodel.vision",
    "zeromodel.video",
    "zeromodel.persistence",
    "research",
    "tests",
]
loaded = [
    name
    for name in forbidden
    if name in sys.modules
    or any(module.startswith(name + ".") for module in sys.modules)
]
print(json.dumps(loaded))
raise SystemExit(1 if loaded else 0)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], text=True, capture_output=True, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == []


def test_core_wheel_contains_only_core_namespace_when_wheel_path_is_provided() -> None:
    wheel_path = os.environ.get("CORE_WHEEL_PATH")
    if not wheel_path:
        return
    names = set(zipfile.ZipFile(wheel_path).namelist())
    forbidden_prefixes = (
        "zeromodel/__init__.py",
        "zeromodel/analysis/",
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
    offenders = [
        name
        for name in names
        for prefix in forbidden_prefixes
        if name == prefix or name.startswith(prefix)
    ]
    assert offenders == []
    assert "zeromodel/core/artifact.py" in names
    assert "zeromodel/core/policy_lookup.py" in names
    assert any(name.startswith("zeromodel-1.0.13.dist-info/") for name in names)
