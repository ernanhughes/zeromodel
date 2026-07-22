from __future__ import annotations

import os
import subprocess
import sys

import zeromodel.observation as observation


def test_observation_public_api_is_deliberate() -> None:
    assert hasattr(observation, "__all__")
    assert "ImageObservation" in observation.__all__
    assert "VisualAddressProvider" in observation.__all__
    assert "VisualSignReader" not in observation.__all__
    assert "VideoPolicyReader" not in observation.__all__
    assert "ScoreTable" not in observation.__all__


def test_observation_import_loads_core_but_no_forbidden_siblings() -> None:
    script = r"""
import json
import sys

import zeromodel.observation

forbidden = [
    "zeromodel.analysis",
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
    if os.environ.get("OBSERVATION_WHEEL_PATH"):
        args.insert(1, "-I")
    result = subprocess.run(args, text=True, capture_output=True, check=False)

    assert result.returncode == 0, result.stdout + result.stderr
