"""Package-boundary regression for the finalization CLI surface.

An installed `zeromodel-video` wheel, on its own, must not carry any
SQLAlchemy-only finalization capability. That capability (the
`video_action_set_final_cli` / `video_action_set_final_admin_cli` entry
points, and everything under `zeromodel.persistence.*`) lives exclusively in
the separate `zeromodel-sqlalchemy` distribution.

This replaces an earlier version of this test that built and inspected the
retired monolithic root distribution (pre package-split, see
`git log --follow -- tests/integration/test_video_finalization_package_boundary.py`,
commit 5f1d9a9). That version could never pass post-split: it built
`packages/video`'s parent checkout as a single root wheel (the root is no
longer a buildable distribution) and asserted the presence of files at flat,
pre-split module paths that do not exist anywhere in the current layout.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import venv

import pytest


pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]


def _environment_without_pythonpath() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def test_video_wheel_alone_carries_no_sqlalchemy_finalization_capability(
    tmp_path: Path,
) -> None:
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--outdir",
            str(wheel_dir),
            str(REPO_ROOT / "packages" / "video"),
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = tuple(wheel_dir.glob("zeromodel_video-*.whl"))
    assert len(wheels) == 1

    environment_dir = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True).create(environment_dir)
    python = _venv_python(environment_dir)

    # --no-deps: this test only needs to prove an ABSENCE (no sqlalchemy
    # persistence capability leaks into a video-only install); it does not
    # need core/observation/numpy actually importable to do that.
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env=_environment_without_pythonpath(),
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr

    script = """
import importlib
import json

forbidden = [
    "zeromodel.persistence",
    "zeromodel.persistence.sqlalchemy",
    "zeromodel.persistence.sqlalchemy.video_action_set_final_cli",
    "zeromodel.persistence.sqlalchemy.video_action_set_final_admin_cli",
]
blocked = []
for name in forbidden:
    try:
        importlib.import_module(name)
    except ImportError:
        blocked.append(name)
print(json.dumps({"blocked": sorted(blocked)}))
"""
    checked = subprocess.run(
        [str(python), "-c", script],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=45,
        env=_environment_without_pythonpath(),
    )
    assert checked.returncode == 0, checked.stderr
    result = json.loads(checked.stdout)
    assert result["blocked"] == [
        "zeromodel.persistence",
        "zeromodel.persistence.sqlalchemy",
        "zeromodel.persistence.sqlalchemy.video_action_set_final_admin_cli",
        "zeromodel.persistence.sqlalchemy.video_action_set_final_cli",
    ]
