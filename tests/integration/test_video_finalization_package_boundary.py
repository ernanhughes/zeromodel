from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import venv
import zipfile

import pytest


pytestmark = pytest.mark.integration
REPO_ROOT = Path(__file__).resolve().parents[2]


def _environment_without_pythonpath() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    return environment


def test_offline_installed_wheel_preserves_finalization_boundaries(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    shutil.copytree(
        REPO_ROOT,
        source,
        ignore=shutil.ignore_patterns(
            ".git",
            ".pytest_cache",
            "__pycache__",
            "*.pyc",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    wheel_dir = tmp_path / "wheel"
    wheel_dir.mkdir()
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_dir),
        ],
        cwd=source,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env=_environment_without_pythonpath(),
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheels = tuple(wheel_dir.glob("zeromodel-*.whl"))
    assert len(wheels) == 1

    required_modules = {
        "zeromodel/video_action_set_final_cli.py",
        "zeromodel/video_action_set_final_admin_cli.py",
        "zeromodel/domains/video_action_set/final_access_service.py",
        "zeromodel/domains/video_action_set/final_publication.py",
        "zeromodel/domains/video_action_set/final_reconstruction.py",
    }
    with zipfile.ZipFile(wheels[0]) as archive:
        assert required_modules.issubset(archive.namelist())

    environment_dir = tmp_path / "environment"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment_dir)
    python = environment_dir / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    installed = subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--no-deps",
            "--no-index",
            "--force-reinstall",
            str(wheels[0]),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
        env=_environment_without_pythonpath(),
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr

    script = """
import json
from pathlib import Path
import tempfile
from zeromodel import build_runtime
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.build_orchestration import build_split
from zeromodel.video_action_set_cli import build_argument_parser
import zeromodel.video_action_set_final_cli
import zeromodel.video_action_set_final_admin_cli
import zeromodel.domains.video_action_set.final_access_service
import zeromodel.domains.video_action_set.final_publication
import zeromodel.domains.video_action_set.final_reconstruction

service = build_runtime().video_action_set.engine.final_access_service
options = {option for action in build_argument_parser()._actions for option in action.option_strings}
blocked = False
with tempfile.TemporaryDirectory() as root:
    try:
        build_split("final", Path(root) / "output", Path(root) / "repository")
    except VPMValidationError as exc:
        blocked = "prohibited" in str(exc)
print(json.dumps({
    "executor_registered": service.final_executor is not None,
    "historical_cli_has_build_final": "--build-final" in options,
    "final_build_blocked": blocked,
}))
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
    assert json.loads(checked.stdout) == {
        "executor_registered": False,
        "historical_cli_has_build_final": False,
        "final_build_blocked": True,
    }
