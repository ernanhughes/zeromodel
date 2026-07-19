from __future__ import annotations

import subprocess
import sys
import textwrap
import venv
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_zeromodel_package_does_not_import_examples_or_tests() -> None:
    for path in (REPO_ROOT / "zeromodel").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "from examples" not in text
        assert "import examples" not in text
        assert "from tests" not in text
        assert "import tests" not in text


@pytest.mark.slow
def test_installed_wheel_imports_prospective_modules(tmp_path: Path) -> None:
    dist_dir = tmp_path / "dist"
    subprocess.run([sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)], cwd=REPO_ROOT, check=True)
    wheel = next(dist_dir.glob("zeromodel-*.whl"))
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    subprocess.run([str(python), "-m", "pip", "install", "--quiet", str(wheel)], cwd=tmp_path, check=True)
    script = textwrap.dedent(
        """
        import os
        import zeromodel
        import zeromodel.video_complete_row_evidence
        import zeromodel.video_prospective_providers
        import zeromodel.video_action_set_benchmark
        import zeromodel.video_action_equivalence
        import zeromodel.video_policy_reachability
        assert 'examples' not in os.listdir('.')
        """
    )
    subprocess.run([str(python), "-c", script], cwd=tmp_path, check=True)
