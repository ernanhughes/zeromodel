from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SOURCE_ROOTS = (
    "packages/core/src",
    "packages/analysis/src",
    "packages/observation/src",
    "packages/vision/src",
    "packages/video/src",
    "packages/sqlalchemy/src",
)


def test_production_packages_do_not_import_examples_tests_or_research() -> None:
    # This originally rglob'd a root `zeromodel/` tree that predates the
    # package split and no longer exists (rglob on a missing path silently
    # returns nothing, so the check was vacuous). It now inspects every real
    # production source root instead. See
    # docs/reviews/post-split-main-audit.md finding F3 for the same class of
    # bug in scripts/check_architecture.py.
    offenders: list[str] = []
    for root in PRODUCTION_SOURCE_ROOTS:
        for path in (REPO_ROOT / root).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            text = path.read_text(encoding="utf-8")
            if (
                "from examples" in text
                or "import examples" in text
                or "from tests" in text
                or "import tests" in text
                or "from research" in text
                or "import research" in text
            ):
                offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == []


@pytest.mark.slow
def test_installed_core_wheel_excludes_research_and_examples(tmp_path: Path) -> None:
    # The original version of this test built the (now unbuildable) monolithic
    # root distribution and asserted that installing it made research-only
    # modules such as `zeromodel.video_prospective_providers` and
    # `zeromodel.video_action_set_benchmark` importable. That assumption is
    # now architecturally backwards: those are research concepts today
    # (research/video/video_prospective_providers.py,
    # research/benchmarks/video_action_set_benchmark.py) and must NOT ship in
    # any production wheel (see FORBIDDEN_WHEEL_PREFIXES in
    # scripts/validate_release_candidate.py). This test now proves the
    # opposite, current invariant: an installed core wheel does not expose
    # them at all.
    dist_dir = tmp_path / "dist"
    subprocess.run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir), str(REPO_ROOT / "packages" / "core")],
        cwd=REPO_ROOT,
        check=True,
    )
    wheel = next(dist_dir.glob("zeromodel-*.whl"))
    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = venv_dir / ("Scripts" if sys.platform == "win32" else "bin") / "python"
    subprocess.run(
        [str(python), "-m", "pip", "install", "--quiet", "--no-deps", str(wheel)],
        cwd=tmp_path,
        check=True,
    )
    script = """
import importlib
import json

forbidden = [
    "zeromodel.video_prospective_providers",
    "zeromodel.video_action_set_benchmark",
    "zeromodel.video_complete_row_evidence",
    "zeromodel.video_action_equivalence",
]
still_importable = []
for name in forbidden:
    try:
        importlib.import_module(name)
    except ImportError:
        pass
    else:
        still_importable.append(name)
print(json.dumps(still_importable))
"""
    result = subprocess.run(
        [str(python), "-c", script],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    import json

    assert json.loads(result.stdout) == []
