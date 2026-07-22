from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

EXECUTABLE_EXAMPLES = [
    "examples/arcade_visual_address_benchmark.py",
    "examples/arcade_visual_video_baseline.py",
    "examples/render_signs_demo.py",
]


@pytest.mark.parametrize("example", EXECUTABLE_EXAMPLES)
def test_executable_example_bootstrap_does_not_crash(example: str) -> None:
    result = subprocess.run(
        [sys.executable, example, "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_nested_research_benchmark_resolves_the_correct_repository_root() -> None:
    path = (
        REPO_ROOT
        / "research"
        / "video_action_set"
        / "benchmarks"
        / "arcade_visual_video_local_correlation_benchmark.py"
    )
    source = path.read_text(encoding="utf-8")
    assert "parents[3]" in source
    assert "Path(__file__).resolve().parents[1]" not in source

    # Prove parents[3] from this exact file's depth actually lands on the
    # repository root (research/video_action_set/benchmarks/<file>.py).
    assert path.resolve().parents[3] == REPO_ROOT


@pytest.mark.parametrize(
    "path",
    [
        "research/video_action_set/tests/test_video_benchmark_facade.py",
        "research/video_action_set/tests/test_video_verification_closure_kernel.py",
    ],
)
def test_research_files_at_the_same_depth_also_use_parents_three(path: str) -> None:
    source = (REPO_ROOT / path).read_text(encoding="utf-8")
    assert "parents[3]" in source
    assert (REPO_ROOT / path).resolve().parents[3] == REPO_ROOT
