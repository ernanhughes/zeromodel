from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FORMAT_LINT_PATHS = [
    Path("scripts/code_quality_report.py"),
    Path("scripts/check_architecture.py"),
    Path("scripts/check_quality.py"),
    Path("packages/core/src"),
    Path("packages/core/tests"),
    Path("packages/analysis/src"),
    Path("packages/analysis/tests"),
    Path("packages/observation/src"),
    Path("packages/observation/tests"),
    Path("packages/vision/src/zeromodel/vision/visual.py"),
    Path("packages/vision/src/zeromodel/vision/visual_policy.py"),
    Path("packages/vision/src/zeromodel/vision/__init__.py"),
    Path("packages/vision/tests"),
    Path("packages/video/src"),
    Path("packages/video/tests"),
]

TYPING_PATHS = [
    Path("packages/core/src"),
    Path("packages/analysis/src"),
    Path("packages/observation/src"),
    Path("packages/vision/src/zeromodel/vision/visual.py"),
    Path("packages/vision/src/zeromodel/vision/visual_policy.py"),
    Path("packages/vision/src/zeromodel/vision/__init__.py"),
    Path("packages/video/src"),
]


def existing_python_paths(paths_to_check: list[Path]) -> list[str]:
    paths: list[str] = []
    for path in paths_to_check:
        resolved = REPO_ROOT / path
        if resolved.is_file() and resolved.suffix in {".py", ".pyi"}:
            paths.append(path.as_posix())
        elif resolved.is_dir() and any(resolved.rglob("*.py")):
            paths.append(path.as_posix())
    return paths


def run_step(label: str, command: list[str]) -> None:
    print(label)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"{label}: passed")


def main() -> int:
    governed_paths = existing_python_paths(FORMAT_LINT_PATHS)
    if not governed_paths:
        raise SystemExit("No governed quality paths exist")
    typing_paths = existing_python_paths(TYPING_PATHS)
    if not typing_paths:
        raise SystemExit("No governed typing paths exist")

    run_step(
        "Formatting",
        [sys.executable, "-m", "ruff", "format", "--check", *governed_paths],
    )
    run_step(
        "Linting",
        [sys.executable, "-m", "ruff", "check", *governed_paths],
    )
    run_step(
        "Typing",
        [sys.executable, "-m", "mypy", *typing_paths],
    )
    run_step("Architecture", [sys.executable, "scripts/check_architecture.py"])
    run_step(
        "Package boundaries",
        [sys.executable, "scripts/check_package_boundaries.py"],
    )
    run_step(
        "Quality limits",
        [
            sys.executable,
            "scripts/code_quality_report.py",
            "--json",
            "build/quality/code-quality-report.json",
            "--markdown",
            "build/quality/code-quality-report.md",
            "--path",
            "packages/core/src",
            "--path",
            "packages/analysis/src",
            "--path",
            "packages/analysis/tests",
            "--path",
            "packages/observation/src",
            "--path",
            "packages/observation/tests",
            "--path",
            "packages/vision/src/zeromodel/vision/visual.py",
            "--path",
            "packages/vision/src/zeromodel/vision/visual_policy.py",
            "--path",
            "packages/vision/src/zeromodel/vision/__init__.py",
            "--path",
            "packages/vision/tests",
            "--path",
            "packages/video/src",
            "--path",
            "packages/video/tests",
        ],
    )
    print("Quality checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
