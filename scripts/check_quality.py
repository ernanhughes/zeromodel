from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Every one of the six production packages, plus the repository-wide
# integration test tree, must appear here. If a package is ever missing from
# this list, tests/test_quality_gate_coverage.py fails.
FORMAT_LINT_PATHS = [
    Path("scripts/code_quality_report.py"),
    Path("scripts/check_architecture.py"),
    Path("scripts/check_package_boundaries.py"),
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
    Path("packages/sqlalchemy/src"),
    Path("packages/sqlalchemy/tests"),
    Path("packages/artifacts/src"),
    Path("packages/artifacts/tests"),
    Path("packages/trust/src"),
    Path("packages/trust/tests"),
    Path("packages/navigation/src"),
    Path("packages/navigation/tests"),
    Path("integration_tests"),
]

TYPING_PATHS = [
    Path("packages/core/src"),
    Path("packages/analysis/src"),
    Path("packages/observation/src"),
    Path("packages/vision/src/zeromodel/vision/visual.py"),
    Path("packages/vision/src/zeromodel/vision/visual_policy.py"),
    Path("packages/vision/src/zeromodel/vision/__init__.py"),
    Path("packages/video/src"),
    Path("packages/sqlalchemy/src"),
    Path("packages/artifacts/src"),
    Path("packages/trust/src"),
    Path("packages/navigation/src"),
]

QUALITY_LIMIT_PATHS = [
    Path("packages/core/src"),
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
    Path("packages/sqlalchemy/src"),
    Path("packages/sqlalchemy/tests"),
    Path("packages/artifacts/src"),
    Path("packages/artifacts/tests"),
    Path("packages/trust/src"),
    Path("packages/trust/tests"),
    Path("packages/navigation/src"),
    Path("packages/navigation/tests"),
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


def run_step(label: str, command: list[str], paths: list[str] | None = None) -> None:
    print(f"== {label} ==")
    if paths is not None:
        print(f"paths checked: {', '.join(paths)}")
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        print(f"{label}: FAILED")
        raise SystemExit(result.returncode)
    print(f"{label}: passed")


def main() -> int:
    governed_paths = existing_python_paths(FORMAT_LINT_PATHS)
    if not governed_paths:
        raise SystemExit("No governed quality paths exist")
    typing_paths = existing_python_paths(TYPING_PATHS)
    if not typing_paths:
        raise SystemExit("No governed typing paths exist")
    quality_limit_paths = existing_python_paths(QUALITY_LIMIT_PATHS)
    if not quality_limit_paths:
        raise SystemExit("No governed quality-limit paths exist")

    # Execution order: format -> lint -> typing -> package boundaries ->
    # architecture -> quality limits. A failure at any stage stops the gate
    # immediately (run_step raises SystemExit), so a later, unrelated
    # "passed" line can never mask an earlier failure.
    run_step(
        "Ruff format check",
        [sys.executable, "-m", "ruff", "format", "--check", *governed_paths],
        governed_paths,
    )
    run_step(
        "Ruff lint check",
        [sys.executable, "-m", "ruff", "check", *governed_paths],
        governed_paths,
    )
    run_step(
        "mypy",
        [sys.executable, "-m", "mypy", *typing_paths],
        typing_paths,
    )
    run_step(
        "Package boundaries",
        [sys.executable, "scripts/check_package_boundaries.py"],
        [
            "packages/core/src",
            "packages/analysis/src",
            "packages/observation/src",
            "packages/vision/src",
            "packages/video/src",
            "packages/sqlalchemy/src",
            "packages/artifacts/src",
            "packages/trust/src",
            "packages/navigation/src",
        ],
    )
    run_step(
        "Architecture rules",
        [sys.executable, "scripts/check_architecture.py"],
        [
            "packages/core/src",
            "packages/analysis/src",
            "packages/observation/src",
            "packages/vision/src",
            "packages/video/src",
            "packages/sqlalchemy/src",
            "packages/artifacts/src",
            "packages/trust/src",
            "packages/navigation/src",
        ],
    )
    run_step(
        "Code-quality limits",
        [
            sys.executable,
            "scripts/code_quality_report.py",
            "--json",
            "build/quality/code-quality-report.json",
            "--markdown",
            "build/quality/code-quality-report.md",
            *(arg for path in quality_limit_paths for arg in ("--path", path)),
        ],
        quality_limit_paths,
    )
    print("Quality checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
