from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GOVERNED_PATHS = [
    Path("scripts/code_quality_report.py"),
    Path("scripts/check_architecture.py"),
    Path("scripts/check_quality.py"),
    Path("zeromodel/video_action_set"),
]


def existing_governed_paths() -> list[str]:
    return [path.as_posix() for path in GOVERNED_PATHS if (REPO_ROOT / path).exists()]


def run_step(label: str, command: list[str]) -> None:
    print(label)
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    print(f"{label}: passed")


def main() -> int:
    governed_paths = existing_governed_paths()
    if not governed_paths:
        raise SystemExit("No governed quality paths exist")

    run_step(
        "Formatting",
        [sys.executable, "-m", "ruff", "format", "--check", *governed_paths],
    )
    run_step(
        "Linting",
        [sys.executable, "-m", "ruff", "check", *governed_paths],
    )
    run_step("Typing", [sys.executable, "-m", "mypy", *governed_paths])
    run_step("Architecture", [sys.executable, "scripts/check_architecture.py"])
    run_step(
        "Quality limits",
        [
            sys.executable,
            "scripts/code_quality_report.py",
            "--json",
            "build/quality/code-quality-report.json",
            "--markdown",
            "build/quality/code-quality-report.md",
        ],
    )
    print("Quality checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
