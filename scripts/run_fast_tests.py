from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TIMEOUT_SECONDS = 60
FAST_TEST_TARGETS = [
    "tests/test_artifact_kernel.py",
    "tests/test_views.py",
    "tests/test_spatial.py",
    "tests/test_manifold.py",
]


def main() -> int:
    command = [sys.executable, "-m", "pytest", "-q", *FAST_TEST_TARGETS]
    try:
        result = subprocess.run(command, cwd=REPO_ROOT, check=False, timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        print(f"Fast test suite exceeded {TIMEOUT_SECONDS} seconds", file=sys.stderr)
        return 124
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
