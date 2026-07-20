from __future__ import annotations

import subprocess
import sys
import time


FAST_SUITE_BUDGET_SECONDS = 60


def main() -> int:
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "--maxfail=1",
        *sys.argv[1:],
    ]
    started = time.monotonic()

    try:
        completed = subprocess.run(
            command,
            check=False,
            timeout=FAST_SUITE_BUDGET_SECONDS,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        print(
            f"\nFAST TEST BUDGET EXCEEDED: {elapsed:.1f}s > "
            f"{FAST_SUITE_BUDGET_SECONDS}s",
            file=sys.stderr,
        )
        print(
            "Move expensive tests to the integration tier or reduce their fixture scope.",
            file=sys.stderr,
        )
        return 124

    elapsed = time.monotonic() - started
    print(
        f"\nFast-suite runtime: {elapsed:.2f}s "
        f"(budget: {FAST_SUITE_BUDGET_SECONDS}s)"
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
