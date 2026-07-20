from __future__ import annotations

import subprocess
import sys
import time


FAST_SUITE_BUDGET_SECONDS = 60
FORBIDDEN_INTEGRATION_FLAGS = {"--run-integration", "--run-slow"}


def main() -> int:
    forbidden = [
        argument
        for argument in sys.argv[1:]
        if argument in FORBIDDEN_INTEGRATION_FLAGS
    ]
    if forbidden:
        print(
            "The fast-test runner does not permit integration opt-in flags: "
            + ", ".join(forbidden),
            file=sys.stderr,
        )
        print(
            "Run integration tests explicitly with pytest instead.",
            file=sys.stderr,
        )
        return 2

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
