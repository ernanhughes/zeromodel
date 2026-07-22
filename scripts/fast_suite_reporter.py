"""Pytest plugin that writes a structured JSON summary of a fast-suite run.

Loaded by scripts/run_fast_tests.py via `-p fast_suite_reporter` in a
subprocess, configured entirely through environment variables so the parent
script does not need to parse pytest's terminal output to learn how many
tests were collected, deselected, passed, failed, or skipped.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


class FastSuiteReport:
    def __init__(
        self, report_path: Path, test_roots: list[str], marker_expression: str
    ) -> None:
        self.report_path = report_path
        self.test_roots = test_roots
        self.marker_expression = marker_expression
        self.collected = 0
        self.deselected = 0
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.collection_errors = 0
        self._started = time.monotonic()

    def pytest_itemcollected(self, item: Any) -> None:
        self.collected += 1

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected += len(items)

    def pytest_collectreport(self, report: Any) -> None:
        if report.failed:
            self.collection_errors += 1

    def pytest_runtest_logreport(self, report: Any) -> None:
        if report.when == "call":
            if report.outcome == "passed":
                self.passed += 1
            elif report.outcome == "failed":
                self.failed += 1
            elif report.outcome == "skipped":
                self.skipped += 1
        elif report.when in ("setup", "teardown") and report.outcome == "failed":
            self.failed += 1
        elif report.when == "setup" and report.outcome == "skipped":
            self.skipped += 1

    def pytest_sessionfinish(self, session: Any, exitstatus: int) -> None:
        duration = time.monotonic() - self._started
        self.report_path.parent.mkdir(parents=True, exist_ok=True)
        self.report_path.write_text(
            json.dumps(
                {
                    "test_roots": self.test_roots,
                    "marker_expression": self.marker_expression,
                    "collected": self.collected,
                    "deselected": self.deselected,
                    "passed": self.passed,
                    "failed": self.failed,
                    "skipped": self.skipped,
                    "duration_seconds": round(duration, 2),
                    "collection_errors": self.collection_errors,
                    "exit_status": int(exitstatus),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )


def pytest_configure(config: Any) -> None:
    report_path = os.environ.get("ZEROMODEL_FAST_SUITE_REPORT_PATH")
    if not report_path:
        return
    test_roots_raw = os.environ.get("ZEROMODEL_FAST_SUITE_TEST_ROOTS", "")
    marker_expression = os.environ.get("ZEROMODEL_FAST_SUITE_MARKER_EXPRESSION", "")
    test_roots = [root for root in test_roots_raw.split(os.pathsep) if root]
    reporter = FastSuiteReport(Path(report_path), test_roots, marker_expression)
    config.pluginmanager.register(reporter, "zeromodel-fast-suite-reporter")
