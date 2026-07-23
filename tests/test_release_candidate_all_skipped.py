from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT = Path("scripts/validate_release_candidate.py")
SPEC = importlib.util.spec_from_file_location("validate_release_candidate_skipped", SCRIPT)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def test_required_layer_with_only_skipped_tests_fails_release_verdict() -> None:
    report = {
        "source_tree_fast_production_tests": {
            "passed": 0,
            "failed": 0,
            "errors": 0,
            "skipped": 12,
            "returncode": 0,
        },
        "package_local_source_tests_by_package": {
            key: {
                "passed": 1,
                "failed": 0,
                "errors": 0,
                "skipped": 0,
                "returncode": 0,
            }
            for key in validator.PACKAGES
        },
        "cross_package_integration_tests": {
            "passed": 1,
            "failed": 0,
            "errors": 0,
            "skipped": 0,
            "returncode": 0,
        },
        "research": {"status": "excluded_by_policy"},
    }

    verdicts = validator.evaluate_release_test_layers(report)
    fast = next(
        verdict
        for verdict in verdicts
        if verdict.name == "source_tree_fast_production_tests"
    )

    assert fast.status == "failed"
    assert any("zero" in reason and "execut" in reason for reason in fast.reasons)
    assert validator.release_verdict_passed(verdicts) is False
