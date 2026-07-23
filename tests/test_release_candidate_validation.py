"""Unit tests for scripts/validate_release_candidate.py's release-truth
logic: the release verdict (fix A), the installed-wheel probe target list
(fix C), and package-authority consistency against package-boundaries.toml
(fix F). None of these tests build packages, create a virtual environment,
or execute the full release-candidate validator - they exercise pure
functions directly, the same pattern test_release_validator_venv_paths.py
already established for this script.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path("scripts/validate_release_candidate.py")
SPEC = importlib.util.spec_from_file_location("validate_release_candidate", SCRIPT)
assert SPEC is not None
validator = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def _passing_counts() -> dict[str, Any]:
    return {"passed": 3, "failed": 0, "skipped": 0, "errors": 0, "returncode": 0}


def _base_report() -> dict[str, Any]:
    """A payload shaped like the committed release-test-layers report, with
    every required layer passing and research correctly excluded."""
    return {
        "source_tree_fast_production_tests": _passing_counts(),
        "package_local_source_tests_by_package": {
            key: _passing_counts() for key in validator.PACKAGES
        },
        "cross_package_integration_tests": _passing_counts(),
        "research": {"status": "excluded_by_policy", "note": "excluded"},
    }


# --- Fix A: release verdict must be truthful --------------------------------


def test_all_required_layers_passing_yields_an_overall_pass() -> None:
    verdicts = validator.evaluate_release_test_layers(_base_report())
    assert validator.release_verdict_passed(verdicts)
    assert all(v.ok for v in verdicts)


def test_a_layer_with_nonzero_returncode_fails_the_verdict() -> None:
    report = _base_report()
    report["source_tree_fast_production_tests"] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 3,
        "returncode": 2,
    }
    verdicts = validator.evaluate_release_test_layers(report)
    assert not validator.release_verdict_passed(verdicts)
    fast = next(v for v in verdicts if v.name == "source_tree_fast_production_tests")
    assert fast.status == "failed"
    assert any("returncode" in reason for reason in fast.reasons)


def test_a_layer_reporting_collection_errors_fails_the_verdict() -> None:
    report = _base_report()
    report["cross_package_integration_tests"] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 3,
        "returncode": 2,
    }
    verdicts = validator.evaluate_release_test_layers(report)
    assert not validator.release_verdict_passed(verdicts)
    layer = next(v for v in verdicts if v.name == "cross_package_integration_tests")
    assert layer.status == "failed"
    assert any("errors=3" in reason for reason in layer.reasons)


def test_a_layer_reporting_failed_tests_fails_the_verdict() -> None:
    report = _base_report()
    report["package_local_source_tests_by_package"]["core"] = {
        "passed": 5,
        "failed": 1,
        "skipped": 0,
        "errors": 0,
        "returncode": 1,
    }
    verdicts = validator.evaluate_release_test_layers(report)
    assert not validator.release_verdict_passed(verdicts)
    layer = next(
        v for v in verdicts if v.name == "package_local_source_tests:core"
    )
    assert layer.status == "failed"
    assert any("failed=1" in reason for reason in layer.reasons)


def test_a_missing_required_layer_fails_the_verdict() -> None:
    report = _base_report()
    del report["cross_package_integration_tests"]
    verdicts = validator.evaluate_release_test_layers(report)
    assert not validator.release_verdict_passed(verdicts)
    layer = next(v for v in verdicts if v.name == "cross_package_integration_tests")
    assert layer.status == "not_executed"


def test_a_layer_with_zero_meaningful_tests_fails_the_verdict() -> None:
    report = _base_report()
    report["package_local_source_tests_by_package"]["trust"] = {
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "errors": 0,
        "returncode": 0,
    }
    verdicts = validator.evaluate_release_test_layers(report)
    assert not validator.release_verdict_passed(verdicts)
    layer = next(
        v for v in verdicts if v.name == "package_local_source_tests:trust"
    )
    assert layer.status == "failed"
    assert any("zero relevant tests" in reason for reason in layer.reasons)


def test_research_excluded_by_policy_does_not_fail_the_production_verdict() -> None:
    verdicts = validator.evaluate_release_test_layers(_base_report())
    research = next(v for v in verdicts if v.name == "research")
    assert research.status == "excluded_by_policy"
    assert research.ok
    assert validator.release_verdict_passed(verdicts)


def test_previously_committed_false_positive_shape_is_now_rejected() -> None:
    """Regression for the exact defect this fix closes: a layer reporting
    `errors=3, passed=0, returncode=2` must never be accompanied by an
    overall pass."""
    report = _base_report()
    report["cross_package_integration_tests"] = {
        "errors": 3,
        "passed": 0,
        "failed": 0,
        "skipped": 0,
        "returncode": 2,
    }
    verdicts = validator.evaluate_release_test_layers(report)
    assert validator.release_verdict_passed(verdicts) is False


# --- Fix C: installed-wheel probe covers every configured package -----------


def test_wheel_smoke_probe_namespaces_covers_every_configured_package() -> None:
    namespaces = validator.wheel_smoke_probe_namespaces()
    expected = {expected["namespace"] for expected in validator.PACKAGES.values()}
    assert set(namespaces) == expected
    assert len(namespaces) == len(validator.PACKAGES) == 9


def test_wheel_smoke_probe_namespaces_includes_the_new_packages() -> None:
    namespaces = set(validator.wheel_smoke_probe_namespaces())
    for namespace in ("zeromodel.artifacts", "zeromodel.trust", "zeromodel.navigation"):
        assert namespace in namespaces


# --- Fix F: package authorities must not silently disagree ------------------


def _boundaries_from(packages: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        key: {
            "distribution": expected["distribution"],
            "namespace": expected["namespace"],
            "source_root": (Path(expected["path"]) / "src").as_posix(),
            "depends_on": list(expected["depends_on"]),
        }
        for key, expected in packages.items()
    }


def test_matching_configuration_passes_consistency_check() -> None:
    boundaries = _boundaries_from(validator.PACKAGES)
    validator.validate_package_boundary_consistency(
        boundaries=boundaries, packages=validator.PACKAGES
    )


def test_package_set_mismatch_is_rejected() -> None:
    boundaries = _boundaries_from(validator.PACKAGES)
    del boundaries["trust"]
    with pytest.raises(SystemExit, match="package key set mismatch"):
        validator.validate_package_boundary_consistency(
            boundaries=boundaries, packages=validator.PACKAGES
        )


def test_namespace_mismatch_is_rejected() -> None:
    boundaries = _boundaries_from(validator.PACKAGES)
    boundaries["core"]["namespace"] = "zeromodel.core.wrong"
    with pytest.raises(SystemExit, match="namespace mismatch"):
        validator.validate_package_boundary_consistency(
            boundaries=boundaries, packages=validator.PACKAGES
        )


def test_source_root_mismatch_is_rejected() -> None:
    boundaries = _boundaries_from(validator.PACKAGES)
    boundaries["video"]["source_root"] = "packages/video/wrong-src"
    with pytest.raises(SystemExit, match="source_root mismatch"):
        validator.validate_package_boundary_consistency(
            boundaries=boundaries, packages=validator.PACKAGES
        )


def test_internal_dependency_edge_mismatch_is_rejected() -> None:
    """Regression for the exact historical drift this check exists to
    catch: PACKAGES declaring an extra internal edge package-boundaries.toml
    does not (sqlalchemy -> observation)."""
    boundaries = _boundaries_from(validator.PACKAGES)
    packages = {
        key: dict(expected) for key, expected in validator.PACKAGES.items()
    }
    packages["sqlalchemy"] = dict(packages["sqlalchemy"])
    packages["sqlalchemy"]["depends_on"] = ("core", "video", "observation")
    with pytest.raises(SystemExit, match="depends_on mismatch"):
        validator.validate_package_boundary_consistency(
            boundaries=boundaries, packages=packages
        )


def test_current_package_boundaries_toml_matches_packages_dict() -> None:
    """End-to-end: the real package-boundaries.toml on disk must agree with
    the real PACKAGES dict right now, not just in a synthetic fixture."""
    validator.validate_package_boundary_consistency()
