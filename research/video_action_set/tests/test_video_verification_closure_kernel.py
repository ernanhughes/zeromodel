from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from research.video_action_set import verification


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_ROOT = REPO_ROOT / "scripts"


def _load_checker() -> ModuleType:
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    spec = importlib.util.spec_from_file_location(
        "stage7b_check_architecture",
        SCRIPTS_ROOT / "check_architecture.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


CHECKER = _load_checker()


def _verification_payload() -> dict[str, object]:
    return {
        "verified": True,
        "version": "reference/v1",
        "verification_digest": "sha256:verification",
        "primary_failure_code": None,
        "gates": [],
        "unavailable_checks": [],
        "authoritative_roots": {
            "benchmark_contract_identity": {"benchmark": "identity"},
            "policy_artifact_id": "policy:1",
            "root_seed_digest": "sha256:seed",
            "episode_family_registry_digest": "sha256:families",
            "reachability_tile_digest": "sha256:tile",
            "provider_versions": {"P1": "p1", "P2": "p2", "P3": "p3"},
        },
        "final_access_measurements": {
            "final_observation_materialization_count": 0,
            "final_provider_score_access_count": 0,
            "final_reachability_execution_count": 0,
            "calibration_execution_count": 0,
            "architecture_selection_execution_count": 0,
            "candidate_tuning_execution_count": 0,
            "final_evaluation_count": 0,
        },
        "measured_counts": {
            "reachability_replay_count": 0,
            "forbidden_final_access_counter": 0,
        },
    }


def _passing_repeated_audit() -> dict[str, object]:
    audit = {
        "status": "passed",
        "declared_mutation_count": 93,
        "executable_mutation_count": 93,
        "expected_detection_count": 91,
        "expected_mutation_count": 91,
        "detected_mutation_count": 91,
        "missed_mutation_count": 0,
        "undetected_mutation_count": 0,
        "unexpected_failure_code_count": 0,
        "invariant_count": 2,
        "invariant_pass_count": 2,
        "digest_laundering_class_closure": {name: {} for name in "abcdefg"},
        "mutation_isolation_passed": True,
        "mutation_audit_digest": "sha256:audit",
    }
    return {"deterministic": True, "audit": audit}


def _set_path(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


def test_conservative_closure_pass_and_unavailable_statuses() -> None:
    passed = verification.build_verification_closure(
        verification=_verification_payload(),
        repeated_mutation_audit=_passing_repeated_audit(),
        read_only={"status": "passed", "read_only": True},
        split_plan_identities={"selection": "sha256:plan"},
    )
    unavailable = verification.build_verification_closure(
        verification=_verification_payload(),
        repeated_mutation_audit=verification.build_unavailable_repeated_mutation_audit(),
        read_only={"status": "passed", "read_only": True},
        split_plan_identities={},
    )

    assert passed["supported_status"] == "reference_instrument_correct"
    assert unavailable["supported_status"] == (
        "reference_instrument_correctness_unresolved"
    )
    assert list(passed)[-1] == "closure_report_digest"


@pytest.mark.parametrize(
    ("component", "path", "value"),
    [
        ("verification", ("verified",), False),
        ("verification", ("unavailable_checks",), ["semantic_outcome"]),
        (
            "verification",
            ("final_access_measurements", "final_observation_materialization_count"),
            1,
        ),
        (
            "verification",
            ("final_access_measurements", "final_provider_score_access_count"),
            1,
        ),
        (
            "verification",
            ("final_access_measurements", "final_reachability_execution_count"),
            1,
        ),
        (
            "verification",
            ("final_access_measurements", "calibration_execution_count"),
            1,
        ),
        (
            "verification",
            ("final_access_measurements", "architecture_selection_execution_count"),
            1,
        ),
        (
            "verification",
            ("final_access_measurements", "candidate_tuning_execution_count"),
            1,
        ),
        (
            "verification",
            ("final_access_measurements", "final_evaluation_count"),
            1,
        ),
        ("repeated", ("deterministic",), False),
        ("repeated", ("audit", "status"), "failed"),
        ("repeated", ("audit", "declared_mutation_count"), 92),
        ("repeated", ("audit", "executable_mutation_count"), 92),
        ("repeated", ("audit", "expected_detection_count"), 90),
        ("repeated", ("audit", "detected_mutation_count"), 90),
        ("repeated", ("audit", "missed_mutation_count"), 1),
        ("repeated", ("audit", "unexpected_failure_code_count"), 1),
        ("repeated", ("audit", "invariant_count"), 1),
        ("repeated", ("audit", "invariant_pass_count"), 1),
        ("repeated", ("audit", "digest_laundering_class_closure"), {}),
        ("repeated", ("audit", "mutation_isolation_passed"), False),
        ("read_only", ("status",), "failed"),
    ],
)
def test_each_failed_or_unavailable_prerequisite_keeps_closure_unresolved(
    component: str,
    path: tuple[str, ...],
    value: Any,
) -> None:
    verification_payload: dict[str, Any] = _verification_payload()
    repeated_payload: dict[str, Any] = _passing_repeated_audit()
    read_only_payload: dict[str, Any] = {
        "status": "passed",
        "read_only": True,
    }
    targets = {
        "verification": verification_payload,
        "repeated": repeated_payload,
        "read_only": read_only_payload,
    }
    _set_path(targets[component], path, value)

    closure = verification.build_verification_closure(
        verification=verification_payload,
        repeated_mutation_audit=repeated_payload,
        read_only=read_only_payload,
        split_plan_identities={},
    )

    assert closure["supported_status"] == (
        "reference_instrument_correctness_unresolved"
    )


def test_verification_summary_preserves_legacy_projection() -> None:
    closure = verification.build_verification_closure(
        verification=_verification_payload(),
        repeated_mutation_audit=_passing_repeated_audit(),
        read_only={"status": "passed", "read_only": True},
        split_plan_identities={},
    )

    summary = verification.verification_summary(closure)

    assert list(summary) == [
        "verified",
        "version",
        "closure_report_version",
        "repository_status",
        "materialization_status",
        "primary_failure_code",
        "gates",
        "final_materialization_count",
        "final_score_access_count",
        "final_reachability_execution_count",
        "candidate_set_selection_count",
        "conformal_calibration_count",
        "reachability_replay_count",
        "final_evaluation_count",
        "forbidden_final_access_counter",
        "read_only",
        "verification_digest",
    ]
    assert summary["repository_status"] == "reference_instrument_correct"


@pytest.mark.parametrize(
    ("importer", "imported"),
    [
        ("zeromodel.domains.video_action_set.reference_verification", "pathlib"),
        ("zeromodel.domains.video_action_set.reference_verification", "sqlite3"),
        ("zeromodel.domains.video_action_set.evidence_audit", "sqlalchemy"),
        (
            "zeromodel.domains.video_action_set.mutation_audit",
            "zeromodel.runtime",
        ),
        (
            "zeromodel.domains.video_action_set.mutation_matrix",
            "zeromodel.domains.video_action_set.verification",
        ),
        (
            "research.video_action_set.provider_measurement",
            "zeromodel.domains.video_action_set.evidence_audit",
        ),
        (
            "zeromodel.runtime",
            "zeromodel.domains.video_action_set.reference_verification",
        ),
    ],
)
def test_stage7b_architecture_rejects_representative_edges(
    importer: str,
    imported: str,
) -> None:
    edge = CHECKER.ImportEdge(importer=importer, imported=imported, line=7)

    violations = CHECKER.forbidden_edge_violations([edge])

    assert any(
        violation.importer == importer and violation.imported == imported
        for violation in violations
    )


def test_stage7b_architecture_allows_expected_direction() -> None:
    edge = CHECKER.ImportEdge(
        importer="zeromodel.domains.video_action_set.verification",
        imported="zeromodel.domains.video_action_set.mutation_matrix",
        line=7,
    )

    assert CHECKER.forbidden_edge_violations([edge]) == []


@pytest.mark.parametrize(
    ("importer", "source", "expected_import"),
    [
        (
            "zeromodel.domains.video_action_set.reference_verification",
            "import os\n",
            "os",
        ),
        (
            "zeromodel.domains.video_action_set.evidence_audit",
            "from shutil import copytree\n",
            "shutil",
        ),
        (
            "zeromodel.domains.video_action_set.mutation_audit",
            "import tempfile\n",
            "tempfile",
        ),
        (
            "zeromodel.domains.video_action_set.mutation_matrix",
            "from subprocess import run\n",
            "subprocess",
        ),
        (
            "zeromodel.domains.video_action_set.verification",
            "import csv\n",
            "csv",
        ),
    ],
)
def test_stage7b_external_imports_are_collected_and_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    importer: str,
    source: str,
    expected_import: str,
) -> None:
    module_path = tmp_path / "module.py"
    module_path.write_text(source, encoding="utf-8")
    monkeypatch.setattr(CHECKER, "relative_path", lambda path: path.name)

    edges = CHECKER.collect_import_edges(importer, module_path, {importer})

    assert [(edge.importer, edge.imported) for edge in edges] == [
        (importer, expected_import)
    ]
    assert CHECKER.forbidden_edge_violations(edges)
