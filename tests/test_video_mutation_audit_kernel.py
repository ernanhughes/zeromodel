from __future__ import annotations

from copy import deepcopy

import research.benchmarks.video_action_set_benchmark as benchmark
from research.video_action_set import mutation_audit, mutation_matrix
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256


def _isolation(
    effect: str,
    *,
    changed: int = 1,
    passed: bool = True,
) -> dict[str, object]:
    return {
        "changed_fields": ["artifact.value"] if changed else [],
        "expected_changed_files": ["artifact.json"],
        "unexpected_changed_fields": [] if passed else ["other.value"],
        "changed_field_count": changed,
        "isolation_passed": passed,
        "mutation_effect_digest": effect,
    }


def _report(
    code: str | None,
    *,
    secondary: tuple[str, ...] = (),
) -> dict[str, object]:
    findings = [] if code is None else [{"code": code}]
    findings.extend({"code": item} for item in secondary)
    return {
        "primary_failure_code": code,
        "primary_failure_gate": None if code is None else "structural_identity",
        "gates": []
        if not findings
        else [{"gate": "structural_identity", "findings": findings}],
    }


def _evaluate(
    case: dict[str, object],
    *,
    code: str | None,
    effect: str,
    changed: int = 1,
    passed: bool = True,
    secondary: tuple[str, ...] = (),
    application_error: str | None = None,
) -> dict[str, object]:
    return mutation_audit.evaluate_mutation_case(
        case=case,
        report=_report(code, secondary=secondary),
        isolation=_isolation(effect, changed=changed, passed=passed),
        application_error=application_error,
    )


def test_mutation_registry_identity_and_legacy_aliases_are_frozen() -> None:
    names = [case["name"] for case in mutation_audit._MUTATION_CASES]

    assert len(names) == 93
    assert names[:3] == [
        "evidence_raw_score_preserve_quantized_bin",
        "evidence_raw_score_cross_quantization_boundary",
        "evidence_quantized_score_changed",
    ]
    assert names[-3:] == [
        "access_change_failed_gate_status_to_passed",
        "access_change_repository_status_to_correct",
        "access_remove_required_gate_from_closure_report",
    ]
    assert canonical_sha256(mutation_audit._MUTATION_CASES) == (
        "sha256:49884af315dc025814fff61d63f64cf8c15bea0a6bbe4422792e3aa4714dc3bc"
    )
    assert benchmark._changed_fields is mutation_audit._changed_fields
    assert (
        benchmark._mutation_isolation_report
        is mutation_audit._mutation_isolation_report
    )
    assert benchmark.mutation_catalogue is mutation_matrix.mutation_catalogue


def test_mutation_isolation_is_structural_and_read_only() -> None:
    before = {"record.json": {"nested": {"value": 1}, "stable": True}}
    after = deepcopy(before)
    after["record.json"]["nested"]["value"] = 2
    case = {
        "artifact_class": "policy",
        "name": "policy_alter_row_to_action_mapping",
        "expected_changed_files": ["record.json"],
    }
    original = deepcopy(before)

    result = mutation_audit._mutation_isolation_report(before, after, case)

    assert before == original
    assert result["changed_fields"] == ["record.json.nested.value"]
    assert result["unexpected_changed_fields"] == []
    assert result["isolation_passed"] is True


def test_mutation_case_expected_detector_precedence_is_frozen() -> None:
    case = mutation_matrix.mutation_catalogue()[0]
    report = {
        "primary_failure_code": case["expected_primary_failure_code"],
        "primary_failure_gate": "structural_identity",
        "gates": [
            {
                "gate": "structural_identity",
                "findings": [
                    {"code": case["expected_primary_failure_code"]},
                    {"code": "ranking_reconstruction_mismatch"},
                ],
            }
        ],
    }
    isolation = {
        "changed_field_count": 1,
        "isolation_passed": True,
        "mutation_effect_digest": "sha256:effect-1",
    }

    result = mutation_audit.evaluate_mutation_case(
        case=case,
        report=report,
        isolation=isolation,
    )

    assert result["expected_code_matched"] is True
    assert result["actual_primary_failure_code"] == "raw_diagnostic_digest_mismatch"
    assert result["secondary_failure_codes"] == [
        {"gate": "structural_identity", "code": "ranking_reconstruction_mismatch"}
    ]


def test_mutation_audit_baseline_failure_remains_unavailable() -> None:
    catalogue = mutation_matrix.mutation_catalogue()
    payload = mutation_audit.build_mutation_audit_payload(
        matrix_version=mutation_matrix.MUTATION_MATRIX_VERSION,
        catalogue=catalogue,
        selected_cases=catalogue[:2],
        catalogue_findings=[],
        results=[],
        base_verified=False,
        base_primary_failure_code="benchmark_manifest_mismatch",
    )

    assert payload["status"] == "unavailable"
    assert payload["base_primary_failure_code"] == "benchmark_manifest_mismatch"
    assert payload["missed_mutation_count"] == 2
    assert payload["mutations"] == []


def test_mutation_application_error_and_wrong_detector_are_not_success() -> None:
    case = mutation_matrix.mutation_catalogue()[0]
    application_error = _evaluate(
        case,
        code=None,
        effect="sha256:application",
        application_error="RuntimeError",
    )
    wrong_detector = _evaluate(
        case,
        code="ranking_reconstruction_mismatch",
        effect="sha256:wrong-detector",
        secondary=("tie_group_reconstruction_mismatch",),
    )

    assert application_error["actual_primary_failure_code"] == (
        "mutation_application_error"
    )
    assert application_error["application_error"] == "RuntimeError"
    assert application_error["expected_code_matched"] is False
    assert wrong_detector["detected"] is True
    assert wrong_detector["expected_code_matched"] is False
    assert wrong_detector["secondary_failure_codes"] == [
        {
            "gate": "structural_identity",
            "code": "tie_group_reconstruction_mismatch",
        }
    ]


def test_semantic_invariant_success_and_failure_are_distinct() -> None:
    invariant = next(
        case
        for case in mutation_matrix.mutation_catalogue()
        if case["expected_result_type"] == "semantic_invariant"
    )
    success = _evaluate(
        invariant,
        code=None,
        effect="sha256:invariant-success",
    )
    failure = _evaluate(
        invariant,
        code="semantic_status_mismatch",
        effect="sha256:invariant-failure",
    )

    assert success["semantic_invariant_passed"] is True
    assert success["expected_code_matched"] is True
    assert failure["semantic_invariant_passed"] is False
    assert failure["expected_code_matched"] is False


def test_duplicate_effects_property_absence_and_isolation_failure_fail_audit() -> None:
    catalogue = mutation_matrix.mutation_catalogue()
    first, second = catalogue[:2]
    duplicate_results = [
        _evaluate(
            first,
            code=str(first["expected_primary_failure_code"]),
            effect="sha256:duplicate",
        ),
        _evaluate(
            second,
            code=str(second["expected_primary_failure_code"]),
            effect="sha256:duplicate",
        ),
    ]
    duplicate = mutation_audit.build_mutation_audit_payload(
        matrix_version=mutation_matrix.MUTATION_MATRIX_VERSION,
        catalogue=catalogue,
        selected_cases=catalogue[:2],
        catalogue_findings=[],
        results=duplicate_results,
    )
    no_property_change = mutation_audit.build_mutation_audit_payload(
        matrix_version=mutation_matrix.MUTATION_MATRIX_VERSION,
        catalogue=catalogue,
        selected_cases=catalogue[:1],
        catalogue_findings=[],
        results=[
            _evaluate(
                first,
                code=str(first["expected_primary_failure_code"]),
                effect="sha256:no-property",
                changed=0,
            )
        ],
    )
    isolation_failure = mutation_audit.build_mutation_audit_payload(
        matrix_version=mutation_matrix.MUTATION_MATRIX_VERSION,
        catalogue=catalogue,
        selected_cases=catalogue[:1],
        catalogue_findings=[],
        results=[
            _evaluate(
                first,
                code=str(first["expected_primary_failure_code"]),
                effect="sha256:isolation-failure",
                passed=False,
            )
        ],
    )

    assert duplicate["status"] == "failed"
    assert [row["code"] for row in duplicate["duplicate_effect_findings"]] == [
        "duplicate_mutation_effect"
    ]
    assert no_property_change["property_change_failure_count"] == 1
    assert no_property_change["status"] == "failed"
    assert isolation_failure["mutation_isolation_failure_count"] == 1
    assert isolation_failure["status"] == "failed"


def test_repeated_mutation_audit_detects_nondeterminism() -> None:
    first = {"mutation_audit_digest": "sha256:first", "status": "passed"}
    second = {"mutation_audit_digest": "sha256:second", "status": "passed"}

    payload = mutation_audit.build_repeated_mutation_audit_payload(
        matrix_version=mutation_matrix.MUTATION_MATRIX_VERSION,
        first=first,
        second=second,
    )

    assert payload["deterministic"] is False
    assert payload["first_audit_digest"] == "sha256:first"
    assert payload["second_audit_digest"] == "sha256:second"
