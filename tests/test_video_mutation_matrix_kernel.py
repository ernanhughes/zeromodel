from __future__ import annotations

from copy import deepcopy

import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set import mutation_matrix


def _finding(
    case: dict[str, object],
    effect: str,
    *,
    primary: str | None = None,
) -> dict[str, object]:
    expected_detected = case["expected_result_type"] == "detected"
    actual_primary = (
        case["expected_primary_failure_code"]
        if primary is None and expected_detected
        else primary
    )
    return {
        "mutation": case["mutation_id"],
        "expected_detected": expected_detected,
        "detected": actual_primary is not None,
        "expected_code_matched": (
            actual_primary == case["expected_primary_failure_code"]
            if expected_detected
            else actual_primary is None
        ),
        "actual_primary_failure_code": actual_primary,
        "secondary_failure_codes": [],
        "mutation_isolation": {
            "changed_field_count": 1,
            "isolation_passed": True,
            "mutation_effect_digest": effect,
        },
    }


def _audit(
    rows: list[dict[str, object]],
    *,
    status: str = "passed",
    base_verified: bool = True,
    executable_count: int | None = None,
) -> dict[str, object]:
    return {
        "status": status,
        "base_verified": base_verified,
        "declared_mutation_count": 93,
        "executable_mutation_count": (
            len(rows) if executable_count is None else executable_count
        ),
        "mutations": rows,
    }


def test_mutation_catalogue_schema_and_counts_are_frozen() -> None:
    catalogue = mutation_matrix.mutation_catalogue()

    assert mutation_matrix.validate_mutation_catalogue() == []
    assert len(catalogue) == 93
    assert sum(case["expected_result_type"] == "detected" for case in catalogue) == 91
    assert (
        sum(case["expected_result_type"] == "semantic_invariant" for case in catalogue)
        == 2
    )
    assert list(catalogue[0]) == [
        "matrix_version",
        "mutation_id",
        "artifact_class",
        "protected_scientific_property",
        "fixture_selector",
        "mutator_id",
        "immediate_digest_recomputed",
        "parent_digest_recomputed",
        "expected_result_type",
        "expected_primary_failure_code",
        "permitted_secondary_failure_codes",
        "expected_changed_files",
        "validation_metadata",
    ]


def test_mutation_matrix_uses_registry_order_without_reexecution() -> None:
    catalogue = mutation_matrix.mutation_catalogue()
    first_case, second_case = catalogue[:2]
    first = str(first_case["mutation_id"])
    second = str(second_case["mutation_id"])
    audit = _audit(
        [
            _finding(second_case, "sha256:second"),
            _finding(first_case, "sha256:first"),
        ]
    )

    payload = mutation_matrix.build_mutation_matrix(audit)

    assert payload["mutation_ids"] == [first, second]
    assert [row["mutation"] for row in payload["mutations"]] == [first, second]
    assert payload["matrix_digest"] == (
        "sha256:88861c3d42cc8ab7e5b51dc0ef565294faa3d1c4f1c42307f0856ae421b94e47"
    )


def test_mutation_matrix_rejects_duplicate_unknown_and_missing_results() -> None:
    catalogue = mutation_matrix.mutation_catalogue()
    first_case = catalogue[0]

    duplicate = _audit(
        [
            _finding(first_case, "sha256:first"),
            _finding(first_case, "sha256:second"),
        ]
    )
    unknown_row = _finding(first_case, "sha256:unknown")
    unknown_row["mutation"] = "unknown-mutation"
    unknown = _audit([unknown_row])
    missing = _audit([_finding(first_case, "sha256:first")], executable_count=2)

    with pytest.raises(VPMValidationError, match="duplicate result ids"):
        mutation_matrix.build_mutation_matrix(duplicate)
    with pytest.raises(VPMValidationError, match="unknown result ids"):
        mutation_matrix.build_mutation_matrix(unknown)
    with pytest.raises(VPMValidationError, match="result count"):
        mutation_matrix.build_mutation_matrix(missing)


def test_mutation_matrix_rejects_malformed_result_and_ambiguous_id() -> None:
    case = mutation_matrix.mutation_catalogue()[0]
    mutation_id = str(case["mutation_id"])
    malformed = _finding(case, "sha256:malformed")
    del malformed["mutation_isolation"]
    ambiguous = _finding(case, "sha256:ambiguous")
    ambiguous["mutation_id"] = mutation_id

    with pytest.raises(VPMValidationError, match="missing required fields"):
        mutation_matrix.build_mutation_matrix(_audit([malformed]))
    with pytest.raises(VPMValidationError, match="exactly one"):
        mutation_matrix.build_mutation_matrix(_audit([ambiguous]))


@pytest.mark.parametrize(
    "audit",
    [
        _audit([], status="unavailable", executable_count=93),
        _audit([], base_verified=False, executable_count=93),
    ],
)
def test_mutation_matrix_rejects_unavailable_baselines(
    audit: dict[str, object],
) -> None:
    with pytest.raises(VPMValidationError, match="unavailable baseline"):
        mutation_matrix.build_mutation_matrix(audit)


def test_mutation_matrix_preserves_wrong_and_multiple_detectors() -> None:
    catalogue = mutation_matrix.mutation_catalogue()
    first = _finding(
        catalogue[0],
        "sha256:first",
        primary="ranking_reconstruction_mismatch",
    )
    second = _finding(catalogue[1], "sha256:second")
    second["secondary_failure_codes"] = [
        {"gate": "structural_identity", "code": "ranking_reconstruction_mismatch"},
        {"gate": "semantic_outcome", "code": "semantic_status_mismatch"},
    ]

    payload = mutation_matrix.build_mutation_matrix(
        _audit([second, first], status="failed")
    )

    by_id = {row["mutation"]: row for row in payload["mutations"]}
    assert by_id[first["mutation"]]["expected_code_matched"] is False
    assert by_id[first["mutation"]]["actual_primary_failure_code"] == (
        "ranking_reconstruction_mismatch"
    )
    assert (
        by_id[second["mutation"]]["secondary_failure_codes"]
        == second["secondary_failure_codes"]
    )

    with pytest.raises(VPMValidationError, match="passed.*failing result"):
        mutation_matrix.build_mutation_matrix(_audit([first]))


@pytest.mark.parametrize(
    "contradiction",
    [
        "detected_false_with_primary",
        "detected_true_without_primary",
        "forged_expected_code_match",
        "wrong_expected_detected",
        "isolation_without_change",
        "non_string_primary",
        "semantic_invariant_contradiction",
    ],
)
def test_mutation_matrix_rejects_internally_contradictory_results(
    contradiction: str,
) -> None:
    catalogue = mutation_matrix.mutation_catalogue()
    case = catalogue[0]
    row = _finding(case, "sha256:contradiction")
    if contradiction == "detected_false_with_primary":
        row["detected"] = False
    elif contradiction == "detected_true_without_primary":
        row["actual_primary_failure_code"] = None
    elif contradiction == "forged_expected_code_match":
        row = _finding(
            case,
            "sha256:contradiction",
            primary="ranking_reconstruction_mismatch",
        )
        row["expected_code_matched"] = True
    elif contradiction == "wrong_expected_detected":
        row["expected_detected"] = False
    elif contradiction == "isolation_without_change":
        row["mutation_isolation"]["changed_field_count"] = 0  # type: ignore[index]
    elif contradiction == "non_string_primary":
        row["actual_primary_failure_code"] = 7
    else:
        case = next(
            item
            for item in catalogue
            if item["expected_result_type"] == "semantic_invariant"
        )
        row = _finding(case, "sha256:semantic-invariant")
        row["expected_code_matched"] = False

    with pytest.raises(VPMValidationError):
        mutation_matrix.build_mutation_matrix(_audit([row], status="failed"))


def test_failed_audit_accepts_accurately_represented_wrong_detector() -> None:
    case = mutation_matrix.mutation_catalogue()[0]
    row = _finding(
        case,
        "sha256:wrong-detector",
        primary="ranking_reconstruction_mismatch",
    )

    payload = mutation_matrix.build_mutation_matrix(_audit([row], status="failed"))

    assert payload["mutations"] == [row]
    assert payload["mutations"][0]["detected"] is True
    assert payload["mutations"][0]["expected_code_matched"] is False


def test_mutation_matrix_does_not_modify_audit_input() -> None:
    case = mutation_matrix.mutation_catalogue()[0]
    audit = _audit([_finding(case, "sha256:stable")])
    before = deepcopy(audit)

    mutation_matrix.build_mutation_matrix(audit)

    assert audit == before
