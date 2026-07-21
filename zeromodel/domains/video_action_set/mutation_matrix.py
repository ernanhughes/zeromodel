from __future__ import annotations

from typing import Any, Mapping

from ...artifact import VPMValidationError
from .canonical_json import canonical_sha256
from .mutation_audit import (
    _MUTATION_CASES,
    _MUTATION_GATE_SCOPE,
    _mutation_expected_files,
    _mutation_property,
)

MUTATION_MATRIX_VERSION = "zeromodel-video-action-set-reference-mutation-matrix/v3"

_REQUIRED_RESULT_FIELDS = {
    "actual_primary_failure_code",
    "detected",
    "expected_code_matched",
    "expected_detected",
    "mutation_isolation",
    "secondary_failure_codes",
}
_REQUIRED_ISOLATION_FIELDS = {
    "changed_field_count",
    "isolation_passed",
    "mutation_effect_digest",
}


def mutation_catalogue() -> list[dict[str, Any]]:
    catalogue = []
    for case in _MUTATION_CASES:
        expected = case.get("expected_primary_failure_code")
        invariant = bool(case.get("invariant"))
        catalogue.append(
            {
                "matrix_version": MUTATION_MATRIX_VERSION,
                "mutation_id": case["name"],
                "artifact_class": case["artifact_class"],
                "protected_scientific_property": _mutation_property(case),
                "fixture_selector": case.get(
                    "fixture_selector",
                    f"{case['artifact_class']}:deterministic-first-matching-record",
                ),
                "mutator_id": f"reference-mutator:{case['name']}",
                "immediate_digest_recomputed": bool(
                    case.get("digest_laundering", False)
                ),
                "parent_digest_recomputed": bool(case.get("digest_laundering", False))
                or case["artifact_class"]
                in {
                    "evidence",
                    "semantic",
                    "observation",
                    "family_output",
                    "reachability_trace",
                },
                "expected_result_type": (
                    "semantic_invariant" if invariant else "detected"
                ),
                "expected_primary_failure_code": expected,
                "permitted_secondary_failure_codes": list(
                    case.get("permitted_secondary_failure_codes", [])
                ),
                "expected_changed_files": list(_mutation_expected_files(case)),
                "validation_metadata": {
                    "digest_laundering": bool(case.get("digest_laundering", False)),
                    "gate_scope": list(
                        case.get(
                            "gate_scope",
                            _MUTATION_GATE_SCOPE.get(
                                str(case["artifact_class"]),
                                (),
                            ),
                        )
                    ),
                },
            }
        )
    return catalogue


def validate_mutation_catalogue() -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    catalogue = mutation_catalogue()
    mutation_ids = [str(case["mutation_id"]) for case in catalogue]
    if len(mutation_ids) != len(set(mutation_ids)):
        findings.append(
            _finding(
                "duplicate_mutation_id",
                "mutation catalogue contains duplicate mutation ids",
            )
        )
    declared_ids = set(mutation_ids)
    mutator_ids = {
        str(case["mutation_id"]) for case in catalogue if case.get("mutator_id")
    }
    for missing in sorted(declared_ids - mutator_ids):
        findings.append(
            _finding(
                "mutation_mutator_missing",
                "declared mutation has no executable mutator",
                mutation=missing,
            )
        )
    for extra in sorted(mutator_ids - declared_ids):
        findings.append(
            _finding(
                "mutation_mutator_orphan",
                "mutator has no declaration",
                mutation=extra,
            )
        )
    for case in catalogue:
        if case["expected_result_type"] == "detected" and not case.get(
            "expected_primary_failure_code"
        ):
            findings.append(
                _finding(
                    "mutation_expected_code_missing",
                    "expected detection lacks an expected primary failure code",
                    mutation=case["mutation_id"],
                )
            )
        if not case.get("expected_changed_files"):
            findings.append(
                _finding(
                    "mutation_expected_change_missing",
                    "mutation lacks expected changed file metadata",
                    mutation=case["mutation_id"],
                )
            )
    detected = sum(
        1 for case in catalogue if case["expected_result_type"] == "detected"
    )
    invariants = sum(
        1 for case in catalogue if case["expected_result_type"] == "semantic_invariant"
    )
    if len(catalogue) != 93 or detected != 91 or invariants != 2:
        findings.append(
            _finding(
                "mutation_matrix_count_mismatch",
                "mutation matrix count changed without an explicit matrix revision",
                declared_count=len(catalogue),
                detected_count=detected,
                invariant_count=invariants,
            )
        )
    return findings


def _finding(code: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "message": message, **details}


def _mutation_result_id(row: Mapping[str, Any]) -> str:
    id_fields = [field for field in ("mutation", "mutation_id") if field in row]
    if len(id_fields) != 1 or not str(row[id_fields[0]]):
        raise VPMValidationError(
            "mutation matrix result must contain exactly one non-empty mutation id"
        )
    return str(row[id_fields[0]])


def _validate_matrix_audit(audit: Mapping[str, Any]) -> list[dict[str, Any]]:
    if audit.get("status") == "unavailable" or audit.get("base_verified") is False:
        raise VPMValidationError(
            "mutation matrix cannot be built from an unavailable baseline audit"
        )
    if audit.get("status") not in {"passed", "failed"}:
        raise VPMValidationError("mutation audit status must be passed or failed")
    if audit.get("base_verified") is not True:
        raise VPMValidationError("mutation audit must declare a verified baseline")
    catalogue_ids = [str(case["mutation_id"]) for case in mutation_catalogue()]
    declared_count = audit.get("declared_mutation_count")
    executable_count = audit.get("executable_mutation_count")
    if declared_count != len(catalogue_ids):
        raise VPMValidationError(
            "mutation audit declared count does not match the mutation catalogue"
        )
    if type(executable_count) is not int or not 0 <= executable_count <= len(
        catalogue_ids
    ):
        raise VPMValidationError("mutation audit executable count is invalid")
    raw_rows = audit.get("mutations")
    if not isinstance(raw_rows, (list, tuple)):
        raise VPMValidationError("mutation audit results must be a sequence")
    rows = [dict(row) for row in raw_rows if isinstance(row, Mapping)]
    if len(rows) != len(raw_rows):
        raise VPMValidationError("mutation audit contains a malformed result")
    if len(rows) != executable_count:
        raise VPMValidationError(
            "mutation audit result count does not match executable count"
        )
    result_ids = [_mutation_result_id(row) for row in rows]
    if len(result_ids) != len(set(result_ids)):
        raise VPMValidationError("mutation audit contains duplicate result ids")
    unknown = sorted(set(result_ids) - set(catalogue_ids))
    if unknown:
        raise VPMValidationError(
            f"mutation audit contains unknown result ids: {', '.join(unknown)}"
        )
    if executable_count == len(catalogue_ids) and set(result_ids) != set(catalogue_ids):
        raise VPMValidationError("complete mutation audit omits required result ids")
    for row in rows:
        missing = sorted(_REQUIRED_RESULT_FIELDS - row.keys())
        if missing:
            raise VPMValidationError(
                f"mutation result is missing required fields: {', '.join(missing)}"
            )
        isolation = row["mutation_isolation"]
        if not isinstance(isolation, Mapping):
            raise VPMValidationError("mutation result isolation must be a mapping")
        missing_isolation = sorted(_REQUIRED_ISOLATION_FIELDS - isolation.keys())
        if missing_isolation:
            raise VPMValidationError(
                "mutation result isolation is missing required fields: "
                + ", ".join(missing_isolation)
            )
        if not isinstance(row["secondary_failure_codes"], (list, tuple)):
            raise VPMValidationError(
                "mutation result secondary failure codes must be a sequence"
            )
        for field in ("detected", "expected_code_matched", "expected_detected"):
            if type(row[field]) is not bool:
                raise VPMValidationError(f"mutation result {field} must be a boolean")
        if any(
            not isinstance(item, Mapping)
            or not isinstance(item.get("code"), str)
            or not isinstance(item.get("gate"), str)
            for item in row["secondary_failure_codes"]
        ):
            raise VPMValidationError(
                "mutation result secondary failure codes are malformed"
            )
        if (
            type(isolation["changed_field_count"]) is not int
            or int(isolation["changed_field_count"]) < 0
        ):
            raise VPMValidationError(
                "mutation result changed field count must be non-negative"
            )
        if type(isolation["isolation_passed"]) is not bool:
            raise VPMValidationError(
                "mutation result isolation status must be a boolean"
            )
        if (
            not isinstance(isolation["mutation_effect_digest"], str)
            or not isolation["mutation_effect_digest"]
        ):
            raise VPMValidationError(
                "mutation result effect digest must be a non-empty string"
            )
    if audit["status"] == "passed" and any(
        not row["expected_code_matched"]
        or not row["mutation_isolation"]["isolation_passed"]
        for row in rows
    ):
        raise VPMValidationError("passed mutation audit contains a failing result")
    return rows


def build_mutation_matrix(audit: Mapping[str, Any]) -> dict[str, Any]:
    """Project mutation findings into their stable registry order."""

    validated = _validate_matrix_audit(audit)
    findings = {_mutation_result_id(row): row for row in validated}
    rows = [
        findings[case["mutation_id"]]
        for case in mutation_catalogue()
        if case["mutation_id"] in findings
    ]
    payload = {
        "version": MUTATION_MATRIX_VERSION,
        "mutation_count": len(rows),
        "mutation_ids": [row.get("mutation") or row.get("mutation_id") for row in rows],
        "mutations": rows,
    }
    payload["matrix_digest"] = canonical_sha256(payload)
    return payload


__all__ = [
    "MUTATION_MATRIX_VERSION",
    "build_mutation_matrix",
    "mutation_catalogue",
    "validate_mutation_catalogue",
]
