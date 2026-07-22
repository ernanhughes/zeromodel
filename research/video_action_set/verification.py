from __future__ import annotations

from typing import Any, Mapping

from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from research.video_action_set.mutation_audit import MUTATION_AUDIT_VERSION, _MUTATION_CASES
from research.video_action_set.mutation_matrix import MUTATION_MATRIX_VERSION

CLOSURE_REPORT_VERSION = "zeromodel-video-action-set-reference-closure/v1"


def build_unavailable_repeated_mutation_audit() -> dict[str, Any]:
    detected = [case for case in _MUTATION_CASES if not case.get("invariant")]
    invariants = [case for case in _MUTATION_CASES if case.get("invariant")]
    audit = {
        "version": MUTATION_AUDIT_VERSION,
        "matrix_version": MUTATION_MATRIX_VERSION,
        "status": "unavailable",
        "declared_mutation_count": len(_MUTATION_CASES),
        "executable_mutation_count": len(_MUTATION_CASES),
        "expected_detection_count": len(detected),
        "expected_mutation_count": len(detected),
        "detected_mutation_count": 0,
        "missed_mutation_count": len(detected),
        "undetected_mutation_count": len(detected),
        "unexpected_failure_code_count": 0,
        "invariant_count": len(invariants),
        "invariant_pass_count": 0,
        "invariant_failure_count": len(invariants),
        "mutations": [],
        "digest_laundering_tests": [],
        "digest_laundering_class_closure": {},
        "mutation_isolation_passed": False,
        "mutation_audit_digest": None,
    }
    return {
        "version": "zeromodel-video-action-set-reference-mutation-audit-repeat/v1",
        "matrix_version": MUTATION_MATRIX_VERSION,
        "deterministic": False,
        "first_audit_digest": None,
        "second_audit_digest": None,
        "audit": audit,
    }


def build_verification_closure(
    *,
    verification: Mapping[str, Any],
    repeated_mutation_audit: Mapping[str, Any],
    read_only: Mapping[str, Any],
    split_plan_identities: Mapping[str, str],
) -> dict[str, Any]:
    mutation_audit = repeated_mutation_audit["audit"]
    final_counts = verification["final_access_measurements"]
    required_zero = _required_access_counts_are_zero(final_counts)
    supported = (
        verification["verified"]
        and mutation_audit.get("status") == "passed"
        and mutation_audit.get("declared_mutation_count") == 93
        and mutation_audit.get("executable_mutation_count") == 93
        and mutation_audit.get("expected_detection_count") == 91
        and mutation_audit.get("detected_mutation_count") == 91
        and mutation_audit.get("missed_mutation_count") == 0
        and mutation_audit.get("unexpected_failure_code_count") == 0
        and mutation_audit.get("invariant_count") == 2
        and mutation_audit.get("invariant_pass_count") == 2
        and len(mutation_audit.get("digest_laundering_class_closure", {})) >= 7
        and mutation_audit.get("mutation_isolation_passed") is True
        and repeated_mutation_audit.get("deterministic") is True
        and read_only["status"] == "passed"
        and required_zero
        and not verification["unavailable_checks"]
    )
    roots = verification["authoritative_roots"]
    payload: dict[str, Any] = {
        "version": CLOSURE_REPORT_VERSION,
        "contract_identity": roots["benchmark_contract_identity"],
        "policy_identity": roots["policy_artifact_id"],
        "root_seed_identity": roots["root_seed_digest"],
        "split_plan_identities": dict(split_plan_identities),
        "family_registry_identity": roots["episode_family_registry_digest"],
        "reachability_identity": roots["reachability_tile_digest"],
        "provider_identities": roots["provider_versions"],
        "verification": dict(verification),
        "mutation_audit": dict(mutation_audit),
        "repeated_mutation_audit": dict(repeated_mutation_audit),
        "read_only_verification": dict(read_only),
        "declared_mutation_count": mutation_audit.get("declared_mutation_count", 0),
        "executable_mutation_count": mutation_audit.get("executable_mutation_count", 0),
        "expected_detection_count": mutation_audit.get("expected_detection_count", 0),
        "expected_mutation_count": mutation_audit.get("expected_mutation_count", 0),
        "detected_mutation_count": mutation_audit.get("detected_mutation_count", 0),
        "missed_mutation_count": mutation_audit.get("missed_mutation_count", 0),
        "undetected_mutation_count": mutation_audit.get("undetected_mutation_count", 0),
        "unexpected_failure_code_count": mutation_audit.get(
            "unexpected_failure_code_count", 0
        ),
        "invariant_count": mutation_audit.get("invariant_count", 0),
        "invariant_pass_count": mutation_audit.get("invariant_pass_count", 0),
        "digest_laundering_class_closure": mutation_audit.get(
            "digest_laundering_class_closure", {}
        ),
        "mutation_audit_report_digest": mutation_audit.get("mutation_audit_digest"),
        "final_materialization_access_counts": final_counts,
        "supported_status": (
            "reference_instrument_correct"
            if supported
            else "reference_instrument_correctness_unresolved"
        ),
        "unsupported_statuses": [
            "materialization_ready",
            "benchmark_utility_verified",
            "provider_selected",
            "calibration_complete",
            "final_evaluation_complete",
        ],
        "materialization_status": "prospective_materialization_prohibited",
    }
    payload["closure_report_digest"] = canonical_sha256(payload)
    return payload


def _required_access_counts_are_zero(counts: Mapping[str, Any]) -> bool:
    return (
        counts["final_observation_materialization_count"] == 0
        and counts["final_provider_score_access_count"] == 0
        and counts["final_reachability_execution_count"] == 0
        and counts["calibration_execution_count"] == 0
        and counts["architecture_selection_execution_count"] == 0
        and counts["candidate_tuning_execution_count"] == 0
        and counts["final_evaluation_count"] == 0
    )


def verification_summary(closure: Mapping[str, Any]) -> dict[str, Any]:
    verification = closure["verification"]
    access = verification["final_access_measurements"]
    measured = verification["measured_counts"]
    return {
        "verified": verification["verified"],
        "version": verification["version"],
        "closure_report_version": closure["version"],
        "repository_status": closure["supported_status"],
        "materialization_status": closure["materialization_status"],
        "primary_failure_code": verification["primary_failure_code"],
        "gates": verification["gates"],
        "final_materialization_count": access[
            "final_observation_materialization_count"
        ],
        "final_score_access_count": access["final_provider_score_access_count"],
        "final_reachability_execution_count": access[
            "final_reachability_execution_count"
        ],
        "candidate_set_selection_count": access["candidate_tuning_execution_count"],
        "conformal_calibration_count": access["calibration_execution_count"],
        "reachability_replay_count": measured["reachability_replay_count"],
        "final_evaluation_count": access["final_evaluation_count"],
        "forbidden_final_access_counter": measured["forbidden_final_access_counter"],
        "read_only": closure["read_only_verification"]["read_only"],
        "verification_digest": verification["verification_digest"],
    }


__all__ = [
    "CLOSURE_REPORT_VERSION",
    "build_unavailable_repeated_mutation_audit",
    "build_verification_closure",
    "verification_summary",
]
