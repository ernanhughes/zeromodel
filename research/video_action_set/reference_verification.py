from __future__ import annotations

from typing import Any, Mapping, Sequence

from zeromodel.core.artifact import VPMValidationError
from research.evidence.video_complete_row_evidence import (
    RowScore,
    VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION,
    build_complete_ranking,
    build_complete_row_evidence,
    build_semantic_top_set_outcome,
    quantize_similarity,
)
from research.video.video_prospective_providers import (
    PROSPECTIVE_P1_VERSION,
    PROSPECTIVE_P2_VERSION,
    PROSPECTIVE_P3_VERSION,
    PROSPECTIVE_PROVIDER_IDS,
)
from zeromodel.video.domains.video_action_set.canonical_json import canonical_json_value, canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import EPISODE_PLAN_VERSION
from zeromodel.video.domains.video_action_set.episode_families import episode_family_registry as _episode_family_registry
from zeromodel.video.domains.video_action_set.reachability_composition import REACHABILITY_TRACE_VERSION

REFERENCE_VERIFICATION_VERSION = "zeromodel-video-action-set-reference-verification/v1"
SEMANTIC_OUTCOME_VERSION = VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION


def _json_ready(value: Any) -> Any:
    return canonical_json_value(value)


def _sha256(value: Any) -> str:
    return canonical_sha256(value)


_NON_FINAL_SPLITS = ("development", "calibration", "selection")
_ALL_SPLITS = ("development", "calibration", "selection", "final")
_REQUIRED_VERIFICATION_GATES = (
    "structural_identity",
    "semantic_outcome",
    "seed_and_plan",
    "episode_regeneration",
    "family_contract",
    "reachability",
    "completeness_orphan",
    "access_prohibition",
)
_PRIMARY_GATE_PRECEDENCE = {
    name: index for index, name in enumerate(_REQUIRED_VERIFICATION_GATES)
}
_PRIMARY_FAILURE_CODE_PRECEDENCE = {
    code: index
    for index, code in enumerate(
        (
            "expected_file_missing",
            "benchmark_contract_identity_mismatch",
            "benchmark_manifest_mismatch",
            "policy_action_mapping_mismatch",
            "provider_contract_mismatch",
            "reachability_tile_mismatch",
            "score_quantizer_mismatch",
            "evidence_schema_mismatch",
            "closure_gate_missing",
            "score_row_universe_mismatch",
            "quantized_score_vector_mismatch",
            "raw_diagnostic_digest_mismatch",
            "ranking_reconstruction_mismatch",
            "tie_group_reconstruction_mismatch",
            "semantic_status_mismatch",
            "resolved_row_not_permitted",
            "resolved_action_not_permitted",
            "semantic_outcome_digest_mismatch",
            "episode_seed_derivation_mismatch",
            "episode_split_reassignment",
            "duplicate_episode_id",
            "final_observation_provenance_mismatch",
            "sealed_episode_identity_mismatch",
            "expected_record_missing",
            "orphan_observation_record",
            "frame_identity_mismatch",
            "gap_event_structure_mismatch",
            "observation_bytes_mismatch",
            "observation_digest_mismatch",
            "family_contract_violation",
            "family_regeneration_mismatch",
            "family_no_op",
            "transition_classification_mismatch",
            "gap_event_has_pixels",
            "control_byte_identity_mismatch",
            "control_denominator_leak",
            "control_provider_visible_leak",
            "control_hidden_history_not_ambiguous",
            "control_hidden_label_not_ambiguous",
            "control_hidden_history_cardinality_mismatch",
            "family_disposition_mismatch",
            "frame_disposition_mismatch",
            "invalid_family_valid_state_collision",
            "invalid_family_valid_transformation_collision",
            "conflicting_action_evidence_absent",
            "consulted_edge_mismatch",
            "reachable_pair_mismatch",
            "reachability_trace_mismatch",
            "executed_action_mismatch",
            "duplicate_observation_record",
            "duplicate_observation_identity_unpermitted",
            "orphan_score_vector_record",
            "forbidden_final_materialization",
            "forbidden_final_score_access",
            "forbidden_final_reachability_access",
            "forbidden_calibration_execution",
            "forbidden_selection_execution",
            "forbidden_final_evaluation",
            "status_claim_not_supported",
        )
    )
}


def policy_row_action_digest(
    policy_artifact_id: str, row_ids: Sequence[str], row_actions: Mapping[str, str]
) -> str:
    return _sha256(
        {
            "policy_artifact_id": policy_artifact_id,
            "row_action": [
                {"row_id": row_id, "action_id": row_actions[row_id]}
                for row_id in row_ids
            ],
        }
    )


def build_reference_context(
    *,
    identity: Any,
    policy: Any,
    row_ids: Sequence[str],
    row_actions: Mapping[str, str],
    reachability_tile: Mapping[str, Any],
    plans: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    """Build the verifier context from authoritative, already-loaded inputs."""

    normalized_row_ids = [str(row_id) for row_id in row_ids]
    return {
        "identity": identity,
        "policy": policy,
        "row_ids": normalized_row_ids,
        "row_actions": dict(row_actions),
        "reachability_tile": dict(reachability_tile),
        "plans": {split: list(split_plans) for split, split_plans in plans.items()},
        "policy_row_action_digest": policy_row_action_digest(
            policy.artifact_id, normalized_row_ids, row_actions
        ),
    }


_policy_row_action_digest = policy_row_action_digest


def _finding(code: str, message: str, **details: Any) -> dict[str, Any]:
    payload = {"code": code, "message": message}
    payload.update(
        {key: _json_ready(value) for key, value in details.items() if value is not None}
    )
    return payload


def _gate(
    name: str,
    findings: Sequence[Mapping[str, Any]],
    *,
    counts: Mapping[str, Any] | None = None,
    unavailable: bool = False,
) -> dict[str, Any]:
    status = "unavailable" if unavailable else ("failed" if findings else "passed")
    return {
        "gate": name,
        "status": status,
        "finding_count": len(findings),
        "findings": [dict(item) for item in findings],
        "counts": dict(counts or {}),
    }


def _first_failure_code(report: Mapping[str, Any]) -> str | None:
    primary = _primary_failure(report)
    return None if primary is None else str(primary["code"])


def _primary_failure(report: Mapping[str, Any]) -> dict[str, Any] | None:
    candidates = []
    for gate in report.get("gates", []):
        gate_name = str(gate.get("gate"))
        for finding in gate.get("findings", []):
            code = str(finding["code"])
            candidates.append(
                (
                    _PRIMARY_GATE_PRECEDENCE.get(gate_name, 10_000),
                    _PRIMARY_FAILURE_CODE_PRECEDENCE.get(code, 10_000),
                    code,
                    gate_name,
                    dict(finding),
                )
            )
    if not candidates:
        return None
    _gate_index, _code_index, code, gate_name, finding = sorted(
        candidates, key=lambda item: item[:4]
    )[0]
    return {"code": code, "gate": gate_name, "finding": finding}


def _report_failure_codes(report: Mapping[str, Any]) -> list[dict[str, str]]:
    rows = []
    for gate in report.get("gates", []):
        for finding in gate.get("findings", []):
            rows.append(
                {"gate": str(gate.get("gate")), "code": str(finding.get("code"))}
            )
    return sorted(
        rows,
        key=lambda row: (
            _PRIMARY_GATE_PRECEDENCE.get(row["gate"], 10_000),
            _PRIMARY_FAILURE_CODE_PRECEDENCE.get(row["code"], 10_000),
            row["gate"],
            row["code"],
        ),
    )


def _raw_score_diagnostic_from_row(
    row: Mapping[str, Any], policy_row_ids: Sequence[str]
) -> str:
    scores = {
        str(row_id): float(score)
        for row_id, score in zip(row["all_112_row_ids"], row["all_112_raw_scores"])
    }
    return build_complete_row_evidence(
        row_scores=[(row_id, scores[row_id]) for row_id in policy_row_ids],
        policy_artifact_id=str(row["policy_artifact_id"]),
        provider_id=str(row["provider_id"]),
        provider_version=str(row["provider_version"]),
        policy_row_ids=policy_row_ids,
    ).raw_score_diagnostic_digest


def _stored_quantized_evidence(
    row: Mapping[str, Any], policy_row_ids: Sequence[str]
) -> tuple[Any | None, list[dict[str, Any]]]:
    vectors, findings = _validated_score_vectors(row, policy_row_ids)
    if vectors is None:
        return None, findings
    row_ids, raw_scores, quantized_scores = vectors
    evidence, build_findings = _build_stored_evidence(
        row,
        policy_row_ids,
        row_ids,
        raw_scores,
    )
    findings.extend(build_findings)
    if evidence is None:
        return None, findings
    findings.extend(_evidence_identity_findings(row, evidence))
    findings.extend(
        _ranking_findings(
            row,
            row_ids,
            raw_scores,
            quantized_scores,
        )
    )
    return evidence, findings


def _validated_score_vectors(
    row: Mapping[str, Any],
    policy_row_ids: Sequence[str],
) -> tuple[
    tuple[list[str], list[Any], list[Any]] | None,
    list[dict[str, Any]],
]:
    row_ids = [str(row_id) for row_id in row.get("all_112_row_ids", [])]
    raw_scores = list(row.get("all_112_raw_scores", []))
    quantized_scores = list(row.get("all_112_quantized_scores", []))
    if (
        row_ids != list(policy_row_ids)
        or len(row_ids) != 112
        or len(set(row_ids)) != 112
        or len(raw_scores) != 112
        or len(quantized_scores) != 112
    ):
        return None, [
            _finding(
                "score_row_universe_mismatch",
                "score vector does not contain the exact frozen 112-row policy universe",
                frame_id=row.get("frame_id"),
            )
        ]
    try:
        expected_quantized = [quantize_similarity(float(score)) for score in raw_scores]
    except VPMValidationError:
        return None, [
            _finding(
                "raw_diagnostic_digest_mismatch",
                "raw scores are not finite or quantizable",
                frame_id=row.get("frame_id"),
            )
        ]
    findings = []
    if [int(score) for score in quantized_scores] != expected_quantized:
        findings.append(
            _finding(
                "quantized_score_vector_mismatch",
                "stored quantized scores do not reconstruct from raw scores",
                frame_id=row.get("frame_id"),
            )
        )
    return (row_ids, raw_scores, quantized_scores), findings


def _build_stored_evidence(
    row: Mapping[str, Any],
    policy_row_ids: Sequence[str],
    row_ids: Sequence[str],
    raw_scores: Sequence[Any],
) -> tuple[Any | None, list[dict[str, Any]]]:
    try:
        evidence = build_complete_row_evidence(
            row_scores=[
                (row_id, float(score)) for row_id, score in zip(row_ids, raw_scores)
            ],
            policy_artifact_id=str(row["policy_artifact_id"]),
            provider_id=str(row["provider_id"]),
            provider_version=str(row["provider_version"]),
            policy_row_ids=policy_row_ids,
        )
    except (KeyError, VPMValidationError) as exc:
        return None, [
            _finding(
                "quantized_score_vector_mismatch",
                "complete row evidence could not be reconstructed",
                frame_id=row.get("frame_id"),
                error=str(exc),
            )
        ]
    return evidence, []


def _evidence_identity_findings(
    row: Mapping[str, Any],
    evidence: Any,
) -> list[dict[str, Any]]:
    findings = []
    stored_score_digest = str(
        row.get("quantized_score_vector_digest", row.get("score_vector_digest"))
    )
    if (
        stored_score_digest != evidence.quantized_score_vector_digest
        or str(row.get("score_vector_digest")) != evidence.quantized_score_vector_digest
    ):
        findings.append(
            _finding(
                "quantized_score_vector_mismatch",
                "quantized score-vector identity does not match reconstructed evidence",
                frame_id=row.get("frame_id"),
            )
        )
    if (
        row.get("raw_score_diagnostic_digest") is not None
        and str(row.get("raw_score_diagnostic_digest"))
        != evidence.raw_score_diagnostic_digest
    ):
        findings.append(
            _finding(
                "raw_diagnostic_digest_mismatch",
                "raw diagnostic identity does not match reconstructed raw scores",
                frame_id=row.get("frame_id"),
            )
        )
    return findings


def _ranking_findings(
    row: Mapping[str, Any],
    row_ids: Sequence[str],
    raw_scores: Sequence[Any],
    quantized_scores: Sequence[Any],
) -> list[dict[str, Any]]:
    ranking_scores = tuple(
        RowScore(row_id=row_id, raw_score=float(raw), quantized_score=int(quantized))
        for row_id, raw, quantized in zip(row_ids, raw_scores, quantized_scores)
    )
    expected_ranking = build_complete_ranking(ranking_scores)
    findings = []
    if list(row.get("complete_ordered_ranking", [])) != list(
        expected_ranking.ranked_row_ids
    ):
        findings.append(
            _finding(
                "ranking_reconstruction_mismatch",
                "stored ranking does not reconstruct from quantized scores",
                frame_id=row.get("frame_id"),
            )
        )
    if list(row.get("tie_groups", [])) != [
        group.to_dict() for group in expected_ranking.tie_groups
    ]:
        findings.append(
            _finding(
                "tie_group_reconstruction_mismatch",
                "stored tie groups do not reconstruct from quantized scores",
                frame_id=row.get("frame_id"),
            )
        )
    if (
        row.get("ranking_digest") is not None
        and str(row.get("ranking_digest")) != expected_ranking.ranking_digest
    ):
        findings.append(
            _finding(
                "ranking_reconstruction_mismatch",
                "ranking digest does not match reconstructed ranking",
                frame_id=row.get("frame_id"),
            )
        )
    return findings


def _semantic_cache_key(row: Mapping[str, Any]) -> str:
    return _sha256(
        {
            "policy_artifact_id": row.get("policy_artifact_id"),
            "provider_id": row.get("provider_id"),
            "provider_version": row.get("provider_version"),
            "all_112_row_ids": row.get("all_112_row_ids"),
            "all_112_raw_scores": row.get("all_112_raw_scores"),
            "all_112_quantized_scores": row.get("all_112_quantized_scores"),
            "complete_ordered_ranking": row.get("complete_ordered_ranking"),
            "tie_groups": row.get("tie_groups"),
            "score_vector_digest": row.get("score_vector_digest"),
            "quantized_score_vector_digest": row.get("quantized_score_vector_digest"),
            "raw_score_diagnostic_digest": row.get("raw_score_diagnostic_digest"),
            "ranking_digest": row.get("ranking_digest"),
        }
    )


def _expected_semantic_for_row(
    row: Mapping[str, Any],
    row_actions: Mapping[str, str],
    policy_row_ids: Sequence[str],
    cache: dict[str, Any] | None = None,
) -> tuple[Any | None, list[dict[str, Any]]]:
    key = _semantic_cache_key(row)
    if cache is not None and key in cache:
        return cache[key], []
    evidence, findings = _stored_quantized_evidence(row, policy_row_ids)
    if evidence is None:
        return None, findings
    try:
        outcome = build_semantic_top_set_outcome(
            evidence=evidence, row_action=row_actions
        )
    except VPMValidationError as exc:
        return None, findings + [
            _finding(
                "semantic_status_mismatch",
                "semantic outcome could not be reconstructed",
                frame_id=row.get("frame_id"),
                error=str(exc),
            )
        ]
    if cache is not None and not findings:
        cache[key] = outcome
    return outcome, findings


def compare_provider_results(
    *,
    provider_id: str,
    observation_id: str,
    reference: Any,
    optimized: Any,
) -> dict[str, Any]:
    """Compare one optimized result with its independently scored reference result."""

    quantized_equal = reference.quantized_scores == optimized.quantized_scores
    ranking_equal = (
        reference.evidence.ranking.ranked_row_ids
        == optimized.evidence.ranking.ranked_row_ids
    )
    tie_groups_equal = tuple(
        group.to_dict() for group in reference.evidence.ranking.tie_groups
    ) == tuple(group.to_dict() for group in optimized.evidence.ranking.tie_groups)
    digests_equal = (
        reference.evidence.score_vector_digest == optimized.evidence.score_vector_digest
        and reference.evidence.ranking.ranking_digest
        == optimized.evidence.ranking.ranking_digest
    )
    return {
        "provider_id": provider_id,
        "observation_id": observation_id,
        "quantized_equal": quantized_equal,
        "ranking_equal": ranking_equal,
        "tie_groups_equal": tie_groups_equal,
        "digests_equal": digests_equal,
    }


def build_provider_equivalence_payload(
    comparisons: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    mismatches: list[str] = []
    for provider_id in PROSPECTIVE_PROVIDER_IDS:
        rows = [row for row in comparisons if row.get("provider_id") == provider_id]
        provider_summary = {
            "quantized_mismatch_count": sum(
                int(not bool(row["quantized_equal"])) for row in rows
            ),
            "ranking_mismatch_count": sum(
                int(not bool(row["ranking_equal"])) for row in rows
            ),
            "tie_group_mismatch_count": sum(
                int(not bool(row["tie_groups_equal"])) for row in rows
            ),
            "digest_mismatch_count": sum(
                int(not bool(row["digests_equal"])) for row in rows
            ),
        }
        summary[provider_id] = provider_summary
        if any(provider_summary.values()):
            mismatches.append(provider_id)
    return {
        "providers_verified": not mismatches,
        "mismatching_providers": mismatches,
        "summary": summary,
    }


def build_reference_verification_payload(
    *,
    context: Mapping[str, Any],
    gates: Sequence[Mapping[str, Any]],
    phase_counts: Mapping[str, Any],
) -> dict[str, Any]:
    normalized_gates = [dict(gate) for gate in gates]
    passed = [gate["gate"] for gate in normalized_gates if gate["status"] == "passed"]
    failed = [gate["gate"] for gate in normalized_gates if gate["status"] == "failed"]
    unavailable = [
        gate["gate"] for gate in normalized_gates if gate["status"] == "unavailable"
    ]
    identity = context["identity"]
    policy = context["policy"]
    payload: dict[str, Any] = {
        "version": REFERENCE_VERIFICATION_VERSION,
        "authoritative_roots": {
            "benchmark_contract_identity": identity.to_dict(),
            "root_seed_digest": identity.seed_digest,
            "policy_artifact_id": policy.artifact_id,
            "policy_row_action_digest": context["policy_row_action_digest"],
            "episode_family_registry_digest": _episode_family_registry()[
                "registry_digest"
            ],
            "reachability_tile_digest": context["reachability_tile"]["tile_digest"],
            "provider_versions": {
                "P1": PROSPECTIVE_P1_VERSION,
                "P2": PROSPECTIVE_P2_VERSION,
                "P3": PROSPECTIVE_P3_VERSION,
            },
            "evidence_schema_versions": {
                "complete_row_evidence": "zeromodel-video-complete-row-evidence/v2",
                "semantic_top_set_outcome": SEMANTIC_OUTCOME_VERSION,
                "reachability_trace": REACHABILITY_TRACE_VERSION,
                "sealed_episode_plan": EPISODE_PLAN_VERSION,
            },
        },
        "checks_executed": [gate["gate"] for gate in normalized_gates],
        "passed_checks": passed,
        "failed_checks": failed,
        "unavailable_checks": unavailable,
        "gates": normalized_gates,
        "measured_counts": dict(phase_counts),
        "final_access_measurements": {
            "final_plan_count": len(context["plans"]["final"]),
            "final_observation_materialization_count": phase_counts[
                "final_materialization_count"
            ],
            "final_provider_score_access_count": phase_counts[
                "final_score_access_count"
            ],
            "final_reachability_execution_count": phase_counts[
                "final_reachability_execution_count"
            ],
            "final_evaluation_count": phase_counts["final_evaluation_count"],
            "calibration_execution_count": phase_counts["calibration_execution_count"],
            "architecture_selection_execution_count": phase_counts[
                "architecture_selection_execution_count"
            ],
            "candidate_tuning_execution_count": phase_counts[
                "candidate_tuning_execution_count"
            ],
        },
        "verified": not failed and not unavailable,
        "primary_failure_code": None,
        "primary_failure_gate": None,
    }
    primary = _primary_failure(payload)
    payload["primary_failure_code"] = None if primary is None else primary["code"]
    payload["primary_failure_gate"] = None if primary is None else primary["gate"]
    payload["verification_digest"] = _sha256(
        {key: value for key, value in payload.items() if key != "verification_digest"}
    )
    return payload


def build_read_only_verification_payload(
    *,
    before: Mapping[str, Any],
    middle: Mapping[str, Any],
    after: Mapping[str, Any],
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "read_only": before == middle == after,
        "deterministic": (
            first.get("verification_digest") == second.get("verification_digest")
            and first == second
        ),
        "first_verification_digest": first.get("verification_digest"),
        "second_verification_digest": second.get("verification_digest"),
        "path_count": len(before),
    }
    payload["status"] = (
        "passed" if payload["read_only"] and payload["deterministic"] else "failed"
    )
    payload["digest"] = _sha256(payload)
    return payload


__all__ = [
    "REFERENCE_VERIFICATION_VERSION",
    "SEMANTIC_OUTCOME_VERSION",
    "_REQUIRED_VERIFICATION_GATES",
    "_expected_semantic_for_row",
    "_finding",
    "_first_failure_code",
    "_gate",
    "_policy_row_action_digest",
    "_primary_failure",
    "_raw_score_diagnostic_from_row",
    "_report_failure_codes",
    "_semantic_cache_key",
    "_sha256",
    "_stored_quantized_evidence",
    "build_provider_equivalence_payload",
    "build_read_only_verification_payload",
    "build_reference_context",
    "build_reference_verification_payload",
    "compare_provider_results",
]
