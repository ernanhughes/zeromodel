from __future__ import annotations

from typing import Any, Mapping, Sequence

from zeromodel.core.artifact import VPMValidationError
from research.evidence.video_complete_row_evidence import (
    QUANTIZATION_SCALE,
    VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION,
    build_complete_row_evidence,
    semantic_top_set_outcome_from_dict,
)
from research.video.video_prospective_providers import (
    PROSPECTIVE_PROVIDER_IDS,
    score_b3_joint_fit,
    score_normalized_pixel,
    score_registered_local_correlation,
)
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from research.video_action_set.provider_measurement import SOURCE_SCOPE
from research.video_action_set.reference_verification import _finding, _gate

PHASE_ACCESS_VERSION = "zeromodel-video-prospective-phase-access/v1"
SEMANTIC_OUTCOME_VERSION = VIDEO_SEMANTIC_TOP_SET_OUTCOME_VERSION

_NON_FINAL_SPLITS = ("development", "calibration", "selection")
_ALL_SPLITS = (*_NON_FINAL_SPLITS, "final")
_CALIBRATION_ARTIFACTS = (
    "selected-calibration.json",
    "calibration-grid.json",
    "calibration-results.json",
    "conformal-calibration.json",
)
_SELECTION_ARTIFACTS = (
    "selected-architecture.json",
    "architecture-grid.json",
    "architecture-selection.json",
    "selected-method.json",
)
_TUNING_ARTIFACTS = (
    "candidate-tuning.json",
    "candidate-set-tuning.json",
    "candidate-grid.json",
    "reachability-replay.json",
)
_FINAL_ARTIFACTS = (
    "final-results.json",
    "final-summary.json",
    "final-evaluation.json",
)


def measured_phase_access_counts(
    *,
    final_plan: Mapping[str, Any],
    frame_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    existing_artifacts: Sequence[str],
) -> dict[str, Any]:
    """Measure protocol access from already-loaded artifact metadata."""

    final_ids = {
        episode_id
        for values in final_plan.get("sealed_episode_ids", {}).values()
        for episode_id in values
    }
    frames = [
        row for split in _ALL_SPLITS for row in frame_rows_by_split.get(split, ())
    ]
    evidence = [
        row for split in _ALL_SPLITS for row in evidence_rows_by_split.get(split, ())
    ]
    final_materialization_count = sum(
        1
        for row in frames
        if row.get("episode_id") in final_ids or row.get("split") == "final"
    )
    final_score_access_count = sum(
        1
        for row in evidence
        if row.get("episode_id") in final_ids or row.get("split") == "final"
    )
    final_reachability_execution_count = sum(
        1
        for row in evidence
        if (row.get("episode_id") in final_ids or row.get("split") == "final")
        and row.get("reachability_composition_trace") is not None
    )
    present = set(existing_artifacts)
    calibration_count = sum(int(name in present) for name in _CALIBRATION_ARTIFACTS)
    selection_count = sum(int(name in present) for name in _SELECTION_ARTIFACTS)
    tuning_count = sum(int(name in present) for name in _TUNING_ARTIFACTS)
    final_evaluation_count = sum(int(name in present) for name in _FINAL_ARTIFACTS)
    return {
        "version": PHASE_ACCESS_VERSION,
        "final_materialization_count": final_materialization_count,
        "final_score_access_count": final_score_access_count,
        "final_reachability_execution_count": final_reachability_execution_count,
        "candidate_set_selection_count": tuning_count,
        "candidate_tuning_execution_count": tuning_count,
        "conformal_calibration_count": calibration_count,
        "calibration_execution_count": calibration_count,
        "architecture_selection_execution_count": selection_count,
        "reachability_replay_count": tuning_count,
        "final_evaluation_count": final_evaluation_count,
        "forbidden_final_access_counter": (
            final_materialization_count
            + final_score_access_count
            + final_reachability_execution_count
        ),
    }


def build_observation_identity_manifest(
    frame_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, Any]:
    frames = {
        split: [row["frame_id"] for row in frame_rows_by_split.get(split, ())]
        for split in _NON_FINAL_SPLITS
    }
    return {
        "development_observation_count": len(frames["development"]),
        "calibration_observation_count": len(frames["calibration"]),
        "selection_observation_count": len(frames["selection"]),
        "all_frame_ids_digest": canonical_sha256(frames),
    }


def build_split_overlap_audit(
    *,
    frame_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    final_plan: Mapping[str, Any],
) -> dict[str, Any]:
    split_sets = {
        split: {row["frame_id"] for row in frame_rows_by_split.get(split, ())}
        for split in _NON_FINAL_SPLITS
    }
    final_ids = {
        episode_id
        for values in final_plan.get("sealed_episode_ids", {}).values()
        for episode_id in values
    }
    materialized_final = sum(
        1
        for split in _NON_FINAL_SPLITS
        for row in frame_rows_by_split.get(split, ())
        if row.get("episode_id") in final_ids or row.get("split") == "final"
    )
    return {
        "development_calibration_overlap": len(
            split_sets["development"] & split_sets["calibration"]
        ),
        "development_selection_overlap": len(
            split_sets["development"] & split_sets["selection"]
        ),
        "calibration_selection_overlap": len(
            split_sets["calibration"] & split_sets["selection"]
        ),
        "materialized_final_plan_overlap": materialized_final,
        "final_episode_ids_digest": canonical_sha256(sorted(final_ids))
        if final_ids
        else None,
    }


def audit_evidence_rows(
    *,
    frame_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    evidence_rows_by_split: Mapping[str, Sequence[Mapping[str, Any]]],
    row_actions: Mapping[str, str],
) -> dict[str, Any]:
    counts = {
        "missing_score_vector_count": 0,
        "invalid_score_count": 0,
        "missing_ranking_count": 0,
        "missing_tie_group_count": 0,
        "missing_semantic_outcome_count": 0,
        "missing_reachability_trace_count": 0,
    }
    summaries = []
    for split in _NON_FINAL_SPLITS:
        rows = list(evidence_rows_by_split.get(split, ()))
        for row in rows:
            if len(row["all_112_row_ids"]) != 112:
                counts["missing_score_vector_count"] += 1
            if (
                len(row["all_112_raw_scores"]) != 112
                or len(row["all_112_quantized_scores"]) != 112
            ):
                counts["missing_score_vector_count"] += 1
            if any(
                score < 0 or score > QUANTIZATION_SCALE
                for score in row["all_112_quantized_scores"]
            ):
                counts["invalid_score_count"] += 1
            if len(row["complete_ordered_ranking"]) != 112:
                counts["missing_ranking_count"] += 1
            if not row["tie_groups"]:
                counts["missing_tie_group_count"] += 1
            counts["missing_semantic_outcome_count"] += _semantic_error_count(
                row, row_actions
            )
        for frame in frame_rows_by_split.get(split, ()):
            if (
                frame.get("expected_disposition") != "information_theoretic_control"
                and "reachability_trace" not in frame.get("metadata", {})
            ):
                counts["missing_reachability_trace_count"] += 1
        summaries.append({"split": split, "provider_frame_records": len(rows)})
    return {
        "complete_score_evidence": (
            counts["missing_score_vector_count"] == 0
            and counts["invalid_score_count"] == 0
            and counts["missing_semantic_outcome_count"] == 0
            and counts["missing_reachability_trace_count"] == 0
        ),
        **counts,
        "split_summaries": summaries,
    }


def _semantic_error_count(
    row: Mapping[str, Any], row_actions: Mapping[str, str]
) -> int:
    if (
        row.get("semantic_top_set_outcome", {}).get("version")
        != SEMANTIC_OUTCOME_VERSION
    ):
        return 1
    try:
        evidence = build_complete_row_evidence(
            row_scores=list(zip(row["all_112_row_ids"], row["all_112_raw_scores"])),
            policy_artifact_id=row["policy_artifact_id"],
            provider_id=row["provider_id"],
            provider_version=row["provider_version"],
            policy_row_ids=row["all_112_row_ids"],
        )
        if evidence.score_vector_digest != row["score_vector_digest"]:
            raise VPMValidationError("foreign score vector digest")
        semantic_top_set_outcome_from_dict(
            row["semantic_top_set_outcome"],
            evidence=evidence,
            row_action=row_actions,
        )
    except (KeyError, VPMValidationError):
        return 1
    return 0


def audit_canonical_provider_results(
    *, prototypes: Mapping[str, Any], policy_artifact_id: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    for provider_id in PROSPECTIVE_PROVIDER_IDS:
        provider_rows = _score_canonical_provider(
            provider_id=provider_id,
            prototypes=prototypes,
            policy_artifact_id=policy_artifact_id,
        )
        rows.extend(provider_rows)
        exact = sum(
            int(row["resolved_row"] == row["expected_row"]) for row in provider_rows
        )
        actions = sum(
            int(row["resolved_action"] == row["expected_action"])
            for row in provider_rows
        )
        summary[provider_id] = {
            "canonical_observation_count": 112,
            "exact_row_resolution_count": exact,
            "action_resolution_count": actions,
            "action_unanimous_tie_resolution_count": sum(
                int(
                    row["semantic_status"] == "action_unanimous_tie"
                    and row["resolved_action"] == row["expected_action"]
                )
                for row in provider_rows
            ),
            "conflicting_action_rejection_count": sum(
                int(row["semantic_status"] == "conflicting_action_tie")
                for row in provider_rows
            ),
            "unresolved_outcome_count": sum(
                int(row["semantic_status"] == "unresolved") for row in provider_rows
            ),
            "exact_top1_count": exact,
            "action_top1_count": actions,
            "maximum_tie_size": max(row["semantic_tie_size"] for row in provider_rows),
            "status": "canonical_diagnostic_pass"
            if provider_id != "P3" or exact == 112
            else "invalid_primary_provider_instrument",
        }
    return summary, rows


def _score_canonical_provider(
    *, provider_id: str, prototypes: Mapping[str, Any], policy_artifact_id: str
) -> list[dict[str, Any]]:
    rows = []
    for observation_id, (row_id, action_id, _digest, observation) in prototypes.items():
        if provider_id == "P1":
            result = score_normalized_pixel(
                observation=observation,
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
            )
        elif provider_id == "P2":
            result = score_registered_local_correlation(
                observation=observation,
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
                source_scope=SOURCE_SCOPE,
            )
        else:
            result = score_b3_joint_fit(
                observation=observation,
                prototypes=prototypes,
                policy_artifact_id=policy_artifact_id,
                source_scope=SOURCE_SCOPE,
            )
        outcome = result.semantic_top_set_outcome
        rows.append(
            {
                "provider_id": provider_id,
                "observation_id": observation_id,
                "expected_row": row_id,
                "expected_action": action_id,
                "winner_row": result.winner_row_id,
                "winner_action": result.winner_action_id,
                "semantic_status": outcome.status,
                "resolved_row": outcome.resolved_row_id,
                "resolved_action": outcome.resolved_action_id,
                "semantic_outcome_digest": outcome.semantic_outcome_digest,
                "semantic_tie_size": result.maximum_tie_size,
                "score_vector_complete": len(result.evidence.row_scores) == 112,
                "ranking_complete": len(result.evidence.ranking.ranked_row_ids) == 112,
                "tie_group_complete": bool(result.evidence.ranking.tie_groups),
            }
        )
    return rows


def access_prohibition_gate(
    measured: Mapping[str, Any],
    stored: Mapping[str, Any],
    *,
    max_findings: int | None = None,
) -> dict[str, Any]:
    checks = (
        (
            "final_materialization_count",
            "forbidden_final_materialization",
            "final split has materialized observations",
        ),
        (
            "final_score_access_count",
            "forbidden_final_score_access",
            "final split has provider score records",
        ),
        (
            "final_reachability_execution_count",
            "forbidden_final_reachability_access",
            "final split has reachability execution traces",
        ),
        (
            "calibration_execution_count",
            "forbidden_calibration_execution",
            "prospective calibration execution artifact is present",
        ),
        (
            "final_evaluation_count",
            "forbidden_final_evaluation",
            "final evaluation artifact is present",
        ),
    )
    findings = [
        _finding(code, message, count=measured[key])
        for key, code, message in checks
        if measured.get(key)
    ]
    selection_count = measured.get(
        "architecture_selection_execution_count", 0
    ) + measured.get("candidate_tuning_execution_count", 0)
    if selection_count:
        findings.append(
            _finding(
                "forbidden_selection_execution",
                "prospective selection or candidate-tuning execution artifact is present",
                count=selection_count,
            )
        )
    stored_projection = {key: stored.get(key) for key in measured if key in stored}
    measured_projection = {key: measured.get(key) for key in measured if key in stored}
    if stored and stored_projection != measured_projection:
        findings.append(
            _finding(
                "status_claim_not_supported",
                "stored phase-access counters do not match counters measured from concrete artifacts",
            )
        )
    if max_findings is not None:
        findings = findings[:max_findings]
    return _gate("access_prohibition", findings, counts=measured)


__all__ = [
    "PHASE_ACCESS_VERSION",
    "access_prohibition_gate",
    "audit_canonical_provider_results",
    "audit_evidence_rows",
    "build_observation_identity_manifest",
    "build_split_overlap_audit",
    "measured_phase_access_counts",
]
