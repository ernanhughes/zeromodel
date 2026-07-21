from __future__ import annotations

from zeromodel.domains.video_action_set import evidence_audit


def test_phase_access_counts_are_built_from_loaded_payloads() -> None:
    final_plan = {"sealed_episode_ids": {"valid": ["final:1"]}}
    frames = {
        "development": [{"episode_id": "dev:1", "split": "development"}],
        "calibration": [],
        "selection": [],
        "final": [{"episode_id": "final:1", "split": "final"}],
    }
    evidence = {
        "development": [],
        "calibration": [],
        "selection": [],
        "final": [
            {
                "episode_id": "final:1",
                "split": "final",
                "reachability_composition_trace": {"trace": True},
            }
        ],
    }

    payload = evidence_audit.measured_phase_access_counts(
        final_plan=final_plan,
        frame_rows_by_split=frames,
        evidence_rows_by_split=evidence,
        existing_artifacts=["selected-calibration.json", "final-results.json"],
    )

    assert list(payload) == [
        "version",
        "final_materialization_count",
        "final_score_access_count",
        "final_reachability_execution_count",
        "candidate_set_selection_count",
        "candidate_tuning_execution_count",
        "conformal_calibration_count",
        "calibration_execution_count",
        "architecture_selection_execution_count",
        "reachability_replay_count",
        "final_evaluation_count",
        "forbidden_final_access_counter",
    ]
    assert payload["final_materialization_count"] == 1
    assert payload["final_score_access_count"] == 1
    assert payload["final_reachability_execution_count"] == 1
    assert payload["calibration_execution_count"] == 1
    assert payload["final_evaluation_count"] == 1


def test_identity_and_overlap_payloads_preserve_split_order() -> None:
    frames = {
        "development": [{"frame_id": "d:1", "episode_id": "dev:1"}],
        "calibration": [{"frame_id": "shared", "episode_id": "cal:1"}],
        "selection": [{"frame_id": "shared", "episode_id": "final:1"}],
    }
    identity = evidence_audit.build_observation_identity_manifest(frames)
    overlap = evidence_audit.build_split_overlap_audit(
        frame_rows_by_split=frames,
        final_plan={"sealed_episode_ids": {"valid": ["final:1"]}},
    )

    assert list(identity) == [
        "development_observation_count",
        "calibration_observation_count",
        "selection_observation_count",
        "all_frame_ids_digest",
    ]
    assert overlap["calibration_selection_overlap"] == 1
    assert overlap["materialized_final_plan_overlap"] == 1


def test_malformed_evidence_counts_are_not_repaired() -> None:
    malformed = {
        "all_112_row_ids": [],
        "all_112_raw_scores": [],
        "all_112_quantized_scores": [],
        "complete_ordered_ranking": [],
        "tie_groups": [],
        "semantic_top_set_outcome": {"version": "foreign"},
    }
    frames = {
        split: [
            {
                "expected_disposition": "valid",
                "metadata": {},
            }
        ]
        for split in ("development", "calibration", "selection")
    }
    evidence = {
        split: [malformed] for split in ("development", "calibration", "selection")
    }

    payload = evidence_audit.audit_evidence_rows(
        frame_rows_by_split=frames,
        evidence_rows_by_split=evidence,
        row_actions={},
    )

    assert payload == {
        "complete_score_evidence": False,
        "missing_score_vector_count": 6,
        "invalid_score_count": 0,
        "missing_ranking_count": 3,
        "missing_tie_group_count": 3,
        "missing_semantic_outcome_count": 3,
        "missing_reachability_trace_count": 3,
        "split_summaries": [
            {"split": "development", "provider_frame_records": 1},
            {"split": "calibration", "provider_frame_records": 1},
            {"split": "selection", "provider_frame_records": 1},
        ],
    }


def test_access_prohibition_finding_order_is_frozen() -> None:
    measured = {
        "final_materialization_count": 1,
        "final_score_access_count": 1,
        "final_reachability_execution_count": 0,
        "calibration_execution_count": 0,
        "architecture_selection_execution_count": 1,
        "candidate_tuning_execution_count": 1,
        "final_evaluation_count": 0,
    }

    gate = evidence_audit.access_prohibition_gate(measured, {})

    assert [finding["code"] for finding in gate["findings"]] == [
        "forbidden_final_materialization",
        "forbidden_final_score_access",
        "forbidden_selection_execution",
    ]
