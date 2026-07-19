from __future__ import annotations

import json
from pathlib import Path

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.video_complete_row_evidence import build_complete_row_evidence, build_semantic_top_set_outcome


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_provider_rows() -> list[dict[str, object]]:
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    evidence = build_complete_row_evidence(
        row_scores=[(row_id, 1.0) for row_id in row_ids],
        policy_artifact_id=policy.artifact_id,
        provider_id="P1",
        provider_version=benchmark.PROSPECTIVE_P1_VERSION,
        policy_row_ids=row_ids,
    )
    outcome = build_semantic_top_set_outcome(evidence=evidence, row_action=row_actions)
    return [
        {
            "frame_id": "frame-001",
            "provider_id": "P1",
            "provider_version": benchmark.PROSPECTIVE_P1_VERSION,
            "policy_artifact_id": policy.artifact_id,
            "all_112_row_ids": row_ids,
            "all_112_raw_scores": [1.0] * 112,
            "all_112_quantized_scores": [1_000_000] * 112,
            "complete_ordered_ranking": list(evidence.ranking.ranked_row_ids),
            "tie_groups": [group.to_dict() for group in evidence.ranking.tie_groups],
            "semantic_top_set_outcome": outcome.to_dict(),
            "semantic_status": outcome.status,
            "resolved_row": outcome.resolved_row_id,
            "resolved_action": outcome.resolved_action_id,
            "top_quantized_score": outcome.top_quantized_score,
            "top_row_ids": list(outcome.top_row_ids),
            "top_action_ids": list(outcome.top_action_ids),
            "semantic_outcome_digest": outcome.semantic_outcome_digest,
            "winner_row": outcome.resolved_row_id,
            "winner_action": outcome.resolved_action_id,
            "winner_quantized_score": None,
            "runner_up_row": evidence.ranking.ranked_row_ids[1],
            "runner_up_quantized_score": 1_000_000,
            "score_vector_digest": evidence.score_vector_digest,
            "ranking_digest": evidence.ranking.to_dict()["ranking_digest"],
            "provider_diagnostics": {},
        }
    ]


def test_instrument_audits_and_verification(tmp_path: Path) -> None:
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    fake_provider_rows = _fake_provider_rows()
    monkeypatch = pytest.MonkeyPatch()
    fake_records = [
        {
            "split": "development",
            "episode_id": "episode-001",
            "clip_id": "clip-001",
            "frame_id": "development:episode-001:frame-00",
            "sequence_number": 0,
            "family": "exact",
            "expected_disposition": "valid",
            "expected_row": "row-000",
            "expected_action": "left",
            "actual_executed_action": "left",
            "action_known": True,
            "gap_declaration": None,
            "metadata": {"episode_seed": 1, "seed_digest": "sha256:test", "reachability_trace": {"reachable_row_ids": ["row-000"]}},
            "pixels": [[0]],
        }
    ]
    fake_records_calibration = [{**record, "split": "calibration", "frame_id": "calibration:episode-001:frame-00"} for record in fake_records]
    fake_records_selection = [{**record, "split": "selection", "frame_id": "selection:episode-001:frame-00"} for record in fake_records]
    monkeypatch.setattr(
        benchmark,
        "_materialize_records",
        lambda split, repo_root: {
            "development": fake_records,
            "calibration": fake_records_calibration,
            "selection": fake_records_selection,
        }[split],
    )
    monkeypatch.setattr(benchmark, "_score_record", lambda record, prototypes, policy_artifact_id, **_kwargs: fake_provider_rows)
    monkeypatch.setattr(benchmark, "canonical_prototypes", lambda: {})
    benchmark.build_split("development", tmp_path, REPO_ROOT)
    benchmark.build_split("calibration", tmp_path, REPO_ROOT)
    benchmark.build_split("selection", tmp_path, REPO_ROOT)
    monkeypatch.undo()
    evidence = benchmark.audit_evidence_completeness(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(benchmark, "audit_canonical_providers", lambda output_dir: {"P1": {"exact_top1_count": 112}, "P2": {"exact_top1_count": 112}, "P3": {"exact_top1_count": 112}})
    monkeypatch.setattr(benchmark, "verify_instrument", lambda output_dir, repo_root: {"verified": True, "final_materialization_count": 0})
    canonical = benchmark.audit_canonical_providers(tmp_path)
    verification = benchmark.verify_instrument(tmp_path, REPO_ROOT)
    monkeypatch.undo()
    assert evidence["complete_score_evidence"] is True
    assert canonical["P3"]["exact_top1_count"] == 112
    assert verification["verified"] is True
    assert json.loads((tmp_path / "phase-access-audits.json").read_text(encoding="utf-8"))["final_materialization_count"] == 0
    assert not (tmp_path / "selected-method.json").exists()
    assert not (tmp_path / "reachability-replay.json").exists()
    assert not (tmp_path / "final-results.json").exists()
def test_mutation_gate_detects_protected_field_changes() -> None:
    cases = {case["name"]: case for case in benchmark._MUTATION_CASES}
    assert cases["evidence_quantized_score_changed"]["expected_primary_failure_code"] == "quantized_score_vector_mismatch"
    assert cases["semantic_resolved_row_for_action_unanimous_tie"]["expected_primary_failure_code"] == "resolved_row_not_permitted"
    assert cases["seed_alter_final_sealed_identity"]["expected_primary_failure_code"] == "sealed_episode_identity_mismatch"
    assert cases["reachability_change_executed_action"]["expected_primary_failure_code"] == "executed_action_mismatch"
    assert cases["access_add_final_observation_artifact"]["expected_primary_failure_code"] == "forbidden_final_materialization"
