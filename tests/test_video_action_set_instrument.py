from __future__ import annotations

import json
from pathlib import Path

import pytest

import zeromodel.video_action_set_benchmark as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_instrument_audits_and_verification(tmp_path: Path) -> None:
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    fake_provider_rows = [
        {
            "frame_id": "frame-001",
            "provider_id": "P1",
            "all_112_row_ids": [f"row-{index:03d}" for index in range(112)],
            "all_112_raw_scores": [1.0] * 112,
            "all_112_quantized_scores": [1_000_000] * 112,
            "complete_ordered_ranking": [f"row-{index:03d}" for index in range(112)],
            "tie_groups": [{"tie_group_index": 0, "quantized_score": 1_000_000, "row_ids": [f"row-{index:03d}" for index in range(112)]}],
            "winner_row": "row-000",
            "winner_action": "left",
            "winner_quantized_score": 1_000_000,
            "runner_up_row": "row-001",
            "runner_up_quantized_score": 1_000_000,
            "score_vector_digest": "sha256:test",
            "ranking_digest": "sha256:test",
            "provider_diagnostics": {},
        }
    ]
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
    monkeypatch.setattr(benchmark, "_score_record", lambda record, prototypes, policy_artifact_id: fake_provider_rows)
    monkeypatch.setattr(benchmark, "canonical_prototypes", lambda: {})
    monkeypatch.setattr(benchmark, "compile_policy_artifact", lambda *args, **kwargs: type("Policy", (), {"artifact_id": "policy"})())
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
