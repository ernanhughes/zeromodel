from __future__ import annotations

import json
from pathlib import Path

import pytest

import zeromodel.video_action_set_benchmark as benchmark


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_materialized_split_counts_and_final_freeze(tmp_path: Path) -> None:
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    development = benchmark._materialize_records("development", REPO_ROOT)
    calibration = benchmark._materialize_records("calibration", REPO_ROOT)
    selection = benchmark._materialize_records("selection", REPO_ROOT)
    assert len(development) == 112
    assert len(calibration) == 448
    assert len(selection) == 1008
    split_manifest = json.loads((tmp_path / "split-manifest.json").read_text(encoding="utf-8"))
    assert split_manifest["calibration_episode_count"] == 112
    assert split_manifest["selection_valid_episode_count"] == 112
    phase_access = json.loads((tmp_path / "phase-access-audits.json").read_text(encoding="utf-8"))
    assert phase_access["final_materialization_count"] == 0
    assert phase_access["final_score_access_count"] == 0


def test_build_split_writes_overlap_and_observation_manifests(tmp_path: Path) -> None:
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    fake_output = [
        {
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
    monkeypatch.setattr(benchmark, "_score_record", lambda record, prototypes, policy_artifact_id: fake_output)
    monkeypatch.setattr(benchmark, "canonical_prototypes", lambda: {})
    monkeypatch.setattr(benchmark, "compile_policy_artifact", lambda *args, **kwargs: type("Policy", (), {"artifact_id": "policy"})())
    benchmark.build_split("development", tmp_path, REPO_ROOT)
    benchmark.build_split("calibration", tmp_path, REPO_ROOT)
    benchmark.build_split("selection", tmp_path, REPO_ROOT)
    monkeypatch.undo()
    overlap = json.loads((tmp_path / "split-overlap-audit.json").read_text(encoding="utf-8"))
    assert overlap["development_calibration_overlap"] == 0
    assert overlap["development_selection_overlap"] == 0
    assert overlap["calibration_selection_overlap"] == 0
    observation_manifest = json.loads((tmp_path / "observation-identity-manifest.json").read_text(encoding="utf-8"))
    assert observation_manifest["development_observation_count"] == 1
    assert observation_manifest["calibration_observation_count"] == 1
    assert observation_manifest["selection_observation_count"] == 1
