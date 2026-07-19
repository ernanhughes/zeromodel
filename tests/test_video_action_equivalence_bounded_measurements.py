from __future__ import annotations

import json
from pathlib import Path

from examples.arcade_visual_action_equivalence_audit import (
    run_audit_evidence_closure,
    run_build_reachability_tile,
    run_replay_reachability,
    run_rescore_supported_top1,
    run_verify_bounded_measurements,
)


def test_unsupported_methods_produce_status_artifacts(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    fixed = json.loads((tmp_path / "fixed-top-k-status.json").read_text(encoding="utf-8"))
    score_gap = json.loads((tmp_path / "score-gap-status.json").read_text(encoding="utf-8"))
    conformal = json.loads((tmp_path / "conformal-viability.json").read_text(encoding="utf-8"))
    assert fixed["status"] == "fixed_top_k_not_supported"
    assert score_gap["status"] == "score_gap_not_supported"
    assert conformal["status"] == "conformal_not_supported"


def test_bounded_measurement_verification_keeps_forbidden_access_counts_zero(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    run_rescore_supported_top1(tmp_path)
    run_build_reachability_tile(tmp_path / "reachability")
    run_replay_reachability(tmp_path)
    payload = run_verify_bounded_measurements(tmp_path, tmp_path / "reachability")
    assert payload["v3_final_access_count"] == 0
    assert payload["new_observation_generation_count"] == 0
    assert payload["pr_42_grid_execution_count"] == 0


def test_bounded_measurement_verification_succeeds_when_outputs_exist(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    run_rescore_supported_top1(tmp_path)
    run_build_reachability_tile(tmp_path / "reachability")
    run_replay_reachability(tmp_path)
    payload = run_verify_bounded_measurements(tmp_path, tmp_path / "reachability")
    assert payload["verified"] is True
