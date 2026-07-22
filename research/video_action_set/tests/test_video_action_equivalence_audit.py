from __future__ import annotations

import json
from pathlib import Path

from research.video_action_set.arcade_visual_action_equivalence_audit import run_finalize_audit, run_verify_audit
import pytest

pytestmark = pytest.mark.research


def test_insufficient_historical_artifacts_is_selected(tmp_path: Path) -> None:
    summary = run_finalize_audit(tmp_path, tmp_path / "reachability")
    assert summary["primary_status"] == "insufficient_historical_artifacts"


def test_raw_action_gaps_do_not_trigger_material_governed_utility_status(tmp_path: Path) -> None:
    summary = run_finalize_audit(tmp_path, tmp_path / "reachability")
    assert summary["primary_status"] != "action_equivalence_materially_changes_protocol_result"


def test_unsupported_top_k_is_not_zero_coverage(tmp_path: Path) -> None:
    run_finalize_audit(tmp_path, tmp_path / "reachability")
    status = json.loads((tmp_path / "fixed-top-k-status.json").read_text(encoding="utf-8"))
    assert status["status"] == "fixed_top_k_not_supported"


def test_unavailable_replay_is_not_treated_as_reachability_failure(tmp_path: Path) -> None:
    run_finalize_audit(tmp_path, tmp_path / "reachability")
    report = (tmp_path / "protocol-sensitivity-report.md").read_text(encoding="utf-8")
    assert "unmeasured rather than positive or negative" in report


def test_verified_reachability_tile_does_not_imply_replay_success(tmp_path: Path) -> None:
    run_finalize_audit(tmp_path, tmp_path / "reachability")
    replay = json.loads((tmp_path / "reachability-replay-summary.json").read_text(encoding="utf-8"))
    assert replay["status"] == "reachability_replay_unavailable"


def test_visual_branch_is_neither_closed_nor_promoted(tmp_path: Path) -> None:
    run_finalize_audit(tmp_path, tmp_path / "reachability")
    recommendation = json.loads((tmp_path / "visual-branch-recommendation.json").read_text(encoding="utf-8"))
    assert recommendation["visual_branch_recommendation"] == "undetermined_due_to_missing_artifacts"


def test_supported_and_unsupported_claims_are_deterministic(tmp_path: Path) -> None:
    first = run_finalize_audit(tmp_path, tmp_path / "reachability")
    second = run_finalize_audit(tmp_path, tmp_path / "reachability")
    assert first["supported_claim"] == second["supported_claim"]
    assert first["unsupported_claims"] == second["unsupported_claims"]


def test_final_verification_is_read_only(tmp_path: Path) -> None:
    run_finalize_audit(tmp_path, tmp_path / "reachability")
    payload = run_verify_audit(tmp_path, tmp_path / "reachability")
    assert payload["read_only"] is True
    assert payload["verified"] is True
