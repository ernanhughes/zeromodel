from __future__ import annotations

import json
from pathlib import Path

from examples.arcade_visual_action_equivalence_audit import run_rescore_supported_top1
from examples.arcade_shooter_policy import compile_policy_artifact
from zeromodel.video_action_equivalence import build_policy_row_action_map, policy_action_for_row


def _results(tmp_path: Path) -> list[dict[str, object]]:
    run_rescore_supported_top1(tmp_path)
    return json.loads((tmp_path / "top1-action-results.json").read_text(encoding="utf-8"))


def test_aggregate_metric_verification_is_labelled_separately(tmp_path: Path) -> None:
    results = _results(tmp_path)
    row = next(item for item in results if item["provider_id"] == "stage3-v1")
    assert row["mode"] == "not_supported"


def test_row_to_action_mapping_uses_exact_policy_artifact() -> None:
    mapping = build_policy_row_action_map(policy_artifact_id=compile_policy_artifact().artifact_id)
    assert policy_action_for_row("tank=0|target=none|cooldown=0", row_action_map=mapping, policy_artifact_id=mapping[0]["policy_artifact_id"]) == "STAY"


def test_same_action_wrong_row_logic_is_recorded(tmp_path: Path) -> None:
    results = _results(tmp_path)
    row = next(item for item in results if item["provider_id"] == "system-b-v2")
    assert row["same_action_wrong_row_count"] > 0


def test_raw_action_gap_calculation_matches_frozen_claims(tmp_path: Path) -> None:
    results = _results(tmp_path)
    system_b = next(item for item in results if item["provider_id"] == "system-b-v2")
    r1 = next(item for item in results if item["provider_id"] == "r1-local-correlation")
    assert system_b["raw_action_gap"] == 0.21875
    assert r1["raw_action_gap"] == 0.109375


def test_canonical_diagnostics_are_not_labelled_benchmark_utility(tmp_path: Path) -> None:
    results = _results(tmp_path)
    row = next(item for item in results if item["provider_id"] == "stage3-v3-b3")
    assert row["mode"] == "canonical_diagnostic_rescore"
    assert row["diagnostic_label"] == "canonical_instrument_diagnostic"


def test_invalid_measurement_provider_retains_invalid_boundary(tmp_path: Path) -> None:
    results = _results(tmp_path)
    row = next(item for item in results if item["provider_id"] == "stage3-v2")
    assert row["invalid_boundary"] is True
