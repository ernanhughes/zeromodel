from __future__ import annotations

from pathlib import Path

from examples.arcade_visual_action_equivalence_audit import run_inventory_evidence, run_verify_inventory
from examples.arcade_shooter_policy import compile_policy_artifact
from zeromodel.artifact import VPMValidationError
from zeromodel.video_action_equivalence import (
    build_policy_row_action_map,
    classify_score_evidence,
    collect_v3_preservation_manifest,
    policy_action_for_row,
    replay_eligibility,
    verify_v3_preservation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_full_score_vectors_are_distinguished_from_top1_evidence() -> None:
    full_vector, full_ranking, top_k = classify_score_evidence(
        full_vector=True,
        full_ranking=False,
        top_k=False,
        top1_only=False,
        aggregate_only=False,
        reproducible=False,
    )
    assert full_vector == "stored_original_scores"
    assert full_ranking == "stored_original_scores"
    assert top_k == "stored_original_scores"


def test_top_k_rankings_are_distinguished_from_full_score_vectors() -> None:
    full_vector, full_ranking, top_k = classify_score_evidence(
        full_vector=False,
        full_ranking=False,
        top_k=True,
        top1_only=False,
        aggregate_only=False,
        reproducible=False,
    )
    assert full_vector == "missing"
    assert full_ranking == "missing"
    assert top_k == "stored_original_scores"


def test_aggregate_prose_is_never_treated_as_raw_evidence() -> None:
    full_vector, full_ranking, top_k = classify_score_evidence(
        full_vector=False,
        full_ranking=False,
        top_k=False,
        top1_only=False,
        aggregate_only=True,
        reproducible=False,
    )
    assert full_vector == "aggregate_only"
    assert full_ranking == "aggregate_only"
    assert top_k == "aggregate_only"


def test_recomputable_frozen_providers_are_identified_separately() -> None:
    full_vector, full_ranking, top_k = classify_score_evidence(
        full_vector=False,
        full_ranking=False,
        top_k=False,
        top1_only=False,
        aggregate_only=False,
        reproducible=True,
    )
    assert full_vector == "recomputed_from_frozen_committed_artifacts"
    assert full_ranking == "recomputed_from_frozen_committed_artifacts"
    assert top_k == "recomputed_from_frozen_committed_artifacts"


def test_missing_sequence_order_blocks_replay_eligibility() -> None:
    eligible, reasons = replay_eligibility(
        frame_order_available=False,
        executed_actions_available=True,
        recommended_actions_available=False,
    )
    assert eligible is False
    assert "missing_sequence_order" in reasons


def test_missing_executed_action_blocks_replay_eligibility() -> None:
    eligible, reasons = replay_eligibility(
        frame_order_available=True,
        executed_actions_available=False,
        recommended_actions_available=False,
    )
    assert eligible is False
    assert "missing_executed_actions" in reasons


def test_recommended_action_does_not_satisfy_executed_action_availability() -> None:
    eligible, reasons = replay_eligibility(
        frame_order_available=True,
        executed_actions_available=False,
        recommended_actions_available=True,
    )
    assert eligible is False
    assert "recommended_action_is_not_executed_action" in reasons


def test_historical_final_data_is_labelled_retrospective(tmp_path: Path) -> None:
    payload = run_inventory_evidence(tmp_path)
    provider_map = {item["system_id"]: item for item in payload["providers"]}
    assert provider_map["stage3-v2"]["historical_final_data_already_unblinded"] is False
    assert provider_map["stage3-v3-b3"]["historical_final_data_already_unblinded"] is False


def test_policy_version_mismatch_blocks_rescoring() -> None:
    wrong_policy = "sha256:not-the-compiled-policy"
    try:
        build_policy_row_action_map(policy_artifact_id=wrong_policy)
    except VPMValidationError as exc:
        assert "mismatched policy version" in str(exc)
    else:
        raise AssertionError("expected mismatched policy version failure")


def test_every_current_policy_row_maps_to_one_action() -> None:
    policy = compile_policy_artifact()
    mapping = build_policy_row_action_map(policy_artifact_id=policy.artifact_id)
    assert len(mapping) == 112
    assert len({entry.row_id for entry in mapping}) == 112
    assert len({entry.action_id for entry in mapping}) == 4


def test_unknown_rows_fail() -> None:
    policy = compile_policy_artifact()
    mapping = build_policy_row_action_map(policy_artifact_id=policy.artifact_id)
    try:
        policy_action_for_row("tank=99|target=99|cooldown=9", row_action_map=mapping, policy_artifact_id=policy.artifact_id)
    except VPMValidationError as exc:
        assert "unknown policy row" in str(exc)
    else:
        raise AssertionError("expected unknown policy row failure")


def test_frozen_v3_files_remain_unchanged() -> None:
    manifest = collect_v3_preservation_manifest(REPO_ROOT)
    verification = verify_v3_preservation(REPO_ROOT, manifest)
    assert verification["verified"] is True


def test_inventory_generation_is_deterministic(tmp_path: Path) -> None:
    first = run_inventory_evidence(tmp_path)
    second = run_inventory_evidence(tmp_path)
    assert first["inventory_digest"] == second["inventory_digest"]


def test_inventory_verification_is_read_only(tmp_path: Path) -> None:
    run_inventory_evidence(tmp_path)
    payload = run_verify_inventory(tmp_path)
    assert payload["read_only"] is True


def test_no_untouched_v3_final_access_occurs(tmp_path: Path) -> None:
    payload = run_verify_inventory(tmp_path) if (tmp_path / "parent-v3-preservation-manifest.json").exists() else None
    if payload is None:
        run_inventory_evidence(tmp_path)
        payload = run_verify_inventory(tmp_path)
    assert payload["v3_final_split_access_count"] == 0


def test_no_pr_42_result_artifact_is_created(tmp_path: Path) -> None:
    run_inventory_evidence(tmp_path)
    assert not (tmp_path / "selected-architecture.json").exists()
    assert not (tmp_path / "selected-operating-point.json").exists()
