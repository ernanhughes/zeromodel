from __future__ import annotations

from pathlib import Path

from examples.arcade_visual_action_equivalence_audit import run_audit_evidence_closure
from examples.arcade_shooter_policy import compile_policy_artifact
from zeromodel.video_action_equivalence import (
    VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION,
    build_policy_row_action_map,
    collect_v3_preservation_manifest,
    verify_v3_preservation,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_aggregate_metrics_do_not_imply_per_observation_evidence(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    closure = __import__("json").loads((tmp_path / "evidence-closure.json").read_text(encoding="utf-8"))
    provider = next(item for item in closure["providers"] if item["provider_id"] == "stage3-v1")
    assert provider["aggregate_metric_evidence"] is True
    assert provider["per_observation_top1_evidence"] is False


def test_sequence_summaries_do_not_imply_visual_beliefs(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    closure = __import__("json").loads((tmp_path / "evidence-closure.json").read_text(encoding="utf-8"))
    provider = next(item for item in closure["providers"] if item["provider_id"] == "stage3-v1")
    assert provider["sequence_metadata_evidence"] is True
    assert provider["frame_level_visual_belief_evidence"] is False
    assert provider["reachability_replay_closure"] is False


def test_unknown_source_commits_remain_unknown(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    inventory = __import__("json").loads((tmp_path / "evidence-inventory-v2.json").read_text(encoding="utf-8"))
    provider = next(item for item in inventory["providers"] if item["provider_id"] == "stage3-v1")
    assert provider["source_commit_status"] == "unknown"
    assert provider["source_commit"] is None


def test_current_branch_sha_is_not_substituted_for_historical_source_identity(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    inventory = __import__("json").loads((tmp_path / "evidence-inventory-v2.json").read_text(encoding="utf-8"))
    provider = next(item for item in inventory["providers"] if item["provider_id"] == "stage3-v1")
    assert provider["source_commit"] != "4790165de78557fce63d64e5f2b7ddfde04f1e98"


def test_empty_reproduction_command_cannot_produce_verified_reproducibility(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    inventory = __import__("json").loads((tmp_path / "evidence-inventory-v2.json").read_text(encoding="utf-8"))
    provider = next(item for item in inventory["providers"] if item["provider_id"] == "system-b-v2")
    assert provider["reproducibility_status"] == "not_tested"


def test_canonical_instrument_reproducibility_does_not_imply_evaluation_scores(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    inventory = __import__("json").loads((tmp_path / "evidence-inventory-v2.json").read_text(encoding="utf-8"))
    provider = next(item for item in inventory["providers"] if item["provider_id"] == "stage3-v3-b3")
    assert provider["canonical_instrument_reproducible"] is True
    assert provider["evaluation_observation_scores_available"] is False


def test_invalid_v2_historical_artifacts_are_not_relabelled_reproducible(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    inventory = __import__("json").loads((tmp_path / "evidence-inventory-v2.json").read_text(encoding="utf-8"))
    provider = next(item for item in inventory["providers"] if item["provider_id"] == "stage3-v2")
    assert provider["historical_package_reproducible"] is False


def test_corrected_inventory_version_is_v2() -> None:
    assert VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION.endswith("/v2")


def test_policy_row_action_map_still_covers_entire_policy() -> None:
    mapping = build_policy_row_action_map(policy_artifact_id=compile_policy_artifact().artifact_id)
    assert len(mapping) == 112


def test_frozen_v3_files_remain_unchanged() -> None:
    manifest = collect_v3_preservation_manifest(REPO_ROOT)
    verification = verify_v3_preservation(REPO_ROOT, manifest)
    assert verification["verified"] is True
