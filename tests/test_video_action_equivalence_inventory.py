from __future__ import annotations

import json
from pathlib import Path

from examples.arcade_visual_action_equivalence_audit import run_audit_evidence_closure


def test_inventory_generation_is_deterministic(tmp_path: Path) -> None:
    first = run_audit_evidence_closure(tmp_path)
    second = run_audit_evidence_closure(tmp_path)
    assert first["inventory_digest"] == second["inventory_digest"]


def test_stage3_v1_replay_is_removed_from_corrected_inventory(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    inventory = json.loads((tmp_path / "evidence-inventory-v2.json").read_text(encoding="utf-8"))
    provider = next(item for item in inventory["providers"] if item["provider_id"] == "stage3-v1")
    assert provider["reachability_replay_eligible"] is False


def test_provider_field_audit_is_written(tmp_path: Path) -> None:
    run_audit_evidence_closure(tmp_path)
    assert (tmp_path / "provider-evidence-files.json").exists()
    assert (tmp_path / "provider-evidence-fields.csv").exists()
