from __future__ import annotations

import json
from pathlib import Path

from examples.arcade_visual_action_equivalence_audit import run_build_reachability_tile, run_replay_reachability


def test_all_448_pairs_are_represented(tmp_path: Path) -> None:
    payload = run_build_reachability_tile(tmp_path)
    assert payload["pair_count"] == 448
    assert payload["verified"] is True


def test_reachable_rows_stay_within_universe(tmp_path: Path) -> None:
    run_build_reachability_tile(tmp_path)
    tile = json.loads((tmp_path / "reachability-tile.json").read_text(encoding="utf-8"))
    sources = {edge["source_row_id"] for edge in tile["edges"]}
    assert all(dest in sources for edge in tile["edges"] for dest in edge["reachable_row_ids"])


def test_tile_regeneration_is_deterministic(tmp_path: Path) -> None:
    first = run_build_reachability_tile(tmp_path)
    second = run_build_reachability_tile(tmp_path)
    assert first["tile_digest"] == second["tile_digest"]


def test_aggregate_sequence_evidence_blocks_replay(tmp_path: Path) -> None:
    payload = run_replay_reachability(tmp_path)
    assert payload["status"] == "reachability_replay_unavailable"
    assert payload["stage3_v1_classification"] == "sequence_metadata_without_visual_beliefs"


def test_missing_visual_sets_cannot_be_reconstructed_from_aggregate_metrics(tmp_path: Path) -> None:
    payload = run_replay_reachability(tmp_path)
    assert payload["replay_eligible_providers"] == []
