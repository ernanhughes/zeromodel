from __future__ import annotations

from pathlib import Path

from examples import arcade_visual_video_discriminative_evidence_benchmark as bench
import pytest

pytestmark = pytest.mark.research

def test_v3_freeze_preserves_counts_and_pairwise_universe(tmp_path: Path) -> None:
    payload = bench.run_freeze_benchmark_v3(tmp_path)
    assert payload["benchmark_digest"].startswith("sha256:")
    pairwise = bench._load_json(tmp_path / "pairwise-mask-summary.json")
    candidate_masks = bench._load_json(tmp_path / "candidate-mask-manifest.json")
    assert len(candidate_masks["mask_specs"]) == 112
    assert pairwise["pairwise_mask_count"] == 6216
    assert pairwise["same_action_pair_count"] + pairwise["different_action_pair_count"] == 6216


def test_v3_ids_are_disjoint_from_v2_ids() -> None:
    v2 = bench._build_stage3_benchmark_v2(materialize_final=False)
    v3 = bench._build_stage3_benchmark_v3(materialize_final=False)
    assert {record.observation_id for record in v2.records}.isdisjoint({record.observation_id for record in v3.records})
    assert {record.clip_id for record in v2.records}.isdisjoint({record.clip_id for record in v3.records})
    assert {record.frame_id for record in v2.records}.isdisjoint({record.frame_id for record in v3.records})
