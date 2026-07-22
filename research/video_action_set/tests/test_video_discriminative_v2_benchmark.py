from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples import arcade_visual_video_discriminative_evidence_benchmark as bench


@pytest.mark.slow
def test_v2_prototype_and_development_counts_match_full_universe(tmp_path: Path) -> None:
    payload = bench._freeze_benchmark_v2_into(tmp_path)
    assert payload["prototype_manifest"]["provider_prototype_count"] == 112
    assert payload["prototype_manifest"]["prototype_manifest_count"] == 112
    assert payload["development_manifest"]["development_record_count"] == 224
    assert payload["development_manifest"]["development_covered_row_count"] == 112
    assert payload["evaluation_sample"]["sample_size"] == 12
    assert len(payload["evaluation_sample"]["selected_row_ids"]) == 12


@pytest.mark.slow
def test_v2_ids_are_disjoint_from_v1_frozen_artifacts(tmp_path: Path) -> None:
    payload = bench._freeze_benchmark_v2_into(tmp_path)
    v1_split = json.loads(Path("docs/results/video-discriminative-local-evidence-v1/split-manifest.json").read_text(encoding="utf-8"))
    v1_ids = {item for values in v1_split["observation_membership"].values() for item in values}
    v2_ids = {record["observation_id"] for record in payload["prototype_manifest"]["rows"]}
    v2_ids.update(record["observation_id"] for record in payload["development_manifest"]["rows"])
    assert v1_ids.isdisjoint(v2_ids)


@pytest.mark.slow
def test_v2_freeze_writes_under_v2_output_only(tmp_path: Path) -> None:
    before = Path("docs/results/video-discriminative-local-evidence-v1/benchmark-manifest.json").read_text(encoding="utf-8")
    bench.run_freeze_benchmark_v2(tmp_path)
    after = Path("docs/results/video-discriminative-local-evidence-v1/benchmark-manifest.json").read_text(encoding="utf-8")
    assert before == after
