from __future__ import annotations

from pathlib import Path

import pytest

from examples import arcade_visual_video_discriminative_evidence_benchmark as bench


def test_stage3_benchmark_descriptors_are_deterministic() -> None:
    left = bench._build_stage3_benchmark(materialize_final=False)
    right = bench._build_stage3_benchmark(materialize_final=False)
    assert bench._benchmark_manifest(left)["benchmark_digest"] == bench._benchmark_manifest(right)["benchmark_digest"]
    assert bench._split_manifest(left)["split_digest"] == bench._split_manifest(right)["split_digest"]


def test_stage3_split_membership_is_disjoint() -> None:
    benchmark = bench._build_stage3_benchmark(materialize_final=False)
    membership = {}
    for record in benchmark.records:
        assert record.observation_id not in membership
        membership[record.observation_id] = record.split
    assert len(membership) == len(benchmark.records)
    assert set(bench.SPLIT_ROLES) == {record.split for record in benchmark.records}


def test_stage3_split_access_blocks_final_in_selection_phase() -> None:
    benchmark = bench._build_stage3_benchmark(materialize_final=False)
    with pytest.raises(Exception):
        benchmark.access(phase="select_architecture", allowed_splits=("final_benign",))


def test_architecture_d_is_frozen_before_selection() -> None:
    active, inactive = bench.zde._architecture_active_gates("D")
    assert "minimum_support" in active
    assert "maximum_contradiction" in active
    assert "maximum_critical_contradiction" in active
    assert "minimum_supporting_regions" in active
    assert inactive == ()


def test_verify_stage2_diagnosis_is_read_only(tmp_path: Path) -> None:
    output_dir = Path("docs/results/video-discriminative-local-evidence-v1")
    payload = bench.run_verify_stage2_diagnosis(output_dir)
    assert payload["verified"] is True
    assert payload["mode"] == "verify-stage2-diagnosis"
