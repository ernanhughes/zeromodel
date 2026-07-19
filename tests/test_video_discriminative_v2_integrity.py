from __future__ import annotations

from pathlib import Path

import pytest

from examples import arcade_visual_video_discriminative_evidence_benchmark as bench


def test_v2_mask_closure_is_complete_with_full_development(tmp_path: Path) -> None:
    benchmark = bench._build_stage3_benchmark_v2(materialize_final=False)
    freeze = bench._freeze_regions_and_masks(benchmark, output_dir=tmp_path)
    closure = bench._mask_closure_v2(benchmark, freeze["masks"])
    assert closure["closure_valid"] is True
    assert closure["masks_with_development_closure"] == 112
    assert closure["remaining_zero_evidence_masks"] == 0


@pytest.mark.slow
def test_v2_exact_sanity_is_reproducibly_invalid_under_full_universe(tmp_path: Path) -> None:
    benchmark = bench._build_stage3_benchmark_v2(materialize_final=False)
    freeze = bench._freeze_regions_and_masks(benchmark, output_dir=tmp_path)
    collision = bench._prototype_collision_atlas_v2(benchmark)
    summary = bench._exact_sanity_v2(benchmark=benchmark, freeze=freeze, collision_atlas=collision, output_dir=tmp_path)
    assert summary["sanity_valid"] is False
    assert summary["direct_provider_equivalence"] is True
    for architecture_id in ("A", "B", "C"):
        assert summary["architectures"][architecture_id]["unique_exact_expected_row_top1_count"] == 1


@pytest.mark.slow
def test_v2_verify_benchmark_is_read_only_and_reproducible(tmp_path: Path) -> None:
    bench.run_freeze_benchmark_v2(tmp_path)
    payload = bench.run_verify_v2_benchmark(tmp_path)
    assert payload["verified"] is True


@pytest.mark.slow
def test_v2_evaluate_remains_blocked_without_valid_selection(tmp_path: Path) -> None:
    bench.run_freeze_benchmark_v2(tmp_path)
    bench.run_select_architecture_v2(tmp_path)
    bench.run_calibrate_v2(tmp_path)
    bench.run_verify_pre_final_v2(tmp_path)
    try:
        bench.run_evaluate_v2(tmp_path)
    except SystemExit as exc:
        assert "no architecture has been validly selected" in str(exc)
    else:
        raise AssertionError("evaluate-v2 should be blocked")
