from __future__ import annotations

import json
from pathlib import Path

from examples import arcade_visual_video_discriminative_evidence_benchmark as bench


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_representation_audit_freezes_expected_v2_failures(tmp_path: Path) -> None:
    bench.run_freeze_benchmark_v2(tmp_path)
    bench.run_select_architecture_v2(tmp_path)
    bench.run_calibrate_v2(tmp_path)
    bench.run_verify_pre_final_v2(tmp_path)
    try:
        bench._run_representation_audit_v2(tmp_path)
    except SystemExit as exc:
        payload = json.loads(str(exc))
    else:
        raise AssertionError("representation audit should fail nonzero for frozen v2 defects")
    assert payload["canonical_observations_audited"] == 112
    assert payload["self_retrieval_summary"]["architectures"]["A"]["top1_count"] == 7
    assert payload["self_retrieval_summary"]["architectures"]["B"]["top1_count"] == 7
    assert payload["self_retrieval_summary"]["architectures"]["C"]["top1_count"] == 7
    assert payload["mask_separation_summary"]["rows_with_positive_stable_informative_mass"] == 112
    assert payload["mask_separation_summary"]["rows_with_positive_separation_weighted_mass"] == 0
    assert payload["distributed_competitor_cover_row_count"] == 112
    assert payload["architecture_a_conformance"]["ruling"] == "architecture_a_implementation_defect"
    assert payload["exact_tie_audit"]["ruling"] == "exact_tie_implementation_defect"
    assert payload["ruling"]["primary_ruling"] == "multiple_representation_failures"
    assert payload["mutated_v2_root_artifacts"] == []


def test_representation_audit_quantifies_separation_collapse(tmp_path: Path) -> None:
    benchmark = bench._build_stage3_benchmark_v2(materialize_final=False)
    freeze = bench._freeze_regions_and_masks(benchmark, output_dir=tmp_path)
    rows = []
    for row_id, mask in sorted(freeze["masks"].items()):
        arrays = bench._effective_mask_arrays(mask)
        rows.append(
            (
                row_id,
                int(((arrays["row_informative"] > 0.0) & (arrays["separation"] <= 0.0)).sum()),
                int((arrays["positive_row_stable"] > 0.0).sum()),
                int((arrays["positive_row_stable_separation"] > 0.0).sum()),
            )
        )
    assert any(zeroed > 0 for _row_id, zeroed, _stable, _effective in rows)
    assert all(stable > 0 for _row_id, _zeroed, stable, _effective in rows)
    assert all(effective == 0 for _row_id, _zeroed, _stable, effective in rows)


def test_strict_exact_tie_check_rejects_lexical_ties(tmp_path: Path) -> None:
    benchmark = bench._build_stage3_benchmark_v2(materialize_final=False)
    freeze = bench._freeze_regions_and_masks(benchmark, output_dir=tmp_path)
    calibration = bench._calibration(
        architecture_id="B",
        benchmark=benchmark,
        region_manifest=freeze["region_manifest"],
        mask_manifest=freeze["mask_manifest"],
        values={
            "minimum_available_mass": 0.0,
            "minimum_available_fraction": 0.0,
            "minimum_support": 0.0,
            "maximum_contradiction": 1.0,
            "maximum_critical_contradiction": 1.0,
            "exact_winner_threshold": 0.0,
            "exact_winner_margin": 0.0,
            "candidate_relative_margin": 0.0,
            "conflicting_action_separation": 0.0,
            "minimum_supporting_regions": 0,
            "maximum_candidate_set_size": bench.MAXIMUM_USEFUL_CANDIDATE_SET_SIZE,
        },
    )
    ranked = next(iter(bench._canonical_rankings_for_architecture(benchmark=benchmark, freeze=freeze, architecture_id="B", calibration=calibration).values()))
    assert ranked[0].eligible_for_exact is True
    assert bench._strict_zero_margin_exact_eligibility(ranked[0], calibration=calibration) is False
