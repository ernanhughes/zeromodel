from __future__ import annotations

from examples import arcade_visual_video_discriminative_evidence_benchmark as bench


def test_committed_v3_audit_reports_outcome_b() -> None:
    summary = bench._load_json(bench.OUTPUT_DIR_V3 / "architecture-self-retrieval-status.json")
    assert summary["shared_instrument_valid"] is True
    assert summary["stop_outcome"] == "Outcome B"
    assert summary["architectures"]["B3"]["self_retrieval_status"] == "eligible_self_retrieval"
    assert summary["architectures"]["B3"]["canonical_top1_count"] == 112
    assert summary["architectures"]["A3"]["self_retrieval_status"] == "ineligible_self_retrieval"
    assert summary["architectures"]["C3"]["self_retrieval_status"] == "ineligible_self_retrieval"
    assert summary["architectures"]["D3"]["self_retrieval_status"] == "ineligible_self_retrieval"


def test_committed_v3_tie_safety_and_pairwise_symmetry_hold() -> None:
    tie = bench._load_json(bench.OUTPUT_DIR_V3 / "tie-safety-results.json")
    symmetry = bench._load_json(bench.OUTPUT_DIR_V3 / "pairwise-symmetry-results.json")
    equivalence = bench._load_json(bench.OUTPUT_DIR_V3 / "provider-equivalence-results.json")
    assert tie["incorrect_lexical_exact_accepts"] == 0
    assert tie["equal_score_exact_accept_count"] == 0
    assert symmetry["mask_symmetry_valid"] is True
    assert symmetry["pairwise_antisymmetry_valid"] is True
    assert equivalence["all_match"] is True
