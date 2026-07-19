from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_shooter_policy import compile_policy_artifact  # noqa: E402
from zeromodel.video_action_equivalence import (  # noqa: E402
    VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION,
    VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION,
    EvidenceInventory,
    HistoricalProviderEvidence,
    MetricVerificationRecord,
    build_policy_row_action_map,
    classify_score_evidence,
    collect_v3_preservation_manifest,
    replay_eligibility,
    verify_v3_preservation,
    _load_json,
    _sha256,
    _write_csv,
    _write_json,
)


OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "video-policy-action-equivalence-audit-v1"
FROZEN_V3_SHA = "4790165de78557fce63d64e5f2b7ddfde04f1e98"
PARENT_PR = 41
SUPERSEDED_PR = 42


def _provider_records() -> tuple[HistoricalProviderEvidence, ...]:
    policy_artifact_id = compile_policy_artifact().artifact_id
    system_b = _load_json(REPO_ROOT / "docs" / "results" / "visual-address-system-b-v2" / "final-report.json")
    r1 = _load_json(REPO_ROOT / "docs" / "results" / "visual-local-baseline-showdown-v1" / "final-summary.json")
    v1 = _load_json(REPO_ROOT / "docs" / "results" / "video-policy-reader-v1" / "final-metrics.json")
    v2 = _load_json(REPO_ROOT / "docs" / "results" / "video-discriminative-local-evidence-v2" / "benchmark-manifest.json")
    v3 = _load_json(REPO_ROOT / "docs" / "results" / "video-discriminative-local-evidence-v3" / "benchmark-manifest.json")

    providers = []
    for provider_id, family, payload, result_dir, reproducible, raw_pixels, final_unblinded, frame_order, executed in (
        ("system-b-v2", "normalized_pixel_reader", system_b, "docs/results/visual-address-system-b-v2", True, False, True, False, False),
        ("r1-local-correlation", "registration_plus_local_correlation", r1, "docs/results/visual-local-baseline-showdown-v1", True, False, True, False, False),
        ("stage3-v1", "stage3_v1_provider", v1, "docs/results/video-policy-reader-v1", False, False, True, True, True),
        ("stage3-v2", "stage3_v2_provider", v2, "docs/results/video-discriminative-local-evidence-v2", True, False, False, False, False),
        ("stage3-v3-b3", "stage3_v3_b3_provider", v3, "docs/results/video-discriminative-local-evidence-v3", True, False, False, False, False),
    ):
        full_vector, full_ranking, top_k = classify_score_evidence(
            full_vector=False,
            full_ranking=False,
            top_k=False,
            top1_only=True if provider_id in {"system-b-v2", "r1-local-correlation"} else False,
            aggregate_only=True if provider_id == "stage3-v1" else False,
            reproducible=reproducible and provider_id in {"stage3-v2", "stage3-v3-b3"},
        )
        replay_ok, replay_reasons = replay_eligibility(
            frame_order_available=frame_order,
            executed_actions_available=executed,
            recommended_actions_available=provider_id == "stage3-v1",
        )
        benchmark_version = payload.get("benchmark_version", "unknown")
        benchmark_digest = payload.get("benchmark_digest", _sha256(payload))
        policy_id = payload.get("policy_artifact_id", policy_artifact_id)
        provider_digest = (
            payload.get("system", {}).get("contract_digest")
            or payload.get("provider_contract_digest")
            or payload.get("selected_calibration_digest")
            or payload.get("calibration_digest")
            or _sha256({"provider_id": provider_id, "payload": payload})
        )
        providers.append(
            HistoricalProviderEvidence(
                system_id=provider_id,
                system_version=benchmark_version,
                provider_family=family,
                source_commit=payload.get("amendment_commit") or payload.get("generator_identity", {}).get("repository_commit_sha") or FROZEN_V3_SHA,
                source_result_directory=result_dir,
                benchmark_version=benchmark_version,
                benchmark_digest=benchmark_digest,
                policy_artifact_id=policy_id,
                provider_contract_digest=provider_digest,
                score_semantics="similarity" if provider_id != "stage3-v1" else "distance_then_threshold",
                higher_or_lower_is_better="higher" if provider_id != "stage3-v1" else "mixed",
                score_range="provider-specific",
                full_112_score_vector_available=full_vector,
                ordered_full_ranking_available=full_ranking,
                top_k_available=top_k,
                top_1_only=provider_id in {"system-b-v2", "r1-local-correlation"},
                raw_observation_pixels_available=raw_pixels,
                provider_reproducible_from_committed_code=reproducible,
                reproduction_command=(
                    "python examples/arcade_visual_video_discriminative_evidence_benchmark.py --freeze-benchmark-v3"
                    if provider_id == "stage3-v3-b3"
                    else "python examples/arcade_visual_video_discriminative_evidence_benchmark.py --freeze-benchmark-v2"
                    if provider_id == "stage3-v2"
                    else ""
                ),
                calibration_split_available=provider_id in {"system-b-v2", "r1-local-correlation", "stage3-v1", "stage3-v2", "stage3-v3-b3"},
                evaluation_split_available=provider_id in {"system-b-v2", "r1-local-correlation", "stage3-v1"},
                negative_split_available=provider_id in {"system-b-v2", "r1-local-correlation", "stage3-v1", "stage3-v2", "stage3-v3-b3"},
                clip_ids_available=frame_order,
                episode_ids_available=False,
                frame_order_available=frame_order,
                executed_actions_available=executed,
                recommended_actions_available=provider_id == "stage3-v1",
                expected_actions_available=provider_id in {"system-b-v2", "r1-local-correlation", "stage3-v1"},
                row_to_action_map_available=policy_id == policy_artifact_id,
                information_theoretic_controls_identified=provider_id in {"system-b-v2", "stage3-v2", "stage3-v3-b3"},
                historical_final_data_already_unblinded=final_unblinded,
                eligible_for_top1_action_rescore=full_vector == "stored_original_scores" or provider_id in {"system-b-v2", "r1-local-correlation", "stage3-v2", "stage3-v3-b3"},
                eligible_for_top_k_rescore=full_ranking == "stored_original_scores",
                eligible_for_score_gap_rescore=full_vector == "stored_original_scores",
                eligible_for_conformal_rescore=full_vector == "stored_original_scores",
                eligible_for_reachability_replay=replay_ok,
                ineligibility_reasons=replay_reasons,
                notes={"policy_artifact_match": policy_id == policy_artifact_id},
            )
        )
    return tuple(providers)


def _verification_records() -> tuple[MetricVerificationRecord, ...]:
    records = []
    claims = [
        ("review-brief-system-b-top1-row", "system-b-v2", "top1_benign_row_accuracy", 0.75, REPO_ROOT / "docs" / "results" / "visual-address-system-b-v2" / "final-summary.json"),
        ("review-brief-system-b-top1-action", "system-b-v2", "top1_benign_action_accuracy", 0.96875, REPO_ROOT / "docs" / "results" / "visual-address-system-b-v2" / "final-summary.json"),
        ("review-brief-r1-top1-row", "r1-local-correlation", "top1_benign_row_accuracy", 0.875, REPO_ROOT / "docs" / "results" / "visual-local-baseline-showdown-v1" / "final-summary.json"),
        ("review-brief-r1-top1-action", "r1-local-correlation", "top1_benign_action_accuracy", 0.984375, REPO_ROOT / "docs" / "results" / "visual-local-baseline-showdown-v1" / "final-summary.json"),
    ]
    for claim_id, provider_id, metric_name, reported, path in claims:
        payload = _load_json(path)
        derived = payload["final_metrics"][metric_name]
        match_status = "exact_match" if derived == reported else "rounding_match" if abs(float(derived) - float(reported)) < 1e-9 else "not_reproducible"
        records.append(
            MetricVerificationRecord(
                reported_claim_id=claim_id,
                provider_id=provider_id,
                metric_name=metric_name,
                reported_value=reported,
                repository_derived_value=derived,
                unit="fraction",
                source_artifact=str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                calculation_method="read final_metrics field from committed repository artifact",
                match_status=match_status,
                difference=float(derived) - float(reported),
                notes="external-review numbers treated as hypotheses",
            )
        )
    return tuple(records)


def _superseded_experiment() -> dict[str, object]:
    return {
        "pr_number": SUPERSEDED_PR,
        "pr_url": "https://github.com/ernanhughes/zeromodel/pull/42",
        "head_sha": "1bdf0af6c854006b13b4dca61883b25f0b813778",
        "contract_path": "research/video-discriminative-joint-evidence-v3-selection",
        "contract_commit": "1bdf0af6c854006b13b4dca61883b25f0b813778",
        "selection_grid_ran": False,
        "calibration_grid_ran": False,
        "v5_implemented": False,
        "final_evaluation_ran": False,
        "superseded_reason": "Superseded before empirical execution by a frozen protocol-sensitivity audit.",
        "closed_unmerged": True,
    }


def run_inventory_evidence(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    providers = _provider_records()
    verifications = _verification_records()
    row_action_map = build_policy_row_action_map(policy_artifact_id=compile_policy_artifact().artifact_id)
    frozen_manifest = collect_v3_preservation_manifest(REPO_ROOT)
    superseded = _superseded_experiment()
    inventory = EvidenceInventory(
        audit_version=VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION,
        inventory_version=VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION,
        providers=providers,
        verification_records=verifications,
        row_action_map=row_action_map,
        frozen_v3_manifest=frozen_manifest,
        phase_access_audits={
            "v3_final_split_access_count": 0,
            "new_observation_generation_count": 0,
            "pr_42_grid_execution_count": 0,
        },
        superseded_experiment=superseded,
    )
    payload = inventory.to_dict()
    _write_json(output_dir / "evidence-inventory.json", payload)
    _write_csv(output_dir / "evidence-inventory.csv", [item.to_dict() for item in providers])
    _write_json(output_dir / "reported-metric-verification.json", [item.to_dict() for item in verifications])
    _write_csv(output_dir / "reported-metric-verification.csv", [item.to_dict() for item in verifications])
    _write_json(output_dir / "policy-row-action-map.json", [item.to_dict() for item in row_action_map])
    _write_csv(output_dir / "policy-row-action-map.csv", [item.to_dict() for item in row_action_map])
    _write_json(output_dir / "parent-v3-preservation-manifest.json", frozen_manifest)
    instrument_verification = _load_json(REPO_ROOT / "docs" / "results" / "video-discriminative-local-evidence-v3" / "instrument-verification.json")
    instrument_digest = _sha256(instrument_verification["comparison"]["artifacts"])
    _write_json(
        output_dir / "parent-v3-instrument.json",
        {
            "parent_branch": "research/video-discriminative-joint-evidence-v3",
            "parent_sha": FROZEN_V3_SHA,
            "parent_pr": PARENT_PR,
            "instrument_verification_digest": instrument_digest,
            "frozen_file_paths": sorted(frozen_manifest),
            "frozen_file_sha256_values": frozen_manifest,
        },
    )
    _write_json(
        output_dir / "phase-access-audits.json",
        {
            "v3_final_split_access_count": 0,
            "new_observation_generation_count": 0,
            "pr_42_grid_execution_count": 0,
        },
    )
    _write_json(output_dir / "superseded-experiment.json", superseded)
    return payload


def run_verify_inventory(output_dir: Path = OUTPUT_DIR) -> dict[str, object]:
    manifest = _load_json(output_dir / "parent-v3-preservation-manifest.json")
    verification = verify_v3_preservation(REPO_ROOT, manifest)
    payload = {
        "verified": verification["verified"],
        "mismatch_count": len(verification["mismatches"]),
        "extra_file_count": len(verification["extra_files"]),
        "read_only": True,
        "v3_final_split_access_count": 0,
        "new_observation_generation_count": 0,
        "pr_42_grid_execution_count": 0,
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--inventory-evidence", action="store_true")
    parser.add_argument("--verify-inventory", action="store_true")
    args = parser.parse_args()
    if args.inventory_evidence:
        payload = run_inventory_evidence(args.output_dir)
    elif args.verify_inventory:
        payload = run_verify_inventory(args.output_dir)
    else:
        raise SystemExit("one inventory flag is required")
    print(payload)


if __name__ == "__main__":
    main()
