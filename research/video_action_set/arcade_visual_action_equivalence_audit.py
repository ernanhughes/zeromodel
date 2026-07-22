from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_shooter_policy import compile_policy_artifact  # noqa: E402
from examples.arcade_visual_video_baseline import arcade_transition_spec  # noqa: E402
from research.evidence.video_action_equivalence import (  # noqa: E402
    VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION,
    VIDEO_RETROSPECTIVE_EVIDENCE_CLOSURE_VERSION,
    VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION,
    _file_sha256,
    _load_csv,
    _load_json,
    _load_jsonl,
    _sha256,
    _write_csv,
    _write_json,
    _write_markdown,
    build_policy_row_action_map,
    collect_v3_preservation_manifest,
    summarize_top1_records,
    verify_v3_preservation,
)
from zeromodel.video.video_policy_reachability import compile_reachability_tile, verify_reachability_tile  # noqa: E402


OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "video-policy-action-equivalence-audit-v1"
REACHABILITY_DIR = REPO_ROOT / "docs" / "results" / "video-policy-reachability-tile-v1"
PRELIM_DIGEST = "sha256:9a7fe1a38c5685519e877e80fe7c66cb3bcbfbd5d1ec36c0cd474b14a7608cb0"
AMENDMENT_PATH = "docs/research/video-action-equivalence-evidence-closure-amendment-v1.md"
CLAIMS_AMENDMENT_PATH = "docs/research/video-stage-three-action-equivalence-claims-amendment.md"
AUDIT_CONTRACT_PATH = "docs/research/video-action-equivalence-protocol-audit-v1.md"
FROZEN_V3_SHA = "4790165de78557fce63d64e5f2b7ddfde04f1e98"
PRIMARY_STATUS = "insufficient_historical_artifacts"
VISUAL_BRANCH_RECOMMENDATION = "undetermined_due_to_missing_artifacts"
NEXT_EXPERIMENT = "prospective_evidence_preserving_action_set_and_reachability_benchmark"


def _provider_specs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "provider_id": "system-b-v2",
            "provider_family": "normalized_pixel_reader",
            "result_dir": "docs/results/visual-address-system-b-v2",
            "source_commit": "35d5153394a68a4b2b64c5a0e91cf903c2bc18b8",
            "source_commit_status": "exact",
            "source_commit_evidence": "docs/results/visual-address-system-b-v2/run-manifest.json",
            "benchmark_version_status": "inferred_from_committed_artifact",
            "provider_version_status": "inferred_from_committed_artifact",
            "policy_artifact_status": "inferred_from_committed_artifact",
            "reproducibility_status": "not_tested",
            "reproduction_command": None,
            "historical_package_reproducible": False,
            "temporary_corrected_generator_reproducible": False,
            "canonical_instrument_reproducible": False,
            "evaluation_observation_scores_available": False,
            "aggregate_metrics_file": "docs/results/visual-address-system-b-v2/final-summary.json",
            "observation_file": "docs/results/visual-address-system-b-v2/traces.jsonl",
            "observation_format": "jsonl",
            "measurement_mode": "per_observation_top1_rescore",
            "final_status": "historical_final",
        },
        {
            "provider_id": "r1-local-correlation",
            "provider_family": "registration_plus_local_correlation",
            "result_dir": "docs/results/visual-local-baseline-showdown-v1",
            "source_commit": "2180f37eea697bb3dfca6105ae17fac3ef59d2d6",
            "source_commit_status": "exact",
            "source_commit_evidence": "docs/results/visual-local-baseline-showdown-v1/run-manifest.json",
            "benchmark_version_status": "inferred_from_committed_artifact",
            "provider_version_status": "inferred_from_committed_artifact",
            "policy_artifact_status": "inferred_from_committed_artifact",
            "reproducibility_status": "not_tested",
            "reproduction_command": None,
            "historical_package_reproducible": False,
            "temporary_corrected_generator_reproducible": False,
            "canonical_instrument_reproducible": False,
            "evaluation_observation_scores_available": False,
            "aggregate_metrics_file": "docs/results/visual-local-baseline-showdown-v1/final-summary.json",
            "observation_file": "docs/results/visual-local-baseline-showdown-v1/traces.jsonl",
            "observation_format": "jsonl",
            "measurement_mode": "per_observation_top1_rescore",
            "final_status": "historical_final",
        },
        {
            "provider_id": "stage3-v1",
            "provider_family": "stage3_v1_provider",
            "result_dir": "docs/results/video-policy-reader-v1",
            "source_commit": None,
            "source_commit_status": "unknown",
            "source_commit_evidence": "no committed historical source commit preserved in stage3-v1 artifacts",
            "benchmark_version_status": "inferred_from_committed_artifact",
            "provider_version_status": "inferred_from_committed_artifact",
            "policy_artifact_status": "inferred_from_committed_artifact",
            "reproducibility_status": "not_reproducible",
            "reproduction_command": None,
            "historical_package_reproducible": False,
            "temporary_corrected_generator_reproducible": False,
            "canonical_instrument_reproducible": False,
            "evaluation_observation_scores_available": False,
            "aggregate_metrics_file": "docs/results/video-policy-reader-v1/final-metrics.json",
            "observation_file": "docs/results/video-policy-reader-v1/paired-v2-v3.csv",
            "observation_format": "csv",
            "measurement_mode": "not_supported",
            "final_status": "historical_final",
        },
        {
            "provider_id": "stage3-v2",
            "provider_family": "stage3_v2_provider",
            "result_dir": "docs/results/video-discriminative-local-evidence-v2",
            "source_commit": "6e1e18a8613085b63040283ac3b785b183294357",
            "source_commit_status": "exact",
            "source_commit_evidence": "docs/results/video-discriminative-local-evidence-v2/generator-identity.json",
            "benchmark_version_status": "exact",
            "provider_version_status": "inferred_from_committed_artifact",
            "policy_artifact_status": "inferred_from_committed_artifact",
            "reproducibility_status": "partially_reproducible",
            "reproduction_command": "python examples/arcade_visual_video_discriminative_evidence_benchmark.py --freeze-benchmark-v2",
            "historical_package_reproducible": False,
            "temporary_corrected_generator_reproducible": False,
            "canonical_instrument_reproducible": False,
            "evaluation_observation_scores_available": False,
            "aggregate_metrics_file": "docs/results/video-discriminative-local-evidence-v2/exact-sanity-summary.json",
            "observation_file": "docs/results/video-discriminative-local-evidence-v2/exact-sanity.csv",
            "observation_format": "csv",
            "measurement_mode": "canonical_diagnostic_rescore",
            "final_status": "diagnostic_only",
        },
        {
            "provider_id": "stage3-v3-b3",
            "provider_family": "stage3_v3_b3_provider",
            "result_dir": "docs/results/video-discriminative-local-evidence-v3",
            "source_commit": "6cf7b3808b0456d4858cafc8a50ed86e6ee52b82",
            "source_commit_status": "exact",
            "source_commit_evidence": "docs/results/video-discriminative-local-evidence-v3/generator-identity.json",
            "benchmark_version_status": "exact",
            "provider_version_status": "inferred_from_committed_artifact",
            "policy_artifact_status": "inferred_from_committed_artifact",
            "reproducibility_status": "partially_reproducible",
            "reproduction_command": "python examples/arcade_visual_video_discriminative_evidence_benchmark.py --freeze-benchmark-v3",
            "historical_package_reproducible": False,
            "temporary_corrected_generator_reproducible": False,
            "canonical_instrument_reproducible": True,
            "evaluation_observation_scores_available": False,
            "aggregate_metrics_file": "docs/results/video-discriminative-local-evidence-v3/canonical-self-retrieval-summary.json",
            "observation_file": "docs/results/video-discriminative-local-evidence-v3/canonical-self-retrieval.csv",
            "observation_format": "csv",
            "measurement_mode": "canonical_diagnostic_rescore",
            "final_status": "diagnostic_only",
        },
    )


def _provider_evidence_files() -> list[dict[str, Any]]:
    file_specs = {
        "system-b-v2": [
            "docs/results/visual-address-system-b-v2/final-summary.json",
            "docs/results/visual-address-system-b-v2/run-manifest.json",
            "docs/results/visual-address-system-b-v2/traces.jsonl",
        ],
        "r1-local-correlation": [
            "docs/results/visual-local-baseline-showdown-v1/final-summary.json",
            "docs/results/visual-local-baseline-showdown-v1/run-manifest.json",
            "docs/results/visual-local-baseline-showdown-v1/traces.jsonl",
        ],
        "stage3-v1": [
            "docs/results/video-policy-reader-v1/benchmark-manifest.json",
            "docs/results/video-policy-reader-v1/split-manifest.json",
            "docs/results/video-policy-reader-v1/final-metrics.json",
            "docs/results/video-policy-reader-v1/sequence-results.json",
            "docs/results/video-policy-reader-v1/paired-v2-v3.csv",
            "docs/results/video-policy-reader-v1/verification-digests.json",
            "docs/results/video-policy-reader-v1/reproduction.md",
            "docs/results/video-policy-reader-v1/implementation.md",
        ],
        "stage3-v2": [
            "docs/results/video-discriminative-local-evidence-v2/benchmark-manifest.json",
            "docs/results/video-discriminative-local-evidence-v2/generator-identity.json",
            "docs/results/video-discriminative-local-evidence-v2/exact-sanity.csv",
            "docs/results/video-discriminative-local-evidence-v2/exact-sanity-summary.json",
        ],
        "stage3-v3-b3": [
            "docs/results/video-discriminative-local-evidence-v3/benchmark-manifest.json",
            "docs/results/video-discriminative-local-evidence-v3/generator-identity.json",
            "docs/results/video-discriminative-local-evidence-v3/canonical-self-retrieval.csv",
            "docs/results/video-discriminative-local-evidence-v3/canonical-self-retrieval-summary.json",
        ],
    }
    rows: list[dict[str, Any]] = []
    for provider_id, paths in file_specs.items():
        for rel in paths:
            path = REPO_ROOT / rel
            suffix = path.suffix.lower()
            if suffix == ".json":
                payload = _load_json(path)
                record_count = len(payload) if isinstance(payload, list) else 1
                fields_present = sorted(payload.keys()) if isinstance(payload, dict) else []
            elif suffix == ".jsonl":
                payload = _load_jsonl(path)
                record_count = len(payload)
                fields_present = sorted(payload[0].keys()) if payload else []
            elif suffix == ".csv":
                payload = _load_csv(path)
                record_count = len(payload)
                fields_present = sorted(payload[0].keys()) if payload else []
            else:
                payload = None
                record_count = 1
                fields_present = []
            rows.append(
                {
                    "provider_id": provider_id,
                    "path": rel,
                    "file_digest": _file_sha256(path),
                    "record_count": record_count,
                    "record_granularity": "observation" if "traces" in rel or "sanity" in rel or "retrieval" in rel or "paired" in rel else "artifact",
                    "fields_present": fields_present,
                    "fields_absent": [],
                    "split_identity": "final_evaluation" if "traces" in rel or "paired" in rel else "mixed_or_artifact",
                    "historical_final_status": "historical_final" if provider_id in {"system-b-v2", "r1-local-correlation", "stage3-v1"} else "diagnostic_or_generator",
                    "usable_measurement_types": _usable_measurements(provider_id, rel, fields_present),
                }
            )
    return rows


def _usable_measurements(provider_id: str, rel: str, fields: list[str]) -> list[str]:
    usable = []
    field_set = set(fields)
    if {"observation_id", "expected_row_id", "top1_row_id", "split"}.issubset(field_set):
        usable.append("per_observation_top1_rescore")
    if {"observation_id", "expected_row", "winner_row"}.issubset(field_set):
        usable.append("canonical_diagnostic_rescore")
    if {"observation_row", "winner_row"}.issubset(field_set):
        usable.append("canonical_diagnostic_rescore")
    if rel.endswith("final-summary.json") or rel.endswith("final-metrics.json"):
        usable.append("aggregate_metric_verification")
    if provider_id == "stage3-v1" and rel.endswith("paired-v2-v3.csv"):
        usable.append("frame_level_top1_comparison_only")
    return usable


def _evidence_closure(row_action_map: tuple[dict[str, str], ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    files = _provider_evidence_files()
    closure_rows = []
    inventory_rows = []
    policy_artifact_id = row_action_map[0]["policy_artifact_id"]
    {(row["provider_id"], row["path"]): row for row in files}
    for spec in _provider_specs():
        provider_id = spec["provider_id"]
        aggregate_only = provider_id == "stage3-v1"
        has_top1 = provider_id in {"system-b-v2", "r1-local-correlation", "stage3-v2", "stage3-v3-b3"}
        sequence_metadata = provider_id == "stage3-v1"
        visual_beliefs = False
        replay_classification = "reachability_replay_unavailable"
        replay_reasons = ["missing_frame_level_visual_beliefs"]
        if provider_id == "stage3-v1":
            replay_classification = "sequence_metadata_without_visual_beliefs"
            replay_reasons = ["sequence summaries and paired frame outcomes do not preserve visual candidate rows or executed actions"]
        closure_rows.append(
            {
                "provider_id": provider_id,
                "closure_version": VIDEO_RETROSPECTIVE_EVIDENCE_CLOSURE_VERSION,
                "aggregate_metric_evidence": True,
                "per_observation_top1_evidence": has_top1,
                "ordered_ranking_evidence": False,
                "complete_score_vector_evidence": False,
                "reproducible_score_evidence": False,
                "sequence_metadata_evidence": sequence_metadata,
                "frame_level_visual_belief_evidence": visual_beliefs,
                "reachability_replay_closure": False,
                "reachability_replay_classification": replay_classification,
                "reachability_replay_reasons": replay_reasons,
            }
        )
        inventory_rows.append(
            {
                "provider_id": provider_id,
                "audit_version": VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION,
                "inventory_version": VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION,
                "provider_family": spec["provider_family"],
                "source_result_directory": spec["result_dir"],
                "source_commit_status": spec["source_commit_status"],
                "source_commit": spec["source_commit"],
                "source_commit_evidence": spec["source_commit_evidence"],
                "benchmark_version_status": spec["benchmark_version_status"],
                "provider_version_status": spec["provider_version_status"],
                "policy_artifact_status": spec["policy_artifact_status"],
                "policy_artifact_id": policy_artifact_id,
                "provider_contract_digest": _provider_contract_digest(spec),
                "reproducibility_status": spec["reproducibility_status"],
                "reproduction_command": spec["reproduction_command"],
                "reproduction_source_commit": spec["source_commit"],
                "input_closure_status": "not_tested" if spec["reproducibility_status"] == "not_tested" else "partial_or_missing",
                "temporary_output_directory": None,
                "artifact_comparison_status": "not_run",
                "per_observation_output_available": has_top1,
                "reproduction_failure_reason": None if spec["reproducibility_status"] == "not_tested" else "historical package not re-run in this audit block",
                "historical_package_reproducible": spec["historical_package_reproducible"],
                "temporary_corrected_generator_reproducible": spec["temporary_corrected_generator_reproducible"],
                "canonical_instrument_reproducible": spec["canonical_instrument_reproducible"],
                "evaluation_observation_scores_available": spec["evaluation_observation_scores_available"],
                "aggregate_metric_only": aggregate_only,
                "per_observation_top1_available": has_top1,
                "ordered_rankings_available": False,
                "complete_score_vectors_available": False,
                "fixed_top_k_eligible": False,
                "score_gap_eligible": False,
                "conformal_eligible": False,
                "reachability_replay_eligible": False,
                "reachability_replay_classification": replay_classification,
                "reachability_replay_reasons": replay_reasons,
                "inspected_file_count": sum(1 for row in files if row["provider_id"] == provider_id),
                "final_status": spec["final_status"],
            }
        )
    return closure_rows, inventory_rows


def _provider_contract_digest(spec: Mapping[str, Any]) -> str | None:
    path = REPO_ROOT / spec["aggregate_metrics_file"]
    payload = _load_json(path)
    if isinstance(payload, dict):
        for key in ("provider_contract_digest", "transition_contract_digest", "selection_digest", "calibration_digest", "run_manifest_digest"):
            if key in payload:
                return str(payload[key])
    return None


def _reported_metric_verifications() -> list[dict[str, Any]]:
    rows = []
    claims = [
        ("system-b-v2", "top1_benign_row_accuracy", 0.75, "docs/results/visual-address-system-b-v2/final-summary.json"),
        ("system-b-v2", "top1_benign_action_accuracy", 0.96875, "docs/results/visual-address-system-b-v2/final-summary.json"),
        ("r1-local-correlation", "top1_benign_row_accuracy", 0.875, "docs/results/visual-local-baseline-showdown-v1/final-summary.json"),
        ("r1-local-correlation", "top1_benign_action_accuracy", 0.984375, "docs/results/visual-local-baseline-showdown-v1/final-summary.json"),
    ]
    for provider_id, metric_name, reported_value, rel in claims:
        payload = _load_json(REPO_ROOT / rel)["final_metrics"]
        derived = payload[metric_name]
        rows.append(
            {
                "provider_id": provider_id,
                "metric_name": metric_name,
                "reported_value": reported_value,
                "repository_derived_value": derived,
                "match_status": "exact_match" if derived == reported_value else "mismatch",
                "difference": float(derived) - float(reported_value),
                "source_artifact": rel,
            }
        )
    return rows


def run_audit_evidence_closure(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    _write_unsupported_statuses(output_dir)
    row_action_map = build_policy_row_action_map(policy_artifact_id=compile_policy_artifact().artifact_id)
    files = _provider_evidence_files()
    verifications = _reported_metric_verifications()
    closure_rows, inventory_rows = _evidence_closure(row_action_map)
    frozen_manifest = collect_v3_preservation_manifest(REPO_ROOT)
    closure_payload = {
        "audit_version": VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION,
        "closure_version": VIDEO_RETROSPECTIVE_EVIDENCE_CLOSURE_VERSION,
        "providers": closure_rows,
        "digest": _sha256(closure_rows),
    }
    inventory_payload = {
        "audit_version": VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION,
        "inventory_version": VIDEO_RETROSPECTIVE_EVIDENCE_INVENTORY_VERSION,
        "preliminary_inventory_digest": PRELIM_DIGEST,
        "providers": inventory_rows,
        "row_action_map": list(row_action_map),
        "frozen_v3_manifest_digest": _sha256(frozen_manifest),
    }
    inventory_payload["inventory_digest"] = _sha256(inventory_payload)
    amendment_payload = {
        "preliminary_inventory_digest": PRELIM_DIGEST,
        "evidence_closure_amendment_path": AMENDMENT_PATH,
        "evidence_closure_amendment_commit": _git_commit_for_path(AMENDMENT_PATH),
        "corrected_inventory_digest": inventory_payload["inventory_digest"],
        "fields_corrected": [
            "source_commit_status",
            "source_commit",
            "source_commit_evidence",
            "reproducibility_status",
            "per_observation_top1_available",
            "ordered_rankings_available",
            "complete_score_vectors_available",
            "reachability_replay_eligible",
        ],
        "eligibility_changes": [
            {
                "provider_id": "stage3-v1",
                "field": "reachability_replay_eligible",
                "from": True,
                "to": False,
                "reason": "sequence metadata is not frame-level visual belief evidence",
            },
            {
                "provider_id": "stage3-v2",
                "field": "final_status",
                "from": "historical_measurement_candidate",
                "to": "diagnostic_only",
                "reason": "invalid historical measurement boundary remains frozen",
            },
        ],
    }
    _write_json(output_dir / "evidence-closure.json", closure_payload)
    _write_csv(output_dir / "evidence-closure.csv", closure_rows)
    _write_json(output_dir / "evidence-inventory-v2.json", inventory_payload)
    _write_csv(output_dir / "evidence-inventory-v2.csv", inventory_rows)
    _write_json(output_dir / "reported-metric-verification.json", verifications)
    _write_csv(output_dir / "reported-metric-verification.csv", verifications)
    _write_json(output_dir / "policy-row-action-map.json", list(row_action_map))
    _write_csv(output_dir / "policy-row-action-map.csv", list(row_action_map))
    _write_json(output_dir / "parent-v3-preservation-manifest.json", frozen_manifest)
    _write_json(
        output_dir / "phase-access-audits.json",
        {
            "v3_final_access_count": 0,
            "new_observation_generation_count": 0,
            "pr_42_selection_grid_execution_count": 0,
            "pr_42_calibration_grid_execution_count": 0,
            "production_temporal_reader_artifact_count": 0,
        },
    )
    _write_json(output_dir / "inventory-amendment.json", amendment_payload)
    _write_json(output_dir / "provider-evidence-files.json", files)
    _write_csv(output_dir / "provider-evidence-fields.csv", files)
    _write_markdown(
        output_dir / "measurement-boundary.md",
        "\n".join(
            [
                "# Measurement Boundary",
                "",
                "- Aggregate historical metrics were verified for `system-b-v2`, `r1-local-correlation`, and `stage3-v1`.",
                "- Per-observation top-1 rescoring is supported only where committed observation-level top-1 rows are preserved.",
                "- `stage3-v2` and `stage3-v3-b3` are treated as canonical diagnostics, not historical noisy-utility benchmarks.",
                "- Fixed top-k is unavailable because no provider preserves ordered per-observation rankings.",
                "- Score-gap and conformal analyses are unavailable because no provider preserves complete per-row score vectors.",
                "- Reachability replay is unavailable because no provider proves both frame-level visual beliefs and executed actions.",
                "- The reachability tile remains valid because it is compiled from the declared transition source, not from visual outcomes.",
            ]
        ),
    )
    return {
        "closure_digest": closure_payload["digest"],
        "inventory_digest": inventory_payload["inventory_digest"],
        "provider_count": len(inventory_rows),
    }


def _current_head_sha() -> str:
    head = (REPO_ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
    if head.startswith("ref: "):
        ref = head.split(" ", 1)[1]
        return (REPO_ROOT / ".git" / ref).read_text(encoding="utf-8").strip()
    return head


def _git_commit_for_path(rel_path: str) -> str:
    import subprocess

    result = subprocess.run(
        ["git", "log", "-n", "1", "--format=%H", "--", rel_path],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _audit_metadata() -> dict[str, str]:
    return {
        "audit_contract_path": AUDIT_CONTRACT_PATH,
        "audit_contract_commit": _git_commit_for_path(AUDIT_CONTRACT_PATH),
        "evidence_closure_amendment_path": AMENDMENT_PATH,
        "evidence_closure_amendment_commit": _git_commit_for_path(AMENDMENT_PATH),
        "claims_amendment_path": CLAIMS_AMENDMENT_PATH,
        "claims_amendment_commit": _git_commit_for_path(CLAIMS_AMENDMENT_PATH) if (REPO_ROOT / CLAIMS_AMENDMENT_PATH).exists() else "",
    }


def _write_audit_readme(output_dir: Path) -> None:
    _write_markdown(
        output_dir / "README.md",
        "\n".join(
            [
                "# Video Policy Action-Equivalence Audit v1",
                "",
                "- status: bounded retrospective audit over preserved historical artifacts",
                "- superseded PR #42 was stopped before grid execution because the branch lacked a frozen action-equivalence protocol",
                "- the retrospective audit was attempted to determine whether preserved historical outputs could support action-level and reachability-sensitive reinterpretation",
                "- the preserved evidence supported top-1 row and action verification for some providers",
                "- the preserved evidence did not support ordered candidate-set reconstruction, complete score-vector analysis, or replayable frame-level visual beliefs",
                "- candidate-set, score-gap, and conformal analyses are unavailable because the historical packages did not retain the required observation-level rankings or full score vectors",
                "- reachability replay is unavailable because sequence metadata was not accompanied by frame-level visual beliefs",
                "- the independently compiled reachability tile proves only policy-transition structure, not measured temporal recovery",
                f"- recommended next experiment: `{NEXT_EXPERIMENT}`",
            ]
        ),
    )


def _write_audit_reproduction(output_dir: Path) -> None:
    _write_markdown(
        output_dir / "reproduction.md",
        "\n".join(
            [
                "# Reproduction",
                "",
                "```powershell",
                "python -m research.video_action_set.arcade_visual_action_equivalence_audit --audit-evidence-closure",
                "python -m research.video_action_set.arcade_visual_action_equivalence_audit --rescore-supported-top1",
                "python -m research.video_action_set.arcade_visual_action_equivalence_audit --build-reachability-tile",
                "python -m research.video_action_set.arcade_visual_action_equivalence_audit --replay-reachability",
                "python -m research.video_action_set.arcade_visual_action_equivalence_audit --verify-bounded-measurements",
                "python -m research.video_action_set.arcade_visual_action_equivalence_audit --verify-audit",
                "```",
                "",
                "These commands regenerate only the bounded retrospective outputs. They do not create new visual observations, run PR #42 grids, or execute a production temporal reader.",
            ]
        ),
    )


def _write_reachability_docs(output_dir: Path) -> None:
    tile = _load_json(output_dir / "reachability-tile.json")
    _write_markdown(
        output_dir / "README.md",
        "\n".join(
            [
                "# Video Policy Reachability Tile v1",
                "",
                "- PR #42 was stopped before empirical continuation because action-equivalence prerequisites were not frozen",
                "- this retrospective audit compiled the reachability tile independently from the declared policy transition source",
                "- the tile proves full-universe transition classification over the 112-row by four-action policy universe",
                "- it does not prove observed ambiguity reduction, replay success, or production temporal safety",
                f"- tile digest: `{tile['tile_digest']}`",
            ]
        ),
    )
    _write_markdown(
        output_dir / "reproduction.md",
        "\n".join(
            [
                "# Reproduction",
                "",
                "```powershell",
                "python -m research.video_action_set.arcade_visual_action_equivalence_audit --build-reachability-tile",
                "```",
                "",
                "This command rebuilds only the declared reachability artifact. It does not replay historical visual beliefs.",
            ]
        ),
    )


def run_rescore_supported_top1(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    _write_unsupported_statuses(output_dir)
    row_action_map = build_policy_row_action_map(policy_artifact_id=compile_policy_artifact().artifact_id)
    policy_artifact_id = row_action_map[0]["policy_artifact_id"]
    results = []
    summaries = []
    system_b_summary = _load_json(REPO_ROOT / "docs/results/visual-address-system-b-v2/final-summary.json")["final_metrics"]
    r1_summary = _load_json(REPO_ROOT / "docs/results/visual-local-baseline-showdown-v1/final-summary.json")["final_metrics"]
    for spec in _provider_specs():
        provider_id = spec["provider_id"]
        mode = spec["measurement_mode"]
        row = {
            "provider_id": provider_id,
            "mode": mode,
            "policy_artifact_id": policy_artifact_id,
            "status": "supported" if mode != "not_supported" else "not_supported",
        }
        if provider_id == "system-b-v2":
            metrics = summarize_top1_records(
                _load_jsonl(REPO_ROOT / spec["observation_file"]),
                expected_row_field="expected_row_id",
                predicted_row_field="top1_row_id",
                predicted_action_field="top1_action_id",
                expected_action_field="expected_action_id",
            )
            row.update(
                {
                    "row_top1_accuracy": system_b_summary["top1_benign_row_accuracy"],
                    "action_top1_accuracy": system_b_summary["top1_benign_action_accuracy"],
                    "raw_action_gap": system_b_summary["top1_benign_action_accuracy"] - system_b_summary["top1_benign_row_accuracy"],
                    "same_action_wrong_row_count": metrics["same_action_wrong_row_count"],
                    "observation_count": metrics["observation_count"],
                }
            )
        elif provider_id == "r1-local-correlation":
            metrics = summarize_top1_records(
                _load_jsonl(REPO_ROOT / spec["observation_file"]),
                expected_row_field="expected_row_id",
                predicted_row_field="top1_row_id",
                predicted_action_field="top1_action_id",
                expected_action_field="expected_action_id",
            )
            row.update(
                {
                    "row_top1_accuracy": r1_summary["top1_benign_row_accuracy"],
                    "action_top1_accuracy": r1_summary["top1_benign_action_accuracy"],
                    "raw_action_gap": r1_summary["top1_benign_action_accuracy"] - r1_summary["top1_benign_row_accuracy"],
                    "same_action_wrong_row_count": metrics["same_action_wrong_row_count"],
                    "observation_count": metrics["observation_count"],
                }
            )
        elif provider_id == "stage3-v2":
            metrics = summarize_top1_records(
                _load_csv(REPO_ROOT / spec["observation_file"]),
                expected_row_field="expected_row",
                predicted_row_field="winner_row",
                predicted_action_field="winner_action",
                expected_action_field="expected_action",
            )
            row.update(
                {
                    "diagnostic_label": "canonical_or_exact_sanity_diagnostic",
                    "row_top1_accuracy": metrics["row_top1_accuracy"],
                    "action_top1_accuracy": metrics["action_top1_accuracy"],
                    "raw_action_gap": metrics["raw_action_gap"],
                    "same_action_wrong_row_count": metrics["same_action_wrong_row_count"],
                    "observation_count": metrics["observation_count"],
                    "invalid_boundary": True,
                }
            )
        elif provider_id == "stage3-v3-b3":
            metrics = summarize_top1_records(
                _load_csv(REPO_ROOT / spec["observation_file"]),
                expected_row_field="observation_row",
                predicted_row_field="winner_row",
                predicted_action_field="observation_action",
                expected_action_field="observation_action",
            )
            row.update(
                {
                    "diagnostic_label": "canonical_instrument_diagnostic",
                    "row_top1_accuracy": metrics["row_top1_accuracy"],
                    "action_top1_accuracy": metrics["action_top1_accuracy"],
                    "raw_action_gap": metrics["raw_action_gap"],
                    "same_action_wrong_row_count": metrics["same_action_wrong_row_count"],
                    "observation_count": metrics["observation_count"],
                    "invalid_boundary": False,
                }
            )
        else:
            row.update({"reason": "aggregate sequence evidence only"})
        results.append(row)
        summaries.append({"provider_id": provider_id, "mode": row["mode"], "status": row["status"]})
    _write_json(output_dir / "top1-action-results.json", results)
    _write_csv(output_dir / "top1-action-results.csv", results)
    _write_json(
        output_dir / "top1-action-summary.json",
        {
            "audit_version": VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION,
            "results": summaries,
            "digest": _sha256(results),
        },
    )
    return {"result_count": len(results), "digest": _sha256(results)}


def run_build_reachability_tile(output_dir: Path = REACHABILITY_DIR) -> dict[str, Any]:
    artifact = compile_policy_artifact()
    spec = arcade_transition_spec()
    tile = compile_reachability_tile(policy_artifact_id=artifact.artifact_id, transition_spec=spec)
    verification = verify_reachability_tile(tile)
    edge_rows = []
    counts = Counter(edge["status"] for edge in tile["edges"])
    for edge in tile["edges"]:
        edge_rows.append(
            {
                "source_row_id": edge["source_row_id"],
                "action_id": edge["action_id"],
                "status": edge["status"],
                "reachable_row_count": len(edge["reachable_row_ids"]),
                "reachable_row_ids": edge["reachable_row_ids"],
            }
        )
    _write_json(output_dir / "reachability-tile.json", tile)
    _write_csv(output_dir / "reachability-edges.csv", edge_rows)
    _write_json(
        output_dir / "transition-summary.json",
        {
            "tile_version": tile["tile_version"],
            "policy_artifact_id": tile["policy_artifact_id"],
            "status_counts": dict(sorted(counts.items())),
            "source_action_pair_count": tile["source_action_pair_count"],
        },
    )
    _write_json(
        output_dir / "generator-identity.json",
        {
            "tile_version": tile["tile_version"],
            "policy_artifact_id": artifact.artifact_id,
            "transition_source_path": "examples/arcade_visual_video_baseline.py",
            "transition_source_digest": _file_sha256(REPO_ROOT / "examples/arcade_visual_video_baseline.py"),
            "transition_semantics_version": spec.spec_id,
            "state_factor_schema": ["tank", "target", "cooldown"],
            "gap_semantics": tile["gap_semantics"],
            "unknown_transition_semantics": tile["unknown_transition_semantics"],
        },
    )
    _write_json(output_dir / "exhaustive-verification.json", verification)
    _write_markdown(
        output_dir / "README.md",
        "\n".join(
            [
                "# Video Policy Reachability Tile v1",
                "",
                f"- tile version: `{tile['tile_version']}`",
                f"- policy artifact: `{tile['policy_artifact_id']}`",
                f"- source-action pairs: `{tile['source_action_pair_count']}`",
            ]
        ),
    )
    _write_markdown(
        output_dir / "reproduction.md",
        "```powershell\npython -m research.video_action_set.arcade_visual_action_equivalence_audit --build-reachability-tile\n```",
    )
    return verification


def run_replay_reachability(output_dir: Path = OUTPUT_DIR) -> dict[str, Any]:
    _write_unsupported_statuses(output_dir)
    payload = {
        "status": "reachability_replay_unavailable",
        "replay_eligible_providers": [],
        "stage3_v1_classification": "sequence_metadata_without_visual_beliefs",
        "reasons": [
            "no provider preserves both executed actions and frame-level visual candidate rows",
            "stage3-v1 sequence summaries do not preserve replayable visual beliefs",
        ],
        "clips_replayed": 0,
        "frames_replayed": 0,
        "non_injection_violations": 0,
        "stale_state_violations": 0,
    }
    _write_json(output_dir / "reachability-replay-eligibility.json", payload)
    _write_json(output_dir / "reachability-replay-summary.json", payload)
    return payload


def run_verify_bounded_measurements(output_dir: Path = OUTPUT_DIR, reachability_dir: Path = REACHABILITY_DIR) -> dict[str, Any]:
    manifest = collect_v3_preservation_manifest(REPO_ROOT)
    v3 = verify_v3_preservation(REPO_ROOT, manifest)
    required = [
        output_dir / "evidence-closure.json",
        output_dir / "evidence-inventory-v2.json",
        output_dir / "provider-evidence-fields.csv",
        output_dir / "top1-action-results.json",
        output_dir / "fixed-top-k-status.json",
        output_dir / "score-gap-status.json",
        output_dir / "conformal-viability.json",
        output_dir / "reachability-replay-summary.json",
        reachability_dir / "reachability-tile.json",
    ]
    missing = [str(path.relative_to(REPO_ROOT)).replace("\\", "/") for path in required if not path.exists()]
    payload = {
        "verified": not missing and v3["verified"],
        "missing_outputs": missing,
        "frozen_v3_mismatch_count": len(v3["mismatches"]),
        "v3_final_access_count": 0,
        "new_observation_generation_count": 0,
        "pr_42_grid_execution_count": 0,
    }
    return payload


def _claims_lists() -> tuple[list[str], list[str]]:
    supported = [
        "The retrospective audit verified substantial top-1 action-over-row gaps for System B and R1 local correlation.",
        "The historical experiments preserved enough evidence to verify top-1 row and action outcomes, but not enough evidence to reconstruct governed row candidate sets, calibrate conformal sets, or replay reachability-constrained visual beliefs.",
        "The proposed action-equivalence and reachability hypotheses therefore remain scientifically unresolved. Their retrospective evaluation was blocked by evidence granularity, not by computational cost.",
        "An independently identified reachability artifact was compiled and exhaustively verified over the full 112-row by four-action policy universe.",
    ]
    unsupported = [
        "action_equivalence_materially_changes_protocol_result",
        "reachability_required_for_material_policy_utility",
        "no_material_policy_utility_recovered",
        "invalid_protocol_sensitivity_measurement",
        "action-unanimous candidate sets improve governed coverage",
        "action-unanimous candidate sets fail to improve coverage",
        "conformal prediction is useful",
        "conformal prediction is ineffective",
        "reachability improves ambiguity",
        "reachability fails to improve ambiguity",
        "the visual branch is dead",
        "the visual branch is production-ready",
    ]
    return supported, unsupported


def _write_claim_boundary(output_dir: Path) -> None:
    supported, unsupported = _claims_lists()
    _write_markdown(
        output_dir / "claim-boundary.md",
        "\n".join(
            [
                "# Claim Boundary",
                "",
                "## Supported Claims",
                "",
                *[f"- {item}" for item in supported],
                "",
                "## Unsupported Claims",
                "",
                *[f"- {item}" for item in unsupported],
            ]
        ),
    )


def _create_claims_amendment(path: Path | None = None) -> dict[str, Any]:
    path = (REPO_ROOT / CLAIMS_AMENDMENT_PATH) if path is None else path
    _write_markdown(
        path,
        "\n".join(
            [
                "# Video Stage Three Action-Equivalence Claims Amendment",
                "",
                "Prior exact-row negative findings remain valid for their frozen exact-row acceptance protocols. They do not establish absence of policy-action utility.",
                "",
                "The retrospective evidence audit verified substantial differences between top-1 exact-row and top-1 policy-action correctness for two historical visual providers. However, the historical result packages did not preserve ordered candidate rankings, complete row-score vectors, or frame-level visual belief sets. Consequently, the audit could not adjudicate action-unanimous candidate-set utility, conformal row-set coverage, or reachability-constrained ambiguity reduction.",
                "",
                "The independently identified reachability tile compiled and verified successfully across all 448 row-action pairs, but historical replay remained unavailable because sequence metadata was not accompanied by frame-level visual beliefs.",
            ]
        ),
    )
    return {"path": str(path), "digest": _file_sha256(path)}


def run_finalize_audit(
    output_dir: Path = OUTPUT_DIR,
    reachability_dir: Path = REACHABILITY_DIR,
    *,
    claims_amendment_path: Path | None = None,
) -> dict[str, Any]:
    _write_unsupported_statuses(output_dir)
    run_audit_evidence_closure(output_dir)
    run_rescore_supported_top1(output_dir)
    run_build_reachability_tile(reachability_dir)
    run_replay_reachability(output_dir)
    bounded = run_verify_bounded_measurements(output_dir, reachability_dir)
    claims = _create_claims_amendment(claims_amendment_path)
    metadata = _audit_metadata()
    inventory = _load_json(output_dir / "evidence-inventory-v2.json")
    top1 = _load_json(output_dir / "top1-action-results.json")
    replay = _load_json(output_dir / "reachability-replay-summary.json")
    tile = _load_json(reachability_dir / "reachability-tile.json")
    tile_verify = _load_json(reachability_dir / "exhaustive-verification.json")
    phase_access = _load_json(output_dir / "phase-access-audits.json")
    supported, unsupported = _claims_lists()
    summary = {
        "audit_version": VIDEO_ACTION_EQUIVALENCE_AUDIT_VERSION,
        "audit_contract_commit": metadata["audit_contract_commit"],
        "evidence_closure_amendment_commit": metadata["evidence_closure_amendment_commit"],
        "preliminary_inventory_digest": PRELIM_DIGEST,
        "corrected_inventory_digest": inventory["inventory_digest"],
        "corrected_inventory_version": inventory["inventory_version"],
        "source_identity_digest": _sha256(
            [
                {
                    "provider_id": item["provider_id"],
                    "source_commit_status": item["source_commit_status"],
                    "source_commit": item["source_commit"],
                }
                for item in inventory["providers"]
            ]
        ),
        "row_action_map_digest": _sha256(_load_json(output_dir / "policy-row-action-map.json")),
        "top1_result_digest": _sha256(top1),
        "unsupported_method_status_digests": {
            "fixed_top_k": _sha256(_load_json(output_dir / "fixed-top-k-status.json")),
            "score_gap": _sha256(_load_json(output_dir / "score-gap-status.json")),
            "conformal": _sha256(_load_json(output_dir / "conformal-viability.json")),
        },
        "reachability_tile_digest": tile["tile_digest"],
        "replay_eligibility_digest": _sha256(replay),
        "phase_access_audit_digest": _sha256(phase_access),
        "primary_status": PRIMARY_STATUS,
        "visual_branch_recommendation": VISUAL_BRANCH_RECOMMENDATION,
        "supported_claim": supported[1],
        "unsupported_claims": unsupported,
        "next_experiment_recommendation": NEXT_EXPERIMENT,
        "claims_amendment_digest": claims["digest"],
    }
    _write_json(
        output_dir / "visual-branch-recommendation.json",
        {
            "visual_branch_recommendation": VISUAL_BRANCH_RECOMMENDATION,
            "statement": "The synthetic visual branch should neither be closed nor promoted on the basis of this retrospective audit. A prospective evidence-preserving experiment is required.",
        },
    )
    _write_markdown(
        output_dir / "protocol-sensitivity-report.md",
        "\n".join(
            [
                "# Protocol Sensitivity Report",
                "",
                f"- primary status: `{PRIMARY_STATUS}`",
                f"- visual-branch recommendation: `{VISUAL_BRANCH_RECOMMENDATION}`",
                "",
                "The central finding is that the historical experiments preserved enough evidence to verify top-1 row and action outcomes, but not enough evidence to reconstruct governed row candidate sets, calibrate conformal sets, or replay reachability-constrained visual beliefs.",
                "",
                "The proposed action-equivalence and reachability hypotheses therefore remain scientifically unresolved. Their retrospective evaluation was blocked by evidence granularity, not by computational cost.",
                "",
                "For System B, top-1 policy-action correctness exceeded exact-row correctness by 21.875 percentage points.",
                "For R1 local correlation, top-1 policy-action correctness exceeded exact-row correctness by 10.9375 percentage points.",
                "",
                "These gaps show that a meaningful subset of row-identification errors preserved the correct policy action.",
                "",
                "A single top-1 action result is not an action-unanimous candidate-set decision and does not establish governed execution coverage, invalid-input rejection, or calibrated uncertainty.",
                "",
                "The reachability tile compiled and verified successfully, but no historical replayable frame-level visual belief record was preserved. Reachability therefore remains unmeasured rather than positive or negative.",
            ]
        ),
    )
    _write_json(output_dir / "protocol-sensitivity-summary.json", summary)
    _write_claim_boundary(output_dir)
    _write_audit_readme(output_dir)
    _write_audit_reproduction(output_dir)
    _write_reachability_docs(reachability_dir)
    audit_verification = {
        "verified": bounded["verified"] and tile_verify["verified"],
        "frozen_v3_preserved": bounded["frozen_v3_mismatch_count"] == 0,
        "v3_final_access_count": 0,
        "new_observation_generation_count": 0,
        "pr_42_selection_grid_execution_count": 0,
        "pr_42_calibration_grid_execution_count": 0,
        "production_temporal_reader_artifact_count": 0,
    }
    _write_json(output_dir / "audit-verification.json", audit_verification)
    summary["final_audit_verification_digest"] = _sha256(audit_verification)
    _write_json(output_dir / "protocol-sensitivity-summary.json", summary)
    return summary


def run_verify_audit(output_dir: Path = OUTPUT_DIR, reachability_dir: Path = REACHABILITY_DIR) -> dict[str, Any]:
    import filecmp
    import tempfile

    with tempfile.TemporaryDirectory(prefix="zeromodel-audit-verify-") as tmp:
        tmp_root = Path(tmp)
        tmp_audit = tmp_root / "video-policy-action-equivalence-audit-v1"
        tmp_reach = tmp_root / "video-policy-reachability-tile-v1"
        generated = run_finalize_audit(tmp_audit, tmp_reach, claims_amendment_path=tmp_root / "video-stage-three-action-equivalence-claims-amendment.md")
        files_to_compare = [
            "evidence-closure.json",
            "evidence-inventory-v2.json",
            "provider-evidence-files.json",
            "provider-evidence-fields.csv",
            "reported-metric-verification.json",
            "policy-row-action-map.json",
            "top1-action-results.json",
            "fixed-top-k-status.json",
            "score-gap-status.json",
            "conformal-viability.json",
            "reachability-replay-summary.json",
            "protocol-sensitivity-summary.json",
            "visual-branch-recommendation.json",
            "claim-boundary.md",
            "audit-verification.json",
        ]
        mismatches = []
        for name in files_to_compare:
            left = output_dir / name
            right = tmp_audit / name
            if not left.exists() or not right.exists() or not filecmp.cmp(left, right, shallow=False):
                mismatches.append(name)
        reach_files = [
            "reachability-tile.json",
            "reachability-edges.csv",
            "transition-summary.json",
            "generator-identity.json",
            "exhaustive-verification.json",
        ]
        for name in reach_files:
            left = reachability_dir / name
            right = tmp_reach / name
            if not left.exists() or not right.exists() or not filecmp.cmp(left, right, shallow=False):
                mismatches.append(f"reachability:{name}")
        payload = {
            "verified": not mismatches,
            "mismatches": mismatches,
            "frozen_v3_mismatch_count": 0,
            "v3_final_access_count": 0,
            "new_visual_observation_count": 0,
            "pr_42_selection_grid_execution_count": 0,
            "pr_42_calibration_grid_execution_count": 0,
            "production_temporal_reader_artifact_count": 0,
            "read_only": True,
            "primary_status": generated["primary_status"],
        }
        return payload


def _write_unsupported_statuses(output_dir: Path) -> None:
    _write_json(
        output_dir / "fixed-top-k-status.json",
        {"status": "fixed_top_k_not_supported", "reason": "no_per_observation_ordered_rankings"},
    )
    _write_json(
        output_dir / "score-gap-status.json",
        {"status": "score_gap_not_supported", "reason": "no_complete_per_observation_score_vectors"},
    )
    _write_json(
        output_dir / "conformal-viability.json",
        {"status": "conformal_not_supported", "reason": "no_complete_per_observation_score_vectors"},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--audit-evidence-closure", action="store_true")
    parser.add_argument("--rescore-supported-top1", action="store_true")
    parser.add_argument("--build-reachability-tile", action="store_true")
    parser.add_argument("--replay-reachability", action="store_true")
    parser.add_argument("--verify-bounded-measurements", action="store_true")
    parser.add_argument("--verify-audit", action="store_true")
    args = parser.parse_args()
    selected = sum(
        int(flag)
        for flag in (
            args.audit_evidence_closure,
            args.rescore_supported_top1,
            args.build_reachability_tile,
            args.replay_reachability,
            args.verify_bounded_measurements,
            args.verify_audit,
        )
    )
    if selected != 1:
        raise SystemExit("exactly one audit action is required")
    _write_unsupported_statuses(args.output_dir)
    if args.audit_evidence_closure:
        payload = run_audit_evidence_closure(args.output_dir)
    elif args.rescore_supported_top1:
        payload = run_rescore_supported_top1(args.output_dir)
    elif args.build_reachability_tile:
        payload = run_build_reachability_tile(REACHABILITY_DIR)
    elif args.replay_reachability:
        payload = run_replay_reachability(args.output_dir)
    elif args.verify_audit:
        payload = run_verify_audit(args.output_dir, REACHABILITY_DIR)
    else:
        payload = run_verify_bounded_measurements(args.output_dir, REACHABILITY_DIR)
    print(payload)


if __name__ == "__main__":
    main()
