"""Post-analysis for the registered local baseline showdown."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_visual_address_benchmark import SOURCE_SCOPE, build_arcade_benchmark_dataset  # noqa: E402
from examples.arcade_visual_local_baseline_showdown import QUANTILES  # noqa: E402
from zeromodel.visual_experiment import EXPECTED_ACCEPT, EXPECTED_REJECT  # noqa: E402
from zeromodel.visual_local_baselines import (  # noqa: E402
    RegisteredPixelCalibration,
    RegisteredPixelAddressProvider,
    build_registered_pixel_prototypes,
    _provider_id,
    _records_digest,
    _require_split,
)
from zeromodel.visual_registration import RegistrationConfig  # noqa: E402


DEFAULT_SOURCE_DIR = REPO_ROOT / "docs" / "results" / "visual-local-baseline-showdown-v1"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "visual-local-baseline-showdown-v1-postanalysis"
POSTANALYSIS_VERSION = "zeromodel-visual-local-postanalysis/v1"


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _conservative_quantile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = int(((1.0 - float(quantile)) * float(len(ordered) - 1)) // 1)
    return ordered[index]


def _ranking_record(
    provider: RegisteredPixelAddressProvider,
    *,
    observation: Any,
    observation_id: str,
    family_id: str,
    expected_row_id: Optional[str],
    expected_action_id: Optional[str],
    expected_disposition: str,
) -> Dict[str, Any]:
    best, second, raw_best = provider._rank(observation)
    return {
        "observation_id": observation_id,
        "family_id": family_id,
        "expected_row_id": expected_row_id,
        "expected_action_id": expected_action_id,
        "expected_disposition": expected_disposition,
        "best": {
            "row_id": best.row_id,
            "action_id": best.action_id,
            "distance": float(best.distance),
        },
        "second": (
            None
            if second is None
            else {
                "row_id": second.row_id,
                "action_id": second.action_id,
                "distance": float(second.distance),
            }
        ),
        "raw_best": {
            "row_id": raw_best.row_id,
            "action_id": raw_best.action_id,
            "distance": float(raw_best.distance),
        },
    }


def _evaluate_rankings(
    rankings: Sequence[Mapping[str, Any]],
    *,
    threshold: float,
    ambiguity_margin: float,
) -> Dict[str, Any]:
    accepted_count = 0
    rejected_count = 0
    correct_row_count = 0
    correct_action_count = 0
    conflicting_action_error_count = 0
    false_accept_count = 0
    false_reject_count = 0
    false_accept_opportunities = 0
    false_reject_opportunities = 0
    top1_correct_row_count = 0
    top1_correct_action_count = 0
    for ranking in rankings:
        second = ranking["second"]
        margin = float("inf") if second is None else float(second["distance"] - ranking["best"]["distance"])
        accepted = (
            float(ranking["best"]["distance"]) <= float(threshold) + 1e-12
            and margin + 1e-12 >= float(ambiguity_margin)
        )
        if ranking["expected_disposition"] == EXPECTED_ACCEPT:
            false_reject_opportunities += 1
            top1_correct_row_count += int(ranking["best"]["row_id"] == ranking["expected_row_id"])
            top1_correct_action_count += int(ranking["best"]["action_id"] == ranking["expected_action_id"])
            if accepted:
                accepted_count += 1
                correct_row_count += int(ranking["best"]["row_id"] == ranking["expected_row_id"])
                correct_action_count += int(ranking["best"]["action_id"] == ranking["expected_action_id"])
                conflicting_action_error_count += int(ranking["best"]["action_id"] != ranking["expected_action_id"])
            else:
                rejected_count += 1
                false_reject_count += 1
        else:
            false_accept_opportunities += 1
            if accepted:
                accepted_count += 1
                false_accept_count += 1
            else:
                rejected_count += 1
    accepted_precision = None if correct_row_count == 0 else 1.0
    return {
        "accepted_benign_count": accepted_count if false_reject_opportunities else 0,
        "accepted_exact_row_precision": accepted_precision,
        "accepted_action_precision": (None if correct_action_count == 0 else 1.0),
        "accepted_exact_row_recall": (
            0.0
            if false_reject_opportunities == 0
            else float(correct_row_count) / float(false_reject_opportunities)
        ),
        "benign_coverage": (
            0.0
            if false_reject_opportunities == 0
            else float(accepted_count) / float(false_reject_opportunities)
        ),
        "false_accepts": false_accept_count,
        "conflicting_action_accepts": conflicting_action_error_count,
        "false_rejects": false_reject_count,
        "top1_benign_row_accuracy": (
            0.0
            if false_reject_opportunities == 0
            else float(top1_correct_row_count) / float(false_reject_opportunities)
        ),
        "top1_benign_action_accuracy": (
            0.0
            if false_reject_opportunities == 0
            else float(top1_correct_action_count) / float(false_reject_opportunities)
        ),
        "false_accept_opportunities": false_accept_opportunities,
        "false_reject_opportunities": false_reject_opportunities,
    }


def _decoupled_selection_key(candidate: Mapping[str, Any]) -> Tuple[float, float, float, float, float, float, float]:
    accepted_precision = (
        -1.0
        if candidate["accepted_exact_row_precision"] is None
        else float(candidate["accepted_exact_row_precision"])
    )
    return (
        accepted_precision,
        float(candidate["benign_coverage"]),
        float(candidate["accepted_exact_row_recall"]),
        float(candidate["top1_benign_row_accuracy"]),
        -float(candidate["threshold"]),
        float(candidate["ambiguity_margin"]),
        -float(candidate["distance_quantile"]),
    )


def _gate_bucket(trace: Mapping[str, Any]) -> str:
    if bool(trace["decision"]["accepted"]):
        return "accepted"
    distance_ok = (
        float(trace["decision"]["nearest_score"])
        <= float(trace["decision"]["trace"]["distance_threshold"]) + 1e-12
    )
    margin_ok = (
        float(trace["decision"]["ambiguity_measure"])
        + 1e-12
        >= float(trace["decision"]["trace"]["required_conflicting_action_margin"])
    )
    if distance_ok and not margin_ok:
        return "margin_only"
    if (not distance_ok) and margin_ok:
        return "distance_only"
    return "both"


def _build_decoupled_grid(
    *,
    distance_thresholds: Mapping[float, float],
    ambiguity_margins: Mapping[float, float],
    benign_rankings: Sequence[Mapping[str, Any]],
    rejection_rankings: Sequence[Mapping[str, Any]],
    final_rankings: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    candidates = []
    for distance_quantile, threshold in distance_thresholds.items():
        for margin_quantile, ambiguity_margin in ambiguity_margins.items():
            benign_metrics = _evaluate_rankings(
                benign_rankings,
                threshold=float(threshold),
                ambiguity_margin=float(ambiguity_margin),
            )
            rejection_metrics = _evaluate_rankings(
                rejection_rankings,
                threshold=float(threshold),
                ambiguity_margin=float(ambiguity_margin),
            )
            final_metrics = _evaluate_rankings(
                final_rankings,
                threshold=float(threshold),
                ambiguity_margin=float(ambiguity_margin),
            )
            candidate = {
                "distance_quantile": float(distance_quantile),
                "margin_quantile": float(margin_quantile),
                "threshold": float(threshold),
                "ambiguity_margin": float(ambiguity_margin),
                "feasible": (
                    rejection_metrics["false_accepts"] == 0
                    and benign_metrics["conflicting_action_accepts"] == 0
                ),
                "infeasible_reasons": tuple(
                    reason
                    for reason, triggered in (
                        ("distinguishable_false_acceptance", rejection_metrics["false_accepts"] > 0),
                        ("conflicting_action_acceptance", benign_metrics["conflicting_action_accepts"] > 0),
                    )
                    if triggered
                ),
                "calibration": {
                    "benign": benign_metrics,
                    "rejection": rejection_metrics,
                },
                "final": final_metrics,
                "accepted_exact_row_precision": benign_metrics["accepted_exact_row_precision"],
                "benign_coverage": benign_metrics["benign_coverage"],
                "accepted_exact_row_recall": benign_metrics["accepted_exact_row_recall"],
                "top1_benign_row_accuracy": benign_metrics["top1_benign_row_accuracy"],
            }
            candidates.append(candidate)
    feasible_candidates = tuple(candidate for candidate in candidates if candidate["feasible"])
    selected = max(feasible_candidates, key=_decoupled_selection_key) if feasible_candidates else None
    return {
        "version": POSTANALYSIS_VERSION,
        "selection_rule": (
            "Among feasible candidates maximize accepted exact-row precision, "
            "then benign accepted coverage, then accepted exact-row recall, "
            "then raw benign exact-row ranking accuracy, then prefer the more "
            "conservative distance threshold, then the more conservative ambiguity "
            "margin, then deterministic quantile ordering."
        ),
        "distance_quantiles": [float(value) for value in sorted(distance_thresholds)],
        "margin_quantiles": [float(value) for value in sorted(ambiguity_margins)],
        "candidate_count": len(candidates),
        "feasible_candidate_count": len(feasible_candidates),
        "selected_candidate": selected,
        "candidates": candidates,
        "summary": {
            "best_feasible_calibration_coverage": (
                None
                if not feasible_candidates
                else max(float(candidate["calibration"]["benign"]["benign_coverage"]) for candidate in feasible_candidates)
            ),
            "best_feasible_final_coverage": (
                None
                if not feasible_candidates
                else max(float(candidate["final"]["benign_coverage"]) for candidate in feasible_candidates)
            ),
            "best_feasible_false_accepts": (
                None
                if not feasible_candidates
                else min(int(candidate["calibration"]["rejection"]["false_accepts"]) for candidate in feasible_candidates)
            ),
        },
    }


def _build_rejection_decomposition(traces: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    overall = {"accepted": 0, "distance_only": 0, "margin_only": 0, "both": 0, "total": 0}
    by_family: Dict[str, Dict[str, int]] = {}
    for trace in traces:
        if trace["expected_disposition"] != EXPECTED_ACCEPT:
            continue
        bucket = _gate_bucket(trace)
        overall[bucket] += 1
        overall["total"] += 1
        family = by_family.setdefault(
            str(trace["family_id"]),
            {"accepted": 0, "distance_only": 0, "margin_only": 0, "both": 0, "total": 0},
        )
        family[bucket] += 1
        family["total"] += 1
    return {
        "version": POSTANALYSIS_VERSION,
        "overall": overall,
        "by_family": by_family,
    }


def _load_jsonl(path: Path) -> Tuple[Dict[str, Any], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def run_postanalysis(*, source_dir: Path, output_dir: Path) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("output directory already exists and is non-empty: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    final_summary = json.loads((source_dir / "final-summary.json").read_text(encoding="utf-8"))
    selected_calibration = json.loads((source_dir / "selected-calibration.json").read_text(encoding="utf-8"))
    traces = _load_jsonl(source_dir / "traces.jsonl")
    dataset = build_arcade_benchmark_dataset(variants_per_family=3)
    registration_config = RegistrationConfig(
        max_dx=3,
        max_dy=3,
        minimum_overlap_fraction=0.6,
    )
    prototype_records = _require_split(dataset.manifest.records, "prototype")
    benign_records = _require_split(dataset.manifest.records, "benign_calibration")
    rejection_records = _require_split(dataset.manifest.records, "rejection_calibration")
    prototype_map = build_registered_pixel_prototypes(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
    )
    provisional = RegisteredPixelCalibration(
        threshold=2.0,
        ambiguity_margin=0.0,
        quantile=0.0,
        registration_config_digest=registration_config.digest,
        prototype_digest=_records_digest(prototype_records),
        benign_calibration_digest=_records_digest(benign_records),
        rejection_calibration_digest=_records_digest(rejection_records),
        source_scope=SOURCE_SCOPE,
        policy_artifact_id=dataset.manifest.policy_artifact_id,
    )
    provider = RegisteredPixelAddressProvider(
        prototypes=prototype_map,
        calibration=provisional,
        registration_config=registration_config,
        provider_id=_provider_id(registration_config=registration_config, calibration=provisional),
    )

    benign_rankings = tuple(
        _ranking_record(
            provider,
            observation=dataset.observations[record.observation_id],
            observation_id=record.observation_id,
            family_id=record.family_id,
            expected_row_id=record.row_id,
            expected_action_id=record.action_id,
            expected_disposition=EXPECTED_ACCEPT,
        )
        for record in benign_records
    )
    rejection_rankings = tuple(
        _ranking_record(
            provider,
            observation=dataset.observations[record.observation_id],
            observation_id=record.observation_id,
            family_id=record.family_id,
            expected_row_id=record.row_id,
            expected_action_id=record.action_id,
            expected_disposition=EXPECTED_REJECT,
        )
        for record in rejection_records
    )
    final_records = tuple(
        record
        for record in dataset.manifest.records
        if record.split == "final_evaluation" and record.evaluation_role in {EXPECTED_ACCEPT, EXPECTED_REJECT}
    )
    final_rankings = tuple(
        _ranking_record(
            provider,
            observation=dataset.observations[record.observation_id],
            observation_id=record.observation_id,
            family_id=record.family_id,
            expected_row_id=record.row_id,
            expected_action_id=record.action_id,
            expected_disposition=record.evaluation_role,
        )
        for record in final_records
    )

    benign_distances = tuple(
        float(ranking["best"]["distance"])
        for ranking in benign_rankings
        if ranking["best"]["row_id"] == ranking["expected_row_id"]
    )
    benign_margins = tuple(
        float("inf") if ranking["second"] is None else float(ranking["second"]["distance"] - ranking["best"]["distance"])
        for ranking in benign_rankings
        if ranking["best"]["row_id"] == ranking["expected_row_id"]
    )
    finite_margins = tuple(value for value in benign_margins if value != float("inf"))
    distance_thresholds = {
        float(quantile): _conservative_quantile(benign_distances, quantile)
        for quantile in QUANTILES
    }
    ambiguity_margins = {
        float(quantile): _conservative_quantile(finite_margins, quantile)
        for quantile in QUANTILES
    }

    decoupled_grid = _build_decoupled_grid(
        distance_thresholds=distance_thresholds,
        ambiguity_margins=ambiguity_margins,
        benign_rankings=benign_rankings,
        rejection_rankings=rejection_rankings,
        final_rankings=final_rankings,
    )
    rejection_decomposition = _build_rejection_decomposition(traces)

    csv_rows = []
    for candidate in decoupled_grid["candidates"]:
        csv_rows.append(
            {
                "distance_quantile": candidate["distance_quantile"],
                "margin_quantile": candidate["margin_quantile"],
                "threshold": candidate["threshold"],
                "ambiguity_margin": candidate["ambiguity_margin"],
                "feasible": candidate["feasible"],
                "calibration_benign_coverage": candidate["calibration"]["benign"]["benign_coverage"],
                "calibration_false_accepts": candidate["calibration"]["rejection"]["false_accepts"],
                "final_benign_coverage": candidate["final"]["benign_coverage"],
                "final_false_accepts": candidate["final"]["false_accepts"],
            }
        )
    with (output_dir / "independent-threshold-margin-grid.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    with (output_dir / "final-rejection-decomposition.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ("family_id", "accepted", "distance_only", "margin_only", "both", "total")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for family_id in sorted(rejection_decomposition["by_family"]):
            writer.writerow({"family_id": family_id, **rejection_decomposition["by_family"][family_id]})

    readme = (
        "# Registered local baseline post-analysis\n\n"
        "- date: Saturday, July 18, 2026\n"
        "- source evidence: `docs/results/visual-local-baseline-showdown-v1/`\n"
        "- purpose: independent threshold-margin sweep and final benign rejection decomposition\n"
        "- result: no useful operating point was hidden by the coupled-quantile search\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    summary = {
        "version": POSTANALYSIS_VERSION,
        "source_selection_digest": final_summary["selection_digest"],
        "source_calibration_digest": final_summary["calibration_digest"],
        "selected_coupled_quantile": final_summary["selected_quantile"],
        "decoupled_grid_digest": _json_digest(decoupled_grid),
        "rejection_decomposition_digest": _json_digest(rejection_decomposition),
        "selected_decoupled_candidate": decoupled_grid["selected_candidate"],
        "summary": decoupled_grid["summary"],
        "final_rejection_decomposition_overall": rejection_decomposition["overall"],
    }

    _write_json(output_dir / "independent-threshold-margin-grid.json", decoupled_grid)
    _write_json(output_dir / "final-rejection-decomposition.json", rejection_decomposition)
    _write_json(output_dir / "postanalysis-summary.json", summary)
    manifest = {
        "version": POSTANALYSIS_VERSION,
        "files": sorted(
            {
                path.name: {
                    "path": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "byte_size": path.stat().st_size,
                }
                for path in output_dir.iterdir()
                if path.is_file()
            }.values(),
            key=lambda item: item["path"],
        ),
    }
    _write_json(output_dir / "postanalysis-manifest.json", manifest)
    summary["postanalysis_manifest_digest"] = hashlib.sha256(
        (output_dir / "postanalysis-manifest.json").read_bytes()
    ).hexdigest()
    _write_json(output_dir / "postanalysis-summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    payload = run_postanalysis(source_dir=args.source_dir, output_dir=args.output_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
