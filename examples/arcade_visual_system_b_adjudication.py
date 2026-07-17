"""Run the corrected System B adjudication and preserve machine-readable evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_visual_address_benchmark import (  # noqa: E402
    SOURCE_SCOPE,
    build_arcade_benchmark_dataset,
)
from zeromodel.visual_analysis import analyze_trace_sets  # noqa: E402
from zeromodel.visual_experiment import (  # noqa: E402
    EXPECTED_ACCEPT,
    EXPECTED_REJECT,
    IMPOSSIBILITY_CONTROL,
    evaluate_visual_provider,
)
from zeromodel.visual_system_b import (  # noqa: E402
    build_system_b_candidates,
    build_system_b_provider,
    select_system_b_operating_point,
)


GENERATOR_VERSION = "arcade_visual_system_b_adjudication/v2"


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


def _environment() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "numpy_version": np.__version__,
    }
    for name in ("torch", "torchvision"):
        try:
            module = __import__(name)
            payload[f"{name}_version"] = getattr(module, "__version__", "?")
            if name == "torch":
                payload["cuda_available"] = bool(module.cuda.is_available())
                if module.cuda.is_available():
                    payload["gpu"] = module.cuda.get_device_name(0)
        except Exception as exc:
            payload[f"{name}_version"] = f"NOT_INSTALLED:{type(exc).__name__}"
    return payload


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()


def _row_confusion_atlas(traces: list[dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[tuple[str, str], Dict[str, Any]] = {}
    for trace in traces:
        if trace.get("expected_disposition") != EXPECTED_ACCEPT:
            continue
        expected = trace.get("expected_row_id")
        predicted = trace.get("top1_row_id")
        if expected is None or predicted is None or expected == predicted:
            continue
        key = (str(expected), str(predicted))
        item = counts.setdefault(
            key,
            {
                "expected_row_id": str(expected),
                "predicted_row_id": str(predicted),
                "expected_action_id": trace.get("expected_action_id"),
                "predicted_action_id": trace.get("top1_action_id"),
                "count": 0,
                "same_action": trace.get("expected_action_id") == trace.get("top1_action_id"),
                "conflicting_action": trace.get("expected_action_id") != trace.get("top1_action_id"),
            },
        )
        item["count"] += 1
    return {
        "atlas_type": "observed_benign_row_confusion",
        "pairs": sorted(counts.values(), key=lambda item: (-item["count"], item["expected_row_id"], item["predicted_row_id"])),
    }


def _translation_family_atlas(traces: list[dict[str, Any]]) -> Dict[str, Any]:
    families = [trace for trace in traces if "shift" in str(trace.get("family_id", ""))]
    summary: Dict[str, Dict[str, int]] = {}
    for trace in families:
        family_id = str(trace["family_id"])
        item = summary.setdefault(
            family_id,
            {
                "count": 0,
                "accepted": 0,
                "rejected": 0,
                "top1_row_correct": 0,
                "top1_action_correct": 0,
                "conflicting_action": 0,
            },
        )
        item["count"] += 1
        item["accepted"] += int(trace["decision"]["accepted"])
        item["rejected"] += int(not trace["decision"]["accepted"])
        item["top1_row_correct"] += int(trace.get("top1_row_id") == trace.get("expected_row_id"))
        item["top1_action_correct"] += int(trace.get("top1_action_id") == trace.get("expected_action_id"))
        item["conflicting_action"] += int(
            trace["decision"]["accepted"]
            and trace.get("top1_action_id") is not None
            and trace.get("top1_action_id") != trace.get("expected_action_id")
        )
    return {"families": summary}


def _run_manifest(
    *,
    argv: Sequence[str],
    command: str,
    started_utc: str,
    completed_utc: str,
    environment_digest: str,
    dataset_digest: str,
    selection_digest: str,
    final_report_digest: str,
    trace_digest: str,
) -> Dict[str, Any]:
    status = _git_output("status", "--short")
    return {
        "version": "zeromodel-visual-benchmark-run-manifest/v1",
        "generator_version": GENERATOR_VERSION,
        "git_commit": _git_output("rev-parse", "HEAD"),
        "branch": _git_output("branch", "--show-current"),
        "dirty": bool(status),
        "argv": list(argv),
        "command": command,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "environment_digest": environment_digest,
        "dataset_digest": dataset_digest,
        "selection_digest": selection_digest,
        "final_report_digest": final_report_digest,
        "trace_digest": trace_digest,
    }


def _classify_outcome(
    *,
    selection_status: str,
    metrics: Any,
) -> tuple[str, str, str]:
    if selection_status != "selected_operating_point":
        return ("C", "no_useful_operating_point", "registration_required_local_baseline_showdown")
    coverage = float(metrics.accepted_benign_count) / float(metrics.false_reject_opportunities or 1)
    transfers_all_gates = (
        metrics.false_accept_count == 0
        and metrics.conflicting_action_error_count == 0
        and metrics.accepted_benign_row_correctness is not None
        and metrics.accepted_benign_row_correctness >= 0.95
    )
    if transfers_all_gates and coverage >= 0.5:
        return ("A", "useful_operating_point_high_coverage", "fixed_camera_and_governance_path")
    if transfers_all_gates and coverage >= 0.1:
        return ("B", "useful_operating_point_low_coverage", "registration_required_local_baseline_showdown")
    return ("C", "no_useful_operating_point", "registration_required_local_baseline_showdown")


def run(
    *,
    output_dir: Path,
    variants_per_family: int,
    argv: Sequence[str] | None = None,
    command: str | None = None,
) -> Dict[str, Any]:
    started_utc = datetime.now(timezone.utc).isoformat()
    dataset = build_arcade_benchmark_dataset(variants_per_family=variants_per_family)
    historical_v1_dataset_digest = "91b1b422482eeeef20eb182162eb2a745f9b50524cc7f94ec95a0aba5f2fa37e"
    candidates = build_system_b_candidates(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        source_scope=SOURCE_SCOPE,
    )
    selection = select_system_b_operating_point(
        dataset_manifest=dataset.manifest,
        candidates=candidates,
        metadata={"variants_per_family": variants_per_family},
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    environment_digest = _json_digest(environment)
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset-manifest.json").write_text(
        json.dumps(dataset.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset-identity.json").write_text(
        json.dumps(
            {
                "source_scope": SOURCE_SCOPE,
                "historical_v1_dataset_digest": historical_v1_dataset_digest,
                "current_v2_dataset_digest": dataset.manifest.digest,
                "digest_compatible": False,
                "reason": (
                    "the v2 adjudication dataset changed split semantics, "
                    "family assignments, calibration roles, evaluation roles, "
                    "and source identity"
                ),
                "changed_families": [
                    "prototype",
                    "calibration",
                    "ood-holdout",
                    "translation",
                    "critical-intervention",
                    "information-theoretic-control",
                ],
                "changed_partitions": [
                    "prototype",
                    "benign_calibration",
                    "rejection_calibration",
                    "final_evaluation",
                ],
                "changed_semantics": [
                    "canonical evaluation dispositions",
                    "independent rejection calibration",
                    "final-only evaluation",
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "candidate-operating-points.json").write_text(
        json.dumps([candidate.to_dict() for candidate in candidates], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per-row-calibration-candidate-grid.json").write_text(
        json.dumps(
            {
                "curve_type": "declared_per_row_calibration_candidate_grid",
                "uses_final_evaluation_traces": False,
                "valid_for_threshold_selection": True,
                "candidates": [candidate.to_dict() for candidate in candidates],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "selected-calibration.json").write_text(
        json.dumps(selection.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    protocol = {
        "protocol_version": "zeromodel-system-b-protocol/v1",
        "question": (
            "Can normalized-pixel retrieval achieve high exact-row precision and "
            "low distinguishable false acceptance at non-negligible benign coverage "
            "when selected entirely on calibration data?"
        ),
        "selection_rule": selection.selection_rule,
        "candidate_quantiles": [candidate["quantile"] for candidate in json.loads((output_dir / "candidate-operating-points.json").read_text(encoding="utf-8"))],
        "permitted_partitions": ["prototype", "benign_calibration", "rejection_calibration"],
        "forbidden_partitions_for_selection": ["final_evaluation"],
        "final_evaluation_partition": "final_evaluation",
        "dataset_identity": SOURCE_SCOPE,
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if selection.selection_status != "selected_operating_point":
        completed_utc = datetime.now(timezone.utc).isoformat()
        run_manifest = _run_manifest(
            argv=(argv or []),
            command=(command or ""),
            started_utc=started_utc,
            completed_utc=completed_utc,
            environment_digest=environment_digest,
            dataset_digest=dataset.manifest.digest,
            selection_digest=selection.digest,
            final_report_digest="",
            trace_digest="",
        )
        run_manifest_digest = _json_digest(run_manifest)
        run_manifest["run_manifest_digest"] = run_manifest_digest
        (output_dir / "run-manifest.json").write_text(
            json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        adjudication = {
            "outcome": "C",
            "selection_status": selection.selection_status,
            "selected_quantile": None,
            "run_manifest_digest": run_manifest_digest,
            "next_action": "registration_required_local_baseline_showdown",
        }
        (output_dir / "adjudication.json").write_text(
            json.dumps(adjudication, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return adjudication

    build, provider, encoder = build_system_b_provider(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        quantile=float(selection.selected_quantile),
    )
    final_result, traces = evaluate_visual_provider(
        provider=provider,
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        system_id="B",
        system_name="normalized_template_matching",
        splits=("final_evaluation",),
        include_traces=True,
    )
    final_report = {
        "selection_digest": selection.digest,
        "selected_quantile": selection.selected_quantile,
        "system": final_result.to_dict(),
        "encoder_manifest": encoder.manifest().to_dict(),
        "address_manifest": build.manifest.to_dict(),
        "calibration": build.calibration.to_dict(),
    }
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")
    trace_digest = hashlib.sha256((output_dir / "traces.jsonl").read_bytes()).hexdigest()

    metrics = final_result.metrics
    trace_payload = [trace.to_dict() for trace in traces]
    benign_count = sum(
        1 for trace in trace_payload if trace["expected_disposition"] == EXPECTED_ACCEPT
    )
    reject_count = sum(
        1 for trace in trace_payload if trace["expected_disposition"] == EXPECTED_REJECT
    )
    control_count = sum(
        1
        for trace in trace_payload
        if trace["expected_disposition"] == IMPOSSIBILITY_CONTROL
    )
    if benign_count != metrics.false_reject_opportunities:
        raise RuntimeError("trace benign denominator does not match final-report benign denominator")
    if reject_count != metrics.false_accept_opportunities:
        raise RuntimeError("trace reject denominator does not match final-report reject denominator")
    metrics_control_count = int(final_result.notes["impossibility_control_count"])
    if control_count != metrics_control_count:
        raise RuntimeError("trace control denominator does not match final-report control denominator")
    operating_atlas = analyze_trace_sets({"B": trace_payload})
    (output_dir / "global-threshold-diagnostic.json").write_text(
        json.dumps(operating_atlas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "operating-atlas.json").write_text(
        json.dumps(operating_atlas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "global-threshold-diagnostic.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "coverage",
                "accepted_row_precision",
                "accepted_action_precision",
                "row_recall",
                "action_recall",
                "false_acceptance_rate",
                "false_rejection_rate",
            ],
        )
        writer.writeheader()
        for row in operating_atlas["systems"]["B"]["global_score_threshold_curve"]:
            writer.writerow(
                {
                    "threshold": row["threshold"],
                    "coverage": row["coverage"],
                    "accepted_row_precision": row["accepted_row_precision"],
                    "accepted_action_precision": row["accepted_action_precision"],
                    "row_recall": row["row_recall"],
                    "action_recall": row["action_recall"],
                    "false_acceptance_rate": row["false_acceptance_rate"],
                    "false_rejection_rate": row["false_rejection_rate"],
                }
            )
    with (output_dir / "operating-atlas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "threshold",
                "coverage",
                "accepted_row_precision",
                "accepted_action_precision",
                "row_recall",
                "action_recall",
                "false_acceptance_rate",
                "false_rejection_rate",
            ],
        )
        writer.writeheader()
        for row in operating_atlas["systems"]["B"]["global_score_threshold_curve"]:
            writer.writerow(
                {
                    "threshold": row["threshold"],
                    "coverage": row["coverage"],
                    "accepted_row_precision": row["accepted_row_precision"],
                    "accepted_action_precision": row["accepted_action_precision"],
                    "row_recall": row["row_recall"],
                    "action_recall": row["action_recall"],
                    "false_acceptance_rate": row["false_acceptance_rate"],
                    "false_rejection_rate": row["false_rejection_rate"],
                }
            )
    row_confusion = _row_confusion_atlas(trace_payload)
    (output_dir / "row-confusion-atlas.json").write_text(
        json.dumps(row_confusion, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    legacy_state_equivalence = output_dir / "state-equivalence-atlas.json"
    if legacy_state_equivalence.exists():
        legacy_state_equivalence.unlink()
    translation_atlas = _translation_family_atlas(trace_payload)
    (output_dir / "translation-family-atlas.json").write_text(
        json.dumps(translation_atlas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    final_report_digest = _json_digest(final_report)
    completed_utc = datetime.now(timezone.utc).isoformat()
    invoked_argv = list(argv or [])
    invoked_command = command or " ".join(invoked_argv)
    run_manifest = _run_manifest(
        argv=invoked_argv,
        command=invoked_command,
        started_utc=started_utc,
        completed_utc=completed_utc,
        environment_digest=environment_digest,
        dataset_digest=dataset.manifest.digest,
        selection_digest=selection.digest,
        final_report_digest=final_report_digest,
        trace_digest=trace_digest,
    )
    run_manifest_digest = _json_digest(run_manifest)
    run_manifest["run_manifest_digest"] = run_manifest_digest
    (output_dir / "run-manifest.json").write_text(
        json.dumps(run_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    final_report["run_manifest_digest"] = run_manifest_digest
    (output_dir / "final-report.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outcome, usefulness_status, next_action = _classify_outcome(
        selection_status=selection.selection_status,
        metrics=metrics,
    )

    summary = {
        "outcome": outcome,
        "selection_status": selection.selection_status,
        "usefulness_status": usefulness_status,
        "selected_quantile": selection.selected_quantile,
        "calibration_digest": build.calibration.digest,
        "run_manifest_digest": run_manifest_digest,
        "final_metrics": metrics.to_dict(),
        "next_action": next_action,
        "translation_families": translation_atlas["families"],
    }
    (output_dir / "final-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "adjudication.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "README.md").write_text(
        "# System B v2 adjudication\n\n"
        f"- outcome: `{outcome}`\n"
        f"- selected quantile: `{selection.selected_quantile}`\n"
        f"- usefulness status: `{usefulness_status}`\n"
        f"- next action: `{next_action}`\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "docs" / "results" / "visual-address-system-b-v2",
    )
    parser.add_argument("--variants-per-family", type=int, default=3)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                output_dir=args.output_dir,
                variants_per_family=args.variants_per_family,
                argv=[sys.executable, *sys.argv],
                command=" ".join([sys.executable, *sys.argv]),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
