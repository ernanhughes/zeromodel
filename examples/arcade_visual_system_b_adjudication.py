"""Run the corrected System B adjudication and preserve machine-readable evidence."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
from typing import Any, Dict

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_visual_address_benchmark import (  # noqa: E402
    SOURCE_SCOPE,
    build_arcade_benchmark_dataset,
)
from zeromodel.visual_analysis import analyze_trace_sets  # noqa: E402
from zeromodel.visual_experiment import evaluate_visual_provider  # noqa: E402
from zeromodel.visual_system_b import (  # noqa: E402
    build_system_b_candidates,
    build_system_b_provider,
    select_system_b_operating_point,
)


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


def _state_equivalence_atlas(traces: list[dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[tuple[str, str], Dict[str, Any]] = {}
    for trace in traces:
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


def run(*, output_dir: Path, variants_per_family: int) -> Dict[str, Any]:
    dataset = build_arcade_benchmark_dataset(variants_per_family=variants_per_family)
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
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset-manifest.json").write_text(
        json.dumps(dataset.manifest.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "candidate-operating-points.json").write_text(
        json.dumps([candidate.to_dict() for candidate in candidates], indent=2, sort_keys=True) + "\n",
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
    }
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    if selection.selection_status != "selected_operating_point":
        adjudication = {
            "outcome": "C",
            "selection_status": selection.selection_status,
            "selected_quantile": None,
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
    (output_dir / "final-report.json").write_text(
        json.dumps(final_report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")

    trace_payload = [trace.to_dict() for trace in traces]
    operating_atlas = analyze_trace_sets({"B": trace_payload})
    (output_dir / "operating-atlas.json").write_text(
        json.dumps(operating_atlas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        for row in operating_atlas["systems"]["B"]["operating_curve"]:
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
    state_atlas = _state_equivalence_atlas(trace_payload)
    (output_dir / "state-equivalence-atlas.json").write_text(
        json.dumps(state_atlas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    translation_atlas = _translation_family_atlas(trace_payload)
    (output_dir / "translation-family-atlas.json").write_text(
        json.dumps(translation_atlas, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    metrics = final_result.metrics
    coverage = float(metrics.accepted_benign_count) / float(metrics.false_reject_opportunities or 1)
    outcome = "B"
    next_action = "registration_required_local_baseline_showdown"
    if (
        metrics.false_accept_count == 0
        and metrics.conflicting_action_error_count == 0
        and metrics.accepted_benign_row_correctness >= 0.95
        and coverage >= 0.5
    ):
        outcome = "A"
        next_action = "fixed_camera_and_governance_path"
    elif metrics.accepted_benign_count == 0:
        outcome = "C"
        next_action = "registration_required_local_baseline_showdown"

    summary = {
        "outcome": outcome,
        "selection_status": selection.selection_status,
        "selected_quantile": selection.selected_quantile,
        "calibration_digest": build.calibration.digest,
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
    print(json.dumps(run(output_dir=args.output_dir, variants_per_family=args.variants_per_family), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
