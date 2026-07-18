"""Complete independent registered-pixel calibration on a fresh v3 fixture."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_visual_local_evidence_benchmark import SOURCE_SCOPE, build_arcade_local_evidence_dataset  # noqa: E402
from zeromodel.visual_experiment import EXPECTED_ACCEPT, EXPECTED_REJECT, evaluate_visual_provider  # noqa: E402
from zeromodel.visual_local_baselines import (  # noqa: E402
    build_registered_pixel_candidates_v2,
    build_registered_pixel_provider,
    select_registered_pixel_candidate_v2,
)
from zeromodel.visual_registration import RegistrationConfig  # noqa: E402


SHOWDOWN_PROTOCOL_VERSION = "zeromodel-visual-local-evidence-protocol/v1"
GENERATOR_VERSION = "arcade_visual_registered_calibration_v2/v1"
SYSTEM_ID = "R1"
SYSTEM_NAME = "registered_local_normalized_pixels"
FROZEN_R1_V1_DIR = REPO_ROOT / "docs" / "results" / "visual-local-baseline-showdown-v1"
FROZEN_SYSTEM_B_DIR = REPO_ROOT / "docs" / "results" / "visual-address-system-b-v2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "visual-registered-calibration-v2"
EXPECTED_STARTING_MAIN_SHA = "3bf83f0b90ea2590a9a6921e9000f8b2ed578696"
QUANTILES = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO_ROOT), text=True).strip()


def _environment() -> Dict[str, Any]:
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }


def _runtime() -> Dict[str, Any]:
    return {"started_utc": datetime.now(timezone.utc).isoformat(), "cwd": str(REPO_ROOT)}


def _calendar_date_label(moment: datetime) -> str:
    return f"{moment.strftime('%A, %B')} {moment.day}, {moment.year}"


def _frozen_reference(path: Path) -> Dict[str, Any]:
    summary = json.loads((path / "final-summary.json").read_text(encoding="utf-8"))
    selected = json.loads((path / "selected-calibration.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((path / "run-manifest.json").read_text(encoding="utf-8")) if (path / "run-manifest.json").exists() else {}
    return {
        "result_directory": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "selection_digest": summary.get("selection_digest", run_manifest.get("selection_digest")),
        "calibration_digest": summary["calibration_digest"],
        "selected_quantile": summary.get("selected_quantile"),
        "outcome": summary["outcome"],
        "final_metrics": summary["final_metrics"],
        "trace_digest": _sha256_file(path / "traces.jsonl"),
        "selected_calibration": selected,
    }


def _classify_outcome(metrics: Mapping[str, Any]) -> Tuple[str, str, str]:
    false_accepts = int(metrics["false_accept_count"])
    conflicting = int(metrics["conflicting_action_error_count"])
    coverage = float(metrics["accepted_benign_count"]) / float(metrics["false_reject_opportunities"] or 1)
    if false_accepts == 0 and conflicting == 0 and coverage >= 0.5:
        return ("A", "registered_pixels_useful_after_complete_calibration", "broader_registered_baseline_validation")
    if false_accepts == 0 and conflicting == 0 and coverage >= 0.1:
        return ("B", "registered_pixels_partially_useful_after_complete_calibration", "translation_equivariant_local_correlation")
    return ("C", "registered_global_pixel_evidence_still_insufficient", "translation_equivariant_local_correlation")


def _load_jsonl(path: Path) -> Tuple[Dict[str, Any], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _risk_coverage_curve(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    return {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "points": [
            {
                "distance_quantile": candidate["distance_quantile"],
                "ambiguity_margin_quantile": candidate["ambiguity_margin_quantile"],
                "feasible": candidate["feasible"],
                "benign_coverage": candidate["benign_coverage"],
                "false_accepts": candidate["false_accepts"],
                "conflicting_action_accepts": candidate["conflicting_action_accepts"],
                "accepted_exact_row_precision": candidate["accepted_exact_row_precision"],
                "accepted_action_precision": candidate["accepted_action_precision"],
            }
            for candidate in candidates
        ],
    }


def _bundle_manifest(output_dir: Path) -> Dict[str, Any]:
    files = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "bundle-manifest.json":
            continue
        files.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "byte_size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {"version": SHOWDOWN_PROTOCOL_VERSION, "files": files}


def _verify_required_files(output_dir: Path, expected: Iterable[str]) -> None:
    missing = [name for name in expected if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError("required evidence output is incomplete: %s" % ", ".join(missing))


def run_registered_calibration_v2(
    *,
    output_dir: Path,
    variants_per_family: int,
    max_dx: int,
    max_dy: int,
    minimum_overlap_fraction: float,
    argv: Sequence[str],
    command: str,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError("output directory already exists and is non-empty: %s" % output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    if _git_output("rev-parse", "HEAD") != EXPECTED_STARTING_MAIN_SHA:
        raise RuntimeError("unexpected starting main sha")
    if _git_output("merge-base", "HEAD", "origin/main") != EXPECTED_STARTING_MAIN_SHA:
        raise RuntimeError("unexpected base main sha")
    if _git_output("status", "--short"):
        raise RuntimeError("final run must start from a clean working tree")

    frozen_r1_v1 = _frozen_reference(FROZEN_R1_V1_DIR)
    frozen_system_b = _frozen_reference(FROZEN_SYSTEM_B_DIR)
    dataset = build_arcade_local_evidence_dataset(variants_per_family=variants_per_family)
    environment = _environment()
    registration_config = RegistrationConfig(max_dx=max_dx, max_dy=max_dy, minimum_overlap_fraction=minimum_overlap_fraction)
    selection_capture_ids: set[str] = set()
    candidates = build_registered_pixel_candidates_v2(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        registration_config=registration_config,
        distance_quantiles=QUANTILES,
        ambiguity_margin_quantiles=QUANTILES,
        source_scope=SOURCE_SCOPE,
        capture_ids=selection_capture_ids,
    )
    final_ids = {record.observation_id for record in dataset.manifest.records if record.split == "final_evaluation"}
    if final_ids.intersection(selection_capture_ids):
        raise RuntimeError("calibration leakage detected before selection freeze")
    selection = select_registered_pixel_candidate_v2(
        dataset_manifest=dataset.manifest,
        registration_config=registration_config,
        candidates=candidates,
        source_scope=SOURCE_SCOPE,
    )
    if selection.selection_status != "selected_operating_point":
        raise RuntimeError("R1-v2 produced no feasible calibration candidate")
    selected = next(candidate for candidate in candidates if candidate.calibration.digest == selection.selected_calibration_digest)
    provider = build_registered_pixel_provider(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        registration_config=registration_config,
        calibration=selected.calibration,
    )
    final_result, traces = evaluate_visual_provider(
        provider=provider,
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        system_id=SYSTEM_ID,
        system_name=SYSTEM_NAME,
        splits=("final_evaluation",),
        include_traces=True,
    )
    traces_path = output_dir / "traces.jsonl"
    with traces_path.open("w", encoding="utf-8") as handle:
        for trace in traces:
            handle.write(json.dumps(trace.to_dict(), sort_keys=True) + "\n")
    trace_digest = _sha256_file(traces_path)
    final_metrics = final_result.metrics.to_dict()
    outcome, usefulness_status, next_action = _classify_outcome(final_metrics)

    protocol = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "research_question": "Can independently calibrated registered global pixels transfer a useful governed operating point on a fresh v3 local-evidence fixture?",
        "system_identity": {"system_id": SYSTEM_ID, "system_name": SYSTEM_NAME},
        "source_scope": SOURCE_SCOPE,
        "distance_quantiles": list(QUANTILES),
        "ambiguity_margin_quantiles": list(QUANTILES),
        "selection_rule": selection.selection_rule,
    }
    runtime = _runtime()
    runtime["completed_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["dirty_at_start"] = False
    runtime["dirty_after_run"] = False
    distance_grid = {"version": SHOWDOWN_PROTOCOL_VERSION, "distance_quantiles": list(QUANTILES)}
    margin_grid = {"version": SHOWDOWN_PROTOCOL_VERSION, "ambiguity_margin_quantiles": list(QUANTILES)}
    candidate_grid = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "registration_config_digest": registration_config.digest,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    selected_calibration = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "selection_digest": selection.digest,
        "selected_distance_quantile": selection.selected_distance_quantile,
        "selected_ambiguity_margin_quantile": selection.selected_ambiguity_margin_quantile,
        "selected_threshold": selection.selected_threshold,
        "selected_ambiguity_margin": selection.selected_ambiguity_margin,
        "calibration_digest": selected.calibration.digest,
    }
    paired = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "comparison_kind": "fresh-v3-versus-frozen-v1-reference",
        "frozen_r1_v1_selection_digest": frozen_r1_v1["selection_digest"],
        "fresh_r1_v2_selection_digest": selection.digest,
        "frozen_r1_v1_trace_digest": frozen_r1_v1["trace_digest"],
        "fresh_r1_v2_trace_digest": trace_digest,
        "notes": "The compared runs use different final splits, so this file records headline deltas rather than per-observation pairing.",
        "headline_deltas": {
            "raw_top1_exact_row_accuracy_delta": float(final_metrics["top1_benign_row_accuracy"]) - float(frozen_r1_v1["final_metrics"]["top1_benign_row_accuracy"]),
            "raw_top1_action_accuracy_delta": float(final_metrics["top1_benign_action_accuracy"]) - float(frozen_r1_v1["final_metrics"]["top1_benign_action_accuracy"]),
            "accepted_benign_count_delta": int(final_metrics["accepted_benign_count"]) - int(frozen_r1_v1["final_metrics"]["accepted_benign_count"]),
        },
    }
    final_report = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "final_metrics": final_metrics,
        "outcome": outcome,
        "usefulness_status": usefulness_status,
        "next_action": next_action,
    }
    final_summary = {
        "selection_digest": selection.digest,
        "calibration_digest": selected.calibration.digest,
        "outcome": outcome,
        "usefulness_status": usefulness_status,
        "next_action": next_action,
        "final_metrics": final_metrics,
    }
    adjudication = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "selection_status": selection.selection_status,
        "outcome": outcome,
        "usefulness_status": usefulness_status,
        "next_action": next_action,
    }
    risk_curve = _risk_coverage_curve([candidate.to_dict() for candidate in candidates])
    readme = (
        "# Registered calibration v2\n\n"
        f"- date: {_calendar_date_label(datetime.now(timezone.utc))}\n"
        f"- source scope: `{SOURCE_SCOPE}`\n"
        f"- outcome: `{outcome}`\n"
        f"- usefulness status: `{usefulness_status}`\n"
        f"- next action: `{next_action}`\n"
    )

    _write_json(output_dir / "protocol.json", protocol)
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "argv.json", {"argv": list(argv)})
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    _write_json(output_dir / "runtime.json", runtime)
    _write_json(output_dir / "run-manifest.json", {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "generator_version": GENERATOR_VERSION,
        "started_utc": runtime["started_utc"],
        "completed_utc": runtime["completed_utc"],
        "exact_command": command,
        "argv": list(argv),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "branch_or_ref": _git_output("branch", "--show-current") or os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or "detached",
        "base_main_sha": EXPECTED_STARTING_MAIN_SHA,
        "dirty_at_start": False,
        "dirty_after_run": False,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": {},
        "dataset_digest": dataset.manifest.digest,
        "config_digest": registration_config.digest,
        "candidate_grid_digest": _json_digest(candidate_grid),
        "selection_digest": selection.digest,
        "calibration_digest": selected.calibration.digest,
        "trace_digest": trace_digest,
        "final_report_payload_digest": _json_digest(final_report),
        "generated_paths": [],
        "bundle_manifest_path": "bundle-manifest.json",
    })
    _write_json(output_dir / "dataset-manifest.json", dataset.manifest.to_dict())
    _write_json(output_dir / "frozen-r1-v1-reference.json", frozen_r1_v1)
    _write_json(output_dir / "registration-config.json", registration_config.to_dict())
    _write_json(output_dir / "distance-quantile-grid.json", distance_grid)
    _write_json(output_dir / "ambiguity-margin-quantile-grid.json", margin_grid)
    _write_json(output_dir / "candidate-grid.json", candidate_grid)
    _write_json(output_dir / "selected-calibration.json", selected_calibration)
    _write_json(output_dir / "final-report.json", final_report)
    _write_json(output_dir / "final-summary.json", final_summary)
    _write_json(output_dir / "adjudication.json", adjudication)
    _write_json(output_dir / "paired-r1-v1-v2-comparison.json", paired)
    _write_json(output_dir / "risk-coverage-curve.json", risk_curve)
    (output_dir / "README.md").write_text(readme, encoding="utf-8")
    bundle_manifest = _bundle_manifest(output_dir)
    _write_json(output_dir / "bundle-manifest.json", bundle_manifest)
    run_manifest = json.loads((output_dir / "run-manifest.json").read_text(encoding="utf-8"))
    run_manifest["generated_paths"] = sorted(
        str(path.relative_to(output_dir)).replace("\\", "/")
        for path in output_dir.rglob("*")
        if path.is_file()
    )
    _write_json(output_dir / "run-manifest.json", run_manifest)

    required = [
        "README.md", "protocol.json", "environment.json", "argv.json", "command.txt", "runtime.json",
        "run-manifest.json", "dataset-manifest.json", "frozen-r1-v1-reference.json", "registration-config.json",
        "distance-quantile-grid.json", "ambiguity-margin-quantile-grid.json", "candidate-grid.json",
        "selected-calibration.json", "final-report.json", "final-summary.json", "adjudication.json",
        "traces.jsonl", "paired-r1-v1-v2-comparison.json", "risk-coverage-curve.json", "bundle-manifest.json",
    ]
    _verify_required_files(output_dir, required)
    return {
        "dataset_digest": dataset.manifest.digest,
        "selection_digest": selection.digest,
        "calibration_digest": selected.calibration.digest,
        "trace_digest": trace_digest,
        "bundle_manifest_digest": _sha256_file(output_dir / "bundle-manifest.json"),
        "outcome": outcome,
        "final_metrics": final_metrics,
        "usefulness_status": usefulness_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants-per-family", type=int, default=3)
    parser.add_argument("--max-dx", type=int, default=3)
    parser.add_argument("--max-dy", type=int, default=3)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.6)
    args = parser.parse_args()
    payload = run_registered_calibration_v2(
        output_dir=args.output_dir,
        variants_per_family=args.variants_per_family,
        max_dx=args.max_dx,
        max_dy=args.max_dy,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
        argv=[sys.executable, *sys.argv],
        command=" ".join([sys.executable, *sys.argv]),
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
