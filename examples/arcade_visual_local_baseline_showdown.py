"""Measure the bounded registered-pixel local baseline against frozen System B."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_visual_address_benchmark import SOURCE_SCOPE, build_arcade_benchmark_dataset  # noqa: E402
from research.visual.visual_experiment import EXPECTED_ACCEPT, EXPECTED_REJECT, IMPOSSIBILITY_CONTROL, evaluate_visual_provider  # noqa: E402
from research.visual.visual_local_baselines import (  # noqa: E402
    RegisteredPixelCalibration,
    build_registered_pixel_candidates,
    build_registered_pixel_provider,
    select_registered_pixel_candidate,
)
from zeromodel.vision.visual_registration import RegistrationConfig  # noqa: E402


SHOWDOWN_PROTOCOL_VERSION = "zeromodel-visual-local-showdown/v1"
SYSTEM_ID = "R1"
SYSTEM_NAME = "registered_local_normalized_pixels"
GENERATOR_VERSION = "arcade_visual_local_baseline_showdown/v1"
FROZEN_SYSTEM_B_DIR = REPO_ROOT / "docs" / "results" / "visual-address-system-b-v2"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "results" / "visual-local-baseline-showdown-v1"
EXPECTED_STARTING_MAIN_SHA = "832bca74fa05a6222ed02c65419bc2f551dfc7c0"
FROZEN_SYSTEM_B_IDENTITIES = {
    "dataset_digest": "b7c0fb2f0c3aaf40862eabf16937ca476ad3266baa682ef9ffad8db93c6cb30b",
    "selection_digest": "b7dfa44942f15cc782eaff3a4d7f4d7224214d4b7b15b6e5c668a76bf191a1b1",
    "calibration_digest": "bb985997a8d56cde073f91fc6611c08c0718d888e0d49d00aa0198f0f603e583",
}
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
    payload: Dict[str, Any] = {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
    }
    return payload


def _runtime() -> Dict[str, Any]:
    return {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cwd": str(REPO_ROOT),
    }


def _calendar_date_label(moment: datetime) -> str:
    return f"{moment.strftime('%A, %B')} {moment.day}, {moment.year}"


def _frozen_system_b_reference() -> Dict[str, Any]:
    summary = json.loads((FROZEN_SYSTEM_B_DIR / "final-summary.json").read_text(encoding="utf-8"))
    run_manifest = json.loads((FROZEN_SYSTEM_B_DIR / "run-manifest.json").read_text(encoding="utf-8"))
    final_report = json.loads((FROZEN_SYSTEM_B_DIR / "final-report.json").read_text(encoding="utf-8"))
    traces_path = FROZEN_SYSTEM_B_DIR / "traces.jsonl"
    trace_digest = _sha256_file(traces_path)
    selection = json.loads((FROZEN_SYSTEM_B_DIR / "selected-calibration.json").read_text(encoding="utf-8"))
    reference = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "result_directory": str(FROZEN_SYSTEM_B_DIR.relative_to(REPO_ROOT)).replace("\\", "/"),
        "dataset_digest": run_manifest["dataset_digest"],
        "selection_digest": run_manifest["selection_digest"],
        "calibration_digest": summary["calibration_digest"],
        "run_manifest_digest": run_manifest["run_manifest_digest"],
        "trace_digest": trace_digest,
        "outcome": summary["outcome"],
        "headline_metrics": {
            "raw_top1_exact_row_accuracy": float(summary["final_metrics"]["top1_benign_row_accuracy"]),
            "raw_top1_action_accuracy": float(summary["final_metrics"]["top1_benign_action_accuracy"]),
            "accepted_benign_count": int(summary["final_metrics"]["accepted_benign_count"]),
            "false_accept_count": int(summary["final_metrics"]["false_accept_count"]),
            "false_reject_count": int(summary["final_metrics"]["false_reject_count"]),
        },
        "selected_quantile": selection["selected_quantile"],
        "frozen_report_digest": _json_digest(final_report),
    }
    for key, expected in FROZEN_SYSTEM_B_IDENTITIES.items():
        if reference[key] != expected:
            raise RuntimeError("frozen System B comparator identity mismatch for %s" % key)
    return reference


def _classify_outcome(summary_metrics: Mapping[str, Any]) -> Tuple[str, str, str]:
    false_accepts = int(summary_metrics["false_accept_count"])
    conflicting = int(summary_metrics["conflicting_action_error_count"])
    coverage = float(summary_metrics["accepted_benign_count"]) / float(summary_metrics["false_reject_opportunities"] or 1)
    if false_accepts == 0 and conflicting == 0 and coverage >= 0.5:
        return ("A", "useful_operating_point_high_coverage", "broader_registered_baseline_validation")
    if false_accepts == 0 and conflicting == 0 and coverage >= 0.1:
        return ("B", "useful_operating_point_low_coverage", "translation_equivariant_template_correlation")
    return ("C", "bounded_registration_insufficient", "translation_equivariant_template_correlation")


def _trace_digest(path: Path) -> str:
    return _sha256_file(path)


def _load_jsonl(path: Path) -> Tuple[Dict[str, Any], ...]:
    return tuple(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _paired_comparison(
    frozen_b_traces: Sequence[Mapping[str, Any]],
    r1_traces: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    b_by_id = {item["observation_id"]: item for item in frozen_b_traces if item["expected_disposition"] == EXPECTED_ACCEPT}
    r_by_id = {item["observation_id"]: item for item in r1_traces if item["expected_disposition"] == EXPECTED_ACCEPT}
    if set(b_by_id) != set(r_by_id):
        raise RuntimeError("paired comparison benign ids do not match")
    row = {"both_correct": 0, "b_only": 0, "r1_only": 0, "neither": 0}
    action = dict(row)
    for observation_id in sorted(b_by_id):
        b = b_by_id[observation_id]
        r = r_by_id[observation_id]
        for target, left_correct, right_correct in (
            (
                row,
                b["top1_row_id"] == b["expected_row_id"],
                r["top1_row_id"] == r["expected_row_id"],
            ),
            (
                action,
                b["top1_action_id"] == b["expected_action_id"],
                r["top1_action_id"] == r["expected_action_id"],
            ),
        ):
            key = (
                "both_correct" if left_correct and right_correct
                else "b_only" if left_correct
                else "r1_only" if right_correct
                else "neither"
            )
            target[key] += 1
    return {
        "observation_count": len(b_by_id),
        "row": row,
        "action": action,
    }


def _translation_family_atlas(traces: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    targets = {"final-shift-two", "benign-calibration-shift-down", "prototype-shift-up"}
    families: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        family_id = str(trace["family_id"])
        if family_id not in targets:
            continue
        item = families.setdefault(
            family_id,
            {
                "count": 0,
                "raw_row_correct": 0,
                "registered_row_correct": 0,
                "raw_action_correct": 0,
                "registered_action_correct": 0,
                "accepted_count": 0,
                "false_rejects": 0,
                "selected_displacements": {},
            },
        )
        raw_row = trace["decision"]["trace"].get("raw_top1_row_id") == trace.get("expected_row_id")
        raw_action = trace["decision"]["trace"].get("raw_top1_action_id") == trace.get("expected_action_id")
        reg_row = trace.get("top1_row_id") == trace.get("expected_row_id")
        reg_action = trace.get("top1_action_id") == trace.get("expected_action_id")
        accepted = bool(trace["decision"]["accepted"])
        dx = int(trace["decision"]["trace"]["dx"])
        dy = int(trace["decision"]["trace"]["dy"])
        key = f"{dx},{dy}"
        item["count"] += 1
        item["raw_row_correct"] += int(raw_row)
        item["registered_row_correct"] += int(reg_row)
        item["raw_action_correct"] += int(raw_action)
        item["registered_action_correct"] += int(reg_action)
        item["accepted_count"] += int(accepted)
        item["false_rejects"] += int(trace["expected_disposition"] == EXPECTED_ACCEPT and not accepted)
        item["selected_displacements"][key] = item["selected_displacements"].get(key, 0) + 1
    return {"families": families}


def _registration_displacement_atlas(traces: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_family: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        if trace["expected_disposition"] != EXPECTED_ACCEPT:
            continue
        family_id = str(trace["family_id"])
        item = by_family.setdefault(
            family_id,
            {
                "count": 0,
                "displacements": {},
                "distance_improvement_total": 0.0,
                "row_correct_before": 0,
                "row_correct_after": 0,
                "action_correct_before": 0,
                "action_correct_after": 0,
            },
        )
        dx = int(trace["decision"]["trace"]["dx"])
        dy = int(trace["decision"]["trace"]["dy"])
        key = f"{dx},{dy}"
        raw_row = trace["decision"]["trace"].get("raw_top1_row_id") == trace.get("expected_row_id")
        raw_action = trace["decision"]["trace"].get("raw_top1_action_id") == trace.get("expected_action_id")
        reg_row = trace.get("top1_row_id") == trace.get("expected_row_id")
        reg_action = trace.get("top1_action_id") == trace.get("expected_action_id")
        item["count"] += 1
        item["displacements"][key] = item["displacements"].get(key, 0) + 1
        item["distance_improvement_total"] += float(trace["decision"]["trace"]["distance_improvement"])
        item["row_correct_before"] += int(raw_row)
        item["row_correct_after"] += int(reg_row)
        item["action_correct_before"] += int(raw_action)
        item["action_correct_after"] += int(reg_action)
    return {"families": by_family}


def _residual_error_atlas(traces: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    counts = {
        "registration_selected_wrong_displacement": 0,
        "correct_action_but_wrong_row": 0,
        "conflicting_action": 0,
        "insufficient_overlap": 0,
        "rejected_despite_correct_raw_candidate": 0,
        "ood_or_distinguishable_rejection": 0,
        "information_theoretic_control": 0,
    }
    for trace in traces:
        disp = trace["decision"]["trace"]
        if trace["expected_disposition"] == IMPOSSIBILITY_CONTROL:
            counts["information_theoretic_control"] += 1
            continue
        if trace["expected_disposition"] == EXPECTED_REJECT:
            counts["ood_or_distinguishable_rejection"] += 1
            continue
        if disp.get("rejection_reason") == "insufficient_overlap" or trace["decision"]["reason"] == "registered_distance_above_threshold":
            counts["insufficient_overlap"] += 1
        raw_row = disp.get("raw_top1_row_id")
        raw_action = disp.get("raw_top1_action_id")
        if raw_row == trace.get("expected_row_id") and not trace["decision"]["accepted"]:
            counts["rejected_despite_correct_raw_candidate"] += 1
        if trace["top1_action_id"] == trace["expected_action_id"] and trace["top1_row_id"] != trace["expected_row_id"]:
            counts["correct_action_but_wrong_row"] += 1
        if trace["top1_action_id"] not in {None, trace["expected_action_id"]}:
            counts["conflicting_action"] += 1
    return {"counts": counts}


def _verify_required_files(output_dir: Path, expected: Iterable[str]) -> None:
    missing = [name for name in expected if not (output_dir / name).exists()]
    if missing:
        raise RuntimeError("required evidence output is incomplete: %s" % ", ".join(missing))


def _bundle_manifest(output_dir: Path) -> Dict[str, Any]:
    entries = []
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "bundle-manifest.json":
            continue
        entries.append(
            {
                "path": str(path.relative_to(output_dir)).replace("\\", "/"),
                "byte_size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "files": entries,
    }


def _command_payload(argv: Sequence[str]) -> Tuple[Sequence[str], str]:
    return (list(argv), " ".join(argv))


def run_showdown(
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

    dirty_at_start = bool(_git_output("status", "--short"))
    if dirty_at_start:
        raise RuntimeError("final run must start from a clean working tree")
    started_utc = datetime.now(timezone.utc).isoformat()
    starting_main_sha = _git_output("merge-base", "HEAD", "origin/main")
    if starting_main_sha != EXPECTED_STARTING_MAIN_SHA:
        raise RuntimeError(
            "unexpected starting main sha: expected %s, got %s"
            % (EXPECTED_STARTING_MAIN_SHA, starting_main_sha)
        )

    dataset = build_arcade_benchmark_dataset(variants_per_family=variants_per_family)
    if dataset.manifest.digest != FROZEN_SYSTEM_B_IDENTITIES["dataset_digest"]:
        raise RuntimeError("dataset identity mismatch against frozen comparator")
    frozen_reference = _frozen_system_b_reference()

    environment = _environment()
    registration_config = RegistrationConfig(
        max_dx=max_dx,
        max_dy=max_dy,
        minimum_overlap_fraction=minimum_overlap_fraction,
    )
    selection_capture_ids: set[str] = set()
    candidates = build_registered_pixel_candidates(
        dataset_manifest=dataset.manifest,
        observations=dataset.observations,
        policy_lookup=dataset.policy_lookup,
        registration_config=registration_config,
        quantiles=QUANTILES,
        source_scope=SOURCE_SCOPE,
        capture_ids=selection_capture_ids,
    )
    final_ids = {
        record.observation_id
        for record in dataset.manifest.records
        if record.split == "final_evaluation"
    }
    if final_ids.intersection(selection_capture_ids):
        raise RuntimeError("calibration leakage detected before selection freeze")
    selection = select_registered_pixel_candidate(
        dataset_manifest=dataset.manifest,
        registration_config=registration_config,
        candidates=candidates,
        source_scope=SOURCE_SCOPE,
    )
    if selection.selection_status != "selected_operating_point":
        raise RuntimeError("R1 produced no feasible calibration candidate")
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
    trace_digest = _trace_digest(traces_path)
    frozen_b_traces = _load_jsonl(FROZEN_SYSTEM_B_DIR / "traces.jsonl")
    r1_traces = _load_jsonl(traces_path)

    outcome, usefulness_status, next_action = _classify_outcome(final_result.metrics.to_dict())
    protocol = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "research_question": (
            "Can bounded, deterministic integer-pixel registration recover a useful "
            "accepted operating point for local normalized-pixel retrieval while "
            "preserving zero observed distinguishable false accepts and zero accepted "
            "conflicting-action errors on the declared fixtures?"
        ),
        "system_identity": {"system_id": SYSTEM_ID, "system_name": SYSTEM_NAME},
        "frozen_comparator_identities": frozen_reference,
        "allowed_data_splits": ["prototype", "benign_calibration", "rejection_calibration", "final_evaluation"],
        "forbidden_data_splits_during_selection": ["final_evaluation"],
        "registration_bounds": {"max_dx": max_dx, "max_dy": max_dy},
        "overlap_rule": {"minimum_overlap_fraction": minimum_overlap_fraction},
        "distance_metric": "normalized_l2",
        "tie_break_rules": {
            "displacement": [
                "smallest Manhattan displacement",
                "smallest abs(dy)",
                "smallest abs(dx)",
                "lowest signed dy",
                "lowest signed dx",
            ],
            "prototype": [
                "lowest registered distance",
                "greatest overlap fraction",
                "smallest displacement magnitude",
                "stable row id ordering",
                "stable prototype observation id ordering",
            ],
        },
        "candidate_grid": list(QUANTILES),
        "feasibility_rules": [
            "distinguishable false accepts = 0",
            "accepted conflicting-action errors = 0",
        ],
        "selection_rule": selection.selection_rule,
        "outcome_rule": {
            "A": "zero observed distinguishable false accepts, zero accepted conflicting-action errors, coverage >= 0.50",
            "B": "zero observed distinguishable false accepts, zero accepted conflicting-action errors, 0.10 <= coverage < 0.50",
            "C": "no feasible transferred operating point or coverage < 0.10",
            "invalid": "leakage, comparator mismatch, dataset mismatch, digest failure, incomplete output, or ambiguous code state",
        },
        "invalid_run_conditions": [
            "calibration leakage detected",
            "frozen comparator identity mismatch",
            "dataset identity mismatch",
            "digest verification fails",
            "required evidence output is incomplete",
        ],
    }
    runtime = _runtime()
    runtime["completed_utc"] = datetime.now(timezone.utc).isoformat()
    runtime["dirty_at_start"] = False
    runtime["dirty_after_run"] = False

    registration_candidate_grid = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "registration_config_digest": registration_config.digest,
        "candidates": [candidate.to_dict() for candidate in candidates],
    }
    selected_calibration = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "selection_digest": selection.digest,
        "selected_quantile": selection.selected_quantile,
        "selected_threshold": selection.selected_threshold,
        "selected_ambiguity_margin": selection.selected_ambiguity_margin,
        "registration_config_digest": registration_config.digest,
        "candidate_grid_digest": _json_digest(registration_candidate_grid),
        "prototype_digest": selection.prototype_digest,
        "benign_calibration_digest": selection.benign_calibration_digest,
        "rejection_calibration_digest": selection.rejection_calibration_digest,
        "selection_rule": selection.selection_rule,
        "selection_status": selection.selection_status,
        "calibration_digest": selected.calibration.digest,
    }

    paired = _paired_comparison(frozen_b_traces, r1_traces)
    translation_atlas = _translation_family_atlas(r1_traces)
    displacement_atlas = _registration_displacement_atlas(r1_traces)
    residual_atlas = _residual_error_atlas(r1_traces)
    final_metrics = final_result.metrics.to_dict()
    final_report = {
        "version": SHOWDOWN_PROTOCOL_VERSION,
        "final_metrics": final_metrics,
        "raw_metrics": {
            "top1_benign_row_accuracy": final_metrics["top1_benign_row_accuracy"],
            "top1_benign_action_accuracy": final_metrics["top1_benign_action_accuracy"],
        },
        "accepted_metrics": {
            "accepted_benign_count": final_metrics["accepted_benign_count"],
            "accepted_benign_row_correctness": final_metrics["accepted_benign_row_correctness"],
            "accepted_benign_action_correctness": final_metrics["accepted_benign_action_correctness"],
        },
        "rejection_metrics": {
            "false_accept_count": final_metrics["false_accept_count"],
            "false_reject_count": final_metrics["false_reject_count"],
        },
        "translation_family_metrics": translation_atlas,
        "registration_metrics": displacement_atlas,
        "frozen_comparator_metrics": frozen_reference["headline_metrics"],
        "paired_deltas": paired,
        "outcome": outcome,
        "usefulness_status": usefulness_status,
        "next_action": next_action,
    }
    final_summary = {
        "outcome": outcome,
        "usefulness_status": usefulness_status,
        "next_action": next_action,
        "selected_quantile": selection.selected_quantile,
        "selection_digest": selection.digest,
        "calibration_digest": selected.calibration.digest,
        "final_metrics": final_metrics,
    }
    adjudication = {
        "outcome": outcome,
        "usefulness_status": usefulness_status,
        "next_action": next_action,
        "selection_status": selection.selection_status,
        "selection_digest": selection.digest,
        "calibration_digest": selected.calibration.digest,
    }

    _write_json(output_dir / "protocol.json", protocol)
    _write_json(output_dir / "environment.json", environment)
    _write_json(output_dir / "argv.json", {"argv": list(argv)})
    (output_dir / "command.txt").write_text(command + "\n", encoding="utf-8")
    _write_json(output_dir / "runtime.json", runtime)
    _write_json(output_dir / "dataset-manifest.json", dataset.manifest.to_dict())
    _write_json(output_dir / "system-b-frozen-reference.json", frozen_reference)
    _write_json(output_dir / "registration-config.json", registration_config.to_dict())
    _write_json(output_dir / "registration-candidate-grid.json", registration_candidate_grid)
    _write_json(output_dir / "selected-calibration.json", selected_calibration)
    _write_json(output_dir / "final-report.json", final_report)
    _write_json(output_dir / "final-summary.json", final_summary)
    _write_json(output_dir / "adjudication.json", adjudication)
    _write_json(output_dir / "paired-comparison.json", paired)
    _write_json(output_dir / "registration-displacement-atlas.json", displacement_atlas)
    _write_json(output_dir / "translation-family-atlas.json", translation_atlas)
    _write_json(output_dir / "residual-error-atlas.json", residual_atlas)
    readme = (
        "# Visual local baseline showdown v1\n\n"
        f"- date: {_calendar_date_label(datetime.now(timezone.utc))}\n"
        f"- system: `{SYSTEM_ID}` / `{SYSTEM_NAME}`\n"
        f"- outcome: `{outcome}`\n"
        f"- usefulness status: `{usefulness_status}`\n"
        f"- next action: `{next_action}`\n"
    )
    (output_dir / "README.md").write_text(readme, encoding="utf-8")

    with (output_dir / "registration-displacement-atlas.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["family_id", "count", "row_correct_before", "row_correct_after", "action_correct_before", "action_correct_after"],
        )
        writer.writeheader()
        for family_id, payload in displacement_atlas["families"].items():
            writer.writerow({"family_id": family_id, **{k: payload[k] for k in writer.fieldnames if k != "family_id"}})

    final_report_payload_digest = _json_digest(final_report)
    dirty_after_run = bool(_git_output("status", "--short"))
    runtime["dirty_after_run"] = dirty_after_run
    _write_json(output_dir / "runtime.json", runtime)
    run_manifest = {
        "version": "zeromodel-visual-benchmark-run-manifest/v1",
        "generator_version": GENERATOR_VERSION,
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "exact_command": command,
        "argv": list(argv),
        "git_commit": _git_output("rev-parse", "HEAD"),
        "branch_or_ref": _git_output("branch", "--show-current") or os.environ.get("GITHUB_HEAD_REF") or os.environ.get("GITHUB_REF_NAME") or "detached",
        "dirty_at_start": False,
        "dirty_after_run": dirty_after_run,
        "generated_paths": sorted(str(path.relative_to(output_dir)).replace("\\", "/") for path in output_dir.rglob("*") if path.is_file()),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "dependency_versions": {},
        "dataset_digest": dataset.manifest.digest,
        "registration_config_digest": registration_config.digest,
        "candidate_grid_digest": _json_digest(registration_candidate_grid),
        "selection_digest": selection.digest,
        "calibration_digest": selected.calibration.digest,
        "final_report_payload_digest": final_report_payload_digest,
        "trace_digest": trace_digest,
        "bundle_manifest_path": "bundle-manifest.json",
        "starting_main_sha": starting_main_sha,
    }
    _write_json(output_dir / "run-manifest.json", run_manifest)
    bundle_manifest = _bundle_manifest(output_dir)
    _write_json(output_dir / "bundle-manifest.json", bundle_manifest)

    required = [
        "README.md",
        "protocol.json",
        "environment.json",
        "argv.json",
        "command.txt",
        "runtime.json",
        "run-manifest.json",
        "dataset-manifest.json",
        "system-b-frozen-reference.json",
        "registration-config.json",
        "registration-candidate-grid.json",
        "selected-calibration.json",
        "final-report.json",
        "final-summary.json",
        "adjudication.json",
        "traces.jsonl",
        "paired-comparison.json",
        "registration-displacement-atlas.json",
        "translation-family-atlas.json",
        "residual-error-atlas.json",
        "bundle-manifest.json",
    ]
    _verify_required_files(output_dir, required)
    return {
        "outcome": outcome,
        "usefulness_status": usefulness_status,
        "next_action": next_action,
        "selection_digest": selection.digest,
        "calibration_digest": selected.calibration.digest,
        "trace_digest": trace_digest,
        "bundle_manifest_digest": _sha256_file(output_dir / "bundle-manifest.json"),
        "final_metrics": final_metrics,
        "registration_config_digest": registration_config.digest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--variants-per-family", type=int, default=3)
    parser.add_argument("--max-dx", type=int, default=3)
    parser.add_argument("--max-dy", type=int, default=3)
    parser.add_argument("--minimum-overlap-fraction", type=float, default=0.6)
    args = parser.parse_args()
    argv, command = _command_payload([sys.executable, *sys.argv])
    payload = run_showdown(
        output_dir=args.output_dir,
        variants_per_family=args.variants_per_family,
        max_dx=args.max_dx,
        max_dy=args.max_dy,
        minimum_overlap_fraction=args.minimum_overlap_fraction,
        argv=argv,
        command=command,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
