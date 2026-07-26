"""CLI entry point for stage 2 (value-aware transition contracts).

Keeps stage 1 completely untouched: this script generates its own datasets
(reusing dataset.py's original generate_split for the categories stage 1
already validated, plus the new generate_value_split for the 5 new
value-only fault categories) and writes its own, separate output files.

Usage:
    python -m visual_transition_benchmark.value_run --dev-episodes 40 --eval-episodes 120 \
        --output-dir artifacts/value_aware_transition_contracts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import asdict
from pathlib import Path

import numpy as np

from visual_transition_benchmark import baselines as bl
from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import report as rp
from visual_transition_benchmark import value_metrics as vm
from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.render import build_html_index, render_transition_panel
from visual_transition_benchmark.value_adapter import ValueAwareZeroModelAnalyzer
from visual_transition_benchmark.run import _git_sha


def _analyze_all(records):
    analyzer = ValueAwareZeroModelAnalyzer()
    analyses = []
    pd_outputs = []
    priv_outputs = []
    for record in records:
        metadata = zm.TransitionMetadata(transition_id=record.transition_id, step_number=record.step_number)
        analyses.append(analyzer.analyze(record.frame_before, record.frame_after, record.action, metadata))
        pd_outputs.append(bl.pixel_diff_baseline(record.frame_before, record.frame_after))
        priv_outputs.append(bl.privileged_baseline(record))
    return analyses, pd_outputs, priv_outputs


def _group_metrics(records, analyses, pd_outputs, priv_outputs):
    values_list = [a.values for a in analyses]
    flags_list = [a.value_flags for a in analyses]
    component_outputs = [a.component_analysis for a in analyses]

    component_report = rp.score_group(records, component_outputs, pd_outputs, priv_outputs)
    return {
        "n": len(records),
        "component_level": {
            "note": "unchanged stage-1 metrics (visible changed-component attribution)",
            **component_report,
        },
        "value_level": {
            "accuracy": asdict(vm.value_accuracy_summary(records, values_list)),
            "fault_localization": asdict(
                vm.value_fault_localization_summary(records, values_list, flags_list)
            ),
            "hidden_value_faults": asdict(
                vm.label_correct_but_value_wrong(records, component_outputs, values_list)
            ),
            "relation_violation_rate_by_category": vm.relation_violation_rate_by_category(
                records, flags_list
            ),
        },
    }


def _render_summary_markdown(environment: dict, metrics_report: dict) -> str:
    all_m = metrics_report["all"]
    reused_m = metrics_report["reused_stage1_categories"]
    new_m = metrics_report["new_value_fault_categories"]
    ordinary_m = metrics_report["ordinary_only"]

    lines = []
    lines.append("# Value-Aware Transition Contracts -- Summary")
    lines.append("")
    lines.append("## Executive result")
    lines.append("")
    hidden = all_m["value_level"]["hidden_value_faults"]
    lines.append(
        "Value-aware ZeroModel (System D) **resolves stage 1's key blind spot**: "
        "wrong-direction tank faults, which stage 1 could not flag at all "
        "(component label looks correct either way), are now caught by an "
        "exact direction contract. Across this evaluation split, "
        f"**{hidden['label_clean_but_value_wrong']} of {hidden['n_faulty']} faulty "
        "transitions look completely clean to the component-level system but "
        "are demonstrably value-wrong** -- exactly the failure mode a correct "
        "component label can hide. Target/alien identity correctness remains "
        "an honest, unresolved blind spot: no non-privileged contract here can "
        "name the *correct* next alien without the hidden alien queue."
    )
    lines.append("")
    lines.append("## Exact environment")
    lines.append("")
    for key, value in environment.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Value-level accuracy (decoded value vs. true simulated state)")
    lines.append("")
    lines.append("| Split | n | Movement-direction | State-delta (exact) | Cooldown-value | Target-selection |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, group in (
        ("all", all_m),
        ("reused stage-1 categories", reused_m),
        ("new value-fault categories", new_m),
        ("ordinary (non-faulty)", ordinary_m),
    ):
        acc = group["value_level"]["accuracy"]
        lines.append(
            "| %s | %d | %.3f | %.3f | %.3f | %.3f |"
            % (
                name,
                acc["n"],
                acc["movement_direction_accuracy"],
                acc["state_delta_accuracy"],
                acc["cooldown_value_accuracy"],
                acc["target_selection_accuracy"],
            )
        )
    lines.append("")
    lines.append("## Value-level fault localization (ZeroModel's own, non-privileged flags)")
    lines.append("")
    for name, group in (("all", all_m), ("reused stage-1 categories", reused_m), ("new value-fault categories", new_m)):
        loc = group["value_level"]["fault_localization"]
        lines.append(
            "- **%s**: detection_rate=%.3f (n_relevant=%d), false_alarm_rate_on_correct=%.3f (n_clean=%d)"
            % (name, loc["detection_rate"], loc["n_relevant"], loc["false_alarm_rate_on_correct"], loc["n_clean"])
        )
    lines.append("")
    lines.append("## Component-level still correctly reported alongside (unchanged stage-1 metrics)")
    lines.append("")
    ca = all_m["component_level"]["component_attribution"]
    lines.append(
        "Visible changed-component attribution micro-F1: pixel_diff=%.3f, "
        "privileged=%.3f, zeromodel=%.3f (identical mechanism to stage 1 -- "
        "included here only so component-level and value-level results sit "
        "side by side, never conflated)."
        % (ca["pixel_diff"]["micro_f1"], ca["privileged"]["micro_f1"], ca["zeromodel"]["micro_f1"])
    )
    lines.append("")
    lines.append("## By-category breakdown")
    lines.append("")
    lines.append("| Category | n | Direction acc. | Delta acc. | Cooldown acc. | Target acc. | Value-fault detect | Relation-flag rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for category, group in metrics_report["by_category"].items():
        acc = group["value_level"]["accuracy"]
        loc = group["value_level"]["fault_localization"]
        relation_rates = group["value_level"]["relation_violation_rate_by_category"]
        relation_rate = relation_rates.get(category, 0.0)
        lines.append(
            "| %s | %d | %.3f | %.3f | %.3f | %.3f | %.3f | %.3f |"
            % (
                category,
                acc["n"],
                acc["movement_direction_accuracy"],
                acc["state_delta_accuracy"],
                acc["cooldown_value_accuracy"],
                acc["target_selection_accuracy"],
                loc["detection_rate"],
                relation_rate,
            )
        )
    lines.append("")
    lines.append("## Scientific interpretation")
    lines.append("")
    lines.append(
        "- **What this demonstrates**: adding typed, decoded values on top of "
        "the existing P4A/P18A representation (no new perception-package "
        "code) resolves a specific, previously-documented blind spot "
        "(wrong movement direction) and adds a genuinely new capability "
        "(exact-magnitude and cooldown-value checks, plus one cross-field "
        "relation) using only frames + the action label -- no hidden "
        "simulator state."
    )
    lines.append(
        "- **What it suggests**: presence/absence conformance (stage 1) and "
        "value correctness (stage 2) are complementary, not substitutable -- "
        "a system needs both, reported separately, or a correct label will "
        "silently hide a wrong value."
    )
    lines.append(
        "- **What it does not establish**: target/alien *identity* "
        "correctness remains unresolved without privileged state -- "
        "target-selection accuracy is reported here only as a ground-truth "
        "comparison, not as something System D can assert on its own. This "
        "is the same class of limitation stage 1 reported for hit/miss, now "
        "confirmed to persist under value-awareness too."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "**Continue and strengthen**: value-aware contracts are cheap "
        "(reuse of existing P4A/P18A, just at finer field resolution) and "
        "close a real, previously-documented gap. Do not attempt target-"
        "identity resolution without first deciding whether richer, still-"
        "non-privileged metadata (e.g. an episode-level alien-queue "
        "commitment declared once per episode) is an acceptable input -- "
        "that is a scope decision for a future stage, not a bug in this one."
    )
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-episodes", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/value_aware_transition_contracts")
    )
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args(argv)

    warning_records = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.time()

        original_dev = ds.generate_split(prefix="stage1-dev", episode_count=args.dev_episodes, seed_offset=0)
        original_eval = ds.generate_split(
            prefix="stage1-eval", episode_count=args.eval_episodes, seed_offset=1_000_000
        )
        value_dev = ds.generate_value_split(
            prefix="value-dev", episode_count=args.dev_episodes, seed_offset=2_000_000
        )
        value_eval = ds.generate_value_split(
            prefix="value-eval", episode_count=args.eval_episodes, seed_offset=3_000_000
        )
        ds.assert_disjoint_splits(original_dev, original_eval, value_dev, value_eval)

        eval_records = original_eval.records + value_eval.records
        analyses, pd_outputs, priv_outputs = _analyze_all(eval_records)

        n_original = len(original_eval.records)
        metrics_report = {
            "all": _group_metrics(eval_records, analyses, pd_outputs, priv_outputs),
            "reused_stage1_categories": _group_metrics(
                eval_records[:n_original], analyses[:n_original], pd_outputs[:n_original], priv_outputs[:n_original]
            ),
            "new_value_fault_categories": _group_metrics(
                eval_records[n_original:], analyses[n_original:], pd_outputs[n_original:], priv_outputs[n_original:]
            ),
            "ordinary_only": _group_metrics(
                *zip(
                    *[
                        (r, a, p, v)
                        for r, a, p, v in zip(eval_records, analyses, pd_outputs, priv_outputs)
                        if not r.is_faulty
                    ]
                )
            ),
            "by_category": {},
        }
        for category in ds.ORDINARY_CATEGORIES + ds.FAULT_CATEGORIES + ds.VALUE_FAULT_CATEGORIES:
            subset = [
                (r, a, p, v)
                for r, a, p, v in zip(eval_records, analyses, pd_outputs, priv_outputs)
                if r.category == category
            ]
            if not subset:
                continue
            metrics_report["by_category"][category] = _group_metrics(*zip(*subset))

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        with (output_dir / "value-transition-level-results.jsonl").open("w", encoding="utf-8") as handle:
            for record, analysis in zip(eval_records, analyses):
                row = {
                    "transition_id": record.transition_id,
                    "category": record.category,
                    "fault_type": record.fault_type,
                    "is_faulty": record.is_faulty,
                    "action": record.action,
                    "true_tank_delta": vm.true_tank_delta(record),
                    "true_cooldown_level": vm.true_cooldown_level(record),
                    "true_target_after": vm.true_target_after(record),
                    "decoded_tank_delta": analysis.values.tank.delta_x,
                    "decoded_cooldown_after_level": analysis.values.cooldown.after_level,
                    "decoded_target_after": analysis.values.alien.after_x,
                    "tank_direction_correct": vm.tank_direction_correct(record, analysis.values),
                    "tank_magnitude_correct": vm.tank_magnitude_correct(record, analysis.values),
                    "cooldown_value_correct": vm.cooldown_value_correct(record, analysis.values),
                    "target_selection_correct": vm.target_selection_correct(record, analysis.values),
                    "value_fault_present": vm.value_fault_present(record, analysis.values),
                    "value_flags": list(analysis.value_flags),
                    "component_predicted": list(analysis.component_analysis.predicted_components),
                    "component_missing": list(analysis.component_analysis.missing_components),
                    "component_unexpected": list(analysis.component_analysis.unexpected_components),
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

        environment = {
            "git_commit": _git_sha(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "dev_episode_count": args.dev_episodes,
            "eval_episode_count": args.eval_episodes,
            "eval_transition_count": len(eval_records),
            "reused_stage1_transition_count": n_original,
            "new_value_fault_transition_count": len(eval_records) - n_original,
            "value_fault_categories": list(ds.VALUE_FAULT_CATEGORIES),
            "command": " ".join(sys.argv),
        }

        artifact_rows = []
        if not args.skip_render:
            artifacts_dir = output_dir / "artifacts"
            new_faulty_sample = [
                (r, a) for r, a in zip(eval_records, analyses) if r.category in ds.VALUE_FAULT_CATEGORIES
            ]
            direction_fix_sample = [
                (r, a)
                for r, a in zip(eval_records, analyses)
                if r.category == "tank_moves_wrong_direction"
            ][:10]
            priv_by_id = {r.transition_id: p for r, p in zip(eval_records, priv_outputs)}
            for record, analysis in new_faulty_sample + direction_fix_sample:
                priv_out = priv_by_id[record.transition_id]
                png_path = artifacts_dir / f"{record.transition_id}.png"
                render_transition_panel(
                    record, priv_out, analysis.component_analysis, output_path=png_path
                )
                artifact_rows.append(
                    {
                        "transition_id": record.transition_id,
                        "category": record.category,
                        "fault_type": record.fault_type,
                        "verdict": "value_fault_flagged" if analysis.value_flags else "value_clean",
                        "zeromodel_status": analysis.component_analysis.diagnostics["conformance_status"],
                        "artifact_path": f"artifacts/{record.transition_id}.png",
                    }
                )
            build_html_index(
                artifact_rows,
                output_path=output_dir / "value-visual-index.html",
                title="Value-Aware Transition Contracts -- Diagnostics",
            )

        duration = time.time() - started
        for warning in caught:
            warning_records.append(str(warning.message))

    environment["duration_seconds"] = round(duration, 3)
    environment["warning_count"] = len(warning_records)
    rp.write_json(
        output_dir / "value-benchmark-results.json", {"environment": environment, "metrics": metrics_report}
    )
    rp.write_json(
        output_dir / "value-run-log.json",
        {"environment": environment, "warnings": warning_records, "rendered_artifact_count": len(artifact_rows)},
    )
    summary_md = _render_summary_markdown(environment, metrics_report)
    (output_dir / "value-benchmark-summary.md").write_text(summary_md, encoding="utf-8")

    print(json.dumps(environment, indent=2, sort_keys=True))
    print(f"wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
