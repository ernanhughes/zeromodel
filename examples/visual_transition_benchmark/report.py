"""Aggregation of per-transition scores into the required output files."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

from visual_transition_benchmark import metrics as mx
from visual_transition_benchmark.baselines import SystemOutput
from visual_transition_benchmark.dataset import FAULT_CATEGORIES, ORDINARY_CATEGORIES, TransitionRecord
from visual_transition_benchmark.zeromodel_adapter import TransitionAnalysis


def _field_metrics(records: Sequence[TransitionRecord], outputs: Sequence) -> Mapping[str, float]:
    precisions = []
    recalls = []
    for record, output in zip(records, outputs):
        truth = mx.ground_truth_changed_fields(record.frame_before, record.frame_after)
        precision, recall = mx.field_precision_recall(output.predicted_fields, truth)
        precisions.append(precision)
        recalls.append(recall)
    n = len(records)
    return {
        "mean_precision": sum(precisions) / n if n else 0.0,
        "mean_recall": sum(recalls) / n if n else 0.0,
        "n": n,
    }


def score_group(
    records: Sequence[TransitionRecord],
    zm_outputs: Sequence[TransitionAnalysis],
    pd_outputs: Sequence[SystemOutput],
    priv_outputs: Sequence[SystemOutput],
) -> Mapping[str, object]:
    if not records:
        return {"n": 0}

    observed = [r.observed_changed_components for r in records]
    improvement, verdicts = mx.compare_to_pixel_diff(records, zm_outputs, pd_outputs)

    return {
        "n": len(records),
        "field_level": {
            "unit": "P4A field tile (4x1 px)",
            "pixel_diff": _field_metrics(records, pd_outputs),
            "zeromodel": _field_metrics(records, zm_outputs),
            "privileged": _field_metrics(records, priv_outputs),
        },
        "component_attribution": {
            "pixel_diff": mx.component_multilabel_metrics(
                [o.predicted_components for o in pd_outputs], observed
            ),
            "zeromodel": mx.component_multilabel_metrics(
                [o.predicted_components for o in zm_outputs], observed
            ),
            "privileged": mx.component_multilabel_metrics(
                [o.predicted_components for o in priv_outputs], observed
            ),
        },
        "unexpected_change_detection": {
            "pixel_diff": asdict(mx.unexpected_change_summary(records, pd_outputs)),
            "zeromodel": asdict(mx.unexpected_change_summary(records, zm_outputs)),
            "privileged": asdict(mx.unexpected_change_summary(records, priv_outputs)),
        },
        "missing_expected_change_detection": {
            "pixel_diff": asdict(mx.missing_change_summary(records, pd_outputs)),
            "zeromodel": asdict(mx.missing_change_summary(records, zm_outputs)),
            "privileged": asdict(mx.missing_change_summary(records, priv_outputs)),
        },
        "false_implicated_components": {
            "pixel_diff": asdict(mx.false_implicated_summary(records, pd_outputs)),
            "zeromodel": asdict(mx.false_implicated_summary(records, zm_outputs)),
            "privileged": asdict(mx.false_implicated_summary(records, priv_outputs)),
        },
        "improvement_over_pixel_diff": {
            "n": improvement.n,
            "better": improvement.better,
            "equal": improvement.equal,
            "worse": improvement.worse,
            "pct_better": improvement.pct_better,
            "pct_equal": improvement.pct_equal,
            "pct_worse": improvement.pct_worse,
        },
    }


def build_metrics_report(
    records: Sequence[TransitionRecord],
    zm_outputs: Sequence[TransitionAnalysis],
    pd_outputs: Sequence[SystemOutput],
    priv_outputs: Sequence[SystemOutput],
) -> Mapping[str, object]:
    def _subset(predicate):
        idx = [i for i, r in enumerate(records) if predicate(r)]
        return (
            [records[i] for i in idx],
            [zm_outputs[i] for i in idx],
            [pd_outputs[i] for i in idx],
            [priv_outputs[i] for i in idx],
        )

    report = {
        "all": score_group(records, zm_outputs, pd_outputs, priv_outputs),
        "ordinary": score_group(*_subset(lambda r: not r.is_faulty)),
        "faulty": score_group(*_subset(lambda r: r.is_faulty)),
        "by_category": {},
        "by_fault_type": {},
    }
    for category in ORDINARY_CATEGORIES + FAULT_CATEGORIES:
        report["by_category"][category] = score_group(*_subset(lambda r, c=category: r.category == c))
    for fault_type in FAULT_CATEGORIES:
        report["by_fault_type"][fault_type] = score_group(
            *_subset(lambda r, f=fault_type: r.fault_type == f)
        )
    return report


def write_transition_level_results(
    path: Path,
    records: Sequence[TransitionRecord],
    zm_outputs: Sequence[TransitionAnalysis],
    pd_outputs: Sequence[SystemOutput],
    priv_outputs: Sequence[SystemOutput],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record, zm_out, pd_out, priv_out in zip(records, zm_outputs, pd_outputs, priv_outputs):
            truth_fields = mx.ground_truth_changed_fields(record.frame_before, record.frame_after)
            row = {
                "transition_id": record.transition_id,
                "episode_id": record.episode_id,
                "step_number": record.step_number,
                "seed": record.seed,
                "action": record.action,
                "category": record.category,
                "fault_type": record.fault_type,
                "is_faulty": record.is_faulty,
                "expected_changed_components": list(record.expected_changed_components),
                "observed_changed_components": list(record.observed_changed_components),
                "ground_truth_changed_field_count": len(truth_fields),
                "systems": {
                    "pixel_diff": {
                        "predicted_components": list(pd_out.predicted_components),
                        "predicted_field_count": len(pd_out.predicted_fields),
                        "missing_components": list(pd_out.missing_components),
                        "unexpected_components": list(pd_out.unexpected_components),
                    },
                    "privileged": {
                        "predicted_components": list(priv_out.predicted_components),
                        "missing_components": list(priv_out.missing_components),
                        "unexpected_components": list(priv_out.unexpected_components),
                    },
                    "zeromodel": {
                        "predicted_components": list(zm_out.predicted_components),
                        "predicted_field_count": len(zm_out.predicted_fields),
                        "expected_components": list(zm_out.expected_components),
                        "missing_components": list(zm_out.missing_components),
                        "unexpected_components": list(zm_out.unexpected_components),
                        "evidence_scores": zm_out.evidence_scores,
                        "conformance_status": zm_out.diagnostics["conformance_status"],
                        "unexplained_components": list(zm_out.diagnostics["unexplained_components"]),
                    },
                },
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def render_summary_markdown(
    environment: Mapping[str, object],
    metrics_report: Mapping[str, object],
) -> str:
    all_m = metrics_report["all"]
    ordinary_m = metrics_report["ordinary"]
    faulty_m = metrics_report["faulty"]
    improve = all_m["improvement_over_pixel_diff"]

    lines = []
    lines.append("# Visual Transition Debugging Benchmark -- Summary")
    lines.append("")
    lines.append("## Executive result")
    lines.append("")
    lines.append(
        "ZeroModel (P4A field partitioning + P18A transition evidence + P18B "
        "action-conditioned conformance) provides a **measurable but narrow** "
        "localization advantage over raw pixel differencing: it adds reliable "
        "component-name attribution and catches two specific classes of fault "
        "(declared-stable-region violations, and declared-must-change absences) "
        "that pixel differencing cannot represent at all. It is blind to faults "
        "in regions it has no crisp expectation for (alien hit/miss, cooldown "
        "state outside FIRE) and to faults that preserve the correct *label* "
        "while flipping direction/target. See 'Where ZeroModel failed' below."
    )
    lines.append("")
    lines.append("## Exact environment")
    lines.append("")
    for key, value in environment.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Main metrics (evaluation split)")
    lines.append("")
    lines.append("| Metric | Pixel diff | Privileged | ZeroModel |")
    lines.append("|---|---:|---:|---:|")
    ca = all_m["component_attribution"]
    lines.append(
        "| Visible changed-component attribution micro-F1 | %.3f | %.3f | %.3f |"
        % (ca["pixel_diff"]["micro_f1"], ca["privileged"]["micro_f1"], ca["zeromodel"]["micro_f1"])
    )
    lines.append(
        "| Component exact-set accuracy | %.3f | %.3f | %.3f |"
        % (
            ca["pixel_diff"]["exact_set_accuracy"],
            ca["privileged"]["exact_set_accuracy"],
            ca["zeromodel"]["exact_set_accuracy"],
        )
    )
    fl = all_m["field_level"]
    lines.append(
        "| Field-level mean recall | %.3f | %.3f | %.3f |"
        % (fl["pixel_diff"]["mean_recall"], fl["privileged"]["mean_recall"], fl["zeromodel"]["mean_recall"])
    )
    lines.append(
        "| Missing-change detection rate (faulty only) | n/a (0 by construction) | %.3f | %.3f |"
        % (
            faulty_m["missing_expected_change_detection"]["privileged"]["detection_rate"],
            faulty_m["missing_expected_change_detection"]["zeromodel"]["detection_rate"],
        )
    )
    lines.append(
        "| Unexpected-change detection rate (faulty only) | n/a (0 by construction) | %.3f | %.3f |"
        % (
            faulty_m["unexpected_change_detection"]["privileged"]["detection_rate"],
            faulty_m["unexpected_change_detection"]["zeromodel"]["detection_rate"],
        )
    )
    lines.append(
        "| False alarm rate on correct transitions | n/a | 0.000 | %.3f |"
        % (ordinary_m["missing_expected_change_detection"]["zeromodel"]["false_alarm_rate_on_correct"])
    )
    lines.append(
        "| Mean false-implicated components | %.3f | %.3f | %.3f |"
        % (
            all_m["false_implicated_components"]["pixel_diff"]["mean_count"],
            all_m["false_implicated_components"]["privileged"]["mean_count"],
            all_m["false_implicated_components"]["zeromodel"]["mean_count"],
        )
    )
    lines.append("")
    lines.append(
        f"ZeroModel vs. pixel-diff, per transition: **{improve['pct_better']:.1%} better**, "
        f"{improve['pct_equal']:.1%} equal, {improve['pct_worse']:.1%} worse "
        f"(n={improve['n']})."
    )
    lines.append("")
    lines.append("## Fault detection results by fault type")
    lines.append("")
    lines.append("| Fault type | n | ZeroModel missing-detect | ZeroModel unexpected-detect | ZeroModel false-implicated (mean) |")
    lines.append("|---|---:|---:|---:|---:|")
    for fault_type, group in metrics_report["by_fault_type"].items():
        if group.get("n", 0) == 0:
            continue
        lines.append(
            "| %s | %d | %.3f | %.3f | %.3f |"
            % (
                fault_type,
                group["n"],
                group["missing_expected_change_detection"]["zeromodel"]["detection_rate"],
                group["unexpected_change_detection"]["zeromodel"]["detection_rate"],
                group["false_implicated_components"]["zeromodel"]["mean_count"],
            )
        )
    lines.append("")
    lines.append("## Scientific interpretation")
    lines.append("")
    lines.append(
        "- **What this demonstrates**: within this controlled arcade environment, "
        "field-partitioned transition evidence plus action-conditioned "
        "conformance checking localizes known visual transitions to named "
        "regions, and catches both an unexpected background mutation and a "
        "declared movement that silently failed to occur -- two things raw "
        "pixel differencing structurally cannot express."
    )
    lines.append(
        "- **What it suggests**: aggregating pixel evidence into declared, "
        "named fields is a cheap, effective way to add component-level "
        "attribution on top of pixel differencing, and declaring per-action "
        "expectations over those fields is enough to catch anomalies in "
        "regions where the expectation is crisp (tank motion, background "
        "stability)."
    )
    lines.append(
        "- **What it does not establish**: it does not establish general "
        "vision, causal discovery, or semantic understanding from pixels. It "
        "does not generalize past this environment's fixed layout. It cannot "
        "resolve faults that require hidden state (hit/miss, exact cooldown "
        "counter, movement direction) that isn't recoverable from frames + "
        "action alone -- those require either richer metadata or acceptance "
        "of the 'unexplained, needs review' bucket instead of a pass/fail claim."
    )
    lines.append("")
    lines.append("## Architecture implications")
    lines.append("")
    lines.append(
        "- **Genuinely needed**: P4A field partitioning (`fields.py`), P18A "
        "transition evidence (`transition_evidence.py`), P18B action-conditioned "
        "conformance (`transition_conformance.py`), and P6 region annotations "
        "(`expectations.py`, used only for declaring static bands)."
    )
    lines.append(
        "- **Used for a secondary demonstration only**: P18C recurrent "
        "unexplained-transition discovery (`transition_discovery.py`) -- see "
        "the discovery note in this file; it is not required for the core "
        "per-transition metrics."
    )
    lines.append(
        "- **Bypassed entirely**: every certification/governance/promotion/"
        "lifecycle stage (P12-P17, P18D-P18G). None of it is needed to answer "
        "this benchmark's question."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    lines.append(
        "**Continue and strengthen the visual-debugging direction**, scoped "
        "narrowly to P4A/P18A/P18B(+P18C): the representation earns its keep "
        "on component attribution and on the two fault families it can "
        "structurally express. Do not extend the certification/governance "
        "chain on the strength of this result -- it was not exercised and "
        "was not needed."
    )
    return "\n".join(lines) + "\n"
