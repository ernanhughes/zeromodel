"""Cross-domain replication CLI: runs the same protocol, systems, and metric
functions over the arcade domain (reused, unmodified from stages 1/2) and the
new warehouse domain, then reports whether each capability class replicates.

Usage:
    python -m visual_transition_benchmark.cross_domain_run \
        --arcade-eval-episodes 100 --warehouse-eval-episodes 100 \
        --output-dir artifacts/cross_domain_visual_contracts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from visual_transition_benchmark import cross_domain_baselines as cdb
from visual_transition_benchmark import cross_domain_metrics as cdm
from visual_transition_benchmark import metrics as mx
from visual_transition_benchmark.domains.arcade.domain import ArcadeTransitionDomain
from visual_transition_benchmark.domains.protocol import AnalysisMetadata, VisualTransitionDomain
from visual_transition_benchmark.domains.warehouse.domain import WarehouseTransitionDomain
from visual_transition_benchmark.render import build_html_index, render_transition_panel
from visual_transition_benchmark.run import _git_sha


def _generate(domain: VisualTransitionDomain, *, prefix: str, episode_count: int, seed_offset: int):
    episode_ids = tuple(f"{prefix}-{index:04d}" for index in range(episode_count))
    transitions = []
    for index, episode_id in enumerate(episode_ids):
        transitions.extend(domain.generate_episode(seed=seed_offset + index, episode_id=episode_id))
    return episode_ids, tuple(transitions)


def _analyze_domain(domain: VisualTransitionDomain, transitions, band_masks):
    component_analyzer = domain.build_component_analyzer()
    value_analyzer = domain.build_value_analyzer()
    component_outputs, value_outputs, pixel_outputs, privileged_outputs = [], [], [], []
    for transition in transitions:
        metadata = AnalysisMetadata(transition.transition_id, transition.step_number)
        component_outputs.append(
            component_analyzer.analyze(transition.frame_before, transition.frame_after, transition.action, metadata)
        )
        value_outputs.append(
            value_analyzer.analyze(transition.frame_before, transition.frame_after, transition.action, metadata)
        )
        pixel_outputs.append(cdb.pixel_diff_baseline(transition.frame_before, transition.frame_after, band_masks))
        privileged_outputs.append(cdb.privileged_baseline(transition))
    return component_outputs, value_outputs, pixel_outputs, privileged_outputs


def _domain_report(domain_name, transitions, component_outputs, value_outputs, pixel_outputs, privileged_outputs):
    observed = [t.observed_changed_components for t in transitions]

    component_attribution = {
        "zeromodel": mx.component_multilabel_metrics([o.predicted_components for o in component_outputs], observed),
        "pixel_diff": mx.component_multilabel_metrics([o.predicted_components for o in pixel_outputs], observed),
        "privileged": mx.component_multilabel_metrics([o.predicted_components for o in privileged_outputs], observed),
    }
    unexpected_summary = mx.unexpected_change_summary(transitions, component_outputs)
    missing_summary = mx.missing_change_summary(transitions, component_outputs)
    false_implicated = mx.false_implicated_summary(transitions, component_outputs)

    capability_rates = {
        capability: asdict(cdm.capability_rate(capability, transitions, value_outputs))
        for capability in ("direction", "magnitude", "value", "relation", "identity")
    }
    value_detection = asdict(cdm.value_fault_detection(transitions, value_outputs))
    hidden = asdict(cdm.label_correct_but_value_wrong(transitions, component_outputs, value_outputs))

    ordinary = [t for t in transitions if not t.is_faulty]
    faulty = [t for t in transitions if t.is_faulty]

    return {
        "domain": domain_name,
        "n": len(transitions),
        "n_ordinary": len(ordinary),
        "n_faulty": len(faulty),
        "component_attribution_micro_f1": {
            system: result["micro_f1"] for system, result in component_attribution.items()
        },
        "component_attribution_full": component_attribution,
        "unexpected_change_detection": asdict(unexpected_summary),
        "missing_change_detection": asdict(missing_summary),
        "false_implicated_components": asdict(false_implicated),
        "value_capability_rates": capability_rates,
        "value_fault_detection": value_detection,
        "hidden_value_faults": hidden,
    }


def _capability_table(arcade_report, warehouse_report) -> list:
    def rate(report, key):
        return report["component_attribution_micro_f1"].get(key)

    rows = [
        {
            "capability": "visible_component_attribution_micro_f1",
            "arcade": rate(arcade_report, "zeromodel"),
            "warehouse": rate(warehouse_report, "zeromodel"),
        },
        {
            "capability": "unexpected_change_detection_rate",
            "arcade": arcade_report["unexpected_change_detection"]["detection_rate"],
            "warehouse": warehouse_report["unexpected_change_detection"]["detection_rate"],
        },
        {
            "capability": "missing_change_detection_rate",
            "arcade": arcade_report["missing_change_detection"]["detection_rate"],
            "warehouse": warehouse_report["missing_change_detection"]["detection_rate"],
        },
    ]
    for capability in ("direction", "magnitude", "value", "relation", "identity"):
        rows.append(
            {
                "capability": f"{capability}_correctness_rate",
                "arcade": arcade_report["value_capability_rates"][capability]["n_correct"]
                / arcade_report["value_capability_rates"][capability]["n_applicable"]
                if arcade_report["value_capability_rates"][capability]["n_applicable"]
                else None,
                "warehouse": warehouse_report["value_capability_rates"][capability]["n_correct"]
                / warehouse_report["value_capability_rates"][capability]["n_applicable"]
                if warehouse_report["value_capability_rates"][capability]["n_applicable"]
                else None,
            }
        )
    return rows


REPLICATION_THRESHOLDS = {
    "visible_component_attribution_micro_f1": 0.95,
    "unexpected_change_detection_rate": 0.90,
    "missing_change_detection_rate": 0.90,
    "direction_correctness_rate": 0.90,
    "value_correctness_rate": 0.90,
    "relation_correctness_rate": 0.90,
}


def _replication_decisions(table: list) -> list:
    decisions = []
    for row in table:
        threshold = REPLICATION_THRESHOLDS.get(row["capability"])
        if threshold is None:
            decisions.append({**row, "replicated": "not_applicable"})
            continue
        arcade_ok = row["arcade"] is not None and row["arcade"] >= threshold
        warehouse_ok = row["warehouse"] is not None and row["warehouse"] >= threshold
        if row["arcade"] is None or row["warehouse"] is None:
            status = "not_measurable_in_both_domains"
        elif arcade_ok and warehouse_ok:
            status = "replicated"
        elif arcade_ok or warehouse_ok:
            status = "domain_specific"
        else:
            status = "not_replicated"
        decisions.append({**row, "threshold": threshold, "replicated": status})
    return decisions


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arcade-dev-episodes", type=int, default=20)
    parser.add_argument("--arcade-eval-episodes", type=int, default=100)
    parser.add_argument("--warehouse-dev-episodes", type=int, default=20)
    parser.add_argument("--warehouse-eval-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/cross_domain_visual_contracts"))
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args(argv)

    started = time.time()

    arcade = ArcadeTransitionDomain()
    warehouse = WarehouseTransitionDomain()

    arcade_dev_ids, _ = _generate(arcade, prefix="cd-arcade-dev", episode_count=args.arcade_dev_episodes, seed_offset=0)
    arcade_eval_ids, arcade_eval = _generate(
        arcade, prefix="cd-arcade-eval", episode_count=args.arcade_eval_episodes, seed_offset=5_000_000
    )
    warehouse_dev_ids, _ = _generate(
        warehouse, prefix="cd-wh-dev", episode_count=args.warehouse_dev_episodes, seed_offset=0
    )
    warehouse_eval_ids, warehouse_eval = _generate(
        warehouse, prefix="cd-wh-eval", episode_count=args.warehouse_eval_episodes, seed_offset=7_000_000
    )
    assert set(arcade_dev_ids).isdisjoint(arcade_eval_ids)
    assert set(warehouse_dev_ids).isdisjoint(warehouse_eval_ids)

    arcade_band_masks = cdb.declared_band_masks_arcade()
    warehouse_band_masks = cdb.declared_band_masks_warehouse()

    arcade_component, arcade_value, arcade_pixel, arcade_priv = _analyze_domain(arcade, arcade_eval, arcade_band_masks)
    warehouse_component, warehouse_value, warehouse_pixel, warehouse_priv = _analyze_domain(
        warehouse, warehouse_eval, warehouse_band_masks
    )

    arcade_report = _domain_report("arcade", arcade_eval, arcade_component, arcade_value, arcade_pixel, arcade_priv)
    warehouse_report = _domain_report(
        "warehouse", warehouse_eval, warehouse_component, warehouse_value, warehouse_pixel, warehouse_priv
    )

    capability_table = _capability_table(arcade_report, warehouse_report)
    decisions = _replication_decisions(capability_table)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "domain-results").mkdir(parents=True, exist_ok=True)

    with (output_dir / "domain-results" / "arcade.json").open("w", encoding="utf-8") as handle:
        json.dump(arcade_report, handle, indent=2, sort_keys=True, default=str)
    with (output_dir / "domain-results" / "warehouse.json").open("w", encoding="utf-8") as handle:
        json.dump(warehouse_report, handle, indent=2, sort_keys=True, default=str)

    with (output_dir / "transition-level-results.jsonl").open("w", encoding="utf-8") as handle:
        for domain_name, transitions, component_outputs, value_outputs in (
            ("arcade", arcade_eval, arcade_component, arcade_value),
            ("warehouse", warehouse_eval, warehouse_component, warehouse_value),
        ):
            for transition, component, value in zip(transitions, component_outputs, value_outputs):
                row = {
                    "domain": domain_name,
                    "transition_id": transition.transition_id,
                    "category": transition.category,
                    "fault_type": transition.fault_type,
                    "is_faulty": transition.is_faulty,
                    "action": transition.action,
                    "predicted_components": list(component.predicted_components),
                    "missing_components": list(component.missing_components),
                    "unexpected_components": list(component.unexpected_components),
                    "value_flags": list(value.value_flags),
                    "decoded": {k: (list(v) if isinstance(v, tuple) else v) for k, v in value.decoded.items()},
                }
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    environment = {
        "git_commit": _git_sha(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "command": " ".join(sys.argv),
        "arcade_dev_episodes": args.arcade_dev_episodes,
        "arcade_eval_episodes": args.arcade_eval_episodes,
        "arcade_eval_transitions": len(arcade_eval),
        "warehouse_dev_episodes": args.warehouse_dev_episodes,
        "warehouse_eval_episodes": args.warehouse_eval_episodes,
        "warehouse_eval_transitions": len(warehouse_eval),
    }

    artifact_rows = []
    if not args.skip_render:
        artifacts_dir = output_dir / "representative-artifacts"
        samples = []
        for transition, component, priv in zip(arcade_eval, arcade_component, arcade_priv):
            if transition.category in ("tank_moves_wrong_direction", "background_changes_unexpectedly", "fire_no_projectile"):
                samples.append((transition, component, priv))
        for transition, component, priv in zip(warehouse_eval, warehouse_component, warehouse_priv):
            if transition.category in (
                "robot_moves_wrong_direction",
                "wall_changes_unexpectedly",
                "wrong_crate_moves",
                "push_advances_robot_without_crate",
            ):
                samples.append((transition, component, priv))
        seen_categories = set()
        for transition, component, priv in samples:
            key = (transition.domain_name, transition.category)
            if key in seen_categories:
                continue
            seen_categories.add(key)
            png_path = artifacts_dir / f"{transition.domain_name}-{transition.transition_id}.png"
            render_transition_panel(transition, priv, component, output_path=png_path)
            artifact_rows.append(
                {
                    "transition_id": transition.transition_id,
                    "category": f"{transition.domain_name}/{transition.category}",
                    "fault_type": transition.fault_type,
                    "verdict": "flagged" if (component.missing_components or component.unexpected_components) else "clean",
                    "zeromodel_status": "n/a",
                    "artifact_path": f"representative-artifacts/{transition.domain_name}-{transition.transition_id}.png",
                }
            )
        build_html_index(
            artifact_rows, output_path=output_dir / "visual-index.html", title="Cross-Domain Visual Contracts"
        )

    duration = time.time() - started
    environment["duration_seconds"] = round(duration, 3)

    results = {
        "environment": environment,
        "capability_table": decisions,
        "domain_reports": {"arcade": arcade_report, "warehouse": warehouse_report},
    }
    (output_dir / "cross-domain-results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    summary = _render_summary(environment, decisions, arcade_report, warehouse_report)
    (output_dir / "cross-domain-summary.md").write_text(summary, encoding="utf-8")

    print(json.dumps(environment, indent=2, sort_keys=True))
    print(f"wrote results to {output_dir}")
    return 0


def _fmt(value) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def _render_summary(environment, decisions, arcade_report, warehouse_report) -> str:
    lines = []
    lines.append("# Cross-Domain Visual Contract Replication -- Summary")
    lines.append("")
    lines.append("## Executive result")
    lines.append("")
    replicated = [d["capability"] for d in decisions if d["replicated"] == "replicated"]
    domain_specific = [d["capability"] for d in decisions if d["replicated"] == "domain_specific"]
    not_replicated = [d["capability"] for d in decisions if d["replicated"] == "not_replicated"]
    not_measurable = [d["capability"] for d in decisions if d["replicated"] == "not_measurable_in_both_domains"]
    lines.append(f"- **Replicated in both domains**: {replicated or 'none'}")
    lines.append(f"- **Domain-specific (one domain only)**: {domain_specific or 'none'}")
    lines.append(f"- **Not replicated in either domain**: {not_replicated or 'none'}")
    lines.append(f"- **Not measurable in both domains** (e.g. identity is arcade-unavailable): {not_measurable or 'none'}")
    lines.append("")
    lines.append("## Exact environment")
    lines.append("")
    for key, value in environment.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Capability table (Arcade vs. Warehouse)")
    lines.append("")
    lines.append("| Capability | Arcade | Warehouse | Threshold | Status |")
    lines.append("|---|---:|---:|---:|---|")
    for row in decisions:
        threshold = row.get("threshold")
        lines.append(
            f"| {row['capability']} | {_fmt(row['arcade'])} | {_fmt(row['warehouse'])} | "
            f"{_fmt(threshold)} | {row['replicated']} |"
        )
    lines.append("")
    lines.append("## Hidden value-fault headline (per domain)")
    lines.append("")
    for name, report in (("arcade", arcade_report), ("warehouse", warehouse_report)):
        hidden = report["hidden_value_faults"]
        lines.append(
            f"- **{name}**: {hidden['label_clean_but_value_wrong']} of {hidden['n_faulty']} faulty transitions "
            f"were component-label-clean yet value-wrong."
        )
    lines.append("")
    lines.append("## What replicated, what did not")
    lines.append("")
    lines.append(
        "See `domain-results/arcade.json` and `domain-results/warehouse.json` for full per-domain metrics, "
        "and `transition-level-results.jsonl` for the per-transition record."
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
