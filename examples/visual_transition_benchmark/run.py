"""CLI entry point: generate the dataset, run all three systems, score, render.

Usage:
    python -m visual_transition_benchmark.run --dev-episodes 40 --eval-episodes 120 \
        --output-dir artifacts/visual_transition_benchmark
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

from visual_transition_benchmark import baselines as bl
from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import discovery_demo as dd
from visual_transition_benchmark import metrics as mx
from visual_transition_benchmark import report as rp
from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.render import build_html_index, render_transition_panel


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2])
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def run_systems(records):
    analyzer = zm.ArcadeBandZeroModelAnalyzer()
    zm_outputs = []
    pd_outputs = []
    priv_outputs = []
    for record in records:
        metadata = zm.TransitionMetadata(transition_id=record.transition_id, step_number=record.step_number)
        zm_outputs.append(analyzer.analyze(record.frame_before, record.frame_after, record.action, metadata))
        pd_outputs.append(bl.pixel_diff_baseline(record.frame_before, record.frame_after))
        priv_outputs.append(bl.privileged_baseline(record))
    return zm_outputs, pd_outputs, priv_outputs


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-episodes", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=120)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/visual_transition_benchmark"))
    parser.add_argument(
        "--render-ordinary-episodes",
        type=int,
        default=2,
        help="number of eval episodes whose ordinary transitions also get a rendered panel",
    )
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args(argv)

    warning_records = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.time()

        dev_split = ds.generate_split(prefix="dev", episode_count=args.dev_episodes, seed_offset=0)
        eval_split = ds.generate_split(
            prefix="eval", episode_count=args.eval_episodes, seed_offset=1_000_000
        )
        ds.assert_disjoint_splits(dev_split, eval_split)

        dev_zm, dev_pd, dev_priv = run_systems(dev_split.records)  # noqa: F841 (calibration-only; unused by design)
        eval_zm, eval_pd, eval_priv = run_systems(eval_split.records)

        metrics_report = rp.build_metrics_report(eval_split.records, eval_zm, eval_pd, eval_priv)

        output_dir = args.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        rp.write_transition_level_results(
            output_dir / "transition-level-results.jsonl", eval_split.records, eval_zm, eval_pd, eval_priv
        )

        environment = {
            "git_commit": _git_sha(),
            "python_version": sys.version.split()[0],
            "numpy_version": np.__version__,
            "dev_episode_count": args.dev_episodes,
            "eval_episode_count": args.eval_episodes,
            "dev_transition_count": len(dev_split.records),
            "eval_transition_count": len(eval_split.records),
            "dev_seeds": [0 + i for i in range(args.dev_episodes)],
            "eval_seeds": [1_000_000 + i for i in range(args.eval_episodes)],
            "categories": list(ds.ALL_CATEGORIES),
            "command": " ".join(sys.argv),
        }

        results_payload = {"environment": environment, "metrics": metrics_report}
        rp.write_json(output_dir / "benchmark-results.json", results_payload)

        artifact_rows = []
        if not args.skip_render:
            artifacts_dir = output_dir / "artifacts"
            faulty = [
                (r, z, p) for r, z, p in zip(eval_split.records, eval_zm, eval_pd) if r.is_faulty
            ]
            render_episode_ids = set(eval_split.episode_ids[: args.render_ordinary_episodes])
            ordinary_sample = [
                (r, z, p)
                for r, z, p in zip(eval_split.records, eval_zm, eval_pd)
                if not r.is_faulty and r.episode_id in render_episode_ids
            ]
            priv_by_id = {r.transition_id: p for r, p in zip(eval_split.records, eval_priv)}
            for record, zm_out, pd_out in faulty + ordinary_sample:
                priv_out = priv_by_id[record.transition_id]
                png_path = artifacts_dir / f"{record.transition_id}.png"
                render_transition_panel(record, priv_out, zm_out, output_path=png_path)
                truth = record.observed_changed_components
                zm_f1 = mx.per_transition_component_f1(zm_out.predicted_components, truth)
                pd_f1 = mx.per_transition_component_f1(pd_out.predicted_components, truth)
                missing_gt = mx.missing_ground_truth(record)
                if missing_gt and (missing_gt & set(zm_out.missing_components)):
                    verdict = "better"
                elif zm_f1 > pd_f1 + 1e-9:
                    verdict = "better"
                elif zm_f1 < pd_f1 - 1e-9:
                    verdict = "worse"
                else:
                    verdict = "equal"
                artifact_rows.append(
                    {
                        "transition_id": record.transition_id,
                        "category": record.category,
                        "fault_type": record.fault_type,
                        "is_faulty": record.is_faulty,
                        "verdict": verdict,
                        "zeromodel_status": zm_out.diagnostics["conformance_status"],
                        "false_positive": bool(not record.is_faulty and (zm_out.missing_components or zm_out.unexpected_components)),
                        "false_negative": bool(
                            record.is_faulty
                            and not (zm_out.missing_components or zm_out.unexpected_components)
                        ),
                        "artifact_path": f"artifacts/{record.transition_id}.png",
                    }
                )
            build_html_index(artifact_rows, output_path=output_dir / "visual-index.html")

        discovery_rows = [
            dd.run_episode_discovery(episode_id, eval_split.records, eval_zm)
            for episode_id in eval_split.episode_ids[:5]
        ]
        rp.write_json(output_dir / "discovery-demo.json", {"episodes": discovery_rows})

        duration = time.time() - started
        for warning in caught:
            warning_records.append(str(warning.message))

    environment["duration_seconds"] = round(duration, 3)
    environment["warning_count"] = len(warning_records)
    rp.write_json(output_dir / "benchmark-results.json", {"environment": environment, "metrics": metrics_report})

    summary_md = rp.render_summary_markdown(environment, metrics_report)
    (output_dir / "benchmark-summary.md").write_text(summary_md, encoding="utf-8")

    run_log = {
        "environment": environment,
        "warnings": warning_records,
        "rendered_artifact_count": len(artifact_rows),
    }
    rp.write_json(output_dir / "run-log.json", run_log)

    print(json.dumps(environment, indent=2, sort_keys=True))
    print(f"wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
