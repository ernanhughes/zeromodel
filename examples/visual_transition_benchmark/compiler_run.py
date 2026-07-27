"""CLI entry point for stage 4 (evidence contract compiler).

For each declared benchmark case (arcade + warehouse), compiles a
representation from development samples only, then measures the selected
candidate's held-out accuracy on a disjoint evaluation split. Also reports two
non-searched reference strategies over the same bounded candidate set --
"fixed_coarse" (the region's own declared cell resolution, naive decoder) and
"always_pixel" (finest 1x1 resolution, naive decoder, no auto-narrowing) --
so the value of the deterministic search-and-select step is visible, not
assumed.

Usage:
    python -m visual_transition_benchmark.compiler_run --dev-samples 12 --eval-samples 30 \
        --output-dir artifacts/evidence_contract_compiler
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from visual_transition_benchmark import report as rp
from visual_transition_benchmark.run import _git_sha
from visual_transition_benchmark.compiler.candidates import RepresentationCandidate, generate_candidates
from visual_transition_benchmark.compiler.compile import compile_requirement
from visual_transition_benchmark.compiler.evaluate import CandidateEvaluationResult, evaluate_candidate
from visual_transition_benchmark.compiler_adapters import arcade as arcade_adapter
from visual_transition_benchmark.compiler_adapters import warehouse as warehouse_adapter

# Naive (non-auto-narrowing) decoders only, in preference order -- used to
# pick the "fixed_coarse"/"always_pixel" reference candidates so those two
# strategies never accidentally benefit from the search's own repair step.
_NAIVE_DECODER_PREFERENCE = (
    "presence_threshold",
    "nearest_permitted_value",
    "categorical_template",
    "argmax_field",
    "signed_delta_over_position",
    "exact_delta_over_position",
    "local_marker_pattern",
    "relation_over_decoded",
    "exact_lookup",
)


def _pick_reference_candidate(
    candidates: Sequence[RepresentationCandidate], *, field_height: int, field_width: int
) -> Optional[RepresentationCandidate]:
    matches = [c for c in candidates if c.field_height == field_height and c.field_width == field_width]
    if not matches:
        return None
    for decoder in _NAIVE_DECODER_PREFERENCE:
        for c in matches:
            if c.decoder_kind == decoder:
                return c
    return sorted(matches, key=lambda c: c.candidate_id)[0]


def _eval_summary(result: Optional[CandidateEvaluationResult]) -> Optional[dict]:
    if result is None:
        return None
    return {
        "decoding_accuracy": result.decoding_accuracy,
        "collision_rate": result.collision_rate,
        "stability_false_change_rate": result.stability_false_change_rate,
        "passed": result.passed,
        "rejection_reasons": list(result.rejection_reasons),
    }


def _candidate_summary(candidate: Optional[RepresentationCandidate]) -> Optional[dict]:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate.candidate_id,
        "field_height": candidate.field_height,
        "field_width": candidate.field_width,
        "aggregation": candidate.aggregation,
        "decoder_kind": candidate.decoder_kind,
        "complexity_cost": candidate.complexity_cost,
    }


def run_case(domain: str, case, *, dev_count: int, eval_count: int, dev_seed: int, eval_seed: int) -> dict:
    dev_samples = case.build_samples(dev_count, dev_seed)
    eval_samples = case.build_samples(eval_count, eval_seed)
    candidates = generate_candidates(case.requirement, case.region)
    sub_offsets = getattr(case, "sub_patch_offsets", ())

    compiled = compile_requirement(
        case.requirement,
        case.region,
        candidates,
        dev_samples,
        canonical_levels=case.canonical_levels,
        sub_patch_offsets=sub_offsets,
        min_decoding_accuracy=case.min_decoding_accuracy,
    )

    def _held_out(candidate: Optional[RepresentationCandidate]) -> Optional[CandidateEvaluationResult]:
        if candidate is None or not eval_samples:
            return None
        return evaluate_candidate(
            candidate,
            case.requirement,
            case.region,
            eval_samples,
            canonical_levels=case.canonical_levels,
            sub_patch_offsets=sub_offsets,
            min_decoding_accuracy=case.min_decoding_accuracy,
        )

    fixed_coarse = _pick_reference_candidate(
        candidates, field_height=case.region.cell_height, field_width=case.region.cell_width
    )
    always_pixel = _pick_reference_candidate(candidates, field_height=1, field_width=1)

    selected = compiled.selected_candidate
    return {
        "domain": domain,
        "case": case.name,
        "requirement_id": case.requirement.requirement_id,
        "evidence_kind": case.requirement.evidence_kind,
        "n_dev_samples": len(dev_samples),
        "n_eval_samples": len(eval_samples),
        "n_candidates_considered": len(candidates),
        "status": compiled.status,
        "selection_rationale": list(compiled.selection_rationale),
        "known_limitations": list(compiled.known_limitations),
        "compiled_strategy": {
            "candidate": _candidate_summary(selected),
            "dev_accuracy": None if compiled.selected_evaluation is None else compiled.selected_evaluation.decoding_accuracy,
            "held_out_eval": _eval_summary(_held_out(selected)),
        },
        "fixed_coarse_strategy": {
            "candidate": _candidate_summary(fixed_coarse),
            "held_out_eval": _eval_summary(_held_out(fixed_coarse)),
        },
        "always_pixel_strategy": {
            "candidate": _candidate_summary(always_pixel),
            "held_out_eval": _eval_summary(_held_out(always_pixel)),
        },
        "all_dev_evaluations": [
            {
                "candidate_id": e.candidate_id,
                "decoding_accuracy": e.decoding_accuracy,
                "passed": e.passed,
                "is_degenerate": e.is_degenerate,
                "rejection_reasons": list(e.rejection_reasons),
            }
            for e in sorted(compiled.all_evaluations, key=lambda e: e.candidate_id)
        ],
    }


def _render_summary_markdown(environment: dict, results: list) -> str:
    lines = []
    lines.append("# Evidence Contract Compiler -- Summary")
    lines.append("")
    lines.append("## Exact environment")
    lines.append("")
    for key, value in environment.items():
        lines.append(f"- **{key}**: {value}")
    lines.append("")
    lines.append("## Per-case outcomes")
    lines.append("")
    lines.append("| Domain | Case | Status | Selected (dev) | Dev acc. | Held-out acc. | Fixed-coarse held-out | Always-pixel held-out |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|")
    for r in results:
        cs = r["compiled_strategy"]
        candidate = cs["candidate"]
        selected_desc = "-" if candidate is None else f"{candidate['field_height']}x{candidate['field_width']} {candidate['decoder_kind']}"
        dev_acc = "-" if cs["dev_accuracy"] is None else f"{cs['dev_accuracy']:.3f}"
        held_out = cs["held_out_eval"]
        held_out_acc = "-" if held_out is None else f"{held_out['decoding_accuracy']:.3f}"
        fc = r["fixed_coarse_strategy"]["held_out_eval"]
        fc_acc = "-" if fc is None else f"{fc['decoding_accuracy']:.3f}"
        ap = r["always_pixel_strategy"]["held_out_eval"]
        ap_acc = "-" if ap is None else f"{ap['decoding_accuracy']:.3f}"
        lines.append(
            f"| {r['domain']} | {r['case']} | {r['status']} | {selected_desc} | {dev_acc} | {held_out_acc} | {fc_acc} | {ap_acc} |"
        )
    lines.append("")
    lines.append("## Status counts")
    lines.append("")
    counts: dict = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    for status, count in sorted(counts.items()):
        lines.append(f"- **{status}**: {count}")
    lines.append("")
    lines.append("## Known limitations surfaced per case")
    lines.append("")
    for r in results:
        if r["known_limitations"]:
            lines.append(f"- **{r['domain']}/{r['case']}**: {'; '.join(r['known_limitations'])}")
    lines.append("")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-samples", type=int, default=12, help="samples per category for development splits")
    parser.add_argument("--eval-samples", type=int, default=30, help="samples per category for evaluation splits")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/evidence_contract_compiler"))
    args = parser.parse_args(argv)

    started = time.time()
    results = []
    for domain, adapter in (("arcade", arcade_adapter), ("warehouse", warehouse_adapter)):
        for case in adapter.build_cases():
            results.append(
                run_case(
                    domain,
                    case,
                    dev_count=args.dev_samples,
                    eval_count=args.eval_samples,
                    dev_seed=0,
                    eval_seed=1_000_000,
                )
            )
    duration = time.time() - started

    environment = {
        "git_commit": _git_sha(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "dev_samples_per_category": args.dev_samples,
        "eval_samples_per_category": args.eval_samples,
        "case_count": len(results),
        "duration_seconds": round(duration, 3),
        "command": " ".join(sys.argv),
    }

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rp.write_json(output_dir / "compiler-results.json", {"environment": environment, "results": results})
    summary_md = _render_summary_markdown(environment, results)
    (output_dir / "compiler-summary.md").write_text(summary_md, encoding="utf-8")

    print(json.dumps(environment, indent=2, sort_keys=True))
    print(f"wrote results to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
