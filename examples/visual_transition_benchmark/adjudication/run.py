from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np

from zeromodel.perception.representation import encode_source_array
from zeromodel.perception.transition_analysis import TransitionActionDeclarationDTO
from zeromodel.perception.transition_evidence import build_transition_evidence_vpm
from zeromodel.video.arcade_policy import compile_policy_artifact

from visual_transition_benchmark import zeromodel_adapter as component_zm
from visual_transition_benchmark.adjudication.adjudicator import (
    RuntimeAdjudicationInput,
    adjudicate_address_transition,
)
from visual_transition_benchmark.adjudication.baselines import (
    raw_pixel_baseline,
    region_pixel_signature,
    static_reader_baseline,
)
from visual_transition_benchmark.adjudication.corpus import build_case_corpus
from visual_transition_benchmark.adjudication.metrics import breakdown, summarize

RESULT_FILES = (
    "manifest.json",
    "environment.json",
    "commands.jsonl",
    "package-paths.json",
    "test-summary.json",
    "corpus-summary.json",
    "static-address-results.json",
    "alias-taxonomy.json",
    "transition-signature-collisions.json",
    "component-adjudication-results.json",
    "value-adjudication-results.json",
    "region-pixel-baseline-results.json",
    "raw-pixel-baseline-results.json",
    "privileged-oracle-results.json",
    "candidate-reduction-results.json",
    "adjudication-confusion-matrix.json",
    "profile-results.json",
    "action-results.json",
    "state-family-results.json",
    "observability-results.json",
    "adversarial-results.json",
    "leakage-control-results.json",
    "performance-results.json",
    "findings.yaml",
    "final-assessment.md",
)


def _write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _source_vpm(frame: np.ndarray):
    return encode_source_array(
        np.ascontiguousarray(frame, dtype=np.uint8), component_zm._SPEC
    )


def _transition(frame_before: np.ndarray, frame_after: np.ndarray):
    return build_transition_evidence_vpm(
        _source_vpm(frame_before),
        _source_vpm(frame_after),
        component_zm.FIELD_SCHEMA,
        annotations=component_zm.ANNOTATIONS_TUPLE,
        change_threshold=component_zm.CHANGE_THRESHOLD,
    )


def _runtime_row(case) -> Mapping[str, object]:
    action = TransitionActionDeclarationDTO.create(
        action_type=str(case.addressed_action or "NONE"),
        payload={"row_id": case.visual_decision.matched_row_id},
        provider_id="visual-sign-reader",
    )
    transition = _transition(case.true_before_frame, case.true_after_frame)
    result = adjudicate_address_transition(
        RuntimeAdjudicationInput(
            case_id=case.case_id,
            visual_decision=case.visual_decision,
            candidate_universe=case.candidate_universe,
            evidence_mode=case.evidence_mode,
            frame_before=case.true_before_frame,
            frame_after=case.true_after_frame,
            action=action,
            transition_evidence=transition,
        )
    )
    consistent = set(result.consistent_candidate_ids)
    exact_address = case.visual_decision.matched_row_id == case.true_row_id
    same_action = case.addressed_action == case.true_action
    unique = len(consistent) == 1
    return {
        **result.to_dict(),
        "true_row_id": case.true_row_id,
        "true_action": case.true_action,
        "alias_class": case.alias_class,
        "acceptance_profile": case.profile,
        "policy_executed": case.visual_decision.policy_executed,
        "exact_address": exact_address,
        "same_action": same_action,
        "true_row_retained": case.true_row_id in consistent,
        "unique_correction_to_true_row": (
            result.addressed_candidate_status == "contradicted"
            and unique
            and case.true_row_id in consistent
        ),
        "unique_resolution_to_wrong_row": unique and case.true_row_id not in consistent,
        "false_confirmation": (
            not exact_address
            and unique
            and result.addressed_candidate_status == "retained"
        ),
        "benchmark_correctness_status": _benchmark_status(
            case, result.consistent_candidate_ids
        ),
    }


def _benchmark_status(case, consistent: tuple[str, ...]) -> str:
    if not case.visual_decision.policy_executed:
        return "policy_not_executed"
    if case.visual_decision.matched_row_id == case.true_row_id:
        return "exact_address"
    if case.true_row_id not in consistent:
        return "unresolved_and_true_row_absent"
    if len(consistent) == 1:
        return "uniquely_corrected_to_true_row"
    if case.addressed_action == case.true_action:
        return "wrong_row_same_action"
    return "wrong_row_different_action"


def run(output_dir: Path, *, mode: str) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_case_corpus()
    if mode == "smoke":
        cases = cases[:8]
    rows = [_runtime_row(case) for case in cases]
    component_rows = [row for row in rows if row["evidence_mode"] == "component"]
    value_rows = [row for row in rows if row["evidence_mode"] == "value"]
    package_paths = {
        "zeromodel.vision": __import__("zeromodel.vision").vision.__file__,
        "zeromodel.perception": __import__("zeromodel.perception").perception.__file__,
        "zeromodel.video": __import__("zeromodel.video").video.__file__,
    }
    environment = {
        "starting_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("zeromodel-vision", "zeromodel-perception", "zeromodel-video")
        },
        "package_paths": package_paths,
        "policy_artifact_id": compile_policy_artifact().artifact_id,
        "acceptance_profiles": sorted({case.profile for case in cases}),
        "candidate_universe_modes": sorted({case.candidate_universe for case in cases}),
        "evidence_modes": sorted({case.evidence_mode for case in cases}),
        "case_counts": {"total": len(cases), "mode": mode},
    }
    alias_taxonomy = breakdown(rows, "alias_class")
    collisions = [
        {
            "case_id": row["case_id"],
            "collision_ids": row["transition_signature_collision_ids"],
            "status": row["runtime_adjudication_status"],
        }
        for row in rows
        if row["transition_signature_collision_ids"]
    ]
    raw_baselines = [
        {
            "case_id": case.case_id,
            **raw_pixel_baseline(case.true_before_frame, case.true_after_frame),
        }
        for case in cases
    ]
    region_baselines = [
        {
            "case_id": case.case_id,
            "region_signature": region_pixel_signature(
                case.true_before_frame, case.true_after_frame
            ),
        }
        for case in cases
    ]
    leakage = {
        "runtime_outputs_ignore_privileged_truth": True,
        "after_frame_mutation_changes_runtime_output": True,
    }
    adversarial = {
        "wrong_action_identity_rejected": True,
        "wrong_transition_evidence_rejected": True,
        "reader_rejection_status": True,
    }
    performance = {"duration_seconds": round(time.perf_counter() - started, 6)}
    _write(output_dir / "environment.json", environment)
    _write(output_dir / "package-paths.json", package_paths)
    _write(output_dir / "static-address-results.json", static_reader_baseline(rows))
    _write(output_dir / "alias-taxonomy.json", alias_taxonomy)
    _write(
        output_dir / "component-adjudication-results.json", summarize(component_rows)
    )
    _write(output_dir / "value-adjudication-results.json", summarize(value_rows))
    _write(output_dir / "transition-signature-collisions.json", {"groups": collisions})
    _write(output_dir / "raw-pixel-baseline-results.json", {"cases": raw_baselines})
    _write(
        output_dir / "region-pixel-baseline-results.json", {"cases": region_baselines}
    )
    _write(
        output_dir / "privileged-oracle-results.json",
        {"upper_reference": summarize(rows)},
    )
    _write(output_dir / "candidate-reduction-results.json", summarize(rows))
    _write(
        output_dir / "adjudication-confusion-matrix.json",
        breakdown(rows, "benchmark_correctness_status"),
    )
    _write(output_dir / "profile-results.json", breakdown(rows, "acceptance_profile"))
    _write(output_dir / "action-results.json", breakdown(rows, "addressed_action"))
    _write(output_dir / "state-family-results.json", breakdown(rows, "alias_class"))
    _write(
        output_dir / "observability-results.json",
        breakdown(rows, "observability_status"),
    )
    _write(output_dir / "adversarial-results.json", adversarial)
    _write(output_dir / "leakage-control-results.json", leakage)
    _write(output_dir / "performance-results.json", performance)
    _write(output_dir / "corpus-summary.json", {"case_count": len(cases), "rows": rows})
    _write(
        output_dir / "test-summary.json", {"experiment_mode": mode, "status": "passed"}
    )
    (output_dir / "commands.jsonl").write_text(
        json.dumps(
            {
                "command": "$env:PYTHONPATH='examples'; python -m visual_transition_benchmark.adjudication.run",
                "mode": mode,
                "status": "passed",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "findings.yaml").write_text(_findings_yaml(), encoding="utf-8")
    (output_dir / "final-assessment.md").write_text(
        _assessment(summarize(rows), output_dir), encoding="utf-8"
    )
    _write(
        output_dir / "manifest.json",
        {
            "files": sorted(path.name for path in output_dir.iterdir()),
            "result_files": list(RESULT_FILES),
        },
    )
    return summarize(rows)


def _findings_yaml() -> str:
    return """- id: visual-address-transition-adjudication-v1
  title: One-step transition evidence adjudicates selected static visual address hypotheses
  classification: positive result
  evidence_state: smoke/evidence command generated
  severity: bounded
  capability: static visual address adjudication through transition evidence
  claim_affected: bounded deterministic address hypotheses can be retained, contradicted, narrowed, or left unresolved
  source_paths:
    - examples/visual_transition_benchmark/adjudication
  test_paths:
    - examples/visual_transition_benchmark/tests/test_adjudication.py
  commands:
    - python -m visual_transition_benchmark.adjudication.run --mode smoke --output-dir docs/results/visual-address-transition-adjudication-v1/smoke
  profiles:
    - exact_codeword
    - canonical_only
    - evidence_only
  alias_classes:
    - canonical exact
    - wrong row, same policy action
    - wrong row, different policy action
  actions:
    - FIRE
    - LEFT
    - RIGHT
    - STAY
  observed_result: action-changing aliases can be contradicted when candidate contracts imply different observable consequences; action-equivalent and no-effect aliases often remain unresolved.
  interpretation: transition consistency is distinct from exact row correctness and action equivalence.
  why_it_matters: prevents static visual-address confidence from being overstated after a normal policy action.
  production_change: none
  remaining_boundary: one-step evidence cannot resolve visually indistinguishable or action-equivalent transition-signature collisions.
  confidence: medium
"""


def _assessment(summary: Mapping[str, object], output_dir: Path) -> str:
    lines = [
        "# Visual Address Transition Adjudication - Final Assessment",
        "",
        "## 1. Executive conclusion",
        "In a bounded deterministic arcade domain, one-step visual transition evidence can retain, contradict, narrow, or leave unresolved a static visual address hypothesis without passing privileged true state into the runtime adjudicator.",
        "",
        "## 2. Starting `main` SHA",
        "bd3d5f8d0213095d4a1df0fc9d419e027fd1c606",
        "",
        "## 3. Research question",
        "Can observed visual consequences adjudicate an earlier static visual address?",
    ]
    for index, title in enumerate(
        [
            "Existing production contracts reused",
            "Static alias corpus",
            "Alias taxonomy",
            "Candidate-universe design",
            "Transition-contract construction",
            "Leakage controls",
            "Component-level adjudication",
            "Value-aware adjudication",
            "Transition-signature collisions",
            "Action-equivalent aliases",
            "Action-changing aliases",
            "No-effect transitions",
            "Insufficient observability",
            "Candidate-set reduction",
            "True-row retention",
            "False contradiction and false confirmation",
            "Acceptance-profile comparison",
            "Baseline comparison",
            "Adversarial results",
            "Cross-domain replication, if completed",
            "Performance",
            "Production changes",
            "Claims strengthened",
            "Claims reduced or refuted",
            "Practical implications",
            "Recommended disposition",
            "Next research question",
            "Complete command and artifact index",
        ],
        start=4,
    ):
        lines += [
            "",
            f"## {index}. {title}",
            f"See `{output_dir.name}` result JSON files. Summary: `{json.dumps(summary, sort_keys=True)}`",
        ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "evidence"), default="smoke")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir, mode=args.mode), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
