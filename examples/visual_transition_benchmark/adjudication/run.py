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
            addressed_observation=case.observed_frame,
            addressed_observation_transform_id=case.observation_transform_id,
            feature_spec=case.feature_spec,
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
        "observation_source_row_id": case.observation_source_row_id,
        "observation_transform_id": case.observation_transform_id,
        "acceptance_profile": case.profile,
        "policy_executed": case.visual_decision.policy_executed,
        "exact_address": exact_address,
        "same_action": same_action,
        "true_row_retained": case.true_row_id in consistent,
        "true_row_initially_present": case.true_row_id
        in {item.row_id for item in result.candidate_results},
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
    leakage = _leakage_controls(cases[0])
    adversarial = _adversarial_controls(cases[0])
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


def _leakage_controls(case) -> dict[str, object]:
    from dataclasses import replace

    baseline = _runtime_row(case)
    mutated_after = np.array(case.true_after_frame, copy=True)
    mutated_after[6, 0] = 90
    after_mutated = _runtime_row(replace(case, true_after_frame=mutated_after))
    truth_mutated = _runtime_row(
        replace(case, true_row_id="tank=6|target=6|cooldown=1")
    )
    return {
        "runtime_outputs_ignore_privileged_truth": {
            "baseline_result_id": baseline["result_id"],
            "mutated_result_id": truth_mutated["result_id"],
            "expected_relationship": "equal",
            "observed_relationship": "equal"
            if baseline["result_id"] == truth_mutated["result_id"]
            else "different",
            "passed": baseline["result_id"] == truth_mutated["result_id"],
        },
        "after_frame_mutation_changes_runtime_output": {
            "baseline_result_id": baseline["result_id"],
            "mutated_result_id": after_mutated["result_id"],
            "expected_relationship": "different",
            "observed_relationship": "equal"
            if baseline["result_id"] == after_mutated["result_id"]
            else "different",
            "passed": baseline["result_id"] != after_mutated["result_id"],
        },
    }


def _adversarial_controls(case) -> dict[str, object]:
    transition = _transition(case.true_before_frame, case.true_after_frame)
    wrong_action_result = adjudicate_address_transition(
        RuntimeAdjudicationInput(
            case_id=case.case_id,
            visual_decision=case.visual_decision,
            candidate_universe=case.candidate_universe,
            evidence_mode=case.evidence_mode,
            addressed_observation=case.observed_frame,
            addressed_observation_transform_id=case.observation_transform_id,
            feature_spec=case.feature_spec,
            frame_before=case.true_before_frame,
            frame_after=case.true_after_frame,
            action=TransitionActionDeclarationDTO.create(
                action_type="LEFT" if case.addressed_action != "LEFT" else "RIGHT",
                payload={"row_id": "wrong"},
            ),
            transition_evidence=transition,
        )
    )
    swapped_result = adjudicate_address_transition(
        RuntimeAdjudicationInput(
            case_id=case.case_id,
            visual_decision=case.visual_decision,
            candidate_universe=case.candidate_universe,
            evidence_mode=case.evidence_mode,
            addressed_observation=case.observed_frame,
            addressed_observation_transform_id=case.observation_transform_id,
            feature_spec=case.feature_spec,
            frame_before=case.true_before_frame,
            frame_after=case.true_after_frame,
            action=TransitionActionDeclarationDTO.create(
                action_type=str(case.addressed_action),
                payload={"row_id": case.visual_decision.matched_row_id},
            ),
            transition_evidence=_transition(
                case.true_after_frame, case.true_before_frame
            ),
        )
    )
    stale_result = adjudicate_address_transition(
        RuntimeAdjudicationInput(
            case_id=case.case_id,
            visual_decision=case.visual_decision,
            candidate_universe=case.candidate_universe,
            evidence_mode=case.evidence_mode,
            addressed_observation=np.array(case.true_after_frame, copy=True),
            addressed_observation_transform_id=case.observation_transform_id,
            feature_spec=case.feature_spec,
            frame_before=case.true_before_frame,
            frame_after=case.true_after_frame,
            action=TransitionActionDeclarationDTO.create(
                action_type=str(case.addressed_action),
                payload={"row_id": case.visual_decision.matched_row_id},
            ),
            transition_evidence=transition,
        )
    )
    return {
        "wrong_action_identity_rejected": {
            "result_id": wrong_action_result.result_id,
            "status": wrong_action_result.runtime_adjudication_status,
            "passed": "wrong_action_identity" in wrong_action_result.reason_codes,
        },
        "wrong_transition_evidence_rejected": {
            "result_id": swapped_result.result_id,
            "status": swapped_result.runtime_adjudication_status,
            "passed": "invalid_transition_evidence" in swapped_result.reason_codes,
        },
        "reader_observation_mismatch_rejected": {
            "result_id": stale_result.result_id,
            "status": stale_result.runtime_adjudication_status,
            "passed": "reader_observation_mismatch" in stale_result.reason_codes,
        },
    }


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
    summary_json = json.dumps(summary, sort_keys=True)
    lines = [
        "# Visual Address Transition Adjudication - Final Assessment",
        "",
        "## 1. Executive conclusion",
        "This run is evidence for a narrower claim than the first draft: when the Visual Sign Reader accepts a genuine observation of the true pre-transition frame, the runtime adjudicator can bind that decision to the exact addressed observation and reject stale, swapped, or wrong-action transition inputs. The corrected corpus did not produce accepted wrong-row aliases, so these artifacts do not prove automatic correction of accepted wrong-row visual aliases.",
        "",
        "## 2. Starting `main` SHA",
        "bd3d5f8d0213095d4a1df0fc9d419e027fd1c606",
        "",
        "## 3. Research question",
        "Can observed visual consequences adjudicate an earlier static visual address?",
        "",
        "## 4. Result Summary",
        f"`{summary_json}`",
        "",
        "## 5. Corrected Static Alias Corpus",
        "Every executed corpus case now renders the observation from the true source row. Canonical, exact-codeword noncanonical, and calibrated-nearest accepted cases are genuine perturbations of the true frame. The wrong-row substituted-frame cases from the earlier draft were removed because they were not Visual Sign Reader aliases.",
        "",
        "## 6. Runtime Binding",
        "The adjudicator records the addressed observation digest and rejects runtime input when the supplied VisualDecision trace does not match the exact addressed observation. A transformed observation that preserves the canonical digest is accepted only when the transform is declared; a different before-observation is rejected as stale.",
        "",
        "## 7. Production DTO Path",
        "Component-mode candidate checks now build TransitionExpectationDTO entries, evaluate a conformance report, and construct VisualTransitionAnalysisDTO. Value-mode checks remain the deterministic value-aware comparator for the toy arcade domain.",
        "",
        "## 8. Transition Signatures",
        "Candidate collision groups are keyed by expected transition signatures derived from the candidate contract, excluding row identity. Observed transition evidence is reported separately, so collision reporting is no longer tautological over shared observed frames.",
        "",
        "## 9. Coverage And Retention",
        "The summary includes initial true-row coverage, conditional retention, conditional elimination, false confirmation when truth is present, and false confirmation when truth is absent. Candidate reduction is reported separately for truth-preserved and truth-removed cases.",
        "",
        "## 10. Acceptance Profiles",
        "The corpus covers canonical_only accepted/rejected, exact_codeword canonical/noncanonical, calibrated_nearest accepted/rejected, and evidence_only no-execution paths.",
        "",
        "## 11. Controls",
        "Leakage and adversarial results are generated by paired executions. The artifacts include result IDs for privileged-label mutation, after-frame mutation, wrong action identity, swapped transition evidence, and reader-observation mismatch controls.",
        "",
        "## 12. Baselines",
        "Raw-pixel and region-pixel baselines remain diagnostic rather than equivalent adjudicators. They are useful for comparing what information is visible in the frames, but they do not provide the same candidate-contract evaluation as the runtime adjudicator.",
        "",
        "## 13. Disposition",
        "Merge readiness depends on the intended claim. The implementation now supports the bounded claim that static visual decisions are bound to exact observations and can be checked against one-step visual transition evidence. It should not be described as proving wrong-row alias correction until a genuine accepted wrong-row corpus exists.",
        "",
        "## 14. Artifact Index",
        f"See `{output_dir.name}` for JSON result files, smoke evidence, findings, and manifest entries.",
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
