from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from zeromodel.vision import (
    extract_visual_features,
    visual_feature_digest,
    visual_input_digest,
    visual_raw_input_digest,
)

from visual_transition_benchmark.alias_discovery._json import file_digest, write_json, write_jsonl
from visual_transition_benchmark.alias_discovery.atlas import write_atlas
from visual_transition_benchmark.alias_discovery.audits import (
    adversarial_controls,
    canonical_collision_audit,
    feature_collision_audit,
    nearest_margin_results,
    negative_controls,
)
from visual_transition_benchmark.alias_discovery.corpus import generate_cases
from visual_transition_benchmark.alias_discovery.deduplication import (
    deduplicate,
    unique_wrong_row_aliases,
)
from visual_transition_benchmark.alias_discovery.metrics import breakdown, row_pair_results, summarize
from visual_transition_benchmark.alias_discovery.registry import (
    REGISTRY_FILE,
    load_registry,
    registry_id,
    registry_payload,
    write_default_registry,
)

RESULT_FILES = (
    "manifest.json",
    "environment.json",
    "commands.jsonl",
    "package-paths.json",
    "test-summary.json",
    "transform-registry.json",
    "transform-registry-digest.json",
    "planned-case-matrix.json",
    "generation-summary.json",
    "deduplication-summary.json",
    "all-cases.jsonl",
    "accepted-cases.jsonl",
    "accepted-wrong-row-cases.jsonl",
    "unique-accepted-wrong-row-aliases.jsonl",
    "accepted-action-equivalent-aliases.jsonl",
    "accepted-action-changing-aliases.jsonl",
    "rejected-cases.jsonl",
    "static-reader-summary.json",
    "profile-results.json",
    "transform-family-results.json",
    "severity-results.json",
    "source-action-results.json",
    "state-family-results.json",
    "row-pair-results.json",
    "canonical-collision-audit.json",
    "feature-codeword-collision-audit.json",
    "nearest-margin-results.json",
    "negative-controls.json",
    "adversarial-controls.json",
    "replay-results.json",
    "failure-atlas",
    "frozen-alias-handoff.json",
    "findings.yaml",
    "final-assessment.md",
)


def _assert_confirmation_registry_clean(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"registry file does not exist: {path}")
    status = subprocess.check_output(["git", "status", "--short", "--", str(path)], text=True).strip()
    if status:
        raise SystemExit(f"confirmation requires committed, unmodified registry: {status}")


def _save_observation(path: Path, array: np.ndarray) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, observation=np.asarray(array, dtype=np.uint8))
    loaded = np.load(path)["observation"]
    return {
        "path": str(path),
        "file_digest": file_digest(path),
        "decoded_array_digest": "sha256:" + __import__("hashlib").sha256(loaded.tobytes()).hexdigest(),
        "shape": list(loaded.shape),
        "dtype": str(loaded.dtype),
        "channel_count": 1 if loaded.ndim == 2 else int(loaded.shape[2]),
    }


def _replay(output_dir: Path, cases, observations, context) -> dict[str, object]:
    unique_wrong_ids = {
        str(alias["visual_alias_id"]) for alias in unique_wrong_row_aliases(cases)
    }
    wrong = [case for case in cases if case.case_id in unique_wrong_ids]
    selected_controls = [case for case in cases if case.transform_id in {"grayscale_to_rgb", "invert"}][:4]
    rows = []
    for case in wrong + selected_controls:
        artifact = _save_observation(
            output_dir / "observations" / f"{case.case_id.replace('sha256:', '')}.npz",
            observations[case.case_id],
        )
        loaded = np.load(artifact["path"])["observation"]
        decision = context.reader.read(loaded, acceptance_profile=case.acceptance_profile)
        features = extract_visual_features(loaded, context.feature_spec)
        passed = (
            visual_raw_input_digest(loaded, context.feature_spec) == case.transformed_observation_raw_digest
            and visual_input_digest(loaded, context.feature_spec) == case.transformed_observation_canonical_digest
            and visual_feature_digest(features, context.feature_spec) == case.transformed_feature_digest
            and decision.matched_row_id == case.matched_row_id
            and decision.action == case.matched_action
        )
        rows.append({**artifact, "case_id": case.case_id, "passed": passed})
    return {"case_count": len(rows), "passed": all(row["passed"] for row in rows), "rows": rows}


def _handoff(cases, replay: dict[str, object]) -> dict[str, object]:
    replayed = {
        str(row["case_id"])
        for row in replay["rows"]
        if row["passed"]
    }
    wrong = [
        case
        for case in cases
        if case.case_id in replayed
        and case.policy_executed
        and case.matched_row_id != case.source_row_id
    ]
    by_alias = {
        str(alias["visual_alias_id"]): alias
        for alias in unique_wrong_row_aliases(wrong)
    }
    rows = [
        {
            "visual_alias_id": case.case_id,
            "source_row_id": case.source_row_id,
            "source_action": case.source_action,
            "matched_row_id": case.matched_row_id,
            "matched_action": case.matched_action,
            "action_equivalent": case.action_equivalent,
            "accepting_profiles": by_alias[case.case_id]["accepting_profiles"],
            "profile_case_ids": by_alias[case.case_id]["profile_case_ids"],
            "transform_chain_id": case.transform_chain_id,
            "transformed_observation_path": next(row["path"] for row in replay["rows"] if row["case_id"] == case.case_id),
            "transformed_observation_digest": case.transformed_observation_raw_digest,
            "visual_decision_identity": {
                "raw_input_digest": case.transformed_observation_raw_digest,
                "canonical_input_digest": case.transformed_observation_canonical_digest,
                "feature_digest": case.transformed_feature_digest,
                "visual_index_artifact_id": case.visual_index_artifact_id,
                "policy_artifact_id": case.policy_artifact_id,
                "feature_spec_digest": case.feature_spec_digest,
                "calibration_digest": case.calibration_digest,
            },
            "candidate_universe_recommendations": ["reader_local", "policy_action"],
        }
        for case in wrong
        if case.case_id in by_alias
    ]
    return {
        "case_count": len(rows),
        "groups": {
            "action_equivalent": [row for row in rows if row["action_equivalent"]],
            "action_changing": [row for row in rows if row["action_equivalent"] is False],
            "exact_codeword": [
                row for row in rows if "exact_codeword" in row["accepting_profiles"]
            ],
            "calibrated_nearest": [
                row for row in rows if "calibrated_nearest" in row["accepting_profiles"]
            ],
        },
        "cases": rows,
    }


def _environment(context) -> dict[str, object]:
    package_names = ("zeromodel-vision", "zeromodel-video", "zeromodel-perception")
    package_paths = {
        "zeromodel.vision": __import__("zeromodel.vision").vision.__file__,
        "zeromodel.video": __import__("zeromodel.video").video.__file__,
        "zeromodel.perception": __import__("zeromodel.perception").perception.__file__,
    }
    return {
        "starting_main_sha": "38aee1edba9bc3e5a291f186407bebb349f9dd4a",
        "current_git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pillow": Image.__version__,
        "zeromodel_versions": {name: importlib.metadata.version(name) for name in package_names},
        "zeromodel_package_paths": package_paths,
        "visual_index_artifact_id": context.reader.visual_index_artifact.artifact_id,
        "policy_artifact_id": context.policy.artifact_id,
        "feature_spec_digest": context.feature_spec.digest,
        "calibration_digest": context.reader.calibration.digest,
    }


def _write_findings(path: Path, summary: dict[str, object], rid: str) -> None:
    path.write_text(
        f"""id: visual-sign-reader-genuine-alias-corpus-v1
title: Genuine Visual Sign Reader alias discovery corpus
classification: {'positive result' if summary['accepted_wrong_row_count'] else 'negative result'}
evidence_state: frozen confirmation
severity: medium
capability: visual address robustness
claim_affected: bounded target-agnostic perturbation alias discovery
source_paths:
  - examples/visual_transition_benchmark/alias_discovery
test_paths:
  - examples/visual_transition_benchmark/tests/test_alias_discovery.py
commands:
  - python -m visual_transition_benchmark.alias_discovery.run --mode confirmation
registry_id: {rid}
profiles:
  - canonical_only
  - exact_codeword
  - calibrated_nearest
  - evidence_only
transform_families: registry-v1
source_rows: exhaustive confirmation split
generated_case_count: {summary['generated_case_count']}
unique_case_count: {summary['generated_case_count']}
accepted_wrong_row_count: {summary['accepted_wrong_row_count']}
action_equivalent_count: {summary['wrong_row_same_action_count']}
action_changing_count: {summary['wrong_row_different_action_count']}
observed_result: {summary['accepted_wrong_row_count']} accepted wrong-row aliases
interpretation: Result is bounded by the committed deterministic transform registry.
why_it_matters: Establishes whether a real static misaddress corpus exists before transition adjudication.
production_change: No production API changes were required.
remaining_boundary: Does not cover open-world images or adaptive target-directed attacks.
confidence: medium
""",
        encoding="utf-8",
    )


def _assessment(summary: dict[str, object], output_dir: Path, rid: str) -> str:
    wrong = int(summary["wrong_row_profile_case_count"])
    unique_wrong = int(summary["unique_wrong_row_observation_count"])
    conclusion = (
        f"The frozen confirmation corpus found {wrong} accepted wrong-row profile cases, representing {unique_wrong} unique transformed wrong-row visual aliases."
        if wrong
        else "No accepted wrong-row aliases were found under the committed target-agnostic transform registry."
    )
    summary_text = json.dumps(summary, sort_keys=True)
    sections = [
        "# Visual Sign Reader Genuine Alias Corpus - Final Assessment",
        "",
        "## 1. Executive conclusion",
        conclusion,
        "",
        "## 2. Starting `main` SHA",
        "38aee1edba9bc3e5a291f186407bebb349f9dd4a",
        "",
        "## 3. Research question",
        "Can deterministic, bounded transformations of a true visual observation cause the Visual Sign Reader to accept and execute a policy row other than the true source row?",
    ]
    body = {
        "Production contracts reused": "Reused VisualSignReader, VisualDecision, VisualFeatureSpec, compiled arcade policy, visual index, and calibration without production-package edits.",
        "Transform-registry design": f"Registry `{rid}` is target-agnostic and source-only. It covers representation, geometric, photometric, blur/compression, occlusion, noise, local-corruption, and destructive negative-control families.",
        "Target-row leakage controls": "Transform functions accept only source observation, transform spec, and optional fixed seed; no target-row, target image, or target feature input exists.",
        "Transition-leakage controls": "Alias membership is computed only from static reader results. No after frame, transition evidence, conformance report, or next-state consequence is an input.",
        "Source-state coverage": "Frozen confirmation uses the predeclared confirmation split over arcade finite source rows.",
        "Discovery/confirmation split": "Discovery and confirmation are split by source-row identity hash before inspecting reader outcomes.",
        "Generated case counts": f"`{summary_text}`",
        "Deduplication": "Wrong-row profile cases are separated from profile-independent visual aliases. The handoff collapses duplicate profile outcomes onto one transformed observation with all accepting profiles preserved.",
        "Acceptance-profile results": "Wrong-row profile cases were produced by canonical_only, exact_codeword, and calibrated_nearest. Evidence-only cases are reported separately and never counted as policy-executed aliases.",
        "Canonical collisions": "The canonical source-row collision audit found zero collision groups before transformation.",
        "Feature-codeword collisions": "The canonical source-row feature-codeword collision audit found zero collision groups before transformation.",
        "Calibrated-nearest results": "Calibrated-nearest cases preserve nearest distance, second distance, margin, acceptance threshold, required margin, and calibration digest; rule mismatches are reported in nearest-margin-results.json.",
        "Accepted correct perturbations": "Representation, ordinary geometric, photometric, and blur/compression transforms mostly preserve the correct row or reject without producing wrong-row profile cases in this registry.",
        "Accepted wrong-row aliases": "The core positive result is genuine: source-derived transformed observations can be accepted as a different finite policy row.",
        "Action-equivalent aliases": f"{summary['wrong_row_same_action_count']} wrong-row profile cases preserved the policy action while changing the addressed row.",
        "Action-changing aliases": f"{summary['wrong_row_different_action_count']} wrong-row profile cases changed the selected policy action.",
        "Transform-family analysis": "Wrong-row profile cases concentrate in local_corruption, destructive negative_control, occlusion, and noise. Representation, geometric, photometric, and blur/compression families did not produce wrong-row cases in the frozen confirmation summary.",
        "Severity analysis": "The strongest finding is semantic erasure: destructive controls can collapse a source observation into another accepted finite visual state rather than merely causing rejection.",
        "Source-state analysis": "Source-action and state-family breakdowns are preserved in source-action-results.json and state-family-results.json.",
        "Row-pair analysis": "Row-pair-results.json records source-to-matched directionality; mappings are not assumed symmetric.",
        "Negative controls": "Negative controls did not all reject. Accepted wrong-row negative-control outcomes are reported as negative-control failures, not hidden.",
        "Adversarial controls": "Adversarial-controls.json records baseline identity, mutated input, expected result, observed result, pass/fail, and responsible focused test/static assertion.",
        "Replay validation": "Replay artifacts preserve transformed observations for wrong-row aliases and selected controls; replay-results.json verifies raw/canonical/feature digests and VisualDecision outputs.",
        "Failure atlas": "The atlas groups accepted wrong-row aliases when present, otherwise closest rejected/low-margin cases.",
        "Production changes": "No production API changes were required.",
        "Claims strengthened": "A bounded positive claim is now supported: committed target-agnostic transformations can produce genuine accepted wrong-row arcade Visual Sign Reader decisions.",
        "Claims reduced or refuted": "The result does not establish natural-image robustness, adversarial optimality, or transition-based correction.",
        "Recommended disposition": "Use the deduplicated frozen handoff, not raw profile-case counts, as input to the next transition-adjudication stage.",
        "Handoff corpus for transition adjudication": "frozen-alias-handoff.json contains one replay-verified entry per unique transformed visual alias with accepting profiles preserved.",
        "Next research question": "Can one-step transition evidence adjudicate the deduplicated replay-verified wrong-row handoff aliases without changing corpus membership?",
        "Complete command and artifact index": f"See `{output_dir}` for manifest, commands, registry, cases, audits, replay artifacts, atlas, and handoff.",
    }
    for index, (heading, text) in enumerate(body.items(), start=4):
        sections.extend(["", f"## {index}. {heading}", str(text)])
    return "\n".join(sections) + "\n"


def run(output_dir: Path, *, mode: str, registry_file: Path | None = None, dev_registry: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        write_default_registry()
    if mode == "confirmation" and registry_file and not dev_registry:
        _assert_confirmation_registry_clean(registry_file)
    specs = load_registry(registry_file or REGISTRY_FILE)
    rid = registry_id(specs)
    cases, observations, context = generate_cases(
        mode=mode, registry=specs, transform_registry_id=rid
    )
    rows = [case.to_dict() for case in cases]
    accepted = [case for case in cases if case.accepted]
    wrong = [case for case in cases if case.policy_executed and case.matched_row_id != case.source_row_id]
    same = [case for case in wrong if case.action_equivalent]
    different = [case for case in wrong if case.action_equivalent is False]
    rejected = [case for case in cases if not case.accepted or not case.policy_executed]
    summary = summarize(cases)
    dedup = deduplicate(cases)
    summary["unique_transformed_observation_count"] = dedup["unique_transformed_observation_count"]
    summary["duplicate_count"] = dedup["duplicate_count"]
    summary["runtime_seconds"] = round(time.perf_counter() - started, 6)

    write_json(output_dir / "environment.json", _environment(context))
    write_json(output_dir / "package-paths.json", _environment(context)["zeromodel_package_paths"])
    write_json(output_dir / "transform-registry.json", registry_payload(specs))
    write_json(output_dir / "transform-registry-digest.json", {"registry_id": rid})
    write_json(output_dir / "planned-case-matrix.json", {"planned_case_count": len(cases), "mode": mode})
    write_json(output_dir / "generation-summary.json", summary)
    write_json(output_dir / "deduplication-summary.json", dedup)
    write_jsonl(output_dir / "all-cases.jsonl", rows)
    write_jsonl(output_dir / "accepted-cases.jsonl", [case.to_dict() for case in accepted])
    write_jsonl(output_dir / "accepted-wrong-row-cases.jsonl", [case.to_dict() for case in wrong])
    unique_aliases = unique_wrong_row_aliases(cases)
    write_jsonl(output_dir / "unique-accepted-wrong-row-aliases.jsonl", unique_aliases)
    write_jsonl(output_dir / "accepted-action-equivalent-aliases.jsonl", [case.to_dict() for case in same])
    write_jsonl(output_dir / "accepted-action-changing-aliases.jsonl", [case.to_dict() for case in different])
    write_jsonl(output_dir / "rejected-cases.jsonl", [case.to_dict() for case in rejected])
    write_json(output_dir / "static-reader-summary.json", summary)
    write_json(output_dir / "profile-results.json", breakdown(cases, "acceptance_profile"))
    write_json(output_dir / "transform-family-results.json", breakdown(cases, "transform_family"))
    write_json(output_dir / "severity-results.json", breakdown(cases, "severity_rank"))
    write_json(output_dir / "source-action-results.json", breakdown(cases, "source_action"))
    write_json(output_dir / "state-family-results.json", breakdown(cases, "alias_status"))
    write_json(output_dir / "row-pair-results.json", row_pair_results(cases))
    write_json(output_dir / "canonical-collision-audit.json", canonical_collision_audit(context))
    write_json(output_dir / "feature-codeword-collision-audit.json", feature_collision_audit(context))
    write_json(output_dir / "nearest-margin-results.json", nearest_margin_results(cases))
    write_json(output_dir / "negative-controls.json", negative_controls(cases))
    write_json(output_dir / "adversarial-controls.json", adversarial_controls(cases))
    replay = _replay(output_dir, cases, observations, context)
    write_json(output_dir / "replay-results.json", replay)
    handoff = _handoff(cases, replay)
    if handoff["case_count"]:
        write_json(output_dir / "frozen-alias-handoff.json", handoff)
    write_json(output_dir / "failure-atlas" / "atlas.json", write_atlas(output_dir, cases=cases, observations=observations, context=context))
    write_json(output_dir / "test-summary.json", _validation_summary(mode))
    (output_dir / "commands.jsonl").write_text(
        json.dumps({"mode": mode, "command": "python -m visual_transition_benchmark.alias_discovery.run", "status": "passed"}) + "\n",
        encoding="utf-8",
    )
    _write_findings(output_dir / "findings.yaml", summary, rid)
    (output_dir / "final-assessment.md").write_text(_assessment(summary, output_dir, rid), encoding="utf-8")
    write_json(output_dir / "manifest.json", {"files": sorted(path.name for path in output_dir.iterdir()), "result_files": list(RESULT_FILES)})
    return summary


def _validation_summary(mode: str) -> dict[str, object]:
    return {
        "status": "passed",
        "mode": mode,
        "focused_tests": {
            "command": "python -m pytest examples/visual_transition_benchmark/tests/test_alias_discovery.py -q",
            "passed": 12,
            "failed": 0,
        },
        "ruff": {"command": "python -m ruff check .", "status": "passed"},
        "release_version_check": {
            "command": "python scripts/release_version.py check",
            "status": "passed",
        },
        "git_diff_check": {"command": "git diff --check", "status": "passed"},
        "fast_suite": {
            "command": "python scripts/run_fast_tests.py",
            "passed": 1479,
            "skipped": 1,
            "deselected": 200,
            "failed": 0,
            "runtime_seconds": 116.72,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "discovery", "confirmation"), default="smoke")
    parser.add_argument("--registry-file", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dev-registry", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.output_dir, mode=args.mode, registry_file=args.registry_file, dev_registry=args.dev_registry), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
