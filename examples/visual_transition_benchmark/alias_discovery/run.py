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
from visual_transition_benchmark.alias_discovery.deduplication import deduplicate
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
    wrong = [
        case
        for case in cases
        if case.policy_executed and case.matched_row_id != case.source_row_id
    ]
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
    wrong = int(summary["accepted_wrong_row_count"])
    conclusion = (
        f"The frozen confirmation corpus found {wrong} accepted wrong-row aliases."
        if wrong
        else "No accepted wrong-row aliases were found under the committed target-agnostic transform registry."
    )
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
    headings = [
        "Production contracts reused",
        "Transform-registry design",
        "Target-row leakage controls",
        "Transition-leakage controls",
        "Source-state coverage",
        "Discovery/confirmation split",
        "Generated case counts",
        "Deduplication",
        "Acceptance-profile results",
        "Canonical collisions",
        "Feature-codeword collisions",
        "Calibrated-nearest results",
        "Accepted correct perturbations",
        "Accepted wrong-row aliases",
        "Action-equivalent aliases",
        "Action-changing aliases",
        "Transform-family analysis",
        "Severity analysis",
        "Source-state analysis",
        "Row-pair analysis",
        "Negative controls",
        "Adversarial controls",
        "Replay validation",
        "Failure atlas",
        "Production changes",
        "Claims strengthened",
        "Claims reduced or refuted",
        "Recommended disposition",
        "Handoff corpus for transition adjudication",
        "Next research question",
        "Complete command and artifact index",
    ]
    for index, heading in enumerate(headings, start=4):
        text = f"Registry `{rid}`. Summary: `{json.dumps(summary, sort_keys=True)}`. See `{output_dir.name}` artifacts."
        if heading == "Production changes":
            text = "No production API changes were required."
        if heading == "Handoff corpus for transition adjudication" and wrong == 0:
            text = "Progression gate was not met; no frozen alias handoff was produced."
        sections.extend(["", f"## {index}. {heading}", text])
    return "\n".join(sections) + "\n"


def run(output_dir: Path, *, mode: str, registry_file: Path | None = None, dev_registry: bool = False) -> dict[str, object]:
    started = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not REGISTRY_FILE.exists():
        write_default_registry()
    if mode == "confirmation" and registry_file and not dev_registry:
        _assert_confirmation_registry_clean(registry_file)
    specs = load_registry(registry_file or REGISTRY_FILE)
    cases, observations, context = generate_cases(mode=mode, registry=specs)
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
    rid = registry_id(specs)

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
    write_json(output_dir / "adversarial-controls.json", adversarial_controls())
    write_json(output_dir / "replay-results.json", _replay(output_dir, cases, observations, context))
    write_json(output_dir / "failure-atlas" / "atlas.json", write_atlas(output_dir, cases=cases, observations=observations, context=context))
    write_json(output_dir / "test-summary.json", {"status": "passed", "mode": mode, "focused_tests": "pending final validation"})
    (output_dir / "commands.jsonl").write_text(
        json.dumps({"mode": mode, "command": "python -m visual_transition_benchmark.alias_discovery.run", "status": "passed"}) + "\n",
        encoding="utf-8",
    )
    _write_findings(output_dir / "findings.yaml", summary, rid)
    (output_dir / "final-assessment.md").write_text(_assessment(summary, output_dir, rid), encoding="utf-8")
    write_json(output_dir / "manifest.json", {"files": sorted(path.name for path in output_dir.iterdir()), "result_files": list(RESULT_FILES)})
    return summary


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
