"""End-to-end inspection, structured report, validation and replay."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import _bootstrap  # noqa: F401
from zeromodel.core import VPMArtifact, VPMValidationError
from zeromodel.core.render import png_bytes

from ._common import EXPERIMENT, UnsupportedInput, digest
from .alignment import AlignmentRecipe, align, warp_template
from .assessment import (
    UNSUPPORTED,
    DecisionContract,
    decide,
)
from .comparison import (
    CandidateEvidence,
    ComparisonRecipe,
    ObservationFeatures,
    compare_candidate,
    dilate,
    observation_features,
)
from .observations import Observation, load_observation
from .pose_library import TemplateLibrary
from .vpm_compiler import (
    CandidateVPM,
    compile_candidate_vpm,
    evidence_rows_from_vpm,
    observation_level_from_vpm,
)

REPORT_SCHEMA = "hand-anatomy-report/1"

LIMITATIONS = [
    "Single 2D silhouette: depth, self-occlusion order, palm-vs-back facing and handedness are not recoverable.",
    "Template library is a bounded 2D-projected capsule-skeleton family; unusual but valid hands may fit poorly (reported as insufficient_evidence, not as anomalies).",
    "supported_plausible means only that the declared tests passed; it is not proof of anatomical correctness.",
    "Missing-structure evidence is ambiguous with ordinary occlusion and is never used to claim an anomaly.",
    "Finger-like projection counts are a heuristic that fails under overlap and perspective.",
    "Mask quality is assumed; segmentation errors can produce false anomalies.",
    "Rotation search is bounded (+/-45 deg in-plane); larger rotations are outside the declared search space.",
]


@dataclass
class Inspection:
    observation: Observation
    features: ObservationFeatures
    candidates: List[CandidateEvidence]
    compiled: CandidateVPM
    decision: Dict[str, Any]
    contract: DecisionContract
    alignment_recipe: AlignmentRecipe
    comparison_recipe: ComparisonRecipe
    library: TemplateLibrary


def inspect_observation(
    observation: Observation,
    library: TemplateLibrary,
    contract: DecisionContract = DecisionContract(),
    alignment_recipe: AlignmentRecipe = AlignmentRecipe(),
    comparison_recipe: ComparisonRecipe = ComparisonRecipe(),
) -> Inspection:
    features = observation_features(observation, comparison_recipe)
    aligned = align(observation, library, alignment_recipe)
    candidates = [
        compare_candidate(observation, features, library.get(a.template_id), a, comparison_recipe)
        for a in aligned
    ]
    compiled = compile_candidate_vpm(
        observation, features, candidates, library.library_id,
        comparison_recipe, alignment_recipe.recipe_id,
    )
    # decision reads RAW evidence back out of the VPM artifact
    decision = decide(
        evidence_rows_from_vpm(compiled.vpm), observation_level_from_vpm(compiled.vpm), contract
    )
    return Inspection(
        observation, features, candidates, compiled, decision, contract,
        alignment_recipe, comparison_recipe, library,
    )


def build_report(result: Inspection, artifact_references: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    vpm = result.compiled.vpm
    rows = evidence_rows_from_vpm(vpm)
    best = rows[0]
    by_id = {c.template_id: c for c in result.candidates}
    explainer_id = result.decision["explainer_template_id"] or best["template_id"]
    body: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "observation_id": result.observation.observation_id,
        "observation": result.observation.summary(),
        "template_library_id": result.library.library_id,
        "comparison_recipe_id": result.comparison_recipe.recipe_id,
        "comparison_recipe": result.comparison_recipe.to_dict(),
        "alignment_recipe_id": result.alignment_recipe.recipe_id,
        "alignment_recipe": result.alignment_recipe.to_dict(),
        "decision_contract": result.contract.to_dict(),
        "decision_contract_id": result.contract.contract_id,
        "vpm_artifact_id": vpm.artifact_id,
        "score_table_digest": "sha256:" + vpm.source.digest,
        "layout_recipe_digest": "sha256:" + vpm.recipe.digest,
        "best_template_id": best["template_id"],
        "best_template_name": best["name"],
        "explainer_template_id": explainer_id,
        "explainer_template_name": by_id[explainer_id].template_name,
        "decision": result.decision["status"],
        "decision_detail": result.decision,
        "evidence": {
            "best_template": best["metrics"],
            "explainer_template": by_id[explainer_id].metrics,
            "observation_level": dict(observation_level_from_vpm(vpm)),
            "diagnostics_explainer": by_id[explainer_id].diagnostics,
        },
        "candidates": [
            {
                "template_id": r["template_id"],
                "name": r["name"],
                "alignment": by_id[r["template_id"]].alignment.params(),
                "metrics": r["metrics"],
                "diagnostics": by_id[r["template_id"]].diagnostics,
            }
            for r in rows
        ],
        "vpm_artifact": vpm.to_dict(),
        "limitations": list(LIMITATIONS)
        + (
            [] if result.contract.calibration is not None
            else ["Decision thresholds are exploratory and uncalibrated."]
        ),
    }
    body["report_digest"] = digest(body)
    body["artifact_references"] = dict(artifact_references or {})
    return body


def unsupported_report(reason: str, contract: DecisionContract = DecisionContract()) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "experiment": EXPERIMENT,
        "decision": UNSUPPORTED,
        "decision_detail": {"status": UNSUPPORTED, "reasons": [reason]},
        "decision_contract_id": contract.contract_id,
        "limitations": list(LIMITATIONS),
    }
    body["report_digest"] = digest(body)
    return body


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, allow_nan=False))


def validate_report(report: Dict[str, Any]) -> List[str]:
    """Self-consistency check of a saved report (no source inputs needed)."""
    problems: List[str] = []
    if report.get("schema_version") != REPORT_SCHEMA:
        return ["unsupported schema_version"]
    body = {k: v for k, v in report.items() if k not in ("report_digest", "artifact_references")}
    if report.get("decision") == UNSUPPORTED:
        if digest(body) != report.get("report_digest"):
            problems.append("report_digest mismatch")
        return problems
    if digest(body) != report.get("report_digest"):
        problems.append("report_digest mismatch (report content was altered)")
    try:
        vpm = VPMArtifact.from_dict(report["vpm_artifact"])
    except (VPMValidationError, KeyError, ValueError) as exc:
        return problems + ["embedded VPM invalid: %s" % exc]
    if vpm.artifact_id != report["vpm_artifact_id"]:
        problems.append("vpm_artifact_id does not match embedded artifact")
    contract = DecisionContract.from_dict(report["decision_contract"])
    if contract.contract_id != report["decision_contract_id"]:
        problems.append("decision contract id mismatch")
    rows = evidence_rows_from_vpm(vpm)
    redo = decide(rows, observation_level_from_vpm(vpm), contract)
    if redo["status"] != report["decision"]:
        problems.append("decision does not follow from VPM evidence under the contract")
    for cand in report["candidates"]:
        raw = next(r for r in rows if r["template_id"] == cand["template_id"])["metrics"]
        if _json_safe(raw) != _json_safe(cand["metrics"]):
            problems.append("candidate metrics differ from VPM raw cells: %s" % cand["name"])
    meta = vpm.source.metadata
    if meta["observation_id"] != report["observation_id"]:
        problems.append("VPM observation binding mismatch")
    if meta["library_id"] != report["template_library_id"]:
        problems.append("VPM library binding mismatch")
    return problems


def replay_report(
    report: Dict[str, Any], observation: Observation, library: TemplateLibrary
) -> List[str]:
    """Recompute from the original inputs; return mismatches (empty = reproduced)."""
    problems = validate_report(report)
    if report.get("decision") == UNSUPPORTED:
        return problems
    if observation.observation_id != report["observation_id"]:
        problems.append("observation_id differs: inputs are not the reported observation")
    if library.library_id != report["template_library_id"]:
        problems.append("template_library_id differs")
    result = inspect_observation(
        observation,
        library,
        DecisionContract.from_dict(report["decision_contract"]),
        AlignmentRecipe.from_dict(report["alignment_recipe"]),
        ComparisonRecipe.from_dict(report["comparison_recipe"]),
    )
    if result.compiled.vpm.artifact_id != report["vpm_artifact_id"]:
        problems.append("replayed vpm_artifact_id differs")
    if result.decision["status"] != report["decision"]:
        problems.append("replayed decision differs")
    return problems


def write_inspection(result: Inspection, out_dir: str | Path) -> Dict[str, str]:
    """Write report.json + images; every image is derived from the reported alignment."""
    from . import visualization as vz

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    obs = result.observation
    best_id = result.decision["best_template_id"]
    by_id = {c.template_id: c for c in result.candidates}
    refs: Dict[str, str] = {}

    def save(name: str, img: Any) -> None:
        path = out / name
        img.save(path)
        refs[name] = path.name

    if obs.image is not None:
        from PIL import Image

        save("01_original.png", Image.fromarray(obs.image, "RGB"))
    save("02_normalized_mask.png", vz.mask_image(obs.work_mask, 2))
    tol = result.comparison_recipe.tolerance_px
    obs_mask = obs.work_mask
    best = by_id[best_id]
    expl = by_id[result.decision["explainer_template_id"] or best_id]
    tpl_masks = {}
    for tag, cand in (("best", best), ("explainer", expl)):
        T = warp_template(result.library.get(cand.template_id), cand.alignment.params())
        tpl_masks[tag] = T
        if tag == "best":
            save("03_best_template.png", vz.mask_image(T, 2))
        save("04_overlay_%s.png" % tag, vz.overlay_image(obs_mask, T, obs_mask & ~dilate(T, tol), T & ~dilate(obs_mask, tol)))
        if tag == "explainer":
            save("05_unexplained_observed.png", vz.mask_image(obs_mask & ~dilate(T, tol), 2))
            save("06_missing_expected.png", vz.mask_image(T & ~dilate(obs_mask, tol), 2))
    tiles = []
    for cand in result.candidates:
        T = warp_template(result.library.get(cand.template_id), cand.alignment.params())
        tiles.append(
            ("%s IoU%.2f" % (cand.template_name.replace("standard", "std"), cand.metrics["silhouette_agreement"]),
             vz.overlay_image(obs_mask, T, obs_mask & ~dilate(T, tol), T & ~dilate(obs_mask, tol), scale=1))
        )
    save("07_candidate_contact_sheet.png", vz.tile_sheet(tiles, 4, tile=150))
    vpm = result.compiled.vpm
    rows = evidence_rows_from_vpm(vpm)
    raw = np.array([[r["metrics"][m] for m in [vpm.source.metric_ids[c] for c in vpm.column_order]] for r in rows])
    vz.vpm_render(
        vpm.normalized_values, [r["name"] for r in rows],
        [vpm.source.metric_ids[c] for c in vpm.column_order], raw, out / "08_vpm_annotated.png",
    )
    refs["08_vpm_annotated.png"] = "08_vpm_annotated.png"
    (out / "08_vpm_core_render.png").write_bytes(png_bytes(vpm))
    refs["08_vpm_core_render.png"] = "08_vpm_core_render.png"
    report = build_report(result, refs)
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    refs["report.json"] = "report.json"
    return refs


def inspect_files(
    library: TemplateLibrary,
    image: Optional[str] = None,
    mask: Optional[str] = None,
    landmarks: Optional[str] = None,
    segment_background: bool = False,
    contract: DecisionContract = DecisionContract(),
    out_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """File-level entry point. Unsupported inputs yield an ``unsupported_input`` report."""
    try:
        obs = load_observation(image, mask, landmarks, segment_background)
    except UnsupportedInput as exc:
        report = unsupported_report(str(exc), contract)
        if out_dir:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            (Path(out_dir) / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        return report
    result = inspect_observation(obs, library, contract)
    if out_dir:
        write_inspection(result, out_dir)
        return json.loads((Path(out_dir) / "report.json").read_text(encoding="utf-8"))
    return build_report(result)
