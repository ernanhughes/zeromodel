"""ZM-HAND-02 inspection pipeline, report, validation, replay and visualisation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from examples.hand_anatomy import _bootstrap  # noqa: F401
from zeromodel.core import VPMArtifact, VPMValidationError

from examples.hand_anatomy import visualization as vz
from examples.hand_anatomy._common import UnsupportedInput, digest
from examples.hand_anatomy.alignment import AlignmentRecipe, align, warp_template
from examples.hand_anatomy.comparison import ComparisonRecipe, observation_features
from examples.hand_anatomy.observations import Observation, load_observation
from examples.hand_anatomy.pose_library import TemplateLibrary

from PIL import Image

from .refine import REFINE_VERSION, _anchor, refine, warp_anchored
from .residual import ResidualRecipe, analyze_residual, warp_label_masks
from .topology import digit_branches
from .vpm02 import (
    Contract02,
    compile_vpm02,
    decide02,
    rows_from_vpm02,
    scores_from_rows,
)

STARTS = 3
REPORT02_SCHEMA = "hand-anatomy-report/2"
LIMITATIONS02 = [
    "Single 2D silhouette: an extra digit fully inside the silhouette of another digit is indistinguishable; no method here can detect it.",
    "Finger-likeness and branch counts are heuristics tied to a 96-template library; adjacent or overlapping digits merge.",
    "supported_plausible only means the declared tests passed.",
    "Missing structure is never used to claim an anomaly (ambiguous with occlusion).",
    "Thresholds are calibrated on synthetic masks only; no real-image validation has been performed.",
]


@dataclass
class Inspection02:
    observation: Observation
    vpm: VPMArtifact
    rows: List[Dict[str, Any]]
    decision: Dict[str, Any]
    scores: Dict[str, float]
    contract: Contract02
    library: TemplateLibrary
    recipes: Dict[str, Any]
    explainer_residual: Dict[str, Any]
    explainer_template_id: str
    explainer_mask: np.ndarray
    residual_diag: Dict[str, Dict[str, Any]]


def _mask_metrics(obs: np.ndarray, T: np.ndarray, tol: int, ref_fraction: float, tau: float) -> Dict[str, float]:
    import math

    from examples.hand_anatomy.comparison import connected_components, dilate, symmetric_chamfer

    inter, union = float((obs & T).sum()), float((obs | T).sum())
    chamfer = symmetric_chamfer(obs, T)
    unexplained = obs & ~dilate(T, tol)
    _, areas = connected_components(unexplained, min_area=10)
    o_area = float(obs.sum())
    return {
        "silhouette_agreement": inter / union if union else 0.0,
        "boundary_agreement": math.exp(-chamfer / tau) if math.isfinite(chamfer) else 0.0,
        "observed_explainability": 1.0 - float(unexplained.sum()) / o_area,
        "protrusion_explainability": 1.0 - min(1.0, (max(areas) if areas else 0) / (ref_fraction * o_area)),
    }


def inspect02(
    observation: Observation, library: TemplateLibrary, contract: Contract02 = Contract02(),
    alignment_recipe: AlignmentRecipe = AlignmentRecipe(),
    comparison_recipe: ComparisonRecipe = ComparisonRecipe(),
    residual_recipe: ResidualRecipe = ResidualRecipe(),
    starts: int = STARTS,
    drop_forearm: bool = False,
) -> Inspection02:
    features = observation_features(observation, comparison_recipe)
    obs = observation.work_mask
    obs_branches = digit_branches(obs)
    rows: List[Dict[str, Any]] = []
    residuals: Dict[str, Dict[str, Any]] = {}
    masks: Dict[str, np.ndarray] = {}
    for a in align(observation, library, alignment_recipe)[:starts]:
        tpl0 = library.get(a.template_id)
        T0 = warp_template(tpl0, a.params())
        lib_res = analyze_residual(obs, T0, warp_label_masks(tpl0, a.params()), residual_recipe)
        fit = refine(obs, tpl0, a, drop_forearm)
        tpl = fit.template
        if tpl.template_id in residuals:
            continue
        T = warp_anchored(Image.fromarray((tpl.mask * 255).astype(np.uint8), "L"), _anchor(tpl), fit.transform)
        labels = {
            cid: warp_anchored(Image.fromarray(((tpl.labels == cid) * 255).astype(np.uint8), "L"),
                               _anchor(tpl), fit.transform)
            for cid in range(1, 8) if (tpl.labels == cid).any()
        }
        res = analyze_residual(obs, T, labels, residual_recipe)
        t_branches = digit_branches(T)
        excess = max(0, obs_branches["count"] - t_branches["count"])
        gm = _mask_metrics(obs, T, comparison_recipe.tolerance_px, comparison_recipe.protrusion_ref_fraction,
                           comparison_recipe.chamfer_tau_px)
        metrics = {
            "library_iou": a.iou_post,
            "library_finger_residual_explainability": 1.0 - lib_res["max_finger_likeness"],
            **gm,
            "digit_coverage": res["digit_coverage"],
            "finger_residual_explainability": 1.0 - res["max_finger_likeness"],
            "branch_topology_agreement": 1.0 - min(1.0, float(excess)),
        }
        diag = {
            "top_residual": res["components"][0] if res["components"] else None,
            "n_residual_components": len(res["components"]),
            "branches_observed": obs_branches["count"], "branches_template": t_branches["count"],
            "digits_present": res["digits_present"], "digits_explained": res["digits_explained"],
            "start_template_id": a.template_id, "refine_evaluations": fit.evaluations,
            "refined_transform": fit.transform, "refined_pose": tpl.pose.to_dict(),
            "refined_yaw": tpl.camera.yaw,
        }
        rows.append({"template_id": tpl.template_id, "name": "refined(%s)" % tpl0.name,
                     "metrics": metrics, "diagnostics": diag})
        residuals[tpl.template_id], masks[tpl.template_id] = res, T
    recipe_ids = {"alignment": alignment_recipe.recipe_id, "comparison": comparison_recipe.recipe_id,
                  "residual": residual_recipe.recipe_id, "refine": REFINE_VERSION}
    obs_level = {"border_contact_fraction": observation.border_contact_fraction,
                 "connectivity": features.connectivity, "branches_observed": obs_branches["count"],
                 "area_px": features.area}
    vpm = compile_vpm02(observation.observation_id, library.library_id, recipe_ids, rows, obs_level)
    ranked = rows_from_vpm02(vpm)
    decision = decide02(ranked, vpm.source.metadata["observation_level"], contract)
    scores = scores_from_rows(ranked, contract.fit_floor)
    exp_id = decision.get("explainer_template_id") or ranked[0]["template_id"]
    return Inspection02(
        observation, vpm, ranked, decision, scores, contract, library,
        {"alignment": alignment_recipe, "comparison": comparison_recipe, "residual": residual_recipe},
        residuals[exp_id], exp_id, masks[exp_id],
        {r["template_id"]: r["diagnostics"] for r in rows},
    )


def build_report02(r: Inspection02, refs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "schema_version": REPORT02_SCHEMA, "experiment": "ZM-HAND-02",
        "observation_id": r.observation.observation_id, "observation": r.observation.summary(),
        "template_library_id": r.library.library_id,
        "recipes": {k: v.to_dict() for k, v in r.recipes.items()},
        "decision_contract": r.contract.to_dict(), "decision_contract_id": r.contract.contract_id,
        "vpm_artifact_id": r.vpm.artifact_id, "decision": r.decision["status"], "decision_detail": r.decision,
        "best_template_id": r.rows[0]["template_id"], "best_template_name": r.rows[0]["name"],
        "explainer_template_id": r.explainer_template_id,
        "explainer_template_name": next(x["name"] for x in r.rows if x["template_id"] == r.explainer_template_id),
        "scores": r.scores,
        "evidence": {"best_template": r.rows[0]["metrics"],
                     "explainer_template": next(x["metrics"] for x in r.rows if x["template_id"] == r.explainer_template_id),
                     "diagnostics_explainer": r.residual_diag[r.explainer_template_id]},
        "vpm_artifact": r.vpm.to_dict(),
        "limitations": list(LIMITATIONS02),
    }
    body["report_digest"] = digest(body)
    body["artifact_references"] = dict(refs or {})
    return body


def validate_report02(report: Dict[str, Any]) -> List[str]:
    problems: List[str] = []
    if report.get("schema_version") != REPORT02_SCHEMA:
        return ["unsupported schema_version"]
    body = {k: v for k, v in report.items() if k not in ("report_digest", "artifact_references")}
    if digest(body) != report.get("report_digest"):
        problems.append("report_digest mismatch (report content was altered)")
    try:
        vpm = VPMArtifact.from_dict(report["vpm_artifact"])
    except (VPMValidationError, KeyError, ValueError) as exc:
        return problems + ["embedded VPM invalid: %s" % exc]
    if vpm.artifact_id != report["vpm_artifact_id"]:
        problems.append("vpm_artifact_id does not match embedded artifact")
    contract = Contract02.from_dict(report["decision_contract"])
    if contract.contract_id != report["decision_contract_id"]:
        problems.append("decision contract id mismatch")
    redo = decide02(rows_from_vpm02(vpm), vpm.source.metadata["observation_level"], contract)
    if redo["status"] != report["decision"]:
        problems.append("decision does not follow from VPM evidence under the contract")
    if json.loads(json.dumps(scores_from_rows(rows_from_vpm02(vpm), contract.fit_floor))) != report["scores"]:
        problems.append("scores differ from VPM raw cells")
    if vpm.source.metadata["observation_id"] != report["observation_id"]:
        problems.append("VPM observation binding mismatch")
    return problems


def replay_report02(report: Dict[str, Any], observation: Observation, library: TemplateLibrary) -> List[str]:
    problems = validate_report02(report)
    if observation.observation_id != report["observation_id"]:
        problems.append("observation_id differs: inputs are not the reported observation")
    if library.library_id != report["template_library_id"]:
        problems.append("template_library_id differs")
    rec = report["recipes"]
    res = inspect02(
        observation, library, Contract02.from_dict(report["decision_contract"]),
        AlignmentRecipe.from_dict(rec["alignment"]), ComparisonRecipe.from_dict(rec["comparison"]),
        ResidualRecipe(**{k: v for k, v in rec["residual"].items()}),
    )
    if res.vpm.artifact_id != report["vpm_artifact_id"]:
        problems.append("replayed vpm_artifact_id differs")
    if res.decision["status"] != report["decision"]:
        problems.append("replayed decision differs")
    return problems


def residual_overlay(r: Inspection02, scale: int = 2):
    from examples.hand_anatomy.comparison import dilate

    obs_mask = r.observation.work_mask
    T = r.explainer_mask
    tol = r.recipes["residual"].tolerance_px
    img = np.asarray(vz.overlay_image(obs_mask, T, obs_mask & ~dilate(T, tol), T & ~dilate(obs_mask, tol), scale=1)).copy()
    top = r.explainer_residual["components"][0] if r.explainer_residual["components"] else None
    if top and top["finger_likeness"] > 0:
        y0, x0, y1, x1 = top["bbox_yxyx"]
        img[y0:y1, x0] = img[y0:y1, x1 - 1] = (255, 220, 0)
        img[y0, x0:x1] = img[y1 - 1, x0:x1] = (255, 220, 0)
    from PIL import Image

    out = Image.fromarray(img, "RGB")
    return out.resize((out.width * scale, out.height * scale), Image.NEAREST) if scale != 1 else out


def write_inspection02(r: Inspection02, out_dir: str | Path) -> Dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    refs: Dict[str, str] = {}
    vz.mask_image(r.observation.work_mask, 2).save(out / "02_normalized_mask.png")
    residual_overlay(r).save(out / "04_residual_overlay.png")
    cols = [r.vpm.source.metric_ids[c] for c in r.vpm.column_order]
    raw = np.array([[row["metrics"][m] for m in cols] for row in r.rows])
    vz.vpm_render(r.vpm.normalized_values, [x["name"] for x in r.rows], cols, raw, out / "08_vpm_annotated.png")
    for n in ("02_normalized_mask.png", "04_residual_overlay.png", "08_vpm_annotated.png"):
        refs[n] = n
    report = build_report02(r, refs)
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return refs


def inspect_files02(library: TemplateLibrary, mask: Optional[str] = None, image: Optional[str] = None,
                    contract: Contract02 = Contract02(), out_dir: Optional[str] = None) -> Dict[str, Any]:
    try:
        obs = load_observation(image, mask)
    except UnsupportedInput as exc:
        body = {"schema_version": REPORT02_SCHEMA, "experiment": "ZM-HAND-02", "decision": "unsupported_input",
                "decision_detail": {"status": "unsupported_input", "reasons": [str(exc)]}}
        return body
    res = inspect02(obs, library, contract)
    if out_dir:
        write_inspection02(res, out_dir)
    return build_report02(res)
