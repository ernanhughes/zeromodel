"""Anatomical constraint and residual VPM, plus the ZM-HAND-02 decision contract.

Rows = shortlisted candidate templates.  Columns (all higher = more compatible,
RAW values in the ScoreTable; decisions never read the normalised view):

* library_iou, library_finger_residual_explainability - the UNREFINED library candidate (aligned)
* silhouette_agreement (IoU), boundary_agreement, observed_explainability,
  protrusion_explainability                - global fit evidence of the constraint-refined pose
* digit_coverage                           - fraction of the template's visible digits present in the observation
* finger_residual_explainability           - 1 - max finger-likeness of residual components
* branch_topology_agreement                - 1 - min(1, max(0, n_branches(obs) - n_branches(template)))
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from examples.hand_anatomy import _bootstrap  # noqa: F401
from zeromodel.core import LayoutRecipe, ScoreTable, VPMArtifact, build_vpm
from zeromodel.core.artifact import LAYOUT_VERSION

from examples.hand_anatomy._common import HandAnatomyError, digest

VPM02_SCHEMA = "hand-residual-vpm/1"
DECISION02_VERSION = "hand-decision/2"
METRICS02 = (
    "library_iou", "library_finger_residual_explainability",
    "silhouette_agreement", "boundary_agreement", "observed_explainability",
    "protrusion_explainability", "digit_coverage", "finger_residual_explainability",
    "branch_topology_agreement",
)
SUPPORTED, SUSPECT, INSUFFICIENT = "supported_plausible", "suspected_anomaly", "insufficient_evidence"


def layout_recipe02() -> LayoutRecipe:
    return LayoutRecipe.from_dict({
        "version": LAYOUT_VERSION, "name": "hand-residual-ranking/1",
        "row_order": {"kind": "weighted_score", "tie_break": "row_id", "keys": [
            {"metric_id": "silhouette_agreement", "direction": "desc", "weight": 0.5},
            {"metric_id": "boundary_agreement", "direction": "desc", "weight": 0.5}]},
        "column_order": {"kind": "explicit", "metric_ids": list(METRICS02)},
        "normalization": {"kind": "per_metric_minmax", "clip": True},
    })


def compile_vpm02(
    observation_id: str, library_id: str, recipe_ids: Mapping[str, str],
    rows: Sequence[Mapping[str, Any]], observation_level: Mapping[str, Any],
) -> VPMArtifact:
    """rows: dicts with template_id, name, metrics (METRICS02), diagnostics."""
    if not rows:
        raise HandAnatomyError("cannot compile a VPM without candidates")
    ordered = sorted(rows, key=lambda r: r["template_id"])
    for r in ordered:
        if set(r["metrics"]) != set(METRICS02):
            raise HandAnatomyError("incompatible comparison inputs: metric sets differ")
    meta = {
        "schema": VPM02_SCHEMA, "observation_id": observation_id, "library_id": library_id,
        "recipe_ids": dict(recipe_ids), "observation_level": dict(observation_level),
        "metrics": {m: {"direction": "higher_is_more_compatible", "range": "[0, 1]"} for m in METRICS02},
        "row_bindings": {r["template_id"]: {"name": r["name"], "diagnostics": r["diagnostics"]} for r in ordered},
        "raw_values_note": "cells are raw; the VPM view is per-column min-max normalised",
    }
    table = ScoreTable([[r["metrics"][m] for m in METRICS02] for r in ordered],
                       [r["template_id"] for r in ordered], METRICS02, meta)
    return build_vpm(table, layout_recipe02(), provenance={
        "experiment": "ZM-HAND-02", "observation_id": observation_id, "library_id": library_id})


def rows_from_vpm02(vpm: VPMArtifact) -> List[Dict[str, Any]]:
    out = []
    for r in range(len(vpm.row_order)):
        cells = [vpm.cell(r, c) for c in range(len(vpm.column_order))]
        rid = cells[0].row_id
        out.append({"template_id": rid, "name": vpm.source.metadata["row_bindings"][rid]["name"],
                    "metrics": {c.metric_id: c.raw_value for c in cells}})
    return out


def eligible_rows(rows: Sequence[Mapping[str, Any]], fit_floor: float) -> List[Mapping[str, Any]]:
    elig = [r for r in rows if r["metrics"]["silhouette_agreement"] >= fit_floor]
    return elig or list(rows)


def scores_from_rows(rows: Sequence[Mapping[str, Any]], fit_floor: float = 0.45) -> Dict[str, float]:
    """Anomaly scores (higher = more anomalous), derived only from RAW VPM cells."""
    elig = eligible_rows(rows, fit_floor)
    return {
        "A_iou": 1.0 - max(r["metrics"]["library_iou"] for r in rows),
        "A_refined_iou": 1.0 - max(r["metrics"]["silhouette_agreement"] for r in rows),
        "B01_protrusion": 1.0 - max(r["metrics"]["protrusion_explainability"] for r in elig),
        "C_residual_unrefined": 1.0 - max(r["metrics"]["library_finger_residual_explainability"] for r in rows),
        "C_residual": 1.0 - max(r["metrics"]["finger_residual_explainability"] for r in elig),
        "C_branch": 1.0 - max(r["metrics"]["branch_topology_agreement"] for r in elig),
    }


@dataclass(frozen=True)
class Contract02:
    fit_floor: float = 0.45
    fit_ok: float = 0.70
    residual_threshold: float = 0.5  # flag when C_residual score > threshold
    border_contact_max: float = 0.10
    calibration: Optional[Mapping[str, Any]] = None
    version: str = DECISION02_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {"version": self.version, "fit_floor": self.fit_floor, "fit_ok": self.fit_ok,
                "residual_threshold": self.residual_threshold, "border_contact_max": self.border_contact_max,
                "calibrated": self.calibration is not None,
                "calibration": None if self.calibration is None else dict(self.calibration)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Contract02":
        return cls(fit_floor=d["fit_floor"], fit_ok=d["fit_ok"], residual_threshold=d["residual_threshold"],
                   border_contact_max=d["border_contact_max"], calibration=d.get("calibration"))

    @property
    def contract_id(self) -> str:
        return digest(self.to_dict())

    def with_values(self, **kw: Any) -> "Contract02":
        return replace(self, **kw)


def decide02(rows: Sequence[Mapping[str, Any]], observation_level: Mapping[str, Any],
             contract: Contract02 = Contract02()) -> Dict[str, Any]:
    out: Dict[str, Any] = {"contract_id": contract.contract_id, "calibrated": contract.calibration is not None,
                           "best_template_id": rows[0]["template_id"] if rows else None}
    if not rows:
        return {**out, "status": INSUFFICIENT, "reasons": ["no_candidates"]}
    if observation_level["border_contact_fraction"] > contract.border_contact_max:
        return {**out, "status": INSUFFICIENT, "reasons": ["possible_truncation_at_image_border"]}
    elig = [r for r in rows if r["metrics"]["silhouette_agreement"] >= contract.fit_floor]
    if not elig:
        return {**out, "status": INSUFFICIENT, "reasons": ["no_permitted_configuration_reaches_fit_floor"]}
    explainer = max(elig, key=lambda r: (r["metrics"]["finger_residual_explainability"],
                                         r["metrics"]["silhouette_agreement"]))
    score = 1.0 - explainer["metrics"]["finger_residual_explainability"]
    out.update(explainer_template_id=explainer["template_id"], residual_score=score)
    if score > contract.residual_threshold:
        return {**out, "status": SUSPECT, "reasons": ["finger_like_unexplained_residual"]}
    if rows[0]["metrics"]["silhouette_agreement"] < contract.fit_ok:
        return {**out, "status": INSUFFICIENT, "reasons": ["best_fit_below_positive_threshold"]}
    return {**out, "status": SUPPORTED, "reasons": ["declared_tests_passed_no_finger_like_residual"]}
