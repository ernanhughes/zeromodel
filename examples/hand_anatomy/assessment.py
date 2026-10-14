"""Versioned, conservative decision contract over candidate-comparison evidence.

Principle: the verdict is not "which known hand looks most similar" but "which
observed structure cannot be explained by ANY permitted configuration in the
shortlist".  Hard contradictions are never averaged away.  The system can
abstain.  ``supported_plausible`` only means the declared tests passed.

Thresholds are EXPLORATORY and UNCALIBRATED unless a calibration record is
attached (the benchmark produces one from a held-out calibration split).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence

from ._common import digest

DECISION_VERSION = "hand-decision/1"
SUPPORTED = "supported_plausible"
SUSPECT = "suspected_anomaly"
INSUFFICIENT = "insufficient_evidence"
UNSUPPORTED = "unsupported_input"
STATUSES = (SUPPORTED, SUSPECT, INSUFFICIENT, UNSUPPORTED)


@dataclass(frozen=True)
class DecisionContract:
    fit_floor: float = 0.55  # a candidate is a "permitted explanation" above this IoU
    fit_ok: float = 0.80  # best candidate must reach this for a positive finding
    protrusion_floor: float = 0.50
    observed_floor: float = 0.93
    missing_floor: float = 0.85
    connectivity_floor: float = 0.97
    border_contact_max: float = 0.10
    use_protrusion: bool = True
    use_observed_area: bool = True
    use_structure: bool = False  # heuristic; enabled only if calibration supports it
    structure_min_template_protrusions: int = 3
    calibration: Optional[Mapping[str, Any]] = None  # None => uncalibrated
    version: str = DECISION_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "fit_floor": self.fit_floor,
            "fit_ok": self.fit_ok,
            "protrusion_floor": self.protrusion_floor,
            "observed_floor": self.observed_floor,
            "missing_floor": self.missing_floor,
            "connectivity_floor": self.connectivity_floor,
            "border_contact_max": self.border_contact_max,
            "use_protrusion": self.use_protrusion,
            "use_observed_area": self.use_observed_area,
            "use_structure": self.use_structure,
            "structure_min_template_protrusions": self.structure_min_template_protrusions,
            "calibrated": self.calibration is not None,
            "calibration": None if self.calibration is None else dict(self.calibration),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DecisionContract":
        fields = {k: v for k, v in data.items() if k not in ("calibrated", "calibration")}
        return cls(calibration=data.get("calibration"), **fields)

    @property
    def contract_id(self) -> str:
        return digest(self.to_dict())

    def with_values(self, **kw: Any) -> "DecisionContract":
        return replace(self, **kw)


def decide(
    rows: Sequence[Mapping[str, Any]],
    observation_level: Mapping[str, Any],
    contract: DecisionContract = DecisionContract(),
) -> Dict[str, Any]:
    """Decide from RAW evidence rows (ranked best-first) plus observation-level evidence."""
    contradictions: List[Dict[str, Any]] = []
    out: Dict[str, Any] = {
        "contract_id": contract.contract_id,
        "calibrated": contract.calibration is not None,
        "best_template_id": rows[0]["template_id"] if rows else None,
        "explainer_template_id": None,
        "contradictions": contradictions,
        "reasons": [],
    }

    def finish(status: str, *reasons: str) -> Dict[str, Any]:
        out["status"] = status
        out["reasons"] = list(reasons)
        return out

    if not rows:
        return finish(INSUFFICIENT, "no_candidates")
    best = rows[0]["metrics"]
    out["best_fit"] = best["silhouette_agreement"]
    out["fit_margin"] = (
        best["silhouette_agreement"] - rows[1]["metrics"]["silhouette_agreement"]
        if len(rows) > 1
        else None
    )
    if observation_level["border_contact_fraction"] > contract.border_contact_max:
        return finish(INSUFFICIENT, "possible_truncation_at_image_border")
    eligible = [r for r in rows if r["metrics"]["silhouette_agreement"] >= contract.fit_floor]
    if not eligible:
        return finish(INSUFFICIENT, "no_permitted_configuration_reaches_fit_floor")
    # best-case explainer: the permitted configuration that leaves least unexplained
    explainer = max(
        eligible,
        key=lambda r: (
            r["metrics"]["protrusion_explainability"],
            r["metrics"]["observed_explainability"],
            r["metrics"]["silhouette_agreement"],
        ),
    )
    out["explainer_template_id"] = explainer["template_id"]
    m = explainer["metrics"]
    if contract.use_protrusion and m["protrusion_explainability"] < contract.protrusion_floor:
        contradictions.append(
            {"code": "unexplained_protrusion", "value": m["protrusion_explainability"],
             "floor": contract.protrusion_floor}
        )
    if contract.use_observed_area and m["observed_explainability"] < contract.observed_floor:
        contradictions.append(
            {"code": "unexplained_observed_area", "value": m["observed_explainability"],
             "floor": contract.observed_floor}
        )
    if observation_level["connectivity"] < contract.connectivity_floor:
        contradictions.append(
            {"code": "disconnected_structure", "value": observation_level["connectivity"],
             "floor": contract.connectivity_floor}
        )
    if (
        contract.use_structure
        and explainer["protrusions_template"] >= contract.structure_min_template_protrusions
        and observation_level["protrusions_observed"] > explainer["protrusions_template"]
    ):
        contradictions.append(
            {"code": "excess_finger_like_projections",
             "value": observation_level["protrusions_observed"],
             "template_value": explainer["protrusions_template"]}
        )
    if contradictions:
        return finish(SUSPECT, *[c["code"] for c in contradictions])
    if best["silhouette_agreement"] < contract.fit_ok:
        return finish(INSUFFICIENT, "best_fit_below_positive_threshold")
    if best["missing_explainability"] < contract.missing_floor:
        return finish(INSUFFICIENT, "expected_structure_missing_occlusion_or_absence_undecidable")
    return finish(SUPPORTED, "declared_tests_passed_no_contradiction")
