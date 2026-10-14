"""Compile candidate-template comparison evidence into a genuine ZeroModel VPM.

Rows    = shortlisted candidate templates (row id = template id)
Columns = independent compatibility metrics (all higher-is-more-compatible)
Cells   = RAW compatibility evidence (``ScoreTable.values``)

The VPM view is min-max normalised per column by the core kernel, which can
make the best of several poor candidates look strong.  Therefore every
decision reads RAW values (``VPMArtifact.cell(...).raw_value``), never the
normalised view.  This candidate-comparison VPM is distinct from any policy
VPM that might later turn an assessment into a review action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence

from . import _bootstrap  # noqa: F401
from zeromodel.core import LayoutRecipe, ScoreTable, VPMArtifact, build_vpm
from zeromodel.core.artifact import LAYOUT_VERSION

from ._common import HandAnatomyError
from .comparison import (
    CORE_METRICS,
    METRICS,
    CandidateEvidence,
    ComparisonRecipe,
    ObservationFeatures,
)
from .observations import Observation

VPM_SCHEMA = "hand-candidate-vpm/1"
RANKING_WEIGHTS = (("silhouette_agreement", 0.5), ("boundary_agreement", 0.5))


def layout_recipe(metric_ids: Sequence[str]) -> LayoutRecipe:
    """Rows ranked by a declared retrieval score (presentation order only)."""
    return LayoutRecipe.from_dict(
        {
            "version": LAYOUT_VERSION,
            "name": "hand-candidate-ranking/1",
            "row_order": {
                "kind": "weighted_score",
                "keys": [
                    {"metric_id": m, "direction": "desc", "weight": w}
                    for m, w in RANKING_WEIGHTS
                ],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "explicit", "metric_ids": list(metric_ids)},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )


@dataclass(frozen=True)
class CandidateVPM:
    table: ScoreTable
    recipe: LayoutRecipe
    vpm: VPMArtifact


def compile_candidate_vpm(
    observation: Observation,
    features: ObservationFeatures,
    candidates: Sequence[CandidateEvidence],
    library_id: str,
    comparison_recipe: ComparisonRecipe,
    alignment_recipe_id: str,
) -> CandidateVPM:
    if not candidates:
        raise HandAnatomyError("cannot compile a VPM without candidates")
    metric_ids = list(CORE_METRICS)
    with_landmarks = all("landmark_compatibility" in c.metrics for c in candidates)
    if with_landmarks:
        metric_ids.append("landmark_compatibility")
    for c in candidates:
        if set(c.metrics) != set(metric_ids):
            raise HandAnatomyError("incompatible comparison inputs: metric sets differ")
    ordered = sorted(candidates, key=lambda c: c.template_id)
    values = [[c.metrics[m] for m in metric_ids] for c in ordered]
    metadata: Dict[str, Any] = {
        "schema": VPM_SCHEMA,
        "observation_id": observation.observation_id,
        "library_id": library_id,
        "comparison_recipe_id": comparison_recipe.recipe_id,
        "alignment_recipe_id": alignment_recipe_id,
        "metrics": {m: dict(METRICS[m]) for m in metric_ids},
        "observation_level": {
            "connectivity": features.connectivity,
            "component_count": features.component_count,
            "protrusions_observed": features.protrusions["count"],
            "border_contact_fraction": observation.border_contact_fraction,
            "area_px": features.area,
        },
        "row_bindings": {
            c.template_id: {
                "name": c.template_name,
                "alignment": c.alignment.params(),
                "iou_pre_alignment": c.alignment.iou_pre,
                "protrusions_template": c.diagnostics["protrusions_template"],
            }
            for c in ordered
        },
        "raw_values_note": "cells are raw compatibilities; the VPM view is per-column min-max normalised",
    }
    table = ScoreTable(values, [c.template_id for c in ordered], metric_ids, metadata)
    recipe = layout_recipe(metric_ids)
    vpm = build_vpm(
        table,
        recipe,
        provenance={
            "experiment": "ZM-HAND-01",
            "observation_id": observation.observation_id,
            "library_id": library_id,
            "comparison_recipe_id": comparison_recipe.recipe_id,
        },
    )
    return CandidateVPM(table, recipe, vpm)


def evidence_rows_from_vpm(vpm: VPMArtifact) -> List[Dict[str, Any]]:
    """Read RAW per-candidate evidence back out of a VPM, in view (ranked) order."""
    rows = []
    bindings = vpm.source.metadata["row_bindings"]
    for view_row in range(len(vpm.row_order)):
        cells = [vpm.cell(view_row, c) for c in range(len(vpm.column_order))]
        rid = cells[0].row_id
        rows.append(
            {
                "template_id": rid,
                "name": bindings[rid]["name"],
                "protrusions_template": bindings[rid]["protrusions_template"],
                "metrics": {cell.metric_id: cell.raw_value for cell in cells},
            }
        )
    return rows


def observation_level_from_vpm(vpm: VPMArtifact) -> Mapping[str, Any]:
    return vpm.source.metadata["observation_level"]
