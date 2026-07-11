# Stephanie ZeroModel incorporation map

This document maps the reusable Stephanie `stephanie/zeromodel` work into the standalone ZeroModel package.

## Source modules reviewed

The Stephanie subtree contains:

- `vpm_builder.py` — score rows to normalized visual matrices and simple image output.
- `vpm_emitter.py` — standardized metric packing plus A/B/C, timeline, panel, and knowledge-progress visualizations.
- `vpm_differential_analyzer.py` — visual subtraction, overlap, contrast, and enriched-difference artifacts.
- `gap_gauge.py` — baseline-vs-target numeric and top-left comparison gauges.
- `vpm_controller.py` — trend-aware signals: EDIT, RESAMPLE, ESCALATE, STOP, SPINOFF, and HOLD.
- `vpm_phos.py` — PHOS packing, robust normalization, concentration metrics, guard sweeps, and model comparison.
- `zeromodel_service.py` — service-level orchestration, timeline sessions, model comparison helpers, PHOS mean images, and report rendering.

## Incorporation decisions

| Stephanie idea | ZeroModel destination | Rationale |
|---|---|---|
| Robust percentile normalization | `zeromodel.phos.robust01` | Useful for outlier-resistant visual packing. |
| PHOS sort-pack | `zeromodel.phos.phos_sort_pack` | Proven top-left concentration primitive. |
| Top-left concentration metric | `zeromodel.phos.top_left_concentration` | Required to make concentration claims measurable. |
| Guarded PHOS sweep | `zeromodel.phos.guarded_pack_artifact` | Keeps packing honest by requiring improvement over raw layout. |
| Visual AND/OR/NOT/XOR/subtract/add | `zeromodel.compose` | Makes visual logic explicit and shape-checked. |
| Good-vs-bad differential analysis | `zeromodel.compare` | Keeps contrast, overlap, gain/loss, and enrichment outside the artifact kernel. |
| Metric alias packing | `zeromodel.metrics` | Converts application-specific metrics into stable score tables. |
| Tiny PNG rendering | `zeromodel.render` | Provides dependency-light artifact visualization and transport. |
| Hierarchical pyramid | `zeromodel.hierarchy` | Supports bounded navigation without claiming magic memory. |
| Edge top-left gate | `zeromodel.edge` | Demonstrates no-model-at-decision-time routing on an artifact. |
| Trend-aware control loop | `zeromodel.controller` | Moves EDIT/RESAMPLE/ESCALATE/STOP/SPINOFF into a reusable consumer layer. |

## Boundary

The artifact kernel remains pure: it stores source scores, layout recipes, normalized views, source mapping, provenance, and deterministic identity. Rendering, PHOS packing, edge gates, composition, comparison, hierarchy, and control are explicit consumers around that artifact.
