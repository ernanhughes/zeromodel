"""ZeroModel v2.

The v2 package is built around one pure artifact kernel plus explicit consumer
modules for the proven blog capabilities: PHOS packing, visual composition,
differential comparison, PNG/SVG rendering, hierarchy, edge gating, metric
packing, bundle serialization, and trend-aware control.
"""
from __future__ import annotations

from .artifact import LayoutRecipe, ScoreTable, VPMArtifact, VPMCell, VPMRegion, build_vpm
from .bundle import from_bundle, to_bundle
from .compare import VPMComparison, compare_fields
from .compose import as_field, vpm_add, vpm_and, vpm_not, vpm_or, vpm_subtract, vpm_xor
from .controller import Decision, Policy, Signal, Thresholds, VPMController, VPMRow, default_controller
from .edge import TopLeftGate, TopLeftGateResult
from .hierarchy import HierarchyLevel, build_pyramid, reduce_blocks
from .metrics import CANONICAL_METRICS, metric_ids_for_rows, pack_metrics, score_table_from_metric_rows
from .phos import PHOSResult, guarded_pack_artifact, image_entropy, pack_artifact, phos_sort_pack, robust01, to_square, top_left_concentration
from .render import png_bytes, svg_text, to_uint8, write_png, write_svg

__all__ = [
    "CANONICAL_METRICS",
    "Decision",
    "HierarchyLevel",
    "LayoutRecipe",
    "PHOSResult",
    "Policy",
    "ScoreTable",
    "Signal",
    "Thresholds",
    "TopLeftGate",
    "TopLeftGateResult",
    "VPMArtifact",
    "VPMCell",
    "VPMComparison",
    "VPMController",
    "VPMRegion",
    "VPMRow",
    "as_field",
    "build_pyramid",
    "build_vpm",
    "compare_fields",
    "default_controller",
    "from_bundle",
    "guarded_pack_artifact",
    "image_entropy",
    "metric_ids_for_rows",
    "pack_artifact",
    "pack_metrics",
    "phos_sort_pack",
    "png_bytes",
    "reduce_blocks",
    "robust01",
    "score_table_from_metric_rows",
    "svg_text",
    "to_bundle",
    "to_square",
    "to_uint8",
    "top_left_concentration",
    "vpm_add",
    "vpm_and",
    "vpm_not",
    "vpm_or",
    "vpm_subtract",
    "vpm_xor",
    "write_png",
    "write_svg",
]
