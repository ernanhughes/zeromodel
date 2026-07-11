"""ZeroModel public package surface."""
from __future__ import annotations

from .artifact import LayoutRecipe, ScoreTable, VPMArtifact, VPMCell, VPMRegion, build_vpm
from .bundle import from_bundle, to_bundle
from .compare import VPMComparison, compare_fields
from .compose import as_field, vpm_add, vpm_and, vpm_not, vpm_or, vpm_subtract, vpm_xor
from .controller import Decision, Policy, Signal, Thresholds, VPMController, VPMRow, default_controller
from .edge import TopLeftGate, TopLeftGateResult
from .hierarchy import HierarchyLevel, build_pyramid, reduce_blocks
from .learning import LEARNING_METRICS, LearningAssessment, LearningObservation, build_learning_vpm, learning_recipe
from .metrics import CANONICAL_METRICS, metric_ids_for_rows, pack_metrics, score_table_from_metric_rows
from .phos import PHOSResult, guarded_pack_artifact, image_entropy, pack_artifact, phos_sort_pack, robust01, to_square, top_left_concentration
from .render import png_bytes, svg_text, to_uint8, write_png, write_svg

__version__ = "2.0.0"

__all__ = [
    "CANONICAL_METRICS",
    "Decision",
    "HierarchyLevel",
    "LAYOUT_VERSION",
    "LEARNING_METRICS",
    "LayoutRecipe",
    "LearningAssessment",
    "LearningObservation",
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
    "build_learning_vpm",
    "build_pyramid",
    "build_vpm",
    "compare_fields",
    "default_controller",
    "from_bundle",
    "guarded_pack_artifact",
    "image_entropy",
    "learning_recipe",
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
