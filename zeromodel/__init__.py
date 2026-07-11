"""ZeroModel public package surface."""
from __future__ import annotations

from .artifact import LayoutRecipe, ScoreTable, VPMArtifact, VPMCell, VPMRegion, build_vpm
from .bundle import from_bundle, to_bundle
from .compare import VPMComparison, compare_fields
from .compose import as_field, vpm_add, vpm_and, vpm_not, vpm_or, vpm_subtract, vpm_xor
from .controller import Decision, Policy, Signal, Thresholds, VPMController, VPMRow, default_controller
from .critic import CRITIC_METRICS, CriticAssessment, CriticObservation, build_critic_vpm, critic_recipe, observations_from_critic_lines
from .edge import TopLeftGate, TopLeftGateResult
from .hierarchy import HierarchyLevel, build_pyramid, reduce_blocks
from .learning import LEARNING_METRICS, LearningAssessment, LearningObservation, build_learning_vpm, learning_recipe
from .manifold import DecisionManifold, ManifoldFrame, ManifoldSummary, ManifoldTransition, build_decision_manifold, find_inflection_points
from .metrics import CANONICAL_METRICS, metric_ids_for_rows, pack_metrics, score_table_from_metric_rows
from .phos import PHOSResult, guarded_pack_artifact, image_entropy, pack_artifact, phos_sort_pack, robust01, to_square, top_left_concentration
from .render import png_bytes, svg_text, to_uint8, write_png, write_svg
from .spatial import SpatialOptimizationResult, SpatialOptimizer, build_optimized_view, optimize_view_profile
from .training import TRAINING_METRICS, TrainingCheckpoint, TrainingProgressAssessment, build_training_progress_vpm, training_progress_recipe
from .views import ViewProfile, ViewSet, build_view, build_views

__version__ = "2.0.0"

__all__ = [
    "CANONICAL_METRICS",
    "CRITIC_METRICS",
    "CriticAssessment",
    "CriticObservation",
    "Decision",
    "DecisionManifold",
    "HierarchyLevel",
    "LEARNING_METRICS",
    "LayoutRecipe",
    "LearningAssessment",
    "LearningObservation",
    "ManifoldFrame",
    "ManifoldSummary",
    "ManifoldTransition",
    "PHOSResult",
    "Policy",
    "ScoreTable",
    "Signal",
    "SpatialOptimizationResult",
    "SpatialOptimizer",
    "TRAINING_METRICS",
    "Thresholds",
    "TopLeftGate",
    "TopLeftGateResult",
    "TrainingCheckpoint",
    "TrainingProgressAssessment",
    "VPMArtifact",
    "VPMCell",
    "VPMComparison",
    "VPMController",
    "VPMRegion",
    "VPMRow",
    "ViewProfile",
    "ViewSet",
    "as_field",
    "build_critic_vpm",
    "build_decision_manifold",
    "build_learning_vpm",
    "build_optimized_view",
    "build_pyramid",
    "build_training_progress_vpm",
    "build_view",
    "build_views",
    "build_vpm",
    "compare_fields",
    "critic_recipe",
    "default_controller",
    "find_inflection_points",
    "from_bundle",
    "guarded_pack_artifact",
    "image_entropy",
    "learning_recipe",
    "metric_ids_for_rows",
    "observations_from_critic_lines",
    "optimize_view_profile",
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
    "training_progress_recipe",
    "vpm_add",
    "vpm_and",
    "vpm_not",
    "vpm_or",
    "vpm_subtract",
    "vpm_xor",
    "write_png",
    "write_svg",
]
