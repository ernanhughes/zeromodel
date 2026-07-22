"""ZeroModel analysis public API."""

from __future__ import annotations

from .adapters.common import (
    checkpoints_from_export,
    load_tracker_records,
    records_to_checkpoints,
)
from .adapters.jsonl import (
    checkpoints_from_csv,
    checkpoints_from_json,
    checkpoints_from_jsonl,
)
from .adapters.tensorboard import (
    TENSORBOARD_DEFAULT_ALIASES,
    checkpoints_from_tensorboard_scalars,
)
from .adapters.trackio import (
    TRACKIO_DEFAULT_ALIASES,
    checkpoints_from_trackio_export,
)
from .adapters.wandb import (
    WANDB_DEFAULT_ALIASES,
    checkpoints_from_wandb_export,
)
from .compare import (
    VPMComparison,
    compare_fields,
)
from .compose import (
    as_field,
    vpm_add,
    vpm_and,
    vpm_not,
    vpm_or,
    vpm_subtract,
    vpm_xor,
)
from .controller import (
    Decision,
    Policy,
    Signal,
    Thresholds,
    VPMController,
    VPMRow,
    default_controller,
)
from .critic import (
    CriticAssessment,
    CriticObservation,
    build_critic_vpm,
    critic_recipe,
    observations_from_critic_lines,
)
from .edge import (
    TopLeftGate,
    TopLeftGateResult,
)
from .hierarchy import (
    HierarchyLevel,
    build_pyramid,
    reduce_blocks,
)
from .learning import (
    LearningAssessment,
    LearningObservation,
    VALID_SPLITS,
    build_learning_vpm,
    learning_recipe,
)
from .manifold import (
    DecisionManifold,
    ManifoldFrame,
    ManifoldSummary,
    ManifoldTransition,
    build_decision_manifold,
    find_inflection_points,
)
from .patterns import (
    MatrixPatternDetector,
    OBJECTIVE_IDS,
    ObjectiveResult,
    PATTERN_CHECKER_VERSION,
    PATTERN_METHOD,
    PatternAnalysisSpec,
    PatternDiscoveryArtifacts,
    PatternReport,
    REPORT_METRICS,
    build_discovered_view,
    detect_patterns,
)
from .phos import (
    PHOSResult,
    guarded_pack_artifact,
    image_entropy,
    pack_artifact,
    phos_sort_pack,
    robust01,
    to_square,
    top_left_concentration,
)
from .policy_diagnostics import (
    CRITICALITY_METRIC_ID,
    DECISION_MARGIN_METRIC_ID,
    with_q_diagnostics,
)
from .policy_properties import (
    CHECKER_VERSION,
    PolicyPropertyChecker,
    PolicyPropertyResult,
    PolicyPropertySpec,
    PolicyPropertyViolation,
    PolicyVerificationReport,
    VERIFICATION_METRICS,
    decode_key_value_row_id,
)
from .spatial import (
    SpatialOptimizationResult,
    SpatialOptimizer,
    build_optimized_view,
    optimize_view_profile,
)
from .training import (
    TrainingCheckpoint,
    TrainingProgressAssessment,
    build_training_progress_vpm,
    training_progress_recipe,
)

__all__ = [
    "CHECKER_VERSION",
    "CRITICALITY_METRIC_ID",
    "CriticAssessment",
    "CriticObservation",
    "DECISION_MARGIN_METRIC_ID",
    "Decision",
    "DecisionManifold",
    "HierarchyLevel",
    "LearningAssessment",
    "LearningObservation",
    "ManifoldFrame",
    "ManifoldSummary",
    "ManifoldTransition",
    "MatrixPatternDetector",
    "OBJECTIVE_IDS",
    "ObjectiveResult",
    "PATTERN_CHECKER_VERSION",
    "PATTERN_METHOD",
    "PHOSResult",
    "PatternAnalysisSpec",
    "PatternDiscoveryArtifacts",
    "PatternReport",
    "Policy",
    "PolicyPropertyChecker",
    "PolicyPropertyResult",
    "PolicyPropertySpec",
    "PolicyPropertyViolation",
    "PolicyVerificationReport",
    "REPORT_METRICS",
    "Signal",
    "SpatialOptimizationResult",
    "SpatialOptimizer",
    "TENSORBOARD_DEFAULT_ALIASES",
    "TRACKIO_DEFAULT_ALIASES",
    "Thresholds",
    "TopLeftGate",
    "TopLeftGateResult",
    "TrainingCheckpoint",
    "TrainingProgressAssessment",
    "VALID_SPLITS",
    "VERIFICATION_METRICS",
    "VPMComparison",
    "VPMController",
    "VPMRow",
    "WANDB_DEFAULT_ALIASES",
    "as_field",
    "build_critic_vpm",
    "build_decision_manifold",
    "build_discovered_view",
    "build_learning_vpm",
    "build_optimized_view",
    "build_pyramid",
    "build_training_progress_vpm",
    "checkpoints_from_csv",
    "checkpoints_from_export",
    "checkpoints_from_json",
    "checkpoints_from_jsonl",
    "checkpoints_from_tensorboard_scalars",
    "checkpoints_from_trackio_export",
    "checkpoints_from_wandb_export",
    "compare_fields",
    "critic_recipe",
    "decode_key_value_row_id",
    "default_controller",
    "detect_patterns",
    "find_inflection_points",
    "guarded_pack_artifact",
    "image_entropy",
    "learning_recipe",
    "load_tracker_records",
    "observations_from_critic_lines",
    "optimize_view_profile",
    "pack_artifact",
    "phos_sort_pack",
    "records_to_checkpoints",
    "reduce_blocks",
    "robust01",
    "to_square",
    "top_left_concentration",
    "training_progress_recipe",
    "vpm_add",
    "vpm_and",
    "vpm_not",
    "vpm_or",
    "vpm_subtract",
    "vpm_xor",
    "with_q_diagnostics",
]
