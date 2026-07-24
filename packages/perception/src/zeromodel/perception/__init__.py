"""ZeroModel perception public API."""

from __future__ import annotations

from .dataset import (
    DATASET_MANIFEST_VERSION, INTERACTION_VERSION, SPLIT_ASSIGNMENT_VERSION,
    DatasetFindingDTO, InMemoryPerceptionDatasetStore, PerceptionDatasetError,
    PerceptionDatasetManifestDTO, PerceptionDatasetStore, RecordedInteractionDTO,
    SplitAssignmentDTO, build_dataset_manifest,
)
from .discovery import (
    DIFFERENCE_SURFACE_SEMANTICS, DISCOVERY_VERSION,
    DISCREPANCY_VPM_VERSION, EXPECTED_SURFACE_SEMANTICS,
    OBSERVED_SURFACE_SEMANTICS, UNEXPLAINED_EVIDENCE_VERSION,
    UNEXPLAINED_SURFACE_SEMANTICS, EvidenceDiscoveryReportDTO,
    EvidenceDiscrepancyVPMDTO, PerceptionDiscoveryError,
    UnexplainedEvidenceDTO, discover_unexpected_evidence,
)
from .evidence import (
    EVIDENCE_RENDER_SEMANTICS, EVIDENCE_VPM_VERSION, FIELD_RELEVANCE_SEMANTICS,
    FIELD_RELEVANCE_VERSION, EvidenceVPMDTO, FieldRelevanceDTO,
    PerceptionEvidenceError, estimate_field_relevance,
)
from .expectations import (
    ANNOTATION_VERSION, CONFORMANCE_REPORT_VERSION, CONFORMANCE_STATUSES,
    EXPECTATION_VERSION, OBSERVED_REGISTRATION_VERSION,
    REGISTRATION_SEMANTICS, RELATION_ANNOTATION_VERSION,
    UNEXPLAINED_REGISTRATION_SEMANTICS, EvidenceConformanceFindingDTO,
    EvidenceConformanceReportDTO, EvidenceExpectationDTO,
    ObservedAnnotationRegistrationDTO, PerceptionConformanceError,
    PerceptionRegionAnnotationDTO, RelationAnnotationDTO,
    evaluate_evidence_conformance,
)
from .fields import (
    FIELD_SAMPLE_VERSION, FIELD_SCHEMA_VERSION, ExtractedFieldDTO,
    PerceptionFieldError, VPMFieldAddressDTO, VPMFieldSchemaDTO,
    build_grid_field_schema, extract_source_fields, mask_source_fields,
    reconstruct_source_array, validate_source_for_schema,
)
from .health import (
    OPERATIONAL_DRIFT_POLICY_VERSION, OPERATIONAL_HEALTH_FINDING_VERSION,
    OPERATIONAL_HEALTH_METRICS, OPERATIONAL_HEALTH_REPORT_VERSION,
    OPERATIONAL_HEALTH_SEMANTICS, OPERATIONAL_HEALTH_STATUSES,
    OPERATIONAL_REFERENCE_PROFILE_VERSION, OPERATIONAL_REFERENCE_SEMANTICS,
    ActionFrequencyDTO, OperationalDriftPolicyDTO,
    OperationalHealthFindingDTO, OperationalHealthReportDTO,
    OperationalReferenceProfileDTO, PerceptionOperationalHealthError,
    build_operational_reference_profile, diagnose_operational_health,
)
from .inference import (
    BASELINE_MODEL_VERSION, CONFIDENCE_SEMANTICS, DISTANCE_SEMANTICS,
    PREDICTION_VERSION, ActionCandidateDTO, BaselineInferenceConfigDTO,
    BaselineNearestNeighborModelDTO, BaselinePredictionDTO,
    BaselineTrainingExampleDTO, NeighborEvidenceDTO, PerceptionInferenceError,
    fit_baseline_nearest_neighbor, predict_baseline_action,
)
from .lifecycle import (
    ACTIVE_MODEL_POINTER_VERSION, ACTIVE_POINTER_SEMANTICS,
    MODEL_LEDGER_ENTRY_VERSION, MODEL_LEDGER_SEMANTICS,
    MODEL_LIFECYCLE_SNAPSHOT_VERSION, MODEL_TRANSITION_KINDS,
    MODEL_TRANSITION_SEMANTICS, MODEL_TRANSITION_VERSION,
    ActiveModelPointerDTO, InMemoryPerceptionModelLifecycleStore,
    ModelLifecycleSnapshotDTO, ModelLifecycleTransitionDTO,
    PerceptionModelLifecycleError, PerceptionModelLifecycleStore,
    PromotedModelLedgerEntryDTO, activate_promoted_model,
    build_model_lifecycle_snapshot, deactivate_active_model,
    register_promoted_model, resolve_active_promoted_model,
    rollback_active_model, supersede_active_model,
)
from .production import (
    PRODUCTION_INFERENCE_RECORD_VERSION, PRODUCTION_INFERENCE_SEMANTICS,
    PRODUCTION_METRICS_REPORT_VERSION, PRODUCTION_METRICS_SEMANTICS,
    PRODUCTION_OUTCOME_RECORD_VERSION, PRODUCTION_OUTCOME_SEMANTICS,
    InMemoryPerceptionProductionLedgerStore, PerceptionProductionLedgerError,
    PerceptionProductionLedgerStore, ProductionInferenceRecordDTO,
    ProductionMetricsReportDTO, ProductionOutcomeRecordDTO,
    build_production_metrics_report, record_production_inference,
    record_production_outcome,
)
from .promoted_inference import (
    PROMOTED_INFERENCE_SEMANTICS, PROMOTED_INFERENCE_VERSION,
    PROMOTED_TEST_EVALUATION_SEMANTICS, PROMOTED_TEST_EVALUATION_VERSION,
    PerceptionPromotedInferenceError, PromotedInferenceResultDTO,
    PromotedTestEvaluationReportDTO, PromotedTestExampleDTO,
    evaluate_promoted_model_on_test, run_promoted_inference,
)
from .promotion import (
    CALIBRATION_SEMANTICS, MODEL_CALIBRATION_VERSION, MODEL_PROMOTION_VERSION,
    PROMOTED_MODEL_KINDS, PROMOTION_DECISION_VERSION, PROMOTION_POLICY_VERSION,
    PROMOTION_SEMANTICS, ModelCalibrationDTO, PerceptionPromotionError,
    PromotedPerceptionModelDTO, PromotionDecisionDTO, PromotionPolicyDTO,
    calibrate_comparison_candidates, promote_perception_model,
)
from .representation import (
    ACTION_SCHEMA_VERSION, SOURCE_ENCODER_VERSION, TARGET_ENCODER_VERSION,
    DiscreteActionSchemaDTO, PerceptionRepresentationError,
    SourceImageEncoderSpecDTO, SourceVPMDTO, TargetVPMDTO,
    decode_discrete_action, encode_discrete_action, encode_source_array,
    encode_source_image_bytes,
)
from .sql_lifecycle import (
    SQL_LIFECYCLE_SCHEMA_VERSION, SQL_LIFECYCLE_SEMANTICS,
    SQL_LIFECYCLE_STORE_VERSION, PerceptionSqlLifecycleError,
    SqlitePerceptionModelLifecycleStore,
)
from .sql_production import (
    SQL_PRODUCTION_SCHEMA_VERSION, SQL_PRODUCTION_SEMANTICS,
    SQL_PRODUCTION_STORE_VERSION, PerceptionSqlProductionError,
    SqlitePerceptionProductionLedgerStore,
)
from .temporal import (
    TEMPORAL_DIAGNOSIS_SEMANTICS, TEMPORAL_DIAGNOSIS_STATUSES,
    TEMPORAL_DIAGNOSIS_VERSION, TEMPORAL_LAYOUT_SEMANTICS,
    TEMPORAL_SOURCE_VERSION, TEMPORAL_WINDOW_SPEC_VERSION,
    PerceptionTemporalError, TemporalConflictGroupDTO,
    TemporalSourceVPMDTO, TemporalStateDiagnosisReportDTO,
    TemporalWindowSpecDTO, build_temporal_source_vpms,
    diagnose_temporal_state_completeness,
)
from .temporal_inference import (
    TEMPORAL_COMPARISON_SEMANTICS, TEMPORAL_COMPARISON_VERSION,
    TEMPORAL_FEATURE_SEMANTICS, TEMPORAL_PREDICTION_VERSION,
    TEMPORAL_REJECTION_SEMANTICS, TEMPORAL_TRANSLATOR_VERSION,
    PerceptionTemporalInferenceError, TemporalComparisonExampleDTO,
    TemporalInferenceComparisonReportDTO, TemporalPredictionDTO,
    TemporalTranslatorDTO, compare_single_and_temporal_inference,
    fit_temporal_translator, predict_temporal_action,
)
from .translator import (
    COEFFICIENT_SEMANTICS, SOURCE_FEATURE_SEMANTICS, TARGET_SCORE_SEMANTICS,
    TRANSLATOR_PREDICTION_VERSION, TRANSLATOR_VERSION,
    PerceptionTranslatorError, PredictedTargetVPMDTO, SourceTargetTranslatorDTO,
    TargetActionScoreDTO, TranslatorConfigDTO, fit_source_target_translator,
    predict_target_vpm,
)
from .translator_evaluation import (
    RECONSTRUCTION_ERROR_SEMANTICS, REJECTED_TRANSLATOR_PREDICTION_VERSION,
    REJECTION_SEMANTICS, SPARSITY_SEMANTICS, TRANSLATOR_CALIBRATION_VERSION,
    TRANSLATOR_EVALUATION_VERSION, CalibratedTranslatorPredictionDTO,
    PerceptionTranslatorEvaluationError, TranslatorCalibrationDTO,
    TranslatorEvaluationReportDTO, TranslatorExampleEvaluationDTO,
    calibrate_translator_rejection, evaluate_source_target_translator,
    predict_calibrated_target_vpm,
)
from .weighted import (
    INTERVENTION_REPORT_VERSION, WEIGHTED_DISTANCE_SEMANTICS,
    WEIGHTED_MODEL_VERSION, WEIGHTED_PREDICTION_VERSION,
    EvidenceInterventionReportDTO, EvidenceWeightedModelDTO,
    InterventionOutcomeDTO, PerceptionWeightedInferenceError,
    evaluate_evidence_interventions, fit_evidence_weighted_nearest_neighbor,
    predict_evidence_weighted_action,
)

PERCEPTION_PACKAGE_VERSION = "1.0.13"
PERCEPTION_STAGE = "P16"

__all__ = [name for name in globals() if not name.startswith("_") and name not in {"annotations"}]
