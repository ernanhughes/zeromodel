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
from .inference import (
    BASELINE_MODEL_VERSION, CONFIDENCE_SEMANTICS, DISTANCE_SEMANTICS,
    PREDICTION_VERSION, ActionCandidateDTO, BaselineInferenceConfigDTO,
    BaselineNearestNeighborModelDTO, BaselinePredictionDTO,
    BaselineTrainingExampleDTO, NeighborEvidenceDTO, PerceptionInferenceError,
    fit_baseline_nearest_neighbor, predict_baseline_action,
)
from .representation import (
    ACTION_SCHEMA_VERSION, SOURCE_ENCODER_VERSION, TARGET_ENCODER_VERSION,
    DiscreteActionSchemaDTO, PerceptionRepresentationError,
    SourceImageEncoderSpecDTO, SourceVPMDTO, TargetVPMDTO,
    decode_discrete_action, encode_discrete_action, encode_source_array,
    encode_source_image_bytes,
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
PERCEPTION_STAGE = "P8"

__all__ = [name for name in globals() if not name.startswith("_") and name not in {"annotations"}]
