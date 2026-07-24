"""ZeroModel perception public API."""

from __future__ import annotations

from .dataset import (
    DATASET_MANIFEST_VERSION, INTERACTION_VERSION, SPLIT_ASSIGNMENT_VERSION,
    DatasetFindingDTO, InMemoryPerceptionDatasetStore, PerceptionDatasetError,
    PerceptionDatasetManifestDTO, PerceptionDatasetStore, RecordedInteractionDTO,
    SplitAssignmentDTO, build_dataset_manifest,
)
from .evidence import (
    EVIDENCE_RENDER_SEMANTICS, EVIDENCE_VPM_VERSION, FIELD_RELEVANCE_SEMANTICS,
    FIELD_RELEVANCE_VERSION, EvidenceVPMDTO, FieldRelevanceDTO,
    PerceptionEvidenceError, estimate_field_relevance,
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
from .translator import (
    COEFFICIENT_SEMANTICS,
    SOURCE_FEATURE_SEMANTICS,
    TARGET_SCORE_SEMANTICS,
    TRANSLATOR_PREDICTION_VERSION,
    TRANSLATOR_VERSION,
    PerceptionTranslatorError,
    PredictedTargetVPMDTO,
    SourceTargetTranslatorDTO,
    TargetActionScoreDTO,
    TranslatorConfigDTO,
    fit_source_target_translator,
    predict_target_vpm,
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
PERCEPTION_STAGE = "P5A"

__all__ = [name for name in globals() if not name.startswith("_") and name not in {"annotations"}]
