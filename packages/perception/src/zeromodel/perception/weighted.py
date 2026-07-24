"""Evidence-weighted inference and deterministic controls for Stage P4C."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .dataset import PerceptionDatasetManifestDTO
from .evidence import EvidenceVPMDTO
from .fields import VPMFieldSchemaDTO, mask_source_fields, validate_source_for_schema
from .inference import (
    ActionCandidateDTO,
    BaselineInferenceConfigDTO,
    BaselinePredictionDTO,
    BaselineTrainingExampleDTO,
    NeighborEvidenceDTO,
)
from .representation import SourceVPMDTO

WEIGHTED_MODEL_VERSION: Final = "perception-evidence-weighted-model/1"
WEIGHTED_PREDICTION_VERSION: Final = "perception-evidence-weighted-prediction/1"
WEIGHTED_DISTANCE_SEMANTICS: Final = "field_relevance_weighted_normalized_mean_absolute_distance"
INTERVENTION_REPORT_VERSION: Final = "perception-evidence-intervention-report/1"


class PerceptionWeightedInferenceError(ValueError):
    """Raised when weighted inference or controls violate the P4C contract."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


def _source_pixels(source: SourceVPMDTO) -> bytes:
    return np.ascontiguousarray(source.to_array(), dtype=np.uint8).reshape(-1).tobytes()


@dataclass(frozen=True)
class EvidenceWeightedModelDTO:
    model_id: str
    dataset_id: str
    action_schema_id: str
    source_encoder_spec_id: str
    field_schema_id: str
    evidence_vpm_id: str
    width: int
    height: int
    channels: int
    action_labels: tuple[str, ...]
    field_weights: tuple[tuple[str, float], ...]
    examples: tuple[BaselineTrainingExampleDTO, ...]
    config: BaselineInferenceConfigDTO
    version: str = WEIGHTED_MODEL_VERSION

    def __post_init__(self) -> None:
        if not self.examples:
            raise PerceptionWeightedInferenceError("weighted model requires examples")
        if sum(weight for _, weight in self.field_weights) <= 0.0:
            raise PerceptionWeightedInferenceError("weighted model requires positive field weight")


@dataclass(frozen=True)
class InterventionOutcomeDTO:
    method: str
    field_ids: tuple[str, ...]
    selected_action: str | None
    status: str
    confidence: float
    nearest_distance: float


@dataclass(frozen=True)
class EvidenceInterventionReportDTO:
    report_id: str
    model_id: str
    source_vpm_id: str
    selected_field_ids: tuple[str, ...]
    random_field_ids: tuple[str, ...]
    full: InterventionOutcomeDTO
    keep_only: InterventionOutcomeDTO
    remove_only: InterventionOutcomeDTO
    random_keep: InterventionOutcomeDTO
    random_remove: InterventionOutcomeDTO
    keep_only_sufficiency: bool
    remove_only_necessity: bool
    random_control_separation: float
    version: str = INTERVENTION_REPORT_VERSION


def fit_evidence_weighted_nearest_neighbor(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    field_schema: VPMFieldSchemaDTO,
    evidence: EvidenceVPMDTO,
    *,
    config: BaselineInferenceConfigDTO | None = None,
    training_split: str = "train",
) -> EvidenceWeightedModelDTO:
    if evidence.dataset_id != manifest.dataset_id or evidence.field_schema_id != field_schema.field_schema_id:
        raise PerceptionWeightedInferenceError("evidence does not match dataset or field schema")
    if training_split not in {"train", "validation", "test", "all"}:
        raise PerceptionWeightedInferenceError("invalid training_split")
    selected = {a.interaction_id for a in manifest.split_assignments if training_split == "all" or a.split == training_split}
    interactions = [item for item in manifest.interactions if item.interaction_id in selected]
    if not interactions:
        raise PerceptionWeightedInferenceError("training split contains no interactions")
    examples: list[BaselineTrainingExampleDTO] = []
    for item in sorted(interactions, key=lambda value: value.interaction_id):
        try:
            source = source_vpms[item.source_vpm_id]
        except KeyError as exc:
            raise PerceptionWeightedInferenceError(f"missing SourceVPMDTO for {item.source_vpm_id}") from exc
        if source.pixel_digest != item.source_pixel_digest:
            raise PerceptionWeightedInferenceError("source pixel identity disagrees with interaction")
        validate_source_for_schema(source, field_schema)
        examples.append(BaselineTrainingExampleDTO(item.interaction_id, source.source_vpm_id, source.pixel_digest, item.action_label, source.width, source.height, source.channels, _source_pixels(source)))
    weights = tuple(sorted(((item.field_id, item.score) for item in evidence.relevances), key=lambda value: value[0]))
    if {field.field_id for field in field_schema.fields} != {field_id for field_id, _ in weights}:
        raise PerceptionWeightedInferenceError("evidence must contain exactly one score per field")
    resolved = config or BaselineInferenceConfigDTO()
    labels = tuple(sorted({item.action_label for item in examples}))
    payload = {
        "dataset_id": manifest.dataset_id,
        "evidence_vpm_id": evidence.evidence_vpm_id,
        "field_schema_id": field_schema.field_schema_id,
        "field_weights": list(weights),
        "training_split": training_split,
        "config": resolved.canonical_payload(),
        "examples": [item.interaction_id for item in examples],
        "version": WEIGHTED_MODEL_VERSION,
    }
    return EvidenceWeightedModelDTO(_digest(_canonical_json(payload)), manifest.dataset_id, manifest.action_schema_id, field_schema.source_encoder_spec_id, field_schema.field_schema_id, evidence.evidence_vpm_id, field_schema.width, field_schema.height, field_schema.channels, labels, weights, tuple(examples), resolved)


def _array_from_example(example: BaselineTrainingExampleDTO) -> np.ndarray:
    shape = (example.height, example.width) if example.channels == 1 else (example.height, example.width, example.channels)
    return np.frombuffer(example.pixels, dtype=np.uint8).reshape(shape)


def _weighted_distance(left: np.ndarray, right: np.ndarray, schema: VPMFieldSchemaDTO, weights: Mapping[str, float]) -> float:
    left3 = left.reshape(schema.height, schema.width, schema.channels)
    right3 = right.reshape(schema.height, schema.width, schema.channels)
    total_weight = 0.0
    total = 0.0
    for field in schema.fields:
        weight = weights[field.field_id]
        if weight <= 0.0:
            continue
        region_left = left3[field.y0:field.y1, field.x0:field.x1, field.channel_start:field.channel_end].astype(np.int16)
        region_right = right3[field.y0:field.y1, field.x0:field.x1, field.channel_start:field.channel_end].astype(np.int16)
        total += weight * float(np.mean(np.abs(region_left - region_right)) / 255.0)
        total_weight += weight
    if total_weight <= 0.0:
        raise PerceptionWeightedInferenceError("no positive evidence weight")
    return total / total_weight


def _predict_array(model: EvidenceWeightedModelDTO, source_id: str, array: np.ndarray, schema: VPMFieldSchemaDTO) -> BaselinePredictionDTO:
    weights = dict(model.field_weights)
    ranked = sorted(((_weighted_distance(array, _array_from_example(item), schema, weights), item.interaction_id, item) for item in model.examples), key=lambda value: (value[0], value[1]))[:min(model.config.neighbor_count, len(model.examples))]
    raw = {label: 0.0 for label in model.action_labels}
    counts = {label: 0 for label in model.action_labels}
    nearest = {label: 1.0 for label in model.action_labels}
    neighbors: list[NeighborEvidenceDTO] = []
    for rank, (distance, _, item) in enumerate(ranked, start=1):
        weight = 1.0 / (model.config.epsilon + distance)
        raw[item.action_label] += weight
        counts[item.action_label] += 1
        nearest[item.action_label] = min(nearest[item.action_label], distance)
        neighbors.append(NeighborEvidenceDTO(item.interaction_id, item.source_vpm_id, item.action_label, distance, weight, rank))
    total = sum(raw.values())
    candidates = tuple(sorted((ActionCandidateDTO(label, raw[label] / total, counts[label], nearest[label]) for label in model.action_labels), key=lambda item: (-item.score, item.action_label)))
    best = candidates[0]
    margin = best.score - (candidates[1].score if len(candidates) > 1 else 0.0)
    nearest_distance = neighbors[0].distance
    if nearest_distance > model.config.maximum_distance:
        status, selected = "rejected_out_of_distribution", None
    elif margin < model.config.minimum_margin:
        status, selected = "rejected_ambiguous", None
    else:
        status, selected = "accepted", best.action_label
    payload = {"model_id": model.model_id, "source_vpm_id": source_id, "selected_action": selected, "status": status, "nearest_distance": nearest_distance, "margin": margin, "candidates": [(c.action_label, c.score) for c in candidates], "version": WEIGHTED_PREDICTION_VERSION}
    return BaselinePredictionDTO(_digest(_canonical_json(payload)), model.model_id, source_id, selected, status, best.score, "winning_inverse_distance_weight_share", WEIGHTED_DISTANCE_SEMANTICS, nearest_distance, margin, candidates, tuple(neighbors), WEIGHTED_PREDICTION_VERSION)


def predict_evidence_weighted_action(model: EvidenceWeightedModelDTO, source: SourceVPMDTO, field_schema: VPMFieldSchemaDTO) -> BaselinePredictionDTO:
    validate_source_for_schema(source, field_schema)
    if field_schema.field_schema_id != model.field_schema_id:
        raise PerceptionWeightedInferenceError("field schema does not match model")
    return _predict_array(model, source.source_vpm_id, source.to_array(), field_schema)


def _outcome(method: str, field_ids: tuple[str, ...], prediction: BaselinePredictionDTO) -> InterventionOutcomeDTO:
    return InterventionOutcomeDTO(method, field_ids, prediction.selected_action, prediction.status, prediction.confidence, prediction.nearest_distance)


def evaluate_evidence_interventions(
    model: EvidenceWeightedModelDTO,
    source: SourceVPMDTO,
    field_schema: VPMFieldSchemaDTO,
    *,
    selected_field_count: int = 1,
    neutral_value: int = 0,
    random_seed: int = 0,
) -> EvidenceInterventionReportDTO:
    if selected_field_count <= 0 or selected_field_count > len(field_schema.fields):
        raise PerceptionWeightedInferenceError("selected_field_count is out of range")
    ranked_ids = tuple(field_id for field_id, _ in sorted(model.field_weights, key=lambda value: (-value[1], value[0])))
    selected = ranked_ids[:selected_field_count]
    remaining = tuple(field_id for field_id in ranked_ids if field_id not in selected)
    rng = np.random.default_rng(random_seed)
    random_ids = tuple(sorted(rng.choice(np.asarray(remaining if len(remaining) >= selected_field_count else ranked_ids, dtype=object), size=selected_field_count, replace=False).tolist()))
    full = predict_evidence_weighted_action(model, source, field_schema)
    arrays = {
        "keep_only": mask_source_fields(source, field_schema, selected, mode="keep", neutral_value=neutral_value),
        "remove_only": mask_source_fields(source, field_schema, selected, mode="remove", neutral_value=neutral_value),
        "random_keep": mask_source_fields(source, field_schema, random_ids, mode="keep", neutral_value=neutral_value),
        "random_remove": mask_source_fields(source, field_schema, random_ids, mode="remove", neutral_value=neutral_value),
    }
    predictions = {name: _predict_array(model, f"{source.source_vpm_id}:{name}", array, field_schema) for name, array in arrays.items()}
    keep_sufficient = full.selected_action is not None and predictions["keep_only"].selected_action == full.selected_action
    remove_necessary = full.selected_action is not None and predictions["remove_only"].selected_action != full.selected_action
    separation = predictions["keep_only"].confidence - predictions["random_keep"].confidence
    payload = {"model_id": model.model_id, "source_vpm_id": source.source_vpm_id, "selected": list(selected), "random": list(random_ids), "keep_sufficient": keep_sufficient, "remove_necessary": remove_necessary, "separation": separation, "version": INTERVENTION_REPORT_VERSION}
    return EvidenceInterventionReportDTO(_digest(_canonical_json(payload)), model.model_id, source.source_vpm_id, selected, random_ids, _outcome("full", (), full), _outcome("keep_only", selected, predictions["keep_only"]), _outcome("remove_only", selected, predictions["remove_only"]), _outcome("random_keep", random_ids, predictions["random_keep"]), _outcome("random_remove", random_ids, predictions["random_remove"]), keep_sufficient, remove_necessary, separation)
