"""Learned temporal inference and held-out single-frame comparison.

P16C canonicalizes temporal training examples before every numerical operation. Caller
order therefore cannot change fitted coefficients or content-addressed model identity.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .fields import VPMFieldSchemaDTO, extract_source_fields, validate_source_for_schema
from .representation import DiscreteActionSchemaDTO, SourceVPMDTO
from .temporal import TemporalSourceVPMDTO, TemporalWindowSpecDTO
from .translator import (
    COEFFICIENT_SEMANTICS,
    SOURCE_FEATURE_SEMANTICS,
    TARGET_SCORE_SEMANTICS,
    PredictedTargetVPMDTO,
    SourceTargetTranslatorDTO,
    TargetActionScoreDTO,
    TranslatorConfigDTO,
    predict_target_vpm,
)

TEMPORAL_TRANSLATOR_VERSION: Final = "perception-temporal-translator/2"
TEMPORAL_PREDICTION_VERSION: Final = "perception-temporal-prediction/1"
TEMPORAL_COMPARISON_VERSION: Final = "perception-temporal-comparison/1"
TEMPORAL_FEATURE_SEMANTICS: Final = (
    "normalized_mean_intensity_per_declared_temporal_montage_field"
)
TEMPORAL_COMPARISON_SEMANTICS: Final = (
    "aligned_held_out_single_frame_vs_fixed_window_ridge_translation"
)
TEMPORAL_REJECTION_SEMANTICS: Final = (
    "reject_when_top_two_margin_below_declared_comparison_threshold"
)
TEMPORAL_FIT_ORDER_SEMANTICS: Final = "canonical_temporal_source_id_order_before_numerical_fit"


class PerceptionTemporalInferenceError(ValueError):
    """Raised when temporal fitting or comparison contracts are violated."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


def _feature_vector(
    source: SourceVPMDTO,
    schema: VPMFieldSchemaDTO,
    field_ids: tuple[str, ...],
) -> np.ndarray:
    validate_source_for_schema(source, schema)
    samples = {item.field_id: item for item in extract_source_fields(source, schema)}
    return np.asarray(
        [float(np.mean(samples[field_id].to_array())) / 255.0 for field_id in field_ids],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class TemporalTranslatorDTO:
    temporal_translator_id: str
    temporal_window_spec_id: str
    temporal_field_schema_id: str
    temporal_encoder_spec_id: str
    action_schema_id: str
    action_labels: tuple[str, ...]
    temporal_field_ids: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    training_split: str
    training_temporal_source_ids: tuple[str, ...]
    source_feature_semantics: str
    target_score_semantics: str
    coefficient_semantics: str
    config: TranslatorConfigDTO
    fit_order_semantics: str = TEMPORAL_FIT_ORDER_SEMANTICS
    version: str = TEMPORAL_TRANSLATOR_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.temporal_translator_id,
                self.temporal_window_spec_id,
                self.temporal_field_schema_id,
                self.temporal_encoder_spec_id,
                self.action_schema_id,
            )
        ):
            raise PerceptionTemporalInferenceError("temporal translator identities must be non-empty")
        if self.training_split not in {"train", "validation", "test", "all"}:
            raise PerceptionTemporalInferenceError("unsupported temporal training split")
        if self.action_labels != tuple(sorted(set(self.action_labels))):
            raise PerceptionTemporalInferenceError("action_labels must be unique and sorted")
        if self.temporal_field_ids != tuple(sorted(set(self.temporal_field_ids))):
            raise PerceptionTemporalInferenceError("temporal_field_ids must be unique and sorted")
        if self.training_temporal_source_ids != tuple(
            sorted(set(self.training_temporal_source_ids))
        ) or not self.training_temporal_source_ids:
            raise PerceptionTemporalInferenceError(
                "training_temporal_source_ids must be non-empty, unique, and sorted"
            )
        if len(self.coefficients) != len(self.action_labels):
            raise PerceptionTemporalInferenceError("one coefficient row is required per action")
        if len(self.intercepts) != len(self.action_labels):
            raise PerceptionTemporalInferenceError("one intercept is required per action")
        if any(len(row) != len(self.temporal_field_ids) for row in self.coefficients):
            raise PerceptionTemporalInferenceError("coefficient rows must match temporal fields")
        if self.source_feature_semantics != TEMPORAL_FEATURE_SEMANTICS:
            raise PerceptionTemporalInferenceError("unsupported temporal feature semantics")
        if self.target_score_semantics != TARGET_SCORE_SEMANTICS:
            raise PerceptionTemporalInferenceError("unsupported target score semantics")
        if self.coefficient_semantics != COEFFICIENT_SEMANTICS:
            raise PerceptionTemporalInferenceError("unsupported coefficient semantics")
        if self.fit_order_semantics != TEMPORAL_FIT_ORDER_SEMANTICS:
            raise PerceptionTemporalInferenceError("unsupported temporal fit-order semantics")
        values = np.asarray(self.coefficients, dtype=np.float64)
        intercepts = np.asarray(self.intercepts, dtype=np.float64)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(intercepts)):
            raise PerceptionTemporalInferenceError("temporal translator parameters must be finite")


@dataclass(frozen=True)
class TemporalPredictionDTO:
    prediction_id: str
    temporal_translator_id: str
    temporal_source_id: str
    target_interaction_id: str
    expected_action: str
    scores: tuple[TargetActionScoreDTO, ...]
    selected_action: str
    margin: float
    status: str
    rejection_threshold: float
    score_semantics: str = TARGET_SCORE_SEMANTICS
    rejection_semantics: str = TEMPORAL_REJECTION_SEMANTICS
    version: str = TEMPORAL_PREDICTION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.prediction_id,
                self.temporal_translator_id,
                self.temporal_source_id,
                self.target_interaction_id,
                self.expected_action,
                self.selected_action,
            )
        ):
            raise PerceptionTemporalInferenceError("temporal prediction identities must be non-empty")
        if self.status not in {"accepted", "rejected_ambiguous"}:
            raise PerceptionTemporalInferenceError("unsupported temporal prediction status")
        if not 0.0 <= self.margin <= 1.0:
            raise PerceptionTemporalInferenceError("margin must be in [0, 1]")
        if not 0.0 <= self.rejection_threshold <= 1.0:
            raise PerceptionTemporalInferenceError("rejection_threshold must be in [0, 1]")
        if not self.scores or self.scores[0].action_label != self.selected_action:
            raise PerceptionTemporalInferenceError("selected action must match top-ranked score")

    @property
    def correct(self) -> bool:
        return self.selected_action == self.expected_action


@dataclass(frozen=True)
class TemporalComparisonExampleDTO:
    interaction_id: str
    expected_action: str
    single_selected_action: str
    temporal_selected_action: str
    single_margin: float
    temporal_margin: float
    single_status: str
    temporal_status: str
    single_correct: bool
    temporal_correct: bool
    conflict_group: bool


@dataclass(frozen=True)
class TemporalInferenceComparisonReportDTO:
    report_id: str
    split: str
    single_translator_id: str
    temporal_translator_id: str
    temporal_window_spec_id: str
    example_count: int
    single_accuracy: float
    temporal_accuracy: float
    accuracy_improvement: float
    single_accepted_accuracy: float | None
    temporal_accepted_accuracy: float | None
    single_coverage: float
    temporal_coverage: float
    mean_single_margin: float
    mean_temporal_margin: float
    conflict_example_count: int
    conflict_single_accuracy: float | None
    conflict_temporal_accuracy: float | None
    conflict_resolution_improvement: float | None
    rejection_threshold: float
    examples: tuple[TemporalComparisonExampleDTO, ...]
    comparison_semantics: str = TEMPORAL_COMPARISON_SEMANTICS
    version: str = TEMPORAL_COMPARISON_VERSION

    def __post_init__(self) -> None:
        if not all((self.report_id, self.single_translator_id, self.temporal_translator_id)):
            raise PerceptionTemporalInferenceError("comparison identities must be non-empty")
        if self.split not in {"validation", "test"}:
            raise PerceptionTemporalInferenceError("comparison split must be validation or test")
        if self.example_count <= 0 or self.example_count != len(self.examples):
            raise PerceptionTemporalInferenceError("comparison example count is invalid")
        if self.examples != tuple(sorted(self.examples, key=lambda item: item.interaction_id)):
            raise PerceptionTemporalInferenceError("comparison examples must be sorted")
        for value in (
            self.single_accuracy,
            self.temporal_accuracy,
            self.single_coverage,
            self.temporal_coverage,
            self.mean_single_margin,
            self.mean_temporal_margin,
        ):
            if not 0.0 <= value <= 1.0:
                raise PerceptionTemporalInferenceError("bounded comparison metric outside [0, 1]")
        if not -1.0 <= self.accuracy_improvement <= 1.0:
            raise PerceptionTemporalInferenceError("accuracy improvement must be in [-1, 1]")


def fit_temporal_translator(
    temporal_sources: tuple[TemporalSourceVPMDTO, ...],
    temporal_window_spec: TemporalWindowSpecDTO,
    temporal_field_schema: VPMFieldSchemaDTO,
    action_schema: DiscreteActionSchemaDTO,
    *,
    training_split: str = "train",
    config: TranslatorConfigDTO | None = None,
) -> TemporalTranslatorDTO:
    """Fit a ridge translator after canonicalizing source order."""

    if training_split not in {"train", "validation", "test", "all"}:
        raise PerceptionTemporalInferenceError("unsupported temporal training split")
    if not temporal_sources:
        raise PerceptionTemporalInferenceError("temporal fitting requires examples")
    ordered_sources = tuple(sorted(temporal_sources, key=lambda item: item.temporal_source_id))
    source_ids = tuple(item.temporal_source_id for item in ordered_sources)
    if len(source_ids) != len(set(source_ids)):
        raise PerceptionTemporalInferenceError("temporal source identities must be unique")
    resolved = config or TranslatorConfigDTO()
    for item in ordered_sources:
        if item.temporal_window_spec_id != temporal_window_spec.temporal_window_spec_id:
            raise PerceptionTemporalInferenceError("temporal source window spec mismatch")
        if item.action_label not in action_schema.labels:
            raise PerceptionTemporalInferenceError("temporal source action outside schema")
        validate_source_for_schema(item.montage_source_vpm, temporal_field_schema)

    field_ids = tuple(sorted(field.field_id for field in temporal_field_schema.fields))
    x = np.vstack(
        [
            _feature_vector(item.montage_source_vpm, temporal_field_schema, field_ids)
            for item in ordered_sources
        ]
    )
    y = np.zeros((len(ordered_sources), len(action_schema.labels)), dtype=np.float64)
    for row_index, item in enumerate(ordered_sources):
        y[row_index, action_schema.index_of(item.action_label)] = 1.0
    design = np.column_stack((np.ones(x.shape[0], dtype=np.float64), x))
    regularizer = np.eye(design.shape[1], dtype=np.float64) * resolved.ridge_alpha
    regularizer[0, 0] = 0.0
    parameters = np.linalg.pinv(design.T @ design + regularizer) @ design.T @ y
    intercepts = tuple(float(value) for value in parameters[0, :])
    coefficients = tuple(
        tuple(float(value) for value in parameters[1:, action_index])
        for action_index in range(len(action_schema.labels))
    )
    payload: Mapping[str, object] = {
        "action_labels": list(action_schema.labels),
        "action_schema_id": action_schema.action_schema_id,
        "coefficient_semantics": COEFFICIENT_SEMANTICS,
        "coefficients": [list(row) for row in coefficients],
        "config": resolved.canonical_payload(),
        "fit_order_semantics": TEMPORAL_FIT_ORDER_SEMANTICS,
        "intercepts": list(intercepts),
        "source_feature_semantics": TEMPORAL_FEATURE_SEMANTICS,
        "target_score_semantics": TARGET_SCORE_SEMANTICS,
        "temporal_encoder_spec_id": temporal_field_schema.source_encoder_spec_id,
        "temporal_field_ids": list(field_ids),
        "temporal_field_schema_id": temporal_field_schema.field_schema_id,
        "temporal_window_spec_id": temporal_window_spec.temporal_window_spec_id,
        "training_split": training_split,
        "training_temporal_source_ids": list(source_ids),
        "version": TEMPORAL_TRANSLATOR_VERSION,
    }
    return TemporalTranslatorDTO(
        temporal_translator_id=_digest(_canonical_json(payload)),
        temporal_window_spec_id=temporal_window_spec.temporal_window_spec_id,
        temporal_field_schema_id=temporal_field_schema.field_schema_id,
        temporal_encoder_spec_id=temporal_field_schema.source_encoder_spec_id,
        action_schema_id=action_schema.action_schema_id,
        action_labels=action_schema.labels,
        temporal_field_ids=field_ids,
        coefficients=coefficients,
        intercepts=intercepts,
        training_split=training_split,
        training_temporal_source_ids=source_ids,
        source_feature_semantics=TEMPORAL_FEATURE_SEMANTICS,
        target_score_semantics=TARGET_SCORE_SEMANTICS,
        coefficient_semantics=COEFFICIENT_SEMANTICS,
        config=resolved,
    )


def predict_temporal_action(
    translator: TemporalTranslatorDTO,
    temporal_source: TemporalSourceVPMDTO,
    temporal_field_schema: VPMFieldSchemaDTO,
    *,
    rejection_threshold: float = 0.0,
) -> TemporalPredictionDTO:
    if not 0.0 <= rejection_threshold <= 1.0:
        raise PerceptionTemporalInferenceError("rejection_threshold must be in [0, 1]")
    if temporal_source.temporal_window_spec_id != translator.temporal_window_spec_id:
        raise PerceptionTemporalInferenceError("temporal source window spec mismatch")
    if temporal_field_schema.field_schema_id != translator.temporal_field_schema_id:
        raise PerceptionTemporalInferenceError("temporal field schema mismatch")
    vector = _feature_vector(
        temporal_source.montage_source_vpm,
        temporal_field_schema,
        translator.temporal_field_ids,
    )
    raw = np.asarray(translator.intercepts, dtype=np.float64) + (
        np.asarray(translator.coefficients, dtype=np.float64) @ vector
    )
    clipped = np.clip(raw, 0.0, 1.0)
    ranked = sorted(zip(translator.action_labels, clipped.tolist()), key=lambda item: (-item[1], item[0]))
    scores = tuple(
        TargetActionScoreDTO(action_label=label, score=float(score), rank=index + 1)
        for index, (label, score) in enumerate(ranked)
    )
    margin = scores[0].score - (scores[1].score if len(scores) > 1 else 0.0)
    status = "accepted" if margin >= rejection_threshold else "rejected_ambiguous"
    payload: Mapping[str, object] = {
        "expected_action": temporal_source.action_label,
        "margin": margin,
        "rejection_threshold": rejection_threshold,
        "scores": [[item.action_label, item.score, item.rank] for item in scores],
        "status": status,
        "target_interaction_id": temporal_source.target_interaction_id,
        "temporal_source_id": temporal_source.temporal_source_id,
        "temporal_translator_id": translator.temporal_translator_id,
        "version": TEMPORAL_PREDICTION_VERSION,
    }
    return TemporalPredictionDTO(
        prediction_id=_digest(_canonical_json(payload)),
        temporal_translator_id=translator.temporal_translator_id,
        temporal_source_id=temporal_source.temporal_source_id,
        target_interaction_id=temporal_source.target_interaction_id,
        expected_action=temporal_source.action_label,
        scores=scores,
        selected_action=scores[0].action_label,
        margin=margin,
        status=status,
        rejection_threshold=rejection_threshold,
    )


def _accepted_accuracy(correct: list[bool], accepted: list[bool]) -> float | None:
    retained = [value for value, keep in zip(correct, accepted) if keep]
    return None if not retained else sum(retained) / len(retained)


def compare_single_and_temporal_inference(
    single_translator: SourceTargetTranslatorDTO,
    temporal_translator: TemporalTranslatorDTO,
    single_field_schema: VPMFieldSchemaDTO,
    temporal_field_schema: VPMFieldSchemaDTO,
    temporal_sources: tuple[TemporalSourceVPMDTO, ...],
    current_sources: Mapping[str, SourceVPMDTO],
    action_schema: DiscreteActionSchemaDTO,
    *,
    split: str,
    rejection_threshold: float = 0.0,
    conflicting_current_pixel_digests: tuple[str, ...] = (),
) -> TemporalInferenceComparisonReportDTO:
    """Compare aligned held-out single-frame and temporal predictions."""

    if split not in {"validation", "test"}:
        raise PerceptionTemporalInferenceError("comparison split must be validation or test")
    if split in {single_translator.training_split, temporal_translator.training_split}:
        raise PerceptionTemporalInferenceError("comparison split must be held out from both translators")
    if single_translator.action_schema_id != action_schema.action_schema_id:
        raise PerceptionTemporalInferenceError("single translator action schema mismatch")
    if temporal_translator.action_schema_id != action_schema.action_schema_id:
        raise PerceptionTemporalInferenceError("temporal translator action schema mismatch")
    if not temporal_sources:
        raise PerceptionTemporalInferenceError("comparison requires temporal examples")

    conflict_set = set(conflicting_current_pixel_digests)
    rows: list[TemporalComparisonExampleDTO] = []
    single_correct: list[bool] = []
    temporal_correct: list[bool] = []
    single_accepted: list[bool] = []
    temporal_accepted: list[bool] = []
    single_margins: list[float] = []
    temporal_margins: list[float] = []

    for temporal_source in sorted(temporal_sources, key=lambda item: item.target_interaction_id):
        try:
            current = current_sources[temporal_source.current_source_vpm_id]
        except KeyError as exc:
            raise PerceptionTemporalInferenceError(
                f"missing current SourceVPMDTO for {temporal_source.current_source_vpm_id}"
            ) from exc
        single_prediction: PredictedTargetVPMDTO = predict_target_vpm(
            single_translator,
            current,
            single_field_schema,
            action_schema,
        )
        temporal_prediction = predict_temporal_action(
            temporal_translator,
            temporal_source,
            temporal_field_schema,
            rejection_threshold=rejection_threshold,
        )
        single_status = (
            "accepted" if single_prediction.margin >= rejection_threshold else "rejected_ambiguous"
        )
        expected = temporal_source.action_label
        single_is_correct = single_prediction.selected_action == expected
        temporal_is_correct = temporal_prediction.selected_action == expected
        rows.append(
            TemporalComparisonExampleDTO(
                interaction_id=temporal_source.target_interaction_id,
                expected_action=expected,
                single_selected_action=single_prediction.selected_action,
                temporal_selected_action=temporal_prediction.selected_action,
                single_margin=single_prediction.margin,
                temporal_margin=temporal_prediction.margin,
                single_status=single_status,
                temporal_status=temporal_prediction.status,
                single_correct=single_is_correct,
                temporal_correct=temporal_is_correct,
                conflict_group=temporal_source.current_pixel_digest in conflict_set,
            )
        )
        single_correct.append(single_is_correct)
        temporal_correct.append(temporal_is_correct)
        single_accepted.append(single_status == "accepted")
        temporal_accepted.append(temporal_prediction.status == "accepted")
        single_margins.append(single_prediction.margin)
        temporal_margins.append(temporal_prediction.margin)

    conflict_rows = [item for item in rows if item.conflict_group]
    conflict_single_accuracy = (
        None if not conflict_rows else sum(item.single_correct for item in conflict_rows) / len(conflict_rows)
    )
    conflict_temporal_accuracy = (
        None if not conflict_rows else sum(item.temporal_correct for item in conflict_rows) / len(conflict_rows)
    )
    conflict_improvement = (
        None
        if conflict_single_accuracy is None or conflict_temporal_accuracy is None
        else conflict_temporal_accuracy - conflict_single_accuracy
    )
    ordered_rows = tuple(sorted(rows, key=lambda item: item.interaction_id))
    single_accuracy = sum(single_correct) / len(single_correct)
    temporal_accuracy = sum(temporal_correct) / len(temporal_correct)
    payload: Mapping[str, object] = {
        "conflicting_current_pixel_digests": sorted(conflict_set),
        "examples": [item.__dict__ for item in ordered_rows],
        "rejection_threshold": rejection_threshold,
        "single_translator_id": single_translator.translator_id,
        "split": split,
        "temporal_translator_id": temporal_translator.temporal_translator_id,
        "temporal_window_spec_id": temporal_translator.temporal_window_spec_id,
        "version": TEMPORAL_COMPARISON_VERSION,
    }
    return TemporalInferenceComparisonReportDTO(
        report_id=_digest(_canonical_json(payload)),
        split=split,
        single_translator_id=single_translator.translator_id,
        temporal_translator_id=temporal_translator.temporal_translator_id,
        temporal_window_spec_id=temporal_translator.temporal_window_spec_id,
        example_count=len(ordered_rows),
        single_accuracy=single_accuracy,
        temporal_accuracy=temporal_accuracy,
        accuracy_improvement=temporal_accuracy - single_accuracy,
        single_accepted_accuracy=_accepted_accuracy(single_correct, single_accepted),
        temporal_accepted_accuracy=_accepted_accuracy(temporal_correct, temporal_accepted),
        single_coverage=sum(single_accepted) / len(single_accepted),
        temporal_coverage=sum(temporal_accepted) / len(temporal_accepted),
        mean_single_margin=float(np.mean(single_margins)),
        mean_temporal_margin=float(np.mean(temporal_margins)),
        conflict_example_count=len(conflict_rows),
        conflict_single_accuracy=conflict_single_accuracy,
        conflict_temporal_accuracy=conflict_temporal_accuracy,
        conflict_resolution_improvement=conflict_improvement,
        rejection_threshold=rejection_threshold,
        examples=ordered_rows,
    )
