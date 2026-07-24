"""Held-out translator evaluation, calibration, rejection, and sparsity diagnostics.

Calibration consumes a declared calibration split. Final evaluation must use a different
split; this module rejects accidental reuse of the translator training split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .dataset import PerceptionDatasetManifestDTO
from .fields import VPMFieldSchemaDTO
from .representation import DiscreteActionSchemaDTO, SourceVPMDTO
from .translator import (
    PredictedTargetVPMDTO,
    SourceTargetTranslatorDTO,
    predict_target_vpm,
)

TRANSLATOR_EVALUATION_VERSION: Final = "perception-translator-evaluation/1"
TRANSLATOR_CALIBRATION_VERSION: Final = "perception-translator-calibration/1"
REJECTED_TRANSLATOR_PREDICTION_VERSION: Final = "perception-calibrated-target-prediction/1"
RECONSTRUCTION_ERROR_SEMANTICS: Final = "mean_absolute_error_against_one_hot_target"
SPARSITY_SEMANTICS: Final = "coefficient_absolute_value_at_or_below_threshold"
REJECTION_SEMANTICS: Final = "reject_when_top_two_margin_below_calibrated_threshold"


class PerceptionTranslatorEvaluationError(ValueError):
    """Raised when held-out evaluation or calibration violates the P5B contract."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class TranslatorExampleEvaluationDTO:
    interaction_id: str
    source_vpm_id: str
    expected_action: str
    predicted_action: str
    correct: bool
    margin: float
    reconstruction_error: float
    prediction_id: str


@dataclass(frozen=True)
class TranslatorEvaluationReportDTO:
    report_id: str
    translator_id: str
    dataset_id: str
    evaluation_split: str
    example_count: int
    correct_count: int
    accuracy: float
    mean_reconstruction_error: float
    mean_margin: float
    minimum_margin: float
    maximum_margin: float
    coefficient_count: int
    near_zero_coefficient_count: int
    sparsity_ratio: float
    sparsity_threshold: float
    reconstruction_error_semantics: str
    sparsity_semantics: str
    examples: tuple[TranslatorExampleEvaluationDTO, ...]
    version: str = TRANSLATOR_EVALUATION_VERSION

    def __post_init__(self) -> None:
        if self.example_count <= 0 or self.correct_count < 0:
            raise PerceptionTranslatorEvaluationError("evaluation counts are invalid")
        if not 0.0 <= self.accuracy <= 1.0 or not 0.0 <= self.sparsity_ratio <= 1.0:
            raise PerceptionTranslatorEvaluationError("evaluation ratios must be in [0, 1]")


@dataclass(frozen=True)
class TranslatorCalibrationDTO:
    calibration_id: str
    translator_id: str
    dataset_id: str
    calibration_split: str
    minimum_margin: float
    retained_accuracy: float
    retained_count: int
    rejected_count: int
    rejection_semantics: str
    version: str = TRANSLATOR_CALIBRATION_VERSION


@dataclass(frozen=True)
class CalibratedTranslatorPredictionDTO:
    prediction_id: str
    translator_id: str
    source_vpm_id: str
    selected_action: str | None
    status: str
    margin: float
    minimum_margin: float
    raw_prediction: PredictedTargetVPMDTO
    rejection_semantics: str
    version: str = REJECTED_TRANSLATOR_PREDICTION_VERSION


def _split_interactions(manifest: PerceptionDatasetManifestDTO, split: str):
    if split not in {"train", "validation", "test"}:
        raise PerceptionTranslatorEvaluationError("split must be train, validation, or test")
    selected = {item.interaction_id for item in manifest.split_assignments if item.split == split}
    interactions = tuple(item for item in manifest.interactions if item.interaction_id in selected)
    if not interactions:
        raise PerceptionTranslatorEvaluationError(f"dataset contains no {split!r} interactions")
    return interactions


def evaluate_source_target_translator(
    translator: SourceTargetTranslatorDTO,
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    source_field_schema: VPMFieldSchemaDTO,
    action_schema: DiscreteActionSchemaDTO,
    *,
    evaluation_split: str = "validation",
    sparsity_threshold: float = 1e-9,
) -> TranslatorEvaluationReportDTO:
    """Evaluate target reconstruction and action decoding on a non-training split."""
    if evaluation_split == translator.training_split:
        raise PerceptionTranslatorEvaluationError("evaluation split must differ from translator training split")
    if sparsity_threshold < 0.0 or not np.isfinite(sparsity_threshold):
        raise PerceptionTranslatorEvaluationError("sparsity_threshold must be finite and non-negative")
    interactions = _split_interactions(manifest, evaluation_split)
    records: list[TranslatorExampleEvaluationDTO] = []
    for interaction in interactions:
        try:
            source = source_vpms[interaction.source_vpm_id]
        except KeyError as exc:
            raise PerceptionTranslatorEvaluationError(f"missing SourceVPMDTO for {interaction.source_vpm_id}") from exc
        if source.pixel_digest != interaction.source_pixel_digest:
            raise PerceptionTranslatorEvaluationError("source pixel identity disagrees with interaction")
        prediction = predict_target_vpm(translator, source, source_field_schema, action_schema)
        expected = np.zeros(len(action_schema.labels), dtype=np.float64)
        expected[action_schema.index_of(interaction.action_label)] = 1.0
        ordered_scores = np.asarray([
            next(item.score for item in prediction.scores if item.action_label == label)
            for label in action_schema.labels
        ], dtype=np.float64)
        error = float(np.mean(np.abs(ordered_scores - expected)))
        records.append(TranslatorExampleEvaluationDTO(
            interaction_id=interaction.interaction_id,
            source_vpm_id=source.source_vpm_id,
            expected_action=interaction.action_label,
            predicted_action=prediction.selected_action,
            correct=prediction.selected_action == interaction.action_label,
            margin=prediction.margin,
            reconstruction_error=error,
            prediction_id=prediction.prediction_id,
        ))
    ordered = tuple(sorted(records, key=lambda item: item.interaction_id))
    margins = np.asarray([item.margin for item in ordered], dtype=np.float64)
    errors = np.asarray([item.reconstruction_error for item in ordered], dtype=np.float64)
    coefficient_values = np.abs(np.asarray(translator.coefficients, dtype=np.float64)).reshape(-1)
    near_zero = int(np.sum(coefficient_values <= sparsity_threshold))
    correct = sum(item.correct for item in ordered)
    payload: Mapping[str, object] = {
        "translator_id": translator.translator_id,
        "dataset_id": manifest.dataset_id,
        "evaluation_split": evaluation_split,
        "examples": [item.__dict__ for item in ordered],
        "sparsity_threshold": sparsity_threshold,
        "version": TRANSLATOR_EVALUATION_VERSION,
    }
    return TranslatorEvaluationReportDTO(
        report_id=_digest(_canonical_json(payload)),
        translator_id=translator.translator_id,
        dataset_id=manifest.dataset_id,
        evaluation_split=evaluation_split,
        example_count=len(ordered),
        correct_count=correct,
        accuracy=correct / len(ordered),
        mean_reconstruction_error=float(np.mean(errors)),
        mean_margin=float(np.mean(margins)),
        minimum_margin=float(np.min(margins)),
        maximum_margin=float(np.max(margins)),
        coefficient_count=int(coefficient_values.size),
        near_zero_coefficient_count=near_zero,
        sparsity_ratio=near_zero / int(coefficient_values.size),
        sparsity_threshold=sparsity_threshold,
        reconstruction_error_semantics=RECONSTRUCTION_ERROR_SEMANTICS,
        sparsity_semantics=SPARSITY_SEMANTICS,
        examples=ordered,
    )


def calibrate_translator_rejection(
    report: TranslatorEvaluationReportDTO,
    *,
    target_retained_accuracy: float = 1.0,
) -> TranslatorCalibrationDTO:
    """Choose the lowest margin threshold meeting retained calibration accuracy."""
    if report.evaluation_split == "test":
        raise PerceptionTranslatorEvaluationError("test split cannot be used for calibration")
    if not 0.0 < target_retained_accuracy <= 1.0:
        raise PerceptionTranslatorEvaluationError("target_retained_accuracy must be in (0, 1]")
    candidates = sorted({0.0, *(item.margin for item in report.examples)})
    chosen = candidates[-1]
    retained = tuple(item for item in report.examples if item.margin >= chosen)
    for threshold in candidates:
        current = tuple(item for item in report.examples if item.margin >= threshold)
        if current and sum(item.correct for item in current) / len(current) >= target_retained_accuracy:
            chosen = threshold
            retained = current
            break
    retained_accuracy = sum(item.correct for item in retained) / len(retained)
    payload: Mapping[str, object] = {
        "report_id": report.report_id,
        "minimum_margin": chosen,
        "target_retained_accuracy": target_retained_accuracy,
        "version": TRANSLATOR_CALIBRATION_VERSION,
    }
    return TranslatorCalibrationDTO(
        calibration_id=_digest(_canonical_json(payload)),
        translator_id=report.translator_id,
        dataset_id=report.dataset_id,
        calibration_split=report.evaluation_split,
        minimum_margin=chosen,
        retained_accuracy=retained_accuracy,
        retained_count=len(retained),
        rejected_count=report.example_count - len(retained),
        rejection_semantics=REJECTION_SEMANTICS,
    )


def predict_calibrated_target_vpm(
    translator: SourceTargetTranslatorDTO,
    calibration: TranslatorCalibrationDTO,
    source: SourceVPMDTO,
    source_field_schema: VPMFieldSchemaDTO,
    action_schema: DiscreteActionSchemaDTO,
) -> CalibratedTranslatorPredictionDTO:
    """Predict and explicitly reject when the calibrated top-two margin is too small."""
    if calibration.translator_id != translator.translator_id:
        raise PerceptionTranslatorEvaluationError("calibration does not belong to translator")
    raw = predict_target_vpm(translator, source, source_field_schema, action_schema)
    accepted = raw.margin >= calibration.minimum_margin
    status = "accepted" if accepted else "rejected_ambiguous"
    selected = raw.selected_action if accepted else None
    payload: Mapping[str, object] = {
        "calibration_id": calibration.calibration_id,
        "raw_prediction_id": raw.prediction_id,
        "selected_action": selected,
        "status": status,
        "version": REJECTED_TRANSLATOR_PREDICTION_VERSION,
    }
    return CalibratedTranslatorPredictionDTO(
        prediction_id=_digest(_canonical_json(payload)),
        translator_id=translator.translator_id,
        source_vpm_id=source.source_vpm_id,
        selected_action=selected,
        status=status,
        margin=raw.margin,
        minimum_margin=calibration.minimum_margin,
        raw_prediction=raw,
        rejection_semantics=REJECTION_SEMANTICS,
    )
