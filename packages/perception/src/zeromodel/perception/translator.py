"""Deterministic source-field to target-field translation for Stage P5A.

P5A learns an explicit ridge-regression coefficient matrix from normalized source
field means to the continuous fields of a discrete Target VPM. The predicted target
surface is distinct from an observed canonical one-hot TargetVPMDTO.
"""

from __future__ import annotations

import hashlib
import io
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np
from PIL import Image

from .dataset import PerceptionDatasetManifestDTO
from .fields import VPMFieldSchemaDTO, extract_source_fields, validate_source_for_schema
from .representation import DiscreteActionSchemaDTO, SourceVPMDTO

TRANSLATOR_VERSION: Final = "perception-source-target-translator/1"
TRANSLATOR_PREDICTION_VERSION: Final = "perception-target-surface-prediction/1"
SOURCE_FEATURE_SEMANTICS: Final = "normalized_mean_intensity_per_declared_source_field"
TARGET_SCORE_SEMANTICS: Final = "clipped_ridge_predicted_one_hot_field_value"
COEFFICIENT_SEMANTICS: Final = "ridge_linear_mapping_with_unregularized_intercept"


class PerceptionTranslatorError(ValueError):
    """Raised when translator fitting or prediction violates the P5A contract."""


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


def _png_bytes(values: np.ndarray) -> bytes:
    output = io.BytesIO()
    Image.fromarray(values, mode="L").save(
        output,
        format="PNG",
        optimize=False,
        compress_level=9,
    )
    return output.getvalue()


@dataclass(frozen=True)
class TranslatorConfigDTO:
    """Deterministic bounded fitting contract."""

    ridge_alpha: float = 1e-6

    def __post_init__(self) -> None:
        if not np.isfinite(self.ridge_alpha) or self.ridge_alpha < 0.0:
            raise PerceptionTranslatorError("ridge_alpha must be finite and non-negative")

    def canonical_payload(self) -> Mapping[str, object]:
        return {"ridge_alpha": self.ridge_alpha}


@dataclass(frozen=True)
class SourceTargetTranslatorDTO:
    """Immutable inspectable linear mapping from source fields to target fields."""

    translator_id: str
    dataset_id: str
    source_field_schema_id: str
    source_encoder_spec_id: str
    action_schema_id: str
    action_labels: tuple[str, ...]
    source_field_ids: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    training_split: str
    training_count: int
    source_feature_semantics: str
    target_score_semantics: str
    coefficient_semantics: str
    config: TranslatorConfigDTO
    version: str = TRANSLATOR_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.translator_id,
                self.dataset_id,
                self.source_field_schema_id,
                self.source_encoder_spec_id,
                self.action_schema_id,
            )
        ):
            raise PerceptionTranslatorError("translator identities must be non-empty")
        if self.action_labels != tuple(sorted(set(self.action_labels))):
            raise PerceptionTranslatorError("action_labels must be unique and sorted")
        if self.source_field_ids != tuple(sorted(set(self.source_field_ids))):
            raise PerceptionTranslatorError("source_field_ids must be unique and sorted")
        if self.training_count <= 0:
            raise PerceptionTranslatorError("training_count must be positive")
        if len(self.coefficients) != len(self.action_labels):
            raise PerceptionTranslatorError("one coefficient row is required per action")
        if len(self.intercepts) != len(self.action_labels):
            raise PerceptionTranslatorError("one intercept is required per action")
        if any(len(row) != len(self.source_field_ids) for row in self.coefficients):
            raise PerceptionTranslatorError("coefficient rows must match source fields")
        if self.source_feature_semantics != SOURCE_FEATURE_SEMANTICS:
            raise PerceptionTranslatorError("unsupported source feature semantics")
        if self.target_score_semantics != TARGET_SCORE_SEMANTICS:
            raise PerceptionTranslatorError("unsupported target score semantics")
        if self.coefficient_semantics != COEFFICIENT_SEMANTICS:
            raise PerceptionTranslatorError("unsupported coefficient semantics")
        values = np.asarray(self.coefficients + (self.intercepts,), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise PerceptionTranslatorError("translator parameters must be finite")

    def coefficient_for(self, action_label: str, field_id: str) -> float:
        try:
            action_index = self.action_labels.index(action_label)
            field_index = self.source_field_ids.index(field_id)
        except ValueError as exc:
            raise KeyError((action_label, field_id)) from exc
        return self.coefficients[action_index][field_index]


@dataclass(frozen=True)
class TargetActionScoreDTO:
    action_label: str
    score: float
    rank: int

    def __post_init__(self) -> None:
        if not self.action_label or self.rank <= 0:
            raise PerceptionTranslatorError("target action score identity/rank is invalid")
        if not 0.0 <= self.score <= 1.0:
            raise PerceptionTranslatorError("target action score must be in [0, 1]")


@dataclass(frozen=True)
class PredictedTargetVPMDTO:
    """Continuous predicted target surface plus deterministic decoding metadata."""

    prediction_id: str
    translator_id: str
    source_vpm_id: str
    action_schema_id: str
    width: int
    height: int
    channels: int
    score_semantics: str
    scores: tuple[TargetActionScoreDTO, ...]
    selected_action: str
    margin: float
    png_digest: str
    png_bytes: bytes
    version: str = TRANSLATOR_PREDICTION_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.prediction_id,
                self.translator_id,
                self.source_vpm_id,
                self.action_schema_id,
                self.selected_action,
            )
        ):
            raise PerceptionTranslatorError("prediction identities must be non-empty")
        if (self.height, self.channels) != (1, 1) or self.width <= 0:
            raise PerceptionTranslatorError("predicted target surface must be 1xN grayscale")
        if self.score_semantics != TARGET_SCORE_SEMANTICS:
            raise PerceptionTranslatorError("unsupported target score semantics")
        if len(self.scores) != self.width:
            raise PerceptionTranslatorError("prediction score count must match width")
        if tuple(item.rank for item in self.scores) != tuple(range(1, self.width + 1)):
            raise PerceptionTranslatorError("prediction scores must be consecutively ranked")
        if self.scores[0].action_label != self.selected_action:
            raise PerceptionTranslatorError("selected_action must match top-ranked score")
        if not 0.0 <= self.margin <= 1.0:
            raise PerceptionTranslatorError("prediction margin must be in [0, 1]")
        if _digest(self.png_bytes) != self.png_digest:
            raise PerceptionTranslatorError("predicted Target VPM PNG digest mismatch")

    def to_array(self) -> np.ndarray:
        with Image.open(io.BytesIO(self.png_bytes)) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8)
        if values.shape != (1, self.width):
            raise PerceptionTranslatorError("predicted Target VPM PNG shape mismatch")
        return values.copy()


def _selected_interactions(
    manifest: PerceptionDatasetManifestDTO,
    training_split: str,
):
    if training_split not in {"train", "validation", "test", "all"}:
        raise PerceptionTranslatorError(
            "training_split must be train, validation, test, or all"
        )
    selected_ids = {
        item.interaction_id
        for item in manifest.split_assignments
        if training_split == "all" or item.split == training_split
    }
    interactions = tuple(
        item for item in manifest.interactions if item.interaction_id in selected_ids
    )
    if not interactions:
        raise PerceptionTranslatorError(
            f"dataset contains no {training_split!r} interactions"
        )
    return interactions


def _source_feature_vector(
    source: SourceVPMDTO,
    field_schema: VPMFieldSchemaDTO,
    field_ids: tuple[str, ...],
) -> np.ndarray:
    validate_source_for_schema(source, field_schema)
    samples = {item.field_id: item for item in extract_source_fields(source, field_schema)}
    return np.asarray(
        [float(np.mean(samples[field_id].to_array())) / 255.0 for field_id in field_ids],
        dtype=np.float64,
    )


def fit_source_target_translator(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    source_field_schema: VPMFieldSchemaDTO,
    action_schema: DiscreteActionSchemaDTO,
    *,
    training_split: str = "train",
    config: TranslatorConfigDTO | None = None,
) -> SourceTargetTranslatorDTO:
    """Fit an explicit deterministic ridge mapping from source fields to actions."""

    resolved = config or TranslatorConfigDTO()
    if manifest.action_schema_id != action_schema.action_schema_id:
        raise PerceptionTranslatorError("action schema does not match dataset manifest")
    interactions = _selected_interactions(manifest, training_split)
    represented = {item.action_label for item in interactions}
    if not represented.issubset(set(action_schema.labels)):
        raise PerceptionTranslatorError("dataset contains action outside supplied schema")

    field_ids = tuple(sorted(field.field_id for field in source_field_schema.fields))
    feature_rows: list[np.ndarray] = []
    target_rows: list[np.ndarray] = []
    example_ids: list[str] = []
    for interaction in interactions:
        try:
            source = source_vpms[interaction.source_vpm_id]
        except KeyError as exc:
            raise PerceptionTranslatorError(
                f"missing SourceVPMDTO for {interaction.source_vpm_id}"
            ) from exc
        if source.pixel_digest != interaction.source_pixel_digest:
            raise PerceptionTranslatorError("source pixel identity disagrees with interaction")
        feature_rows.append(_source_feature_vector(source, source_field_schema, field_ids))
        target = np.zeros(len(action_schema.labels), dtype=np.float64)
        target[action_schema.index_of(interaction.action_label)] = 1.0
        target_rows.append(target)
        example_ids.append(interaction.interaction_id)

    x = np.vstack(feature_rows)
    y = np.vstack(target_rows)
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
        "dataset_id": manifest.dataset_id,
        "example_ids": sorted(example_ids),
        "intercepts": list(intercepts),
        "source_encoder_spec_id": source_field_schema.source_encoder_spec_id,
        "source_feature_semantics": SOURCE_FEATURE_SEMANTICS,
        "source_field_ids": list(field_ids),
        "source_field_schema_id": source_field_schema.field_schema_id,
        "target_score_semantics": TARGET_SCORE_SEMANTICS,
        "training_split": training_split,
        "version": TRANSLATOR_VERSION,
    }
    return SourceTargetTranslatorDTO(
        translator_id=_digest(_canonical_json(payload)),
        dataset_id=manifest.dataset_id,
        source_field_schema_id=source_field_schema.field_schema_id,
        source_encoder_spec_id=source_field_schema.source_encoder_spec_id,
        action_schema_id=action_schema.action_schema_id,
        action_labels=action_schema.labels,
        source_field_ids=field_ids,
        coefficients=coefficients,
        intercepts=intercepts,
        training_split=training_split,
        training_count=len(interactions),
        source_feature_semantics=SOURCE_FEATURE_SEMANTICS,
        target_score_semantics=TARGET_SCORE_SEMANTICS,
        coefficient_semantics=COEFFICIENT_SEMANTICS,
        config=resolved,
    )


def predict_target_vpm(
    translator: SourceTargetTranslatorDTO,
    source: SourceVPMDTO,
    source_field_schema: VPMFieldSchemaDTO,
    action_schema: DiscreteActionSchemaDTO,
) -> PredictedTargetVPMDTO:
    """Translate one source VPM into a continuous target surface and ranked action."""

    if translator.source_field_schema_id != source_field_schema.field_schema_id:
        raise PerceptionTranslatorError("source field schema does not match translator")
    if translator.action_schema_id != action_schema.action_schema_id:
        raise PerceptionTranslatorError("action schema does not match translator")
    if translator.action_labels != action_schema.labels:
        raise PerceptionTranslatorError("action labels do not match translator")
    features = _source_feature_vector(
        source,
        source_field_schema,
        translator.source_field_ids,
    )
    coefficient_matrix = np.asarray(translator.coefficients, dtype=np.float64)
    raw = np.asarray(translator.intercepts, dtype=np.float64) + coefficient_matrix @ features
    clipped = np.clip(raw, 0.0, 1.0)
    ranked_indices = sorted(
        range(len(action_schema.labels)),
        key=lambda index: (-float(clipped[index]), action_schema.labels[index]),
    )
    ranked_scores = tuple(
        TargetActionScoreDTO(
            action_label=action_schema.labels[index],
            score=float(clipped[index]),
            rank=rank,
        )
        for rank, index in enumerate(ranked_indices, start=1)
    )
    selected_action = ranked_scores[0].action_label
    second = ranked_scores[1].score if len(ranked_scores) > 1 else 0.0
    margin = ranked_scores[0].score - second
    rendered = np.rint(clipped * 255.0).astype(np.uint8).reshape(1, -1)
    png_bytes = _png_bytes(rendered)
    png_digest = _digest(png_bytes)
    payload: Mapping[str, object] = {
        "action_schema_id": action_schema.action_schema_id,
        "margin": margin,
        "png_digest": png_digest,
        "scores": [
            {
                "action_label": item.action_label,
                "rank": item.rank,
                "score": item.score,
            }
            for item in ranked_scores
        ],
        "selected_action": selected_action,
        "source_vpm_id": source.source_vpm_id,
        "target_score_semantics": TARGET_SCORE_SEMANTICS,
        "translator_id": translator.translator_id,
        "version": TRANSLATOR_PREDICTION_VERSION,
    }
    return PredictedTargetVPMDTO(
        prediction_id=_digest(_canonical_json(payload)),
        translator_id=translator.translator_id,
        source_vpm_id=source.source_vpm_id,
        action_schema_id=action_schema.action_schema_id,
        width=len(action_schema.labels),
        height=1,
        channels=1,
        score_semantics=TARGET_SCORE_SEMANTICS,
        scores=ranked_scores,
        selected_action=selected_action,
        margin=margin,
        png_digest=png_digest,
        png_bytes=png_bytes,
    )
