"""Shared action/future field relevance (joint representation experiment).

The representation used to predict the action and the representation used to
predict the future should not remain completely independent. This module
learns one small, deterministic, frozen per-field relevance vector from a
joint objective:

    weight ~= action_weight * action_discrimination
            + future_weight * future_transition_predictiveness

- action discrimination reuses ``estimate_field_relevance`` (eta-squared of
  before field means across actions);
- future predictiveness is eta-squared of per-transition field delta means
  across actions: a field whose change is action-determined scores near one,
  while an appearance-only nuisance field scores near zero.

The result stays inspectable (plain per-field weights, mean 1.0), frozen
after training, and applies as field-weighted distance in both the baseline
predictor and the future projector. Nuisance fields receive near-zero weight
by construction.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

import numpy as np

from .dataset import PerceptionDatasetManifestDTO
from .evidence import estimate_field_relevance
from .fields import VPMFieldSchemaDTO, validate_source_for_schema
from .inference import (
    ActionCandidateDTO,
    BaselineNearestNeighborModelDTO,
    BaselinePredictionDTO,
    BaselineTrainingExampleDTO,
    NeighborEvidenceDTO,
)
from .representation import SourceVPMDTO

SHARED_RELEVANCE_VERSION: Final = "perception-shared-field-relevance/1"
SHARED_RELEVANCE_PREDICTION_VERSION: Final = "perception-shared-relevance-prediction/1"
SHARED_RELEVANCE_DISTANCE_SEMANTICS: Final = (
    "shared_action_future_relevance_weighted_normalized_mean_absolute_pixel_distance"
)
SHARED_RELEVANCE_FUTURE_SEMANTICS: Final = "eta_squared_of_field_delta_mean_by_action"


class PerceptionSharedRelevanceError(ValueError):
    """Raised when shared relevance cannot be learned or applied."""


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


def _eta_squared(values: np.ndarray, labels: tuple[str, ...]) -> float:
    grand_mean = float(np.mean(values))
    total = float(np.sum((values - grand_mean) ** 2))
    if total <= 0.0:
        return 0.0
    between = 0.0
    for label in sorted(set(labels)):
        group = values[np.asarray([item == label for item in labels], dtype=bool)]
        group_mean = float(np.mean(group))
        between += float(group.size * ((group_mean - grand_mean) ** 2))
    return min(1.0, max(0.0, between / total))


@dataclass(frozen=True)
class SharedFieldRelevanceDTO:
    """Frozen per-field relevance shared by action and future prediction."""

    relevance_id: str
    field_schema_id: str
    training_dataset_id: str
    action_schema_id: str
    weights: tuple[tuple[str, float], ...]
    action_weight: float
    future_weight: float
    training_split: str
    version: str = SHARED_RELEVANCE_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.relevance_id,
                self.field_schema_id,
                self.training_dataset_id,
                self.action_schema_id,
            )
        ):
            raise PerceptionSharedRelevanceError(
                "shared relevance identities must be non-empty"
            )
        if self.weights != tuple(sorted(set(self.weights))):
            raise PerceptionSharedRelevanceError(
                "shared relevance weights must be unique and sorted"
            )
        total = 0.0
        for field_id, weight in self.weights:
            if not field_id or not np.isfinite(weight) or weight < 0.0:
                raise PerceptionSharedRelevanceError(
                    "shared relevance weights must be finite and non-negative"
                )
            total += weight
        if total <= 0.0:
            raise PerceptionSharedRelevanceError(
                "shared relevance requires positive weight"
            )
        if self.version != SHARED_RELEVANCE_VERSION:
            raise PerceptionSharedRelevanceError(
                "unsupported shared field relevance version"
            )

    def weight_for(self, field_id: str) -> float:
        for candidate_id, weight in self.weights:
            if candidate_id == field_id:
                return weight
        raise KeyError(field_id)


def _field_delta_means(
    before: SourceVPMDTO,
    after: SourceVPMDTO,
    field_schema: VPMFieldSchemaDTO,
) -> dict[str, float]:
    before_array = np.asarray(before.to_array(), dtype=np.float64).reshape(
        field_schema.height, field_schema.width, field_schema.channels
    )
    after_array = np.asarray(after.to_array(), dtype=np.float64).reshape(
        field_schema.height, field_schema.width, field_schema.channels
    )
    means: dict[str, float] = {}
    for field in field_schema.fields:
        before_region = before_array[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ]
        after_region = after_array[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ]
        means[field.field_id] = float(np.mean(after_region - before_region)) / 255.0
    return means


def _transition_delta_labels(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    field_schema: VPMFieldSchemaDTO,
    selected: set[str],
) -> tuple[dict[str, list[float]], tuple[str, ...]]:
    interactions = {item.interaction_id: item for item in manifest.interactions}
    deltas_by_field: dict[str, list[float]] = {
        field.field_id: [] for field in field_schema.fields
    }
    labels: list[str] = []
    for interaction_id in sorted(selected):
        interaction = interactions[interaction_id]
        if interaction.next_source_vpm_id is None:
            continue  # future term needs an authoritative next state
        try:
            before = source_vpms[interaction.source_vpm_id]
            after = source_vpms[interaction.next_source_vpm_id]
        except KeyError as exc:
            raise PerceptionSharedRelevanceError(
                f"missing SourceVPMDTO for {interaction_id}"
            ) from exc
        try:
            validate_source_for_schema(before, field_schema)
            validate_source_for_schema(after, field_schema)
        except Exception as exc:
            raise PerceptionSharedRelevanceError(str(exc)) from exc
        delta_means = _field_delta_means(before, after, field_schema)
        for field_id, value in delta_means.items():
            deltas_by_field[field_id].append(value)
        labels.append(interaction.action_label)
    return deltas_by_field, tuple(labels)


def fit_shared_field_relevance(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    field_schema: VPMFieldSchemaDTO,
    *,
    training_split: str = "train",
    action_weight: float = 0.3,
    future_weight: float = 0.7,
) -> SharedFieldRelevanceDTO:
    """Learn frozen per-field relevance from the joint action/future objective."""
    if action_weight < 0.0 or future_weight < 0.0:
        raise PerceptionSharedRelevanceError(
            "action and future weights must be non-negative"
        )
    if action_weight + future_weight <= 0.0:
        raise PerceptionSharedRelevanceError(
            "action and future weights cannot both be zero"
        )
    if training_split not in {"train", "validation", "test", "all"}:
        raise PerceptionSharedRelevanceError(
            "training_split must be train, validation, test, or all"
        )
    evidence = estimate_field_relevance(
        manifest, source_vpms, field_schema, training_split=training_split
    )
    discrimination = {item.field_id: item.score for item in evidence.relevances}
    selected = {
        assignment.interaction_id
        for assignment in manifest.split_assignments
        if training_split == "all" or assignment.split == training_split
    }
    deltas_by_field, label_tuple = _transition_delta_labels(
        manifest, source_vpms, field_schema, selected
    )
    if len(set(label_tuple)) < 2:
        raise PerceptionSharedRelevanceError(
            "future predictiveness requires at least two actions"
        )
    predictiveness = {
        field_id: _eta_squared(np.asarray(values, dtype=np.float64), label_tuple)
        for field_id, values in deltas_by_field.items()
    }
    ordered_ids = tuple(field.field_id for field in field_schema.fields)
    disc = np.array([discrimination[field_id] for field_id in ordered_ids])
    pred = np.array([predictiveness[field_id] for field_id in ordered_ids])
    disc_norm = disc / disc.max() if disc.max() > 0.0 else np.zeros_like(disc)
    pred_norm = pred / pred.max() if pred.max() > 0.0 else np.zeros_like(pred)
    raw = action_weight * disc_norm + future_weight * pred_norm
    if raw.sum() <= 0.0:
        weights = tuple((field_id, 1.0) for field_id in sorted(ordered_ids))
    else:
        scaled = len(ordered_ids) * raw / raw.sum()
        weights = tuple(
            sorted(
                (field_id, float(scaled[position]))
                for position, field_id in enumerate(ordered_ids)
            )
        )
    payload = {
        "action_schema_id": manifest.action_schema_id,
        "action_weight": action_weight,
        "field_schema_id": field_schema.field_schema_id,
        "future_weight": future_weight,
        "training_dataset_id": manifest.dataset_id,
        "training_split": training_split,
        "version": SHARED_RELEVANCE_VERSION,
        "weights": [[field_id, round(weight, 12)] for field_id, weight in weights],
    }
    return SharedFieldRelevanceDTO(
        relevance_id=_digest(_canonical_json(payload)),
        field_schema_id=field_schema.field_schema_id,
        training_dataset_id=manifest.dataset_id,
        action_schema_id=manifest.action_schema_id,
        weights=weights,
        action_weight=float(action_weight),
        future_weight=float(future_weight),
        training_split=training_split,
    )


def _relevance_weighted_distance(
    left: np.ndarray,
    right: np.ndarray,
    schema: VPMFieldSchemaDTO,
    weights: Mapping[str, float],
) -> float:
    left3 = left.reshape(schema.height, schema.width, schema.channels)
    right3 = right.reshape(schema.height, schema.width, schema.channels)
    total_weight = 0.0
    total = 0.0
    for field in schema.fields:
        weight = weights[field.field_id]
        if weight <= 0.0:
            continue
        region_left = left3[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ].astype(np.int16)
        region_right = right3[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ].astype(np.int16)
        total += weight * float(np.mean(np.abs(region_left - region_right)) / 255.0)
        total_weight += weight
    if total_weight <= 0.0:
        raise PerceptionSharedRelevanceError("no positive shared relevance weight")
    return total / total_weight


def _tally_relevance_votes(
    ranked: list[tuple[float, str, BaselineTrainingExampleDTO]],
    action_labels: tuple[str, ...],
    epsilon: float,
) -> tuple[tuple[ActionCandidateDTO, ...], tuple[NeighborEvidenceDTO, ...]]:
    raw = {label: 0.0 for label in action_labels}
    counts = {label: 0 for label in action_labels}
    nearest = {label: 1.0 for label in action_labels}
    neighbors: list[NeighborEvidenceDTO] = []
    for rank, (distance, _, item) in enumerate(ranked, start=1):
        weight = 1.0 / (epsilon + distance)
        raw[item.action_label] += weight
        counts[item.action_label] += 1
        nearest[item.action_label] = min(nearest[item.action_label], distance)
        neighbors.append(
            NeighborEvidenceDTO(
                item.interaction_id,
                item.source_vpm_id,
                item.action_label,
                distance,
                weight,
                rank,
            )
        )
    total = sum(raw.values())
    candidates = tuple(
        sorted(
            (
                ActionCandidateDTO(
                    label, raw[label] / total, counts[label], nearest[label]
                )
                for label in action_labels
            ),
            key=lambda item: (-item.score, item.action_label),
        )
    )
    return candidates, tuple(neighbors)


def _rank_relevance_neighbors(
    baseline_model: BaselineNearestNeighborModelDTO,
    unknown: np.ndarray,
    field_schema: VPMFieldSchemaDTO,
    weights: dict[str, float],
) -> list[tuple[float, str, BaselineTrainingExampleDTO]]:
    def _example_array(pixels: bytes) -> np.ndarray:
        shape = (
            (baseline_model.height, baseline_model.width)
            if baseline_model.channels == 1
            else (
                baseline_model.height,
                baseline_model.width,
                baseline_model.channels,
            )
        )
        return np.frombuffer(pixels, dtype=np.uint8).reshape(shape)

    ranked = sorted(
        (
            (
                _relevance_weighted_distance(
                    unknown, _example_array(item.pixels), field_schema, weights
                ),
                item.interaction_id,
                item,
            )
            for item in baseline_model.examples
        ),
        key=lambda value: (value[0], value[1]),
    )
    return ranked[
        : min(baseline_model.config.neighbor_count, len(baseline_model.examples))
    ]


def predict_relevance_weighted_action(
    baseline_model: BaselineNearestNeighborModelDTO,
    source: SourceVPMDTO,
    field_schema: VPMFieldSchemaDTO,
    relevance: SharedFieldRelevanceDTO,
) -> BaselinePredictionDTO:
    """Rank actions with shared-relevance field distance (System C predictor).

    Reuses the frozen P3 action memory and its vote/reject contract; only the
    distance carries the jointly learned representation.
    """
    try:
        validate_source_for_schema(source, field_schema)
    except Exception as exc:
        raise PerceptionSharedRelevanceError(str(exc)) from exc
    if field_schema.field_schema_id != relevance.field_schema_id:
        raise PerceptionSharedRelevanceError(
            "field schema does not match shared relevance"
        )
    if {field.field_id for field in field_schema.fields} != {
        field_id for field_id, _ in relevance.weights
    }:
        raise PerceptionSharedRelevanceError(
            "shared relevance must cover exactly the schema fields"
        )
    if source.encoder_spec_id != baseline_model.source_encoder_spec_id:
        raise PerceptionSharedRelevanceError(
            "source encoder spec does not match baseline model"
        )
    if (source.width, source.height, source.channels) != (
        baseline_model.width,
        baseline_model.height,
        baseline_model.channels,
    ):
        raise PerceptionSharedRelevanceError("source shape does not match model")
    weights = dict(relevance.weights)
    unknown = np.ascontiguousarray(source.to_array(), dtype=np.uint8).reshape(-1)
    ranked = _rank_relevance_neighbors(baseline_model, unknown, field_schema, weights)
    candidates, neighbors = _tally_relevance_votes(
        ranked, baseline_model.action_labels, baseline_model.config.epsilon
    )
    best = candidates[0]
    margin = best.score - (candidates[1].score if len(candidates) > 1 else 0.0)
    nearest_distance = neighbors[0].distance
    if nearest_distance > baseline_model.config.maximum_distance:
        status, selected = "rejected_out_of_distribution", None
    elif margin < baseline_model.config.minimum_margin:
        status, selected = "rejected_ambiguous", None
    else:
        status, selected = "accepted", best.action_label
    payload = {
        "baseline_model_id": baseline_model.model_id,
        "candidates": [(item.action_label, item.score) for item in candidates],
        "margin": margin,
        "nearest_distance": nearest_distance,
        "relevance_id": relevance.relevance_id,
        "selected_action": selected,
        "source_vpm_id": source.source_vpm_id,
        "status": status,
        "version": SHARED_RELEVANCE_PREDICTION_VERSION,
    }
    return BaselinePredictionDTO(
        _digest(_canonical_json(payload)),
        baseline_model.model_id,
        source.source_vpm_id,
        selected,
        status,
        best.score,
        "winning_inverse_distance_weight_share",
        SHARED_RELEVANCE_DISTANCE_SEMANTICS,
        nearest_distance,
        margin,
        candidates,
        tuple(neighbors),
        SHARED_RELEVANCE_PREDICTION_VERSION,
    )
