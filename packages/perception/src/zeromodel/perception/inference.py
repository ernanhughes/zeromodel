"""Deterministic whole-VPM nearest-neighbour inference for Stage P3.

P3 is the first complete prediction slice. It uses whole-image normalized mean
absolute distance only. Field weighting, sparse translation, semantic evidence,
and temporal models belong to later stages.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np

from .dataset import PerceptionDatasetManifestDTO, RecordedInteractionDTO
from .representation import SourceVPMDTO

BASELINE_MODEL_VERSION: Final = "perception-nearest-neighbour-model/1"
PREDICTION_VERSION: Final = "perception-nearest-neighbour-prediction/1"
DISTANCE_SEMANTICS: Final = "normalized_mean_absolute_pixel_distance"
CONFIDENCE_SEMANTICS: Final = "winning_inverse_distance_weight_share"


class PerceptionInferenceError(ValueError):
    """Raised when a model or prediction request violates the P3 contract."""


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


@dataclass(frozen=True)
class BaselineInferenceConfigDTO:
    """Bounded deterministic inference and rejection contract."""

    neighbor_count: int = 3
    maximum_distance: float = 0.35
    minimum_margin: float = 0.05
    epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if self.neighbor_count <= 0:
            raise PerceptionInferenceError("neighbor_count must be positive")
        if not 0.0 <= self.maximum_distance <= 1.0:
            raise PerceptionInferenceError("maximum_distance must be in [0, 1]")
        if not 0.0 <= self.minimum_margin <= 1.0:
            raise PerceptionInferenceError("minimum_margin must be in [0, 1]")
        if self.epsilon <= 0.0:
            raise PerceptionInferenceError("epsilon must be positive")

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "epsilon": self.epsilon,
            "maximum_distance": self.maximum_distance,
            "minimum_margin": self.minimum_margin,
            "neighbor_count": self.neighbor_count,
        }


@dataclass(frozen=True)
class BaselineTrainingExampleDTO:
    interaction_id: str
    source_vpm_id: str
    source_pixel_digest: str
    action_label: str
    width: int
    height: int
    channels: int
    pixels: bytes

    def __post_init__(self) -> None:
        if not self.interaction_id or not self.source_vpm_id or not self.action_label:
            raise PerceptionInferenceError("training example identities must be non-empty")
        expected = self.width * self.height * self.channels
        if expected <= 0 or len(self.pixels) != expected:
            raise PerceptionInferenceError("training example pixel payload has invalid size")


@dataclass(frozen=True)
class BaselineNearestNeighborModelDTO:
    model_id: str
    dataset_id: str
    action_schema_id: str
    source_encoder_spec_id: str
    width: int
    height: int
    channels: int
    action_labels: tuple[str, ...]
    examples: tuple[BaselineTrainingExampleDTO, ...]
    config: BaselineInferenceConfigDTO
    version: str = BASELINE_MODEL_VERSION

    def __post_init__(self) -> None:
        if not self.model_id or not self.dataset_id or not self.action_schema_id:
            raise PerceptionInferenceError("model identities must be non-empty")
        if not self.examples:
            raise PerceptionInferenceError("model requires at least one training example")
        ids = tuple(item.interaction_id for item in self.examples)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise PerceptionInferenceError("model examples must be unique and sorted")
        if self.action_labels != tuple(sorted(set(self.action_labels))):
            raise PerceptionInferenceError("action_labels must be unique and sorted")


@dataclass(frozen=True)
class NeighborEvidenceDTO:
    interaction_id: str
    source_vpm_id: str
    action_label: str
    distance: float
    weight: float
    rank: int


@dataclass(frozen=True)
class ActionCandidateDTO:
    action_label: str
    score: float
    support_count: int
    nearest_distance: float


@dataclass(frozen=True)
class BaselinePredictionDTO:
    prediction_id: str
    model_id: str
    source_vpm_id: str
    selected_action: str | None
    status: str
    confidence: float
    confidence_semantics: str
    distance_semantics: str
    nearest_distance: float
    margin: float
    candidates: tuple[ActionCandidateDTO, ...]
    neighbors: tuple[NeighborEvidenceDTO, ...]
    version: str = PREDICTION_VERSION


_ALLOWED_STATUSES: Final = {
    "accepted",
    "rejected_out_of_distribution",
    "rejected_ambiguous",
}


def _source_pixels(source: SourceVPMDTO) -> bytes:
    array = source.to_array()
    return np.ascontiguousarray(array, dtype=np.uint8).reshape(-1).tobytes()


def _interaction_by_id(
    manifest: PerceptionDatasetManifestDTO,
) -> dict[str, RecordedInteractionDTO]:
    return {item.interaction_id: item for item in manifest.interactions}


def fit_baseline_nearest_neighbor(
    manifest: PerceptionDatasetManifestDTO,
    source_vpms: Mapping[str, SourceVPMDTO],
    *,
    config: BaselineInferenceConfigDTO | None = None,
    training_split: str = "train",
) -> BaselineNearestNeighborModelDTO:
    """Materialize an immutable whole-image nearest-neighbour model."""

    resolved = config or BaselineInferenceConfigDTO()
    if training_split not in {"train", "validation", "test", "all"}:
        raise PerceptionInferenceError("training_split must be train, validation, test, or all")

    interactions = _interaction_by_id(manifest)
    selected_ids = tuple(
        assignment.interaction_id
        for assignment in manifest.split_assignments
        if training_split == "all" or assignment.split == training_split
    )
    if not selected_ids:
        raise PerceptionInferenceError(f"dataset contains no {training_split!r} examples")

    examples: list[BaselineTrainingExampleDTO] = []
    shape: tuple[int, int, int] | None = None
    encoder_spec_id: str | None = None
    for interaction_id in sorted(selected_ids):
        interaction = interactions[interaction_id]
        try:
            source = source_vpms[interaction.source_vpm_id]
        except KeyError as exc:
            raise PerceptionInferenceError(
                f"missing SourceVPMDTO for {interaction.source_vpm_id}"
            ) from exc
        if source.pixel_digest != interaction.source_pixel_digest:
            raise PerceptionInferenceError("source pixel identity disagrees with interaction")
        current_shape = (source.width, source.height, source.channels)
        if shape is None:
            shape = current_shape
            encoder_spec_id = source.encoder_spec_id
        elif current_shape != shape:
            raise PerceptionInferenceError("baseline model requires one source VPM shape")
        elif source.encoder_spec_id != encoder_spec_id:
            raise PerceptionInferenceError("baseline model requires one source encoder spec")
        examples.append(
            BaselineTrainingExampleDTO(
                interaction_id=interaction.interaction_id,
                source_vpm_id=source.source_vpm_id,
                source_pixel_digest=source.pixel_digest,
                action_label=interaction.action_label,
                width=source.width,
                height=source.height,
                channels=source.channels,
                pixels=_source_pixels(source),
            )
        )

    assert shape is not None and encoder_spec_id is not None
    ordered_examples = tuple(sorted(examples, key=lambda item: item.interaction_id))
    labels = tuple(sorted({item.action_label for item in ordered_examples}))
    payload: Mapping[str, object] = {
        "action_labels": list(labels),
        "action_schema_id": manifest.action_schema_id,
        "config": resolved.canonical_payload(),
        "dataset_id": manifest.dataset_id,
        "examples": [
            {
                "action_label": item.action_label,
                "interaction_id": item.interaction_id,
                "source_pixel_digest": item.source_pixel_digest,
                "source_vpm_id": item.source_vpm_id,
            }
            for item in ordered_examples
        ],
        "shape": list(shape),
        "source_encoder_spec_id": encoder_spec_id,
        "training_split": training_split,
        "version": BASELINE_MODEL_VERSION,
    }
    return BaselineNearestNeighborModelDTO(
        model_id=_digest(_canonical_json(payload)),
        dataset_id=manifest.dataset_id,
        action_schema_id=manifest.action_schema_id,
        source_encoder_spec_id=encoder_spec_id,
        width=shape[0],
        height=shape[1],
        channels=shape[2],
        action_labels=labels,
        examples=ordered_examples,
        config=resolved,
    )


def _distance(left: bytes, right: bytes) -> float:
    left_values = np.frombuffer(left, dtype=np.uint8).astype(np.int16)
    right_values = np.frombuffer(right, dtype=np.uint8).astype(np.int16)
    return float(np.mean(np.abs(left_values - right_values)) / 255.0)


def predict_baseline_action(
    model: BaselineNearestNeighborModelDTO,
    source: SourceVPMDTO,
) -> BaselinePredictionDTO:
    """Rank actions for one unknown source VPM with explicit rejection."""

    if source.encoder_spec_id != model.source_encoder_spec_id:
        raise PerceptionInferenceError("source encoder spec does not match model")
    if (source.width, source.height, source.channels) != (
        model.width,
        model.height,
        model.channels,
    ):
        raise PerceptionInferenceError("source shape does not match model")

    unknown = _source_pixels(source)
    ranked = sorted(
        (
            (_distance(unknown, item.pixels), item.interaction_id, item)
            for item in model.examples
        ),
        key=lambda value: (value[0], value[1]),
    )[: min(model.config.neighbor_count, len(model.examples))]

    raw_weights: dict[str, float] = {label: 0.0 for label in model.action_labels}
    support_counts: dict[str, int] = {label: 0 for label in model.action_labels}
    nearest_by_action: dict[str, float] = {label: 1.0 for label in model.action_labels}
    neighbors: list[NeighborEvidenceDTO] = []
    for rank, (distance, _, item) in enumerate(ranked, start=1):
        weight = 1.0 / (model.config.epsilon + distance)
        raw_weights[item.action_label] += weight
        support_counts[item.action_label] += 1
        nearest_by_action[item.action_label] = min(
            nearest_by_action[item.action_label], distance
        )
        neighbors.append(
            NeighborEvidenceDTO(
                interaction_id=item.interaction_id,
                source_vpm_id=item.source_vpm_id,
                action_label=item.action_label,
                distance=distance,
                weight=weight,
                rank=rank,
            )
        )

    total_weight = sum(raw_weights.values())
    candidates = tuple(
        sorted(
            (
                ActionCandidateDTO(
                    action_label=label,
                    score=raw_weights[label] / total_weight,
                    support_count=support_counts[label],
                    nearest_distance=nearest_by_action[label],
                )
                for label in model.action_labels
            ),
            key=lambda item: (-item.score, item.action_label),
        )
    )
    best = candidates[0]
    second_score = candidates[1].score if len(candidates) > 1 else 0.0
    margin = best.score - second_score
    nearest_distance = neighbors[0].distance

    if nearest_distance > model.config.maximum_distance:
        status = "rejected_out_of_distribution"
        selected_action = None
    elif margin < model.config.minimum_margin:
        status = "rejected_ambiguous"
        selected_action = None
    else:
        status = "accepted"
        selected_action = best.action_label
    assert status in _ALLOWED_STATUSES

    payload: Mapping[str, object] = {
        "candidates": [
            {
                "action_label": item.action_label,
                "nearest_distance": item.nearest_distance,
                "score": item.score,
                "support_count": item.support_count,
            }
            for item in candidates
        ],
        "model_id": model.model_id,
        "neighbors": [
            {
                "action_label": item.action_label,
                "distance": item.distance,
                "interaction_id": item.interaction_id,
                "rank": item.rank,
                "source_vpm_id": item.source_vpm_id,
                "weight": item.weight,
            }
            for item in neighbors
        ],
        "selected_action": selected_action,
        "source_vpm_id": source.source_vpm_id,
        "status": status,
        "version": PREDICTION_VERSION,
    }
    return BaselinePredictionDTO(
        prediction_id=_digest(_canonical_json(payload)),
        model_id=model.model_id,
        source_vpm_id=source.source_vpm_id,
        selected_action=selected_action,
        status=status,
        confidence=best.score,
        confidence_semantics=CONFIDENCE_SEMANTICS,
        distance_semantics=DISTANCE_SEMANTICS,
        nearest_distance=nearest_distance,
        margin=margin,
        candidates=candidates,
        neighbors=tuple(neighbors),
    )
