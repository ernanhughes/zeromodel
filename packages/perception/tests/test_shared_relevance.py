"""Unit tests for shared action/future field relevance (System C)."""

from __future__ import annotations

import numpy as np
import pytest

from zeromodel.perception import (
    build_dataset_manifest,
    build_grid_field_schema,
    encode_discrete_action,
    encode_source_array,
    fit_baseline_nearest_neighbor,
)
from zeromodel.perception.dataset import RecordedInteractionDTO
from zeromodel.perception.representation import (
    DiscreteActionSchemaDTO,
    SourceImageEncoderSpecDTO,
)
from zeromodel.perception.shared_relevance import (
    PerceptionSharedRelevanceError,
    fit_shared_field_relevance,
    predict_relevance_weighted_action,
)

_SPEC = SourceImageEncoderSpecDTO(color_space="L")
_WIDTH, _HEIGHT = 12, 8
_ACTION_SCHEMA = DiscreteActionSchemaDTO.from_labels(["left", "right"])


def _pattern(marker: int, action: str, rng: np.random.RandomState):
    before_array = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    before_array[marker % _HEIGHT, (marker * 5) % _WIDTH] = 200
    # Middle band is appearance-only nuisance: random every observation,
    # carrying no action signal and no transition signal.
    before_array[:, 4:8] = rng.randint(0, 256, size=(_HEIGHT, 4)).astype(np.uint8)
    after_array = before_array.copy()
    if action == "left":
        after_array[:, 0:4] = 60
    else:
        after_array[:, 8:12] = 60
    after_array[:, 4:8] = rng.randint(0, 256, size=(_HEIGHT, 4)).astype(np.uint8)
    return encode_source_array(before_array, _SPEC), encode_source_array(
        after_array, _SPEC
    )


def _build(count_per_action: int = 12):
    rng = np.random.RandomState(7)
    interactions = []
    sources = {}
    step = 0
    marker = 0
    for action in ("left", "right"):
        for _ in range(count_per_action):
            before, after = _pattern(marker, action, rng)
            sources[before.source_vpm_id] = before
            sources[after.source_vpm_id] = after
            interactions.append(
                RecordedInteractionDTO.from_vpms(
                    sequence_id=f"seq-{action}",
                    step_index=step,
                    source=before,
                    target=encode_discrete_action(action, _ACTION_SCHEMA),
                    next_source=after,
                )
            )
            step += 1
            marker += 1
    manifest = build_dataset_manifest(
        interactions, source_encoder_spec_ids=[_SPEC.encoder_spec_id]
    )
    schema = build_grid_field_schema(
        next(iter(sources.values())), tile_width=4, tile_height=4, channel_mode="joint"
    )
    return manifest, sources, schema


def _band(schema, channel: int):
    return {field.field_id for field in schema.fields if field.x0 // 4 == channel}


def test_shared_relevance_downweights_nuisance() -> None:
    manifest, sources, schema = _build()
    relevance = fit_shared_field_relevance(
        manifest, sources, schema, training_split="all"
    )
    assert abs(sum(weight for _, weight in relevance.weights) / 6 - 1.0) < 1e-9
    by_id = dict(relevance.weights)
    nuisance = max(by_id[field_id] for field_id in _band(schema, 1))
    semantic = min(by_id[field_id] for field_id in _band(schema, 0) | _band(schema, 2))
    assert nuisance < semantic
    second = fit_shared_field_relevance(manifest, sources, schema, training_split="all")
    assert second.relevance_id == relevance.relevance_id


def test_shared_relevance_rejects_invalid_weights() -> None:
    manifest, sources, schema = _build()
    with pytest.raises(PerceptionSharedRelevanceError):
        fit_shared_field_relevance(
            manifest,
            sources,
            schema,
            training_split="all",
            action_weight=-0.1,
        )
    with pytest.raises(PerceptionSharedRelevanceError):
        fit_shared_field_relevance(
            manifest,
            sources,
            schema,
            training_split="all",
            action_weight=0.0,
            future_weight=0.0,
        )


def test_relevance_weighted_predictor_returns_baseline_contract() -> None:
    manifest, sources, schema = _build()
    predictor = fit_baseline_nearest_neighbor(manifest, sources, training_split="all")
    relevance = fit_shared_field_relevance(
        manifest, sources, schema, training_split="all"
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    prediction = predict_relevance_weighted_action(predictor, query, schema, relevance)
    assert prediction.source_vpm_id == query.source_vpm_id
    assert prediction.model_id == predictor.model_id
    assert prediction.status == "accepted"
    assert prediction.selected_action == "left"
    assert {item.action_label for item in prediction.candidates} == {"left", "right"}
    again = predict_relevance_weighted_action(predictor, query, schema, relevance)
    assert again.prediction_id == prediction.prediction_id
