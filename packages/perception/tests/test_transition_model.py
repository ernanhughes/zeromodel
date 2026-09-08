"""Unit tests for compiled action-conditioned transition memory."""

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
from zeromodel.perception.transition_model import (
    PerceptionTransitionModelError,
    TransitionModelConfigDTO,
    fit_action_conditioned_transition_model,
    fit_compiled_transition_model,
)
from zeromodel.perception.transition_projection import project_expected_transition

_SPEC = SourceImageEncoderSpecDTO(color_space="L")
_WIDTH, _HEIGHT = 12, 8
_ACTION_SCHEMA = DiscreteActionSchemaDTO.from_labels(["left", "right"])


def _pattern(marker: int, action: str) -> tuple[object, object]:
    before_array = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    before_array[marker % _HEIGHT, (marker * 5) % _WIDTH] = 200
    after_array = before_array.copy()
    if action == "left":
        after_array[:, 0:4] = 60
    else:
        after_array[:, 8:12] = 60
    from zeromodel.perception import encode_source_array as _encode

    return _encode(before_array, _SPEC), _encode(after_array, _SPEC)


def _build(
    count_per_action: int = 6,
    *,
    missing_next: bool = False,
):
    interactions = []
    sources = {}
    step = 0
    marker = 0
    for action in ("left", "right"):
        for _ in range(count_per_action):
            before, after = _pattern(marker, action)
            sources[before.source_vpm_id] = before
            sources[after.source_vpm_id] = after
            interactions.append(
                RecordedInteractionDTO.from_vpms(
                    sequence_id=f"seq-{action}",
                    step_index=step,
                    source=before,
                    target=encode_discrete_action(action, _ACTION_SCHEMA),
                    next_source=None if missing_next else after,
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


def test_fit_requires_authoritative_next_state() -> None:
    manifest, sources, schema = _build(missing_next=True)
    with pytest.raises(
        PerceptionTransitionModelError, match="authoritative next state"
    ):
        fit_action_conditioned_transition_model(
            manifest, sources, schema, training_split="all"
        )
    with pytest.raises(
        PerceptionTransitionModelError, match="authoritative next state"
    ):
        fit_compiled_transition_model(manifest, sources, schema, training_split="all")


def test_fit_rejects_mixed_source_schemas() -> None:
    manifest, sources, _ = _build(count_per_action=2)
    other_spec = SourceImageEncoderSpecDTO(color_space="L", max_pixels=100)
    other_schema_source = encode_source_array(
        np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8), other_spec
    )
    assert other_schema_source.encoder_spec_id != _SPEC.encoder_spec_id
    first = manifest.interactions[0]
    sources[other_schema_source.source_vpm_id] = other_schema_source
    from zeromodel.perception.dataset import RecordedInteractionDTO as _RID

    tampered = _RID(
        interaction_id=first.interaction_id,
        sequence_id=first.sequence_id,
        step_index=first.step_index,
        source_vpm_id=other_schema_source.source_vpm_id,
        target_vpm_id=first.target_vpm_id,
        action_schema_id=first.action_schema_id,
        action_label=first.action_label,
        source_pixel_digest=other_schema_source.pixel_digest,
        next_source_vpm_id=first.next_source_vpm_id,
    )
    interactions = [
        tampered if item.interaction_id == first.interaction_id else item
        for item in manifest.interactions
    ]
    mixed = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[
            _SPEC.encoder_spec_id,
            other_spec.encoder_spec_id,
        ],
    )
    schema = build_grid_field_schema(
        next(iter(sources.values())), tile_width=4, tile_height=4, channel_mode="joint"
    )
    with pytest.raises(PerceptionTransitionModelError, match="one source encoder spec"):
        fit_action_conditioned_transition_model(
            mixed, sources, schema, training_split="all"
        )


def test_fit_is_deterministic_and_identified() -> None:
    manifest, sources, schema = _build()
    first = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    second = fit_action_conditioned_transition_model(
        manifest, dict(reversed(list(sources.items()))), schema, training_split="all"
    )
    assert first.model_id == second.model_id
    assert first.examples == second.examples
    ridge_first = fit_compiled_transition_model(
        manifest, sources, schema, training_split="all"
    )
    ridge_second = fit_compiled_transition_model(
        manifest, sources, schema, training_split="all"
    )
    assert ridge_first.model_id == ridge_second.model_id
    assert ridge_first.model_id != first.model_id


def test_config_rejects_invalid_bounds() -> None:
    with pytest.raises(PerceptionTransitionModelError):
        TransitionModelConfigDTO(neighbor_count=0)
    with pytest.raises(PerceptionTransitionModelError):
        TransitionModelConfigDTO(min_support=0)


def test_projection_is_deterministic_and_action_sensitive() -> None:
    manifest, sources, schema = _build()
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    first = project_expected_transition(model, query, "left", schema)
    second = project_expected_transition(model, query, "left", schema)
    assert first == second
    assert first.status == "supported"
    other = project_expected_transition(model, query, "right", schema)
    assert other.expected_transition_id != first.expected_transition_id
    left_delta = first.field_evidence(
        next(
            field.field_id for field in schema.fields if field.x0 == 0 and field.y0 == 0
        )
    )
    assert left_delta.expected_mean_signed_change > 0.0
    assert other.status == "supported"


def test_projection_statuses_are_explicit() -> None:
    manifest, sources, schema = _build(count_per_action=6)
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(iter(sources.values()))
    assert (
        project_expected_transition(model, query, "jump", schema).status
        == "unsupported_action"
    )
    scarce = project_expected_transition(model, query, "left", schema, min_support=500)
    assert scarce.status == "insufficient_examples"
    assert scarce.confidence <= 0.25
    far = encode_source_array(np.full((_HEIGHT, _WIDTH), 255, dtype=np.uint8), _SPEC)
    assert (
        project_expected_transition(model, far, "left", schema).status
        == "out_of_distribution"
    )


def test_projection_performs_no_runtime_fitting() -> None:
    manifest, sources, schema = _build()
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    snapshot = (model.model_id, model.examples, model.action_spreads)
    query = next(iter(sources.values()))
    project_expected_transition(model, query, "left", schema)
    project_expected_transition(model, query, "right", schema)
    assert (model.model_id, model.examples, model.action_spreads) == snapshot


def test_compiled_projection_supported_and_distinct() -> None:
    manifest, sources, schema = _build()
    model = fit_compiled_transition_model(
        manifest, sources, schema, training_split="all"
    )
    assert all(model.action_has_linear_fit)
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    projected = project_expected_transition(model, query, "left", schema)
    assert projected.status == "supported"
    deltas = [field.expected_mean_signed_change for field in projected.fields]
    assert any(value > 0.0 for value in deltas)
    other = project_expected_transition(model, query, "right", schema)
    assert other.expected_transition_id != projected.expected_transition_id


def test_compiled_projection_clamps_extrapolation() -> None:
    from zeromodel.perception import encode_source_array as _encode

    manifest, sources, schema = _build()
    model = fit_compiled_transition_model(
        manifest, sources, schema, training_split="all"
    )
    far = _encode(
        np.full((_HEIGHT, _WIDTH), 255, dtype=np.uint8),
        SourceImageEncoderSpecDTO(color_space="L"),
    )
    projected = project_expected_transition(model, far, "left", schema)
    assert projected.status == "out_of_distribution"
    for field in projected.fields:
        assert 0.0 <= field.expected_mean_absolute_change <= 1.0
        assert -1.0 <= field.expected_mean_signed_change <= 1.0
        assert 0.0 <= field.expected_after_mean <= 1.0


def test_projection_rejects_schema_mismatch() -> None:
    manifest, sources, schema = _build(count_per_action=2)
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    other_schema = build_grid_field_schema(
        next(iter(sources.values())), tile_width=2, tile_height=2, channel_mode="joint"
    )
    query = next(iter(sources.values()))
    with pytest.raises(
        PerceptionTransitionModelError, match="field schema does not match"
    ):
        project_expected_transition(model, query, "left", other_schema)


def test_baseline_predictor_trains_on_same_manifest() -> None:
    manifest, sources, _ = _build(count_per_action=3)
    predictor = fit_baseline_nearest_neighbor(manifest, sources, training_split="all")
    assert set(predictor.action_labels) == {"left", "right"}
