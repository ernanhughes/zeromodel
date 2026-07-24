from __future__ import annotations

import numpy as np
import pytest

from zeromodel.perception import (
    BaselineInferenceConfigDTO,
    DiscreteActionSchemaDTO,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    build_dataset_manifest,
    build_grid_field_schema,
    encode_discrete_action,
    encode_source_array,
    estimate_field_relevance,
    evaluate_evidence_interventions,
    fit_evidence_weighted_nearest_neighbor,
    predict_evidence_weighted_action,
)


def _fixture():
    spec = SourceImageEncoderSpecDTO(color_space="L")
    actions = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    arrays = [
        np.array([[0, 0], [0, 255]], dtype=np.uint8),
        np.array([[0, 255], [0, 0]], dtype=np.uint8),
        np.array([[255, 0], [255, 255]], dtype=np.uint8),
        np.array([[255, 255], [255, 0]], dtype=np.uint8),
    ]
    labels = ["LEFT", "LEFT", "RIGHT", "RIGHT"]
    sources = [encode_source_array(array, spec) for array in arrays]
    interactions = [
        RecordedInteractionDTO.from_vpms(
            sequence_id="weighted",
            step_index=index,
            source=source,
            target=encode_discrete_action(label, actions),
        )
        for index, (source, label) in enumerate(zip(sources, labels, strict=True))
    ]
    manifest = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[spec.encoder_spec_id],
    )
    schema = build_grid_field_schema(sources[0], tile_width=1, tile_height=2)
    mapping = {source.source_vpm_id: source for source in sources}
    evidence = estimate_field_relevance(manifest, mapping, schema, training_split="all")
    model = fit_evidence_weighted_nearest_neighbor(
        manifest,
        mapping,
        schema,
        evidence,
        training_split="all",
        config=BaselineInferenceConfigDTO(
            neighbor_count=3,
            maximum_distance=1.0,
            minimum_margin=0.0,
        ),
    )
    return spec, schema, evidence, model


def test_weighted_inference_uses_discriminative_field() -> None:
    spec, schema, evidence, model = _fixture()
    query = encode_source_array(
        np.array([[5, 255], [5, 255]], dtype=np.uint8),
        spec,
    )

    prediction = predict_evidence_weighted_action(model, query, schema)

    assert prediction.selected_action == "LEFT"
    assert prediction.distance_semantics == (
        "field_relevance_weighted_normalized_mean_absolute_distance"
    )
    assert max(item.score for item in evidence.relevances) == pytest.approx(1.0)


def test_interventions_are_deterministic_and_preserve_controls() -> None:
    spec, schema, _, model = _fixture()
    query = encode_source_array(
        np.array([[5, 255], [5, 0]], dtype=np.uint8),
        spec,
    )

    first = evaluate_evidence_interventions(
        model,
        query,
        schema,
        selected_field_count=1,
        random_seed=17,
    )
    second = evaluate_evidence_interventions(
        model,
        query,
        schema,
        selected_field_count=1,
        random_seed=17,
    )

    assert first == second
    assert first.keep_only_sufficiency is True
    assert len(first.selected_field_ids) == 1
    assert len(first.random_field_ids) == 1
    assert first.selected_field_ids != first.random_field_ids
    assert first.full.method == "full"
    assert first.keep_only.method == "keep_only"
    assert first.remove_only.method == "remove_only"
