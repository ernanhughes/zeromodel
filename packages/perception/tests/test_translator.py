from __future__ import annotations

import numpy as np
import pytest

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    PerceptionTranslatorError,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    TranslatorConfigDTO,
    build_dataset_manifest,
    build_grid_field_schema,
    encode_discrete_action,
    encode_source_array,
    fit_source_target_translator,
    predict_target_vpm,
)


def _fixture():
    spec = SourceImageEncoderSpecDTO(color_space="L")
    actions = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    arrays = [
        np.array([[0, 30, 120], [0, 30, 120]], dtype=np.uint8),
        np.array([[10, 80, 20], [10, 80, 20]], dtype=np.uint8),
        np.array([[245, 30, 120], [245, 30, 120]], dtype=np.uint8),
        np.array([[255, 80, 20], [255, 80, 20]], dtype=np.uint8),
    ]
    labels = ["LEFT", "LEFT", "RIGHT", "RIGHT"]
    sources = [encode_source_array(array, spec) for array in arrays]
    interactions = [
        RecordedInteractionDTO.from_vpms(
            sequence_id="translator",
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
    field_schema = build_grid_field_schema(
        sources[0], tile_width=1, tile_height=2
    )
    mapping = {source.source_vpm_id: source for source in sources}
    return spec, actions, sources, manifest, field_schema, mapping


def test_translator_is_deterministic_and_rectangular() -> None:
    _, actions, _, manifest, field_schema, mapping = _fixture()

    first = fit_source_target_translator(
        manifest,
        mapping,
        field_schema,
        actions,
        training_split="all",
        config=TranslatorConfigDTO(ridge_alpha=1e-6),
    )
    second = fit_source_target_translator(
        manifest,
        mapping,
        field_schema,
        actions,
        training_split="all",
        config=TranslatorConfigDTO(ridge_alpha=1e-6),
    )

    assert first == second
    assert len(first.coefficients) == 2
    assert all(len(row) == 3 for row in first.coefficients)
    assert first.training_count == 4


def test_translator_predicts_continuous_target_surface() -> None:
    spec, actions, _, manifest, field_schema, mapping = _fixture()
    translator = fit_source_target_translator(
        manifest, mapping, field_schema, actions, training_split="all"
    )
    query = encode_source_array(
        np.array([[5, 255, 0], [5, 255, 0]], dtype=np.uint8), spec
    )

    prediction = predict_target_vpm(translator, query, field_schema, actions)

    assert prediction.selected_action == "LEFT"
    assert prediction.width == 2
    assert prediction.to_array().shape == (1, 2)
    assert prediction.scores[0].score >= prediction.scores[1].score
    assert prediction.margin >= 0.0


def test_translator_coefficients_are_addressable() -> None:
    _, actions, _, manifest, field_schema, mapping = _fixture()
    translator = fit_source_target_translator(
        manifest, mapping, field_schema, actions, training_split="all"
    )

    value = translator.coefficient_for("RIGHT", translator.source_field_ids[0])

    assert isinstance(value, float)
    with pytest.raises(KeyError):
        translator.coefficient_for("MISSING", translator.source_field_ids[0])


def test_translator_rejects_schema_mismatch() -> None:
    _, actions, _, manifest, field_schema, mapping = _fixture()
    other = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT", "WAIT"])

    with pytest.raises(PerceptionTranslatorError):
        fit_source_target_translator(
            manifest,
            mapping,
            field_schema,
            other,
            training_split="all",
        )


def test_translator_rejects_invalid_alpha() -> None:
    with pytest.raises(PerceptionTranslatorError):
        TranslatorConfigDTO(ridge_alpha=-1.0)
