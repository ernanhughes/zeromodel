from __future__ import annotations

import numpy as np
import pytest

from zeromodel.perception import (
    COEFFICIENT_SEMANTICS,
    SOURCE_FEATURE_SEMANTICS,
    TARGET_SCORE_SEMANTICS,
    DiscreteActionSchemaDTO,
    PerceptionTemporalInferenceError,
    SourceImageEncoderSpecDTO,
    SourceTargetTranslatorDTO,
    TemporalSourceVPMDTO,
    TemporalWindowSpecDTO,
    TranslatorConfigDTO,
    build_grid_field_schema,
    compare_single_and_temporal_inference,
    encode_source_array,
    fit_temporal_translator,
    predict_temporal_action,
)


def _temporal(value0: int, value1: int, action: str, index: int) -> TemporalSourceVPMDTO:
    base_spec = SourceImageEncoderSpecDTO(color_space="L")
    first = encode_source_array(np.full((1, 1), value0, dtype=np.uint8), base_spec)
    current = encode_source_array(np.full((1, 1), value1, dtype=np.uint8), base_spec)
    montage_spec = SourceImageEncoderSpecDTO(
        color_space="L",
        max_width=2,
        max_height=1,
        max_pixels=2,
        version="test-temporal-montage/1",
    )
    montage = encode_source_array(
        np.asarray([[value0, value1]], dtype=np.uint8), montage_spec
    )
    spec = TemporalWindowSpecDTO(frame_count=2)
    return TemporalSourceVPMDTO(
        temporal_source_id=f"sha256:temporal-{index}",
        temporal_window_spec_id=spec.temporal_window_spec_id,
        sequence_id=f"sequence-{index}",
        target_interaction_id=f"interaction-{index}",
        target_step_index=1,
        action_label=action,
        frame_source_vpm_ids=(first.source_vpm_id, current.source_vpm_id),
        frame_pixel_digests=(first.pixel_digest, current.pixel_digest),
        current_source_vpm_id=current.source_vpm_id,
        current_pixel_digest=current.pixel_digest,
        montage_source_vpm=montage,
    )


def test_temporal_translator_is_deterministic_and_predicts_context() -> None:
    action_schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    spec = TemporalWindowSpecDTO(frame_count=2)
    examples = (
        _temporal(0, 128, "LEFT", 0),
        _temporal(255, 128, "RIGHT", 1),
        _temporal(10, 128, "LEFT", 2),
        _temporal(245, 128, "RIGHT", 3),
    )
    schema = build_grid_field_schema(
        examples[0].montage_source_vpm, tile_width=1, tile_height=1
    )

    first = fit_temporal_translator(examples, spec, schema, action_schema)
    second = fit_temporal_translator(tuple(reversed(examples)), spec, schema, action_schema)

    assert first.temporal_translator_id == second.temporal_translator_id
    assert first.coefficient_semantics == COEFFICIENT_SEMANTICS
    assert len(first.coefficients) == 2
    assert len(first.coefficients[0]) == 2

    prediction = predict_temporal_action(first, examples[1], schema)
    assert prediction.selected_action == "RIGHT"
    assert prediction.correct
    assert prediction.status == "accepted"


def test_temporal_prediction_supports_explicit_ambiguity_rejection() -> None:
    action_schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    spec = TemporalWindowSpecDTO(frame_count=2)
    examples = (
        _temporal(0, 128, "LEFT", 0),
        _temporal(255, 128, "RIGHT", 1),
    )
    schema = build_grid_field_schema(
        examples[0].montage_source_vpm, tile_width=1, tile_height=1
    )
    translator = fit_temporal_translator(examples, spec, schema, action_schema)

    prediction = predict_temporal_action(
        translator, examples[0], schema, rejection_threshold=1.0
    )

    assert prediction.status == "rejected_ambiguous"
    assert prediction.rejection_threshold == 1.0


def test_comparison_rejects_training_split_reuse() -> None:
    action_schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    spec = TemporalWindowSpecDTO(frame_count=2)
    examples = (
        _temporal(0, 128, "LEFT", 0),
        _temporal(255, 128, "RIGHT", 1),
    )
    temporal_schema = build_grid_field_schema(
        examples[0].montage_source_vpm, tile_width=1, tile_height=1
    )
    temporal = fit_temporal_translator(
        examples, spec, temporal_schema, action_schema, training_split="validation"
    )
    current = encode_source_array(
        np.asarray([[128]], dtype=np.uint8), SourceImageEncoderSpecDTO(color_space="L")
    )
    single_schema = build_grid_field_schema(current, tile_width=1, tile_height=1)
    single = SourceTargetTranslatorDTO(
        translator_id="sha256:single",
        dataset_id="sha256:dataset",
        source_field_schema_id=single_schema.field_schema_id,
        source_encoder_spec_id=single_schema.source_encoder_spec_id,
        action_schema_id=action_schema.action_schema_id,
        action_labels=action_schema.labels,
        source_field_ids=tuple(field.field_id for field in single_schema.fields),
        coefficients=((0.0,), (0.0,)),
        intercepts=(0.5, 0.5),
        training_split="train",
        training_count=2,
        source_feature_semantics=SOURCE_FEATURE_SEMANTICS,
        target_score_semantics=TARGET_SCORE_SEMANTICS,
        coefficient_semantics=COEFFICIENT_SEMANTICS,
        config=TranslatorConfigDTO(),
    )

    with pytest.raises(PerceptionTemporalInferenceError, match="held out"):
        compare_single_and_temporal_inference(
            single,
            temporal,
            single_schema,
            temporal_schema,
            examples,
            {examples[0].current_source_vpm_id: current},
            action_schema,
            split="validation",
        )
