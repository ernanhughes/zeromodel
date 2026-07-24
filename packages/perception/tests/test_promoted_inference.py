from __future__ import annotations

import numpy as np
import pytest

from zeromodel.perception import (
    COEFFICIENT_SEMANTICS,
    SOURCE_FEATURE_SEMANTICS,
    TARGET_SCORE_SEMANTICS,
    DiscreteActionSchemaDTO,
    PerceptionPromotedInferenceError,
    PromotedPerceptionModelDTO,
    SourceImageEncoderSpecDTO,
    SourceTargetTranslatorDTO,
    TranslatorConfigDTO,
    build_grid_field_schema,
    encode_source_array,
    run_promoted_inference,
)


def _single_fixture() -> tuple[
    PromotedPerceptionModelDTO,
    SourceTargetTranslatorDTO,
    object,
    DiscreteActionSchemaDTO,
    object,
]:
    source = encode_source_array(
        np.asarray([[255]], dtype=np.uint8),
        SourceImageEncoderSpecDTO(color_space="L"),
    )
    field_schema = build_grid_field_schema(source, tile_width=1, tile_height=1)
    action_schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    translator = SourceTargetTranslatorDTO(
        translator_id="sha256:single",
        dataset_id="sha256:dataset",
        source_field_schema_id=field_schema.field_schema_id,
        source_encoder_spec_id=field_schema.source_encoder_spec_id,
        action_schema_id=action_schema.action_schema_id,
        action_labels=action_schema.labels,
        source_field_ids=tuple(field.field_id for field in field_schema.fields),
        coefficients=((-1.0,), (1.0,)),
        intercepts=(1.0, 0.0),
        training_split="train",
        training_count=2,
        source_feature_semantics=SOURCE_FEATURE_SEMANTICS,
        target_score_semantics=TARGET_SCORE_SEMANTICS,
        coefficient_semantics=COEFFICIENT_SEMANTICS,
        config=TranslatorConfigDTO(),
    )
    promoted = PromotedPerceptionModelDTO(
        promoted_model_id="sha256:promoted",
        model_kind="single_frame",
        model_id=translator.translator_id,
        rejection_threshold=0.5,
        calibration_id="sha256:calibration",
        promotion_decision_id="sha256:decision",
        validation_comparison_report_id="sha256:validation",
        training_split="train",
        evaluation_split="validation",
    )
    return promoted, translator, field_schema, action_schema, source


def test_promoted_single_frame_runtime_applies_frozen_threshold() -> None:
    promoted, translator, field_schema, action_schema, source = _single_fixture()

    first = run_promoted_inference(
        promoted,
        action_schema,
        single_translator=translator,
        single_field_schema=field_schema,
        source=source,
    )
    second = run_promoted_inference(
        promoted,
        action_schema,
        single_translator=translator,
        single_field_schema=field_schema,
        source=source,
    )

    assert first == second
    assert first.selected_action == "RIGHT"
    assert first.status == "accepted"
    assert first.rejection_threshold == 0.5
    assert first.model_id == promoted.model_id
    assert first.calibration_id == promoted.calibration_id


def test_promoted_runtime_rejects_model_identity_mismatch() -> None:
    promoted, translator, field_schema, action_schema, source = _single_fixture()
    wrong = PromotedPerceptionModelDTO(
        promoted_model_id=promoted.promoted_model_id,
        model_kind="single_frame",
        model_id="sha256:wrong",
        rejection_threshold=promoted.rejection_threshold,
        calibration_id=promoted.calibration_id,
        promotion_decision_id=promoted.promotion_decision_id,
        validation_comparison_report_id=promoted.validation_comparison_report_id,
        training_split="train",
        evaluation_split="validation",
    )

    with pytest.raises(PerceptionPromotedInferenceError, match="does not match"):
        run_promoted_inference(
            wrong,
            action_schema,
            single_translator=translator,
            single_field_schema=field_schema,
            source=source,
        )


def test_promoted_runtime_rejects_temporal_input_for_single_model() -> None:
    promoted, translator, field_schema, action_schema, source = _single_fixture()

    with pytest.raises(PerceptionPromotedInferenceError, match="temporal source"):
        run_promoted_inference(
            promoted,
            action_schema,
            single_translator=translator,
            single_field_schema=field_schema,
            source=source,
            temporal_source=object(),  # type: ignore[arg-type]
        )
