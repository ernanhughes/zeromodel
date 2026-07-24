from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    PerceptionTranslatorEvaluationError,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    SplitAssignmentDTO,
    build_dataset_manifest,
    build_grid_field_schema,
    calibrate_translator_rejection,
    encode_discrete_action,
    encode_source_array,
    evaluate_source_target_translator,
    fit_source_target_translator,
    predict_calibrated_target_vpm,
)


def _fixture():
    spec = SourceImageEncoderSpecDTO(color_space="L")
    actions = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    arrays = [
        np.array([[0, 0]], dtype=np.uint8),
        np.array([[10, 0]], dtype=np.uint8),
        np.array([[245, 0]], dtype=np.uint8),
        np.array([[255, 0]], dtype=np.uint8),
        np.array([[5, 0]], dtype=np.uint8),
        np.array([[250, 0]], dtype=np.uint8),
    ]
    labels = ["LEFT", "LEFT", "RIGHT", "RIGHT", "LEFT", "RIGHT"]
    sources = [encode_source_array(array, spec) for array in arrays]
    interactions = [
        RecordedInteractionDTO.from_vpms(
            sequence_id="evaluation",
            step_index=index,
            source=source,
            target=encode_discrete_action(label, actions),
        )
        for index, (source, label) in enumerate(zip(sources, labels, strict=True))
    ]
    base = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[spec.encoder_spec_id],
    )
    ordered = base.interactions
    assignments = tuple(
        SplitAssignmentDTO(
            interaction_id=item.interaction_id,
            split="train" if item.step_index < 4 else "validation",
        )
        for item in ordered
    )
    manifest = replace(base, split_assignments=assignments)
    mapping = {source.source_vpm_id: source for source in sources}
    schema = build_grid_field_schema(sources[0], tile_width=1, tile_height=1)
    translator = fit_source_target_translator(
        manifest,
        mapping,
        schema,
        actions,
        training_split="train",
    )
    return actions, manifest, mapping, schema, translator, sources


def test_evaluation_is_held_out_and_deterministic() -> None:
    actions, manifest, mapping, schema, translator, _ = _fixture()
    first = evaluate_source_target_translator(
        translator,
        manifest,
        mapping,
        schema,
        actions,
        evaluation_split="validation",
    )
    second = evaluate_source_target_translator(
        translator,
        manifest,
        mapping,
        schema,
        actions,
        evaluation_split="validation",
    )
    assert first == second
    assert first.example_count == 2
    assert 0.0 <= first.accuracy <= 1.0
    assert first.coefficient_count == len(translator.action_labels) * len(translator.source_field_ids)


def test_calibration_and_rejection_contract() -> None:
    actions, manifest, mapping, schema, translator, sources = _fixture()
    report = evaluate_source_target_translator(
        translator,
        manifest,
        mapping,
        schema,
        actions,
        evaluation_split="validation",
    )
    calibration = calibrate_translator_rejection(report, target_retained_accuracy=0.5)
    prediction = predict_calibrated_target_vpm(
        translator,
        calibration,
        sources[0],
        schema,
        actions,
    )
    assert prediction.status in {"accepted", "rejected_ambiguous"}
    assert prediction.minimum_margin == calibration.minimum_margin


def test_training_split_cannot_be_reported_as_held_out() -> None:
    actions, manifest, mapping, schema, translator, _ = _fixture()
    with pytest.raises(PerceptionTranslatorEvaluationError):
        evaluate_source_target_translator(
            translator,
            manifest,
            mapping,
            schema,
            actions,
            evaluation_split="train",
        )
