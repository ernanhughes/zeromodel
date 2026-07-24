from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    BaselineInferenceConfigDTO,
    DiscreteActionSchemaDTO,
    PerceptionInferenceError,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    build_dataset_manifest,
    encode_discrete_action,
    encode_source_array,
    fit_baseline_nearest_neighbor,
    predict_baseline_action,
)


def _dataset(values_and_actions: list[tuple[int, str]]):
    spec = SourceImageEncoderSpecDTO(color_space="L")
    schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    sources = {}
    interactions = []
    for index, (value, action) in enumerate(values_and_actions):
        source = encode_source_array(
            np.full((2, 2), value, dtype=np.uint8), spec
        )
        target = encode_discrete_action(action, schema)
        sources[source.source_vpm_id] = source
        interactions.append(
            RecordedInteractionDTO.from_vpms(
                sequence_id="episode",
                step_index=index,
                source=source,
                target=target,
            )
        )
    manifest = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[spec.encoder_spec_id],
    )
    return manifest, sources, spec


def test_model_identity_is_deterministic() -> None:
    manifest, sources, _ = _dataset([(10, "LEFT"), (240, "RIGHT")])

    first = fit_baseline_nearest_neighbor(manifest, sources, training_split="all")
    second = fit_baseline_nearest_neighbor(manifest, dict(reversed(list(sources.items()))), training_split="all")

    assert first == second
    assert first.model_id == second.model_id
    assert tuple(item.interaction_id for item in first.examples) == tuple(
        sorted(item.interaction_id for item in first.examples)
    )


def test_prediction_selects_nearest_action_and_retains_evidence() -> None:
    manifest, sources, spec = _dataset(
        [(10, "LEFT"), (20, "LEFT"), (230, "RIGHT"), (240, "RIGHT")]
    )
    model = fit_baseline_nearest_neighbor(
        manifest,
        sources,
        training_split="all",
        config=BaselineInferenceConfigDTO(
            neighbor_count=3,
            maximum_distance=0.5,
            minimum_margin=0.05,
        ),
    )
    unknown = encode_source_array(np.full((2, 2), 15, dtype=np.uint8), spec)

    prediction = predict_baseline_action(model, unknown)

    assert prediction.status == "accepted"
    assert prediction.selected_action == "LEFT"
    assert prediction.candidates[0].action_label == "LEFT"
    assert prediction.confidence > 0.5
    assert len(prediction.neighbors) == 3
    assert tuple(item.rank for item in prediction.neighbors) == (1, 2, 3)
    assert prediction.neighbors[0].distance <= prediction.neighbors[1].distance


def test_prediction_identity_is_deterministic() -> None:
    manifest, sources, spec = _dataset([(0, "LEFT"), (255, "RIGHT")])
    model = fit_baseline_nearest_neighbor(manifest, sources, training_split="all")
    unknown = encode_source_array(np.full((2, 2), 5, dtype=np.uint8), spec)

    first = predict_baseline_action(model, unknown)
    second = predict_baseline_action(model, unknown)

    assert first == second
    assert first.prediction_id == second.prediction_id


def test_far_unknown_is_rejected_out_of_distribution() -> None:
    manifest, sources, spec = _dataset([(0, "LEFT"), (10, "LEFT")])
    model = fit_baseline_nearest_neighbor(
        manifest,
        sources,
        training_split="all",
        config=BaselineInferenceConfigDTO(
            neighbor_count=2,
            maximum_distance=0.1,
            minimum_margin=0.0,
        ),
    )
    unknown = encode_source_array(np.full((2, 2), 255, dtype=np.uint8), spec)

    prediction = predict_baseline_action(model, unknown)

    assert prediction.status == "rejected_out_of_distribution"
    assert prediction.selected_action is None
    assert prediction.nearest_distance > 0.1


def test_symmetric_unknown_is_rejected_as_ambiguous() -> None:
    manifest, sources, spec = _dataset([(0, "LEFT"), (255, "RIGHT")])
    model = fit_baseline_nearest_neighbor(
        manifest,
        sources,
        training_split="all",
        config=BaselineInferenceConfigDTO(
            neighbor_count=2,
            maximum_distance=1.0,
            minimum_margin=0.01,
        ),
    )
    unknown = encode_source_array(np.full((2, 2), 127, dtype=np.uint8), spec)

    prediction = predict_baseline_action(model, unknown)

    assert prediction.status == "rejected_ambiguous"
    assert prediction.selected_action is None
    assert prediction.margin < 0.01


def test_fit_rejects_missing_source_and_identity_mismatch() -> None:
    manifest, sources, _ = _dataset([(10, "LEFT")])

    with pytest.raises(PerceptionInferenceError, match="missing SourceVPMDTO"):
        fit_baseline_nearest_neighbor(manifest, {}, training_split="all")

    source_id, source = next(iter(sources.items()))
    changed = replace(source, pixel_digest="sha256:wrong")
    with pytest.raises(PerceptionInferenceError, match="pixel identity"):
        fit_baseline_nearest_neighbor(
            manifest,
            {source_id: changed},
            training_split="all",
        )


def test_fit_and_predict_reject_incompatible_shapes_or_specs() -> None:
    manifest, sources, spec = _dataset([(10, "LEFT")])
    model = fit_baseline_nearest_neighbor(manifest, sources, training_split="all")

    wrong_shape = encode_source_array(np.zeros((3, 3), dtype=np.uint8), spec)
    with pytest.raises(PerceptionInferenceError, match="shape"):
        predict_baseline_action(model, wrong_shape)

    other_spec = SourceImageEncoderSpecDTO(color_space="L", max_width=100)
    wrong_spec = encode_source_array(np.zeros((2, 2), dtype=np.uint8), other_spec)
    with pytest.raises(PerceptionInferenceError, match="encoder spec"):
        predict_baseline_action(model, wrong_spec)


def test_config_rejects_invalid_thresholds() -> None:
    with pytest.raises(PerceptionInferenceError, match="neighbor_count"):
        BaselineInferenceConfigDTO(neighbor_count=0)
    with pytest.raises(PerceptionInferenceError, match="maximum_distance"):
        BaselineInferenceConfigDTO(maximum_distance=1.1)
    with pytest.raises(PerceptionInferenceError, match="minimum_margin"):
        BaselineInferenceConfigDTO(minimum_margin=-0.1)
