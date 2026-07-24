from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    PerceptionEvidenceError,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    build_dataset_manifest,
    build_grid_field_schema,
    encode_discrete_action,
    encode_source_array,
    estimate_field_relevance,
)


def _fixture():
    spec = SourceImageEncoderSpecDTO(color_space="L")
    actions = DiscreteActionSchemaDTO.from_labels(["A", "B"])
    arrays = [
        np.array([[0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.uint8),
        np.array([[0, 0, 255, 255], [0, 0, 255, 255]], dtype=np.uint8),
        np.array([[255, 255, 0, 0], [255, 255, 0, 0]], dtype=np.uint8),
        np.full((2, 4), 255, dtype=np.uint8),
    ]
    labels = ["A", "A", "B", "B"]
    sources = [encode_source_array(array, spec) for array in arrays]
    interactions = [
        RecordedInteractionDTO.from_vpms(
            sequence_id="evidence",
            step_index=index,
            source=source,
            target=encode_discrete_action(label, actions),
        )
        for index, (source, label) in enumerate(zip(sources, labels, strict=True))
    ]
    manifest = build_dataset_manifest(
        interactions,
        source_encoder_spec_ids=[spec.encoder_spec_id],
        split_seed="p4b-fixture",
    )
    schema = build_grid_field_schema(
        sources[0], tile_width=2, tile_height=2, channel_mode="joint"
    )
    return manifest, {item.source_vpm_id: item for item in sources}, schema


def _left_and_right(schema):
    left = next(field for field in schema.fields if field.x0 == 0)
    right = next(field for field in schema.fields if field.x0 == 2)
    return left, right


def test_relevance_identifies_action_discriminative_field() -> None:
    manifest, sources, schema = _fixture()
    evidence = estimate_field_relevance(
        manifest, sources, schema, training_split="all"
    )
    left, right = _left_and_right(schema)

    assert evidence.relevance_for(left.field_id).score == pytest.approx(1.0)
    assert evidence.relevance_for(right.field_id).score == pytest.approx(0.0)
    rendered = evidence.to_array()
    assert np.all(rendered[:, :2] == 255)
    assert np.all(rendered[:, 2:] == 0)


def test_evidence_identity_and_png_are_deterministic() -> None:
    manifest, sources, schema = _fixture()
    first = estimate_field_relevance(manifest, sources, schema, training_split="all")
    second = estimate_field_relevance(manifest, sources, schema, training_split="all")

    assert first == second
    assert first.evidence_vpm_id == second.evidence_vpm_id
    assert first.png_bytes == second.png_bytes


def test_separate_channel_render_uses_max_while_preserving_exact_scores() -> None:
    spec = SourceImageEncoderSpecDTO(color_space="RGB")
    actions = DiscreteActionSchemaDTO.from_labels(["A", "B"])
    arrays = [
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.array([[[0, 255, 0]]], dtype=np.uint8),
        np.array([[[255, 0, 0]]], dtype=np.uint8),
        np.array([[[255, 255, 0]]], dtype=np.uint8),
    ]
    labels = ["A", "A", "B", "B"]
    sources = [encode_source_array(array, spec) for array in arrays]
    interactions = [
        RecordedInteractionDTO.from_vpms(
            sequence_id="channels",
            step_index=index,
            source=source,
            target=encode_discrete_action(label, actions),
        )
        for index, (source, label) in enumerate(zip(sources, labels, strict=True))
    ]
    manifest = build_dataset_manifest(
        interactions, source_encoder_spec_ids=[spec.encoder_spec_id]
    )
    schema = build_grid_field_schema(
        sources[0], tile_width=1, tile_height=1, channel_mode="separate"
    )
    evidence = estimate_field_relevance(
        manifest,
        {item.source_vpm_id: item for item in sources},
        schema,
        training_split="all",
    )

    scores = {
        field.channel_start: evidence.relevance_for(field.field_id).score
        for field in schema.fields
    }
    assert scores[0] == pytest.approx(1.0)
    assert scores[1] == pytest.approx(0.0)
    assert scores[2] == pytest.approx(0.0)
    assert evidence.to_array()[0, 0] == 255


def test_relevance_rejects_missing_or_tampered_sources() -> None:
    manifest, sources, schema = _fixture()
    missing = dict(sources)
    missing.pop(next(iter(missing)))
    with pytest.raises(PerceptionEvidenceError, match="missing SourceVPMDTO"):
        estimate_field_relevance(manifest, missing, schema, training_split="all")

    first_id = manifest.interactions[0].source_vpm_id
    tampered = dict(sources)
    tampered[first_id] = replace(tampered[first_id], pixel_digest="sha256:wrong")
    with pytest.raises(PerceptionEvidenceError, match="pixel identity"):
        estimate_field_relevance(manifest, tampered, schema, training_split="all")


def test_relevance_requires_valid_split_and_two_actions() -> None:
    manifest, sources, schema = _fixture()
    with pytest.raises(PerceptionEvidenceError, match="training_split"):
        estimate_field_relevance(manifest, sources, schema, training_split="invalid")

    one_action_manifest = replace(
        manifest,
        interactions=tuple(
            replace(item, action_label="A") for item in manifest.interactions
        ),
    )
    with pytest.raises(PerceptionEvidenceError, match="at least two actions"):
        estimate_field_relevance(
            one_action_manifest, sources, schema, training_split="all"
        )


def test_evidence_dto_detects_png_tampering() -> None:
    manifest, sources, schema = _fixture()
    evidence = estimate_field_relevance(
        manifest, sources, schema, training_split="all"
    )
    with pytest.raises(PerceptionEvidenceError, match="PNG digest"):
        replace(evidence, png_bytes=evidence.png_bytes + b"tamper")
