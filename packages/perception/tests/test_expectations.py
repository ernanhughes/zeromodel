from __future__ import annotations

import numpy as np

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    EvidenceExpectationDTO,
    PerceptionRegionAnnotationDTO,
    RecordedInteractionDTO,
    SourceImageEncoderSpecDTO,
    TranslatorConfigDTO,
    build_dataset_manifest,
    build_grid_field_schema,
    encode_discrete_action,
    encode_source_array,
    evaluate_evidence_conformance,
    fit_source_target_translator,
)


def _fixture():
    spec = SourceImageEncoderSpecDTO(color_space="L")
    actions = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    arrays = [
        np.array([[0, 0], [0, 0]], dtype=np.uint8),
        np.array([[16, 0], [16, 0]], dtype=np.uint8),
        np.array([[255, 0], [255, 0]], dtype=np.uint8),
        np.array([[240, 0], [240, 0]], dtype=np.uint8),
    ]
    labels = ["LEFT", "LEFT", "RIGHT", "RIGHT"]
    sources = [encode_source_array(array, spec) for array in arrays]
    interactions = [
        RecordedInteractionDTO.from_vpms(
            sequence_id="conformance",
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
    translator = fit_source_target_translator(
        manifest,
        {source.source_vpm_id: source for source in sources},
        schema,
        actions,
        training_split="all",
        config=TranslatorConfigDTO(ridge_alpha=1e-6),
    )
    fields = tuple(sorted(schema.fields, key=lambda field: field.x0))
    signal = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[0].field_id,),
        label="signal",
        role="expected",
    )
    irrelevant = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[1].field_id,),
        label=None,
        role="control",
    )
    return schema, translator, signal, irrelevant


def test_confirmed_expectation_is_deterministic() -> None:
    schema, translator, signal, irrelevant = _fixture()
    expectation = EvidenceExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        source_annotation_ids=(signal.annotation_id,),
        expected_action_labels=("RIGHT",),
        minimum_registration=0.5,
        maximum_unexplained_registration=0.1,
    )

    first = evaluate_evidence_conformance(
        translator,
        schema,
        (signal, irrelevant),
        (),
        expectation,
    )
    second = evaluate_evidence_conformance(
        translator,
        schema,
        (signal, irrelevant),
        (),
        expectation,
    )

    assert first == second
    assert first.overall_status == "confirmed"
    assert first.findings[0].status == "confirmed"


def test_wrong_target_placement_is_preserved_as_finding() -> None:
    schema, translator, signal, irrelevant = _fixture()
    expectation = EvidenceExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        source_annotation_ids=(signal.annotation_id,),
        expected_action_labels=("LEFT",),
        minimum_registration=0.5,
    )

    report = evaluate_evidence_conformance(
        translator,
        schema,
        (signal, irrelevant),
        (),
        expectation,
    )

    assert report.overall_status in {"wrong_target_placement", "missing_expected_evidence"}
    assert report.overall_status != "confirmed"


def test_missing_threshold_is_inconclusive() -> None:
    schema, translator, signal, irrelevant = _fixture()
    expectation = EvidenceExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        source_annotation_ids=(signal.annotation_id,),
        expected_action_labels=("RIGHT",),
    )

    report = evaluate_evidence_conformance(
        translator,
        schema,
        (signal, irrelevant),
        (),
        expectation,
    )

    assert report.overall_status == "inconclusive"


def test_forbidden_evidence_is_not_collapsed_into_boolean() -> None:
    schema, translator, signal, irrelevant = _fixture()
    expectation = EvidenceExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        source_annotation_ids=(irrelevant.annotation_id,),
        expected_action_labels=("RIGHT",),
        forbidden_annotation_ids=(signal.annotation_id,),
        minimum_registration=0.2,
    )

    report = evaluate_evidence_conformance(
        translator,
        schema,
        (signal, irrelevant),
        (),
        expectation,
    )

    assert report.overall_status == "forbidden_evidence_present"
    assert any(item.status == "forbidden_evidence_present" for item in report.findings)
