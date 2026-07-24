from __future__ import annotations

import numpy as np

from zeromodel.perception import (
    COEFFICIENT_SEMANTICS,
    SOURCE_FEATURE_SEMANTICS,
    TARGET_SCORE_SEMANTICS,
    EvidenceConformanceFindingDTO,
    EvidenceConformanceReportDTO,
    EvidenceExpectationDTO,
    PerceptionRegionAnnotationDTO,
    SourceImageEncoderSpecDTO,
    SourceTargetTranslatorDTO,
    TranslatorConfigDTO,
    build_grid_field_schema,
    discover_unexpected_evidence,
    encode_source_array,
)


def _fixture():
    spec = SourceImageEncoderSpecDTO(color_space="L")
    source = encode_source_array(np.zeros((1, 2), dtype=np.uint8), spec)
    schema = build_grid_field_schema(source, tile_width=1, tile_height=1)
    field_ids = tuple(field.field_id for field in schema.fields)
    translator = SourceTargetTranslatorDTO(
        translator_id="translator",
        dataset_id="dataset",
        source_field_schema_id=schema.field_schema_id,
        source_encoder_spec_id=schema.source_encoder_spec_id,
        action_schema_id="actions",
        action_labels=("LEFT", "RIGHT"),
        source_field_ids=field_ids,
        coefficients=((0.9, 0.1), (0.2, 0.8)),
        intercepts=(0.0, 0.0),
        training_split="train",
        training_count=4,
        source_feature_semantics=SOURCE_FEATURE_SEMANTICS,
        target_score_semantics=TARGET_SCORE_SEMANTICS,
        coefficient_semantics=COEFFICIENT_SEMANTICS,
        config=TranslatorConfigDTO(),
    )
    annotation = PerceptionRegionAnnotationDTO.create(
        schema,
        (field_ids[0],),
        label="declared",
    )
    expectation = EvidenceExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        source_annotation_ids=(annotation.annotation_id,),
        expected_action_labels=("LEFT",),
        minimum_registration=0.5,
    )
    finding = EvidenceConformanceFindingDTO(
        finding_id="finding",
        status="confirmed",
        expectation_id=expectation.expectation_id,
        action_labels=("LEFT",),
        annotation_ids=(annotation.annotation_id,),
        detail="fixture",
    )
    conformance = EvidenceConformanceReportDTO(
        report_id="conformance",
        translator_id=translator.translator_id,
        field_schema_id=schema.field_schema_id,
        expectation_id=expectation.expectation_id,
        registrations=(),
        unexplained_registration_by_action=(("LEFT", 0.1), ("RIGHT", 0.8)),
        findings=(finding,),
        overall_status="confirmed",
    )
    return schema, translator, annotation, expectation, conformance, field_ids


def test_discovery_materializes_four_surfaces_per_action() -> None:
    schema, translator, annotation, expectation, conformance, _ = _fixture()

    report = discover_unexpected_evidence(
        translator,
        schema,
        (annotation,),
        expectation,
        conformance,
        contribution_threshold=0.25,
    )

    assert len(report.surfaces) == 8
    assert {
        item.surface_kind for item in report.surfaces
    } == {"observed", "expected", "difference", "unexplained"}
    assert all(item.to_array().shape == (1, 2) for item in report.surfaces)


def test_discovery_preserves_addressable_unexplained_evidence() -> None:
    schema, translator, annotation, expectation, conformance, field_ids = _fixture()

    report = discover_unexpected_evidence(
        translator,
        schema,
        (annotation,),
        expectation,
        conformance,
        contribution_threshold=0.25,
        recurrence_by_action_field={("RIGHT", field_ids[1]): 7},
        stability_by_action_field={("RIGHT", field_ids[1]): 0.75},
        intervention_effect_by_action_field={("RIGHT", field_ids[1]): -0.4},
    )

    item = next(
        value
        for value in report.unexplained_evidence
        if value.action_label == "RIGHT" and value.field_ids == (field_ids[1],)
    )
    assert item.contribution_score == 0.8
    assert item.recurrence_count == 7
    assert item.stability == 0.75
    assert item.intervention_effect == -0.4
    assert item.suggested_labels == ()


def test_discovery_identity_is_deterministic() -> None:
    schema, translator, annotation, expectation, conformance, _ = _fixture()

    first = discover_unexpected_evidence(
        translator,
        schema,
        (annotation,),
        expectation,
        conformance,
    )
    second = discover_unexpected_evidence(
        translator,
        schema,
        (annotation,),
        expectation,
        conformance,
    )

    assert first == second
