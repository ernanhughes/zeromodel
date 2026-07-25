from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    PerceptionRegionAnnotationDTO,
    RelationAnnotationDTO,
    SourceImageEncoderSpecDTO,
    build_grid_field_schema,
    build_transition_evidence_vpm,
    encode_source_array,
)
from zeromodel.perception.transition_conformance import (
    PerceptionTransitionConformanceError,
    TransitionExpectationDTO,
    evaluate_transition_conformance,
)


def _source(values: list[int]):
    encoder = SourceImageEncoderSpecDTO(color_space="L")
    return encode_source_array(np.array([values], dtype=np.uint8), encoder)


def _annotations(schema, labels: tuple[str, ...]):
    fields = tuple(sorted(schema.fields, key=lambda item: item.x0))
    annotations = tuple(
        PerceptionRegionAnnotationDTO.create(
            schema,
            (field.field_id,),
            label=label,
            role="component",
        )
        for field, label in zip(fields, labels, strict=True)
    )
    return fields, annotations


def test_conformance_distinguishes_confirmed_missing_and_unexplained() -> None:
    before = _source([0, 0, 0, 0, 0, 0])
    after = _source([255, 255, 0, 0, 128, 128])
    schema = build_grid_field_schema(before, tile_width=2, tile_height=1)
    fields = tuple(sorted(schema.fields, key=lambda item: item.x0))
    tank = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[0].field_id,),
        label="tank",
        role="actor",
    )
    alien = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[1].field_id,),
        label="alien",
        role="target",
    )
    transition = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(tank, alien),
    )
    tank_expectation = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(tank.annotation_id,),
        expected_change="increase",
        minimum_mean_absolute_change=0.5,
        minimum_changed_fraction=1.0,
        minimum_signed_change_magnitude=0.5,
    )
    alien_expectation = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(alien.annotation_id,),
        expected_change="change",
        minimum_mean_absolute_change=0.1,
    )

    first = evaluate_transition_conformance(
        transition,
        (tank_expectation, alien_expectation),
        (tank, alien),
        minimum_unexplained_mean_absolute_change=0.1,
    )
    second = evaluate_transition_conformance(
        transition,
        (alien_expectation, tank_expectation),
        (alien, tank),
        minimum_unexplained_mean_absolute_change=0.1,
    )

    assert first == second
    assert first.status == "nonconformant"
    assert {item.status for item in first.findings} == {
        "confirmed",
        "missing_expected_change",
        "unexplained_change",
    }
    unexplained = first.findings_for_status("unexplained_change")
    assert len(unexplained) == 1
    assert unexplained[0].field_ids == (fields[2].field_id,)
    assert transition.fields == transition.fields


def test_conformance_preserves_distinct_failure_modes() -> None:
    before = _source([0, 0, 0, 0, 255, 255, 0, 255])
    after = _source([102, 102, 204, 204, 102, 102, 153, 102])
    schema = build_grid_field_schema(before, tile_width=2, tile_height=1)
    _, annotations = _annotations(schema, ("control", "projectile", "tank", "mixed"))
    transition = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=annotations,
    )
    control, projectile, tank, mixed = annotations
    expectations = (
        TransitionExpectationDTO.create(
            field_schema_id=schema.field_schema_id,
            annotation_ids=(control.annotation_id,),
            expected_change="stable",
            maximum_mean_absolute_change=0.1,
            maximum_changed_fraction=0.1,
        ),
        TransitionExpectationDTO.create(
            field_schema_id=schema.field_schema_id,
            annotation_ids=(projectile.annotation_id,),
            expected_change="change",
            maximum_mean_absolute_change=0.5,
        ),
        TransitionExpectationDTO.create(
            field_schema_id=schema.field_schema_id,
            annotation_ids=(tank.annotation_id,),
            expected_change="increase",
            minimum_mean_absolute_change=0.2,
            minimum_signed_change_magnitude=0.1,
        ),
        TransitionExpectationDTO.create(
            field_schema_id=schema.field_schema_id,
            annotation_ids=(mixed.annotation_id,),
            expected_change="increase",
            minimum_mean_absolute_change=0.2,
            minimum_signed_change_magnitude=0.1,
        ),
    )

    report = evaluate_transition_conformance(
        transition,
        expectations,
        annotations,
    )

    assert report.status == "nonconformant"
    assert {item.status for item in report.findings} == {
        "unexpected_change",
        "excessive_change",
        "wrong_change_direction",
        "inconclusive",
    }


def test_relation_expectation_uses_explicit_derived_fields() -> None:
    before = _source([0, 0, 0, 0, 0, 0])
    after = _source([0, 0, 0, 0, 102, 102])
    schema = build_grid_field_schema(before, tile_width=2, tile_height=1)
    fields = tuple(sorted(schema.fields, key=lambda item: item.x0))
    tank = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[0].field_id,),
        label="tank",
    )
    alien = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[1].field_id,),
        label="alien",
    )
    relation = RelationAnnotationDTO(
        relation_id="sha256:tank-alien-distance",
        relation_type="relative_distance",
        member_annotation_ids=tuple(sorted((tank.annotation_id, alien.annotation_id))),
        derived_field_ids=(fields[2].field_id,),
    )
    transition = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(tank, alien),
    )
    expectation = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        relation_ids=(relation.relation_id,),
        expected_change="change",
        minimum_mean_absolute_change=0.2,
    )

    report = evaluate_transition_conformance(
        transition,
        (expectation,),
        (tank, alien),
        (relation,),
    )

    assert report.status == "conformant"
    assert report.findings[0].status == "confirmed"
    assert report.findings[0].field_ids == (fields[2].field_id,)
    assert report.findings[0].relation_ids == (relation.relation_id,)


def test_conformance_rejects_binding_mismatch_and_identity_tampering() -> None:
    before = _source([0, 0, 0, 0])
    after = _source([255, 255, 0, 0])
    schema = build_grid_field_schema(before, tile_width=2, tile_height=1)
    fields = tuple(sorted(schema.fields, key=lambda item: item.x0))
    tank = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[0].field_id,),
        label="tank",
    )
    transition = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(tank,),
    )
    expectation = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(tank.annotation_id,),
        expected_change="change",
    )
    report = evaluate_transition_conformance(
        transition,
        (expectation,),
        (tank,),
    )

    with pytest.raises(
        PerceptionTransitionConformanceError,
        match="field bindings",
    ):
        evaluate_transition_conformance(
            transition,
            (expectation,),
            (replace(tank, field_ids=(fields[1].field_id,)),),
        )
    with pytest.raises(
        PerceptionTransitionConformanceError,
        match="at least one expectation",
    ):
        evaluate_transition_conformance(transition, (), (tank,))
    with pytest.raises(
        PerceptionTransitionConformanceError,
        match="expectation identity",
    ):
        replace(expectation, expectation_id="sha256:tampered")
    with pytest.raises(
        PerceptionTransitionConformanceError,
        match="report identity",
    ):
        replace(report, report_id="sha256:tampered")
