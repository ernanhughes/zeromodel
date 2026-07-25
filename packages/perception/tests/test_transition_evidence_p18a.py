from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import zeromodel.perception as perception
from zeromodel.perception import (
    PerceptionRegionAnnotationDTO,
    SourceImageEncoderSpecDTO,
    build_grid_field_schema,
    encode_source_array,
)
from zeromodel.perception.transition_evidence import (
    PerceptionTransitionEvidenceError,
    TRANSITION_CHANGE_SEMANTICS,
    TRANSITION_EVIDENCE_VPM_VERSION,
    TransitionEvidenceVPMDTO,
    TransitionFieldEvidenceDTO,
    build_transition_evidence_vpm,
)


def _fixture():
    encoder = SourceImageEncoderSpecDTO(color_space="L")
    before = encode_source_array(np.zeros((2, 4), dtype=np.uint8), encoder)
    after_array = np.zeros((2, 4), dtype=np.uint8)
    after_array[:, :2] = 255
    after = encode_source_array(after_array, encoder)
    schema = build_grid_field_schema(before, tile_width=2, tile_height=2)
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
    return before, after, schema, fields, tank, alien


def test_transition_evidence_is_deterministic_addressable_and_annotated() -> None:
    before, after, schema, fields, tank, alien = _fixture()

    first = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(tank, alien),
    )
    second = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(tank, alien),
    )

    assert first == second
    assert first.transition_evidence_id == second.transition_evidence_id
    assert first.annotation_ids == tuple(
        sorted((tank.annotation_id, alien.annotation_id))
    )

    tank_field = first.field_evidence(fields[0].field_id)
    alien_field = first.field_evidence(fields[1].field_id)
    assert tank_field.mean_absolute_change == 1.0
    assert tank_field.mean_signed_change == 1.0
    assert tank_field.changed_fraction == 1.0
    assert tank_field.annotation_ids == (tank.annotation_id,)
    assert alien_field.mean_absolute_change == 0.0
    assert alien_field.changed_fraction == 0.0
    assert alien_field.annotation_ids == (alien.annotation_id,)
    assert first.changed_field_ids() == (fields[0].field_id,)

    rendered = first.to_array()
    assert np.all(rendered[:, :2] == 255)
    assert np.all(rendered[:, 2:] == 0)


def test_transition_preserves_signed_change_and_threshold_counts() -> None:
    encoder = SourceImageEncoderSpecDTO(color_space="L")
    before = encode_source_array(np.array([[100, 100]], dtype=np.uint8), encoder)
    after = encode_source_array(np.array([[90, 120]], dtype=np.uint8), encoder)
    schema = build_grid_field_schema(before, tile_width=2, tile_height=1)

    report = build_transition_evidence_vpm(
        before,
        after,
        schema,
        change_threshold=15,
    )
    field = report.fields[0]

    assert field.changed_value_count == 1
    assert field.total_value_count == 2
    assert field.changed_fraction == 0.5
    assert field.before_mean == pytest.approx(100.0 / 255.0)
    assert field.after_mean == pytest.approx(105.0 / 255.0)
    assert field.mean_absolute_change == pytest.approx(15.0 / 255.0)
    assert field.mean_signed_change == pytest.approx(5.0 / 255.0)

    lower_threshold = build_transition_evidence_vpm(
        before,
        after,
        schema,
        change_threshold=5,
    )
    assert lower_threshold.transition_evidence_id != report.transition_evidence_id
    assert lower_threshold.fields[0].changed_value_count == 2


def test_transition_rejects_schema_and_annotation_mismatches() -> None:
    before, after, schema, fields, tank, _ = _fixture()
    other_encoder = SourceImageEncoderSpecDTO(color_space="L", max_width=8)
    incompatible_after = encode_source_array(after.to_array(), other_encoder)

    with pytest.raises(PerceptionTransitionEvidenceError):
        build_transition_evidence_vpm(before, incompatible_after, schema)

    mismatched_annotation = replace(tank, field_schema_id="sha256:other-schema")
    with pytest.raises(
        PerceptionTransitionEvidenceError,
        match="annotation field schema",
    ):
        build_transition_evidence_vpm(
            before,
            after,
            schema,
            annotations=(mismatched_annotation,),
        )

    with pytest.raises(
        PerceptionTransitionEvidenceError,
        match="unique identities",
    ):
        build_transition_evidence_vpm(
            before,
            after,
            schema,
            annotations=(tank, tank),
        )

    with pytest.raises(PerceptionTransitionEvidenceError, match="change_threshold"):
        build_transition_evidence_vpm(before, after, schema, change_threshold=0)

    valid = build_transition_evidence_vpm(before, after, schema)
    with pytest.raises(
        PerceptionTransitionEvidenceError,
        match="identity disagrees",
    ):
        replace(valid, transition_evidence_id="sha256:tampered")
    with pytest.raises(
        PerceptionTransitionEvidenceError,
        match="PNG digest mismatch",
    ):
        replace(valid, png_bytes=b"not-a-png")


def test_transition_evidence_public_contract() -> None:
    assert TRANSITION_EVIDENCE_VPM_VERSION == "perception-transition-evidence-vpm/1"
    assert TRANSITION_CHANGE_SEMANTICS
    assert TransitionFieldEvidenceDTO is not None
    assert TransitionEvidenceVPMDTO is not None
    assert callable(build_transition_evidence_vpm)
    assert perception.TransitionEvidenceVPMDTO is TransitionEvidenceVPMDTO
    assert perception.build_transition_evidence_vpm is build_transition_evidence_vpm
