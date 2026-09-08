"""Unit tests for expected-versus-observed transition verification."""

from __future__ import annotations

import numpy as np

from zeromodel.perception import (
    build_dataset_manifest,
    build_grid_field_schema,
    build_transition_evidence_vpm,
    encode_discrete_action,
    encode_source_array,
    evaluate_transition_conformance,
)
from zeromodel.perception.dataset import RecordedInteractionDTO
from zeromodel.perception.expectations import PerceptionRegionAnnotationDTO
from zeromodel.perception.representation import (
    DiscreteActionSchemaDTO,
    SourceImageEncoderSpecDTO,
)
from zeromodel.perception.transition_analysis import (
    TransitionActionDeclarationDTO,
    TransitionExpectationSetDTO,
    VisualTransitionAnalysisDTO,
)
from zeromodel.perception.transition_conformance import TransitionExpectationDTO
from zeromodel.perception.transition_model import (
    fit_action_conditioned_transition_model,
)
from zeromodel.perception.transition_projection import project_expected_transition
from zeromodel.perception.transition_verification import verify_expected_transition

_SPEC = SourceImageEncoderSpecDTO(color_space="L")
_WIDTH, _HEIGHT = 12, 8
_ACTION_SCHEMA = DiscreteActionSchemaDTO.from_labels(["left", "right"])


def _pattern(marker: int, action: str):
    before_array = np.zeros((_HEIGHT, _WIDTH), dtype=np.uint8)
    before_array[marker % _HEIGHT, (marker * 5) % _WIDTH] = 200
    after_array = before_array.copy()
    if action == "left":
        after_array[:, 0:4] = 60
    else:
        after_array[:, 8:12] = 60
    return encode_source_array(before_array, _SPEC), encode_source_array(
        after_array, _SPEC
    )


def _build(count_per_action: int = 6):
    interactions = []
    sources = {}
    step = 0
    marker = 0
    for action in ("left", "right"):
        for _ in range(count_per_action):
            before, after = _pattern(marker, action)
            sources[before.source_vpm_id] = before
            sources[after.source_vpm_id] = after
            interactions.append(
                RecordedInteractionDTO.from_vpms(
                    sequence_id=f"seq-{action}",
                    step_index=step,
                    source=before,
                    target=encode_discrete_action(action, _ACTION_SCHEMA),
                    next_source=after,
                )
            )
            step += 1
            marker += 1
    manifest = build_dataset_manifest(
        interactions, source_encoder_spec_ids=[_SPEC.encoder_spec_id]
    )
    schema = build_grid_field_schema(
        next(iter(sources.values())), tile_width=4, tile_height=4, channel_mode="joint"
    )
    return manifest, sources, schema


def _band_field_ids(schema, channel: int):
    return tuple(
        sorted(field.field_id for field in schema.fields if field.x0 // 4 == channel)
    )


def _analyze(before, after, schema, action_label, expectations, annotations):
    evidence = build_transition_evidence_vpm(
        before, after, schema, annotations=annotations
    )
    report = evaluate_transition_conformance(evidence, expectations, annotations)
    expectation_set = TransitionExpectationSetDTO.create(expectations)
    action = TransitionActionDeclarationDTO.create(
        action_type="discrete_action", payload={"action_label": action_label}
    )
    return VisualTransitionAnalysisDTO.create(
        transition=evidence,
        action=action,
        expectation_set=expectation_set,
        conformance_report=report,
    )


def _middle_stable(schema, annotations):
    return TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(annotations["middle"].annotation_id,),
        expected_change="stable",
        maximum_mean_absolute_change=0.0,
        maximum_changed_fraction=0.0,
    )


def _annotations(schema):
    return {
        "left": PerceptionRegionAnnotationDTO.create(
            schema, _band_field_ids(schema, 0), label="left-band"
        ),
        "middle": PerceptionRegionAnnotationDTO.create(
            schema, _band_field_ids(schema, 1), label="middle-band"
        ),
        "right": PerceptionRegionAnnotationDTO.create(
            schema, _band_field_ids(schema, 2), label="right-band"
        ),
    }


def test_verification_confirms_exact_match() -> None:
    manifest, sources, schema = _build()
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    expected = project_expected_transition(model, query, "left", schema)
    assert expected.status == "supported"
    annotations = _annotations(schema)
    after = next(
        sources[interaction.next_source_vpm_id]
        for interaction in manifest.interactions
        if interaction.source_vpm_id == query.source_vpm_id
    )
    analysis = _analyze(
        query,
        after,
        schema,
        "left",
        (_middle_stable(schema, annotations),),
        (annotations["middle"],),
    )
    assert analysis.status != "nonconformant"
    verification = verify_expected_transition(expected, analysis)
    assert verification.status == "confirmed"


def test_verification_flags_wrong_direction() -> None:
    manifest, sources, schema = _build()
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    # Query whose marker sits inside the left band: removing it observes a
    # negative delta where the remembered future projects an increase.
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
        and sources[interaction.source_vpm_id].to_array()[0, 0] == 200
    )
    expected = project_expected_transition(model, query, "left", schema)
    assert expected.status == "supported"
    after_array = np.asarray(query.to_array(), dtype=np.uint8).copy()
    after_array[0, 0] = 0
    after = encode_source_array(after_array, _SPEC)
    annotations = _annotations(schema)
    analysis = _analyze(
        query,
        after,
        schema,
        "left",
        (_middle_stable(schema, annotations),),
        (annotations["middle"],),
    )
    verification = verify_expected_transition(expected, analysis)
    assert verification.status == "future_projection_mismatch"
    assert any("wrong_direction" in note for note in verification.field_notes)


def test_verification_flags_missing_expected_change() -> None:
    manifest, sources, schema = _build()
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    expected = project_expected_transition(model, query, "left", schema)
    annotations = _annotations(schema)
    analysis = _analyze(
        query,
        query,
        schema,
        "left",
        (_middle_stable(schema, annotations),),
        (annotations["middle"],),
    )
    verification = verify_expected_transition(expected, analysis)
    assert verification.status == "future_projection_mismatch"
    assert any("missing_expected_change" in note for note in verification.field_notes)


def test_verification_notes_unexpected_change() -> None:
    manifest, sources, schema = _build()
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    expected = project_expected_transition(model, query, "left", schema)
    after_array = np.asarray(query.to_array(), dtype=np.uint8).copy()
    after_array[:, 0:4] = 60  # expected left change
    after_array[:, 8:12] += 3  # tiny unexpected change on the static band
    after = encode_source_array(after_array, _SPEC)
    annotations = _annotations(schema)
    analysis = _analyze(
        query,
        after,
        schema,
        "left",
        (_middle_stable(schema, annotations),),
        (annotations["middle"],),
    )
    verification = verify_expected_transition(expected, analysis)
    assert verification.status in {
        "confirmed_with_unexpected_change",
        "future_projection_mismatch",
    }
    assert any("unexpected_change" in note for note in verification.field_notes)


def test_verification_reports_declared_violation_first() -> None:
    manifest, sources, schema = _build()
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    expected = project_expected_transition(model, query, "left", schema)
    after = next(
        sources[interaction.next_source_vpm_id]
        for interaction in manifest.interactions
        if interaction.source_vpm_id == query.source_vpm_id
    )
    annotations = _annotations(schema)
    left_stable = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(annotations["left"].annotation_id,),
        expected_change="stable",
        maximum_mean_absolute_change=0.0,
        maximum_changed_fraction=0.0,
    )
    analysis = _analyze(
        query,
        after,
        schema,
        "left",
        (left_stable,),
        (annotations["left"],),
    )
    assert analysis.status == "nonconformant"
    verification = verify_expected_transition(expected, analysis)
    assert verification.status == "declared_expectation_violation"


def test_verification_insufficient_without_supported_projection() -> None:
    manifest, sources, schema = _build(count_per_action=2)
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(iter(sources.values()))
    expected = project_expected_transition(model, query, "jump", schema)
    assert expected.status == "unsupported_action"
    annotations = _annotations(schema)
    analysis = _analyze(
        query,
        query,
        schema,
        "jump",
        (_middle_stable(schema, annotations),),
        (annotations["middle"],),
    )
    verification = verify_expected_transition(expected, analysis)
    assert verification.status == "insufficient_evidence"


def test_verification_reports_error_magnitudes() -> None:
    manifest, sources, schema = _build()
    model = fit_action_conditioned_transition_model(
        manifest, sources, schema, training_split="all"
    )
    query = next(
        sources[interaction.source_vpm_id]
        for interaction in manifest.interactions
        if interaction.action_label == "left"
    )
    expected = project_expected_transition(model, query, "left", schema)
    annotations = _annotations(schema)
    after = next(
        sources[interaction.next_source_vpm_id]
        for interaction in manifest.interactions
        if interaction.source_vpm_id == query.source_vpm_id
    )
    analysis = _analyze(
        query,
        after,
        schema,
        "left",
        (_middle_stable(schema, annotations),),
        (annotations["middle"],),
    )
    verification = verify_expected_transition(expected, analysis)
    assert verification.status == "confirmed"
    assert verification.mean_absolute_error < 0.05
    assert verification.direction_error_rate == 0.0
    assert verification.changed_field_error_rate < 0.2
    for name in (
        "mean_absolute_error",
        "direction_error_rate",
        "changed_field_error_rate",
    ):
        assert 0.0 <= getattr(verification, name) <= 1.0
