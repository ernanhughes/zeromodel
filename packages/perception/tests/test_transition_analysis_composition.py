from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    PerceptionRegionAnnotationDTO,
    PerceptionTransitionAnalysisError,
    SourceImageEncoderSpecDTO,
    TransitionActionDeclarationDTO,
    TransitionExpectationDTO,
    TransitionExpectationSetDTO,
    VisualTransitionAnalysisDTO,
    VisualTransitionReaderTraceDTO,
    build_grid_field_schema,
    build_transition_evidence_vpm,
    encode_source_array,
    evaluate_transition_conformance,
)
from zeromodel.perception.transition_analysis import _digest as _analysis_digest


def _sha(label: str) -> str:
    return "sha256:" + label * 64


def _source(values: list[int]):
    encoder = SourceImageEncoderSpecDTO(color_space="L")
    return encode_source_array(np.array([values], dtype=np.uint8), encoder)


def _analysis():
    before = _source([0, 0, 0, 0])
    after = _source([255, 255, 0, 0])
    schema = build_grid_field_schema(before, tile_width=2, tile_height=1)
    field = tuple(sorted(schema.fields, key=lambda item: item.x0))[0]
    annotation = PerceptionRegionAnnotationDTO.create(
        schema,
        (field.field_id,),
        label="declared component",
        role="component",
    )
    transition = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(annotation,),
    )
    expectation = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(annotation.annotation_id,),
        expected_change="increase",
        minimum_mean_absolute_change=0.5,
        minimum_changed_fraction=1.0,
        minimum_signed_change_magnitude=0.5,
    )
    expectation_set = TransitionExpectationSetDTO.create((expectation,))
    report = evaluate_transition_conformance(
        transition,
        expectation_set.expectations,
        (annotation,),
    )
    action = TransitionActionDeclarationDTO.create(
        action_type="MOVE",
        payload={"direction": "RIGHT"},
        schema_version="arcade-test/v1",
        provider_id="fixture",
    )
    reader_trace = VisualTransitionReaderTraceDTO(
        accepted=True,
        reason="evidence_only",
        raw_input_digest=_sha("0"),
        canonical_input_digest=_sha("1"),
        feature_digest=_sha("2"),
        reader_version="visual-reader-test/v1",
        visual_index_artifact_id="visual-index",
        policy_artifact_id="policy",
        feature_spec_digest="feature-spec",
        calibration_digest="calibration",
        acceptance_profile="evidence_only",
        policy_executed=False,
        nearest_row_id="row-a",
        matched_row_id=None,
        canonical_input_match=True,
        exact_feature_match=True,
    )
    analysis = VisualTransitionAnalysisDTO.create(
        transition=transition,
        action=action,
        expectation_set=expectation_set,
        conformance_report=report,
        before_reader_trace=reader_trace,
        after_reader_trace=reader_trace,
    )
    return transition, action, expectation, expectation_set, report, analysis


def test_action_identity_is_canonical_and_parameter_sensitive() -> None:
    first = TransitionActionDeclarationDTO.create(
        action_type="MOVE",
        payload={"direction": "LEFT", "amount": 1},
        schema_version="test/v1",
    )
    reordered = TransitionActionDeclarationDTO.create(
        action_type="MOVE",
        payload={"amount": 1, "direction": "LEFT"},
        schema_version="test/v1",
    )
    changed = TransitionActionDeclarationDTO.create(
        action_type="MOVE",
        payload={"direction": "RIGHT", "amount": 1},
        schema_version="test/v1",
    )

    assert first.action_id == reordered.action_id
    assert first.action_id != changed.action_id
    assert TransitionActionDeclarationDTO.from_dict(first.to_dict()) == first
    with pytest.raises(PerceptionTransitionAnalysisError):
        TransitionActionDeclarationDTO.create(action_type="")
    with pytest.raises(PerceptionTransitionAnalysisError):
        TransitionActionDeclarationDTO.create(
            action_type="MOVE",
            payload={1: "first"},  # type: ignore[dict-item]
        )
    with pytest.raises(PerceptionTransitionAnalysisError):
        TransitionActionDeclarationDTO.create(
            action_type="MOVE",
            payload={"outer": {1: "nested"}},  # type: ignore[dict-item]
        )


def test_expectation_set_identity_is_order_insensitive_and_rejects_conflicts() -> None:
    _, _, expectation, expectation_set, _, _ = _analysis()
    stable = TransitionExpectationDTO.create(
        field_schema_id=expectation.field_schema_id,
        annotation_ids=("zz-extra",),
        expected_change="stable",
    )

    first = TransitionExpectationSetDTO.create((expectation, stable))
    second = TransitionExpectationSetDTO.create((stable, expectation))

    assert first.expectation_set_id == second.expectation_set_id
    assert TransitionExpectationSetDTO.from_dict(first.to_dict()) == first
    with pytest.raises(PerceptionTransitionAnalysisError):
        TransitionExpectationSetDTO.create((expectation, expectation))


def test_visual_transition_analysis_binds_ordered_transition_action_and_report() -> (
    None
):
    transition, action, _, _, report, analysis = _analysis()

    assert analysis.before_source_vpm_id == transition.before_source_vpm_id
    assert analysis.after_source_vpm_id == transition.after_source_vpm_id
    assert analysis.transition_evidence == transition
    assert analysis.action_id == action.action_id
    assert analysis.status == report.status
    assert VisualTransitionAnalysisDTO.from_dict(analysis.to_dict()) == analysis

    with pytest.raises(PerceptionTransitionAnalysisError):
        replace(
            analysis,
            before_source_vpm_id=analysis.after_source_vpm_id,
            after_source_vpm_id=analysis.before_source_vpm_id,
        )

    wrong_action = TransitionActionDeclarationDTO.create(action_type="WAIT")
    with pytest.raises(PerceptionTransitionAnalysisError):
        replace(analysis, action=wrong_action)


def test_visual_transition_analysis_rejects_rehashed_cross_evidence_pairing() -> None:
    transition, action, _, expectation_set, report, analysis = _analysis()
    other_before = _source([0, 0, 0, 0])
    other_after = _source([0, 0, 255, 255])
    other_schema = build_grid_field_schema(other_before, tile_width=2, tile_height=1)
    other_transition = build_transition_evidence_vpm(
        other_before,
        other_after,
        other_schema,
    )
    forged_payload = dict(analysis.canonical_payload())
    forged_payload["before_source_vpm_id"] = other_transition.before_source_vpm_id
    forged_payload["after_source_vpm_id"] = other_transition.after_source_vpm_id
    forged = analysis.to_dict()
    forged.update(forged_payload)
    forged["analysis_id"] = _analysis_digest(forged_payload)
    with pytest.raises(PerceptionTransitionAnalysisError):
        VisualTransitionAnalysisDTO.from_dict(forged)


def test_reader_trace_preserves_evidence_only_boundary() -> None:
    trace = VisualTransitionReaderTraceDTO(
        accepted=True,
        reason="evidence_only",
        raw_input_digest=_sha("a"),
        canonical_input_digest=_sha("b"),
        feature_digest=_sha("c"),
        reader_version="reader/v1",
        visual_index_artifact_id="index",
        policy_artifact_id="policy",
        feature_spec_digest="feature-spec",
        calibration_digest="calibration",
        acceptance_profile="evidence_only",
        policy_executed=False,
    )

    assert VisualTransitionReaderTraceDTO.from_dict(trace.to_dict()) == trace
    with pytest.raises(PerceptionTransitionAnalysisError):
        VisualTransitionReaderTraceDTO(
            accepted=True,
            reason="evidence_only",
            raw_input_digest=_sha("a"),
            canonical_input_digest=_sha("b"),
            feature_digest=_sha("c"),
            reader_version="reader/v1",
            visual_index_artifact_id="index",
            policy_artifact_id="policy",
            feature_spec_digest="feature-spec",
            calibration_digest="calibration",
            acceptance_profile="evidence_only",
            policy_executed=True,
        )
    with pytest.raises(PerceptionTransitionAnalysisError):
        VisualTransitionReaderTraceDTO(
            accepted=False,
            reason="rejected",
            raw_input_digest=_sha("a"),
            canonical_input_digest=_sha("b"),
            feature_digest=_sha("c"),
            reader_version="reader/v1",
            visual_index_artifact_id="index",
            policy_artifact_id="policy",
            feature_spec_digest="feature-spec",
            calibration_digest="calibration",
            acceptance_profile="unsupported",
            policy_executed=False,
        )


def test_reader_trace_rejects_impossible_visual_decision_states() -> None:
    valid = {
        "accepted": True,
        "reason": "accepted_canonical_input",
        "raw_input_digest": _sha("d"),
        "canonical_input_digest": _sha("e"),
        "feature_digest": _sha("f"),
        "reader_version": "reader/v1",
        "visual_index_artifact_id": "index",
        "policy_artifact_id": "policy",
        "feature_spec_digest": "feature-spec",
        "calibration_digest": "calibration",
        "acceptance_profile": "canonical_only",
        "policy_executed": True,
        "nearest_row_id": "row",
        "matched_row_id": "row",
        "canonical_input_match": True,
        "exact_feature_match": True,
    }
    assert VisualTransitionReaderTraceDTO(**valid).policy_executed
    for bad in (
        valid | {"raw_input_digest": "raw"},
        valid | {"matched_row_id": None},
        valid | {"canonical_input_match": False},
        valid | {"exact_feature_match": False},
        valid
        | {
            "acceptance_profile": "exact_codeword",
            "canonical_input_match": False,
            "exact_feature_match": False,
        },
        valid
        | {
            "policy_executed": False,
            "matched_row_id": "row",
        },
    ):
        with pytest.raises(PerceptionTransitionAnalysisError):
            VisualTransitionReaderTraceDTO(**bad)
