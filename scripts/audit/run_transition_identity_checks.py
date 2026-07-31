"""Generate transition identity and pairing hardening results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

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


def _sha(label: str) -> str:
    return "sha256:" + label * 64


def _source(values: list[int]):
    return encode_source_array(
        np.array([values], dtype=np.uint8),
        SourceImageEncoderSpecDTO(color_space="L"),
    )


def _case():
    before = _source([0, 0, 0, 0])
    after = _source([255, 255, 0, 0])
    schema = build_grid_field_schema(before, tile_width=2, tile_height=1)
    field = tuple(sorted(schema.fields, key=lambda item: item.x0))[0]
    annotation = PerceptionRegionAnnotationDTO.create(
        schema,
        (field.field_id,),
        label="component",
        role="component",
    )
    transition = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(annotation,),
    )
    swapped = build_transition_evidence_vpm(
        after,
        before,
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
        schema_version="audit/v1",
    )
    trace = VisualTransitionReaderTraceDTO(
        accepted=True,
        reason="evidence_only",
        raw_input_digest=_sha("0"),
        canonical_input_digest=_sha("1"),
        feature_digest=_sha("2"),
        reader_version="reader/audit",
        visual_index_artifact_id="visual-index",
        policy_artifact_id="policy",
        feature_spec_digest="feature-spec",
        calibration_digest="calibration",
        acceptance_profile="evidence_only",
        policy_executed=False,
    )
    analysis = VisualTransitionAnalysisDTO.create(
        transition=transition,
        action=action,
        expectation_set=expectation_set,
        conformance_report=report,
        before_reader_trace=trace,
        after_reader_trace=trace,
    )
    return transition, swapped, expectation, expectation_set, report, action, analysis


def _rejects(callback) -> bool:
    try:
        callback()
    except (PerceptionTransitionAnalysisError, ValueError):
        return True
    return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "docs/results/visual-transition-evidence-hardening/"
            "identity-and-pairing-results.json"
        ),
    )
    args = parser.parse_args(argv)

    transition, swapped, expectation, expectation_set, report, action, analysis = (
        _case()
    )
    changed_threshold = TransitionExpectationDTO.create(
        field_schema_id=expectation.field_schema_id,
        annotation_ids=expectation.annotation_ids,
        expected_change=expectation.expected_change,
        minimum_mean_absolute_change=0.75,
        minimum_changed_fraction=1.0,
        minimum_signed_change_magnitude=0.5,
    )
    changed_action = TransitionActionDeclarationDTO.create(
        action_type=action.action_type,
        payload={"direction": "LEFT"},
        schema_version=action.schema_version,
    )
    changed_field = max(
        transition.fields,
        key=lambda item: item.mean_absolute_change,
    )
    swapped_changed_field = swapped.field_evidence(changed_field.field_id)

    payload = {
        "before_after_ordered": {
            "transition_evidence_id": transition.transition_evidence_id,
            "swapped_transition_evidence_id": swapped.transition_evidence_id,
            "identity_changed": transition.transition_evidence_id
            != swapped.transition_evidence_id,
            "field_id": changed_field.field_id,
            "signed_change": changed_field.mean_signed_change,
            "swapped_signed_change": swapped_changed_field.mean_signed_change,
        },
        "action_identity": {
            "action_id": action.action_id,
            "changed_payload_action_id": changed_action.action_id,
            "identity_changed": action.action_id != changed_action.action_id,
            "non_string_payload_key_rejected": _rejects(
                lambda: TransitionActionDeclarationDTO.create(
                    action_type="MOVE",
                    payload={1: "collides"},  # type: ignore[dict-item]
                )
            ),
            "nested_non_string_payload_key_rejected": _rejects(
                lambda: TransitionActionDeclarationDTO.create(
                    action_type="MOVE",
                    payload={"outer": {1: "nested"}},  # type: ignore[dict-item]
                )
            ),
        },
        "expectation_set_identity": {
            "expectation_set_id": expectation_set.expectation_set_id,
            "changed_threshold_expectation_id": changed_threshold.expectation_id,
            "changed_threshold_changes_identity": expectation.expectation_id
            != changed_threshold.expectation_id,
            "duplicate_expectations_rejected": _rejects(
                lambda: TransitionExpectationSetDTO.create((expectation, expectation))
            ),
        },
        "analysis_binding": {
            "analysis_id": analysis.analysis_id,
            "round_trip_preserves_identity": VisualTransitionAnalysisDTO.from_dict(
                analysis.to_dict()
            ).analysis_id
            == analysis.analysis_id,
            "report_from_other_evidence_rejected": _rejects(
                lambda: VisualTransitionAnalysisDTO.create(
                    transition=swapped,
                    action=action,
                    expectation_set=expectation_set,
                    conformance_report=report,
                )
            ),
            "embedded_transition_evidence_preserved": (
                analysis.transition_evidence.transition_evidence_id
                == transition.transition_evidence_id
            ),
        },
        "visual_reader_boundary": {
            "evidence_only_policy_executed_rejected": _rejects(
                lambda: VisualTransitionReaderTraceDTO(
                    accepted=True,
                    reason="evidence_only",
                    raw_input_digest=_sha("a"),
                    canonical_input_digest=_sha("b"),
                    feature_digest=_sha("c"),
                    reader_version="reader/audit",
                    visual_index_artifact_id="index",
                    policy_artifact_id="policy",
                    feature_spec_digest="feature-spec",
                    calibration_digest="calibration",
                    acceptance_profile="evidence_only",
                    policy_executed=True,
                )
            ),
            "invalid_digest_rejected": _rejects(
                lambda: VisualTransitionReaderTraceDTO(
                    accepted=False,
                    reason="rejected",
                    raw_input_digest="raw",
                    canonical_input_digest=_sha("b"),
                    feature_digest=_sha("c"),
                    reader_version="reader/audit",
                    visual_index_artifact_id="index",
                    policy_artifact_id="policy",
                    feature_spec_digest="feature-spec",
                    calibration_digest="calibration",
                    acceptance_profile="calibrated_nearest",
                    policy_executed=False,
                )
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
