from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    PerceptionRegionAnnotationDTO,
    SourceImageEncoderSpecDTO,
    TransitionExpectationDTO,
    build_grid_field_schema,
    build_transition_evidence_vpm,
    encode_source_array,
    evaluate_transition_conformance,
)
from zeromodel.perception.transition_discovery import (
    PerceptionTransitionDiscoveryError,
    TransitionDiscoveryObservationDTO,
    TransitionDiscoveryPolicyDTO,
    discover_recurrent_unexplained_transitions,
)


def _fixture():
    encoder = SourceImageEncoderSpecDTO(color_space="L")
    source = encode_source_array(np.zeros((1, 6), dtype=np.uint8), encoder)
    schema = build_grid_field_schema(source, tile_width=2, tile_height=1)
    fields = tuple(sorted(schema.fields, key=lambda item: item.x0))
    control = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[0].field_id,),
        label="control",
        role="stable_control",
    )
    expectation = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(control.annotation_id,),
        expected_change="stable",
        maximum_mean_absolute_change=0.0,
        maximum_changed_fraction=0.0,
    )
    return encoder, schema, fields, control, expectation


def _observation(
    *,
    interaction_id: str,
    before_values: list[int],
    after_values: list[int],
    cohort_id: str = "discovery/train",
):
    encoder, schema, fields, control, expectation = _fixture()
    before = encode_source_array(
        np.array([before_values], dtype=np.uint8),
        encoder,
    )
    after = encode_source_array(
        np.array([after_values], dtype=np.uint8),
        encoder,
    )
    transition = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(control,),
    )
    conformance = evaluate_transition_conformance(
        transition,
        (expectation,),
        (control,),
        minimum_unexplained_mean_absolute_change=0.05,
        minimum_unexplained_changed_fraction=0.5,
    )
    observation = TransitionDiscoveryObservationDTO.create(
        interaction_id=interaction_id,
        cohort_id=cohort_id,
        transition=transition,
        conformance=conformance,
    )
    return observation, transition, conformance, fields


def _cohort():
    first = _observation(
        interaction_id="interaction-1",
        before_values=[0, 0, 0, 0, 0, 0],
        after_values=[0, 0, 255, 255, 128, 128],
    )
    second = _observation(
        interaction_id="interaction-2",
        before_values=[0, 0, 0, 0, 0, 0],
        after_values=[0, 0, 255, 255, 128, 128],
    )
    third = _observation(
        interaction_id="interaction-3",
        before_values=[0, 0, 0, 0, 0, 0],
        after_values=[0, 0, 200, 200, 0, 0],
    )
    fourth = _observation(
        interaction_id="interaction-4",
        before_values=[0, 0, 0, 0, 0, 0],
        after_values=[0, 0, 0, 0, 0, 0],
    )
    return first, second, third, fourth


def _policy(**changes: object) -> TransitionDiscoveryPolicyDTO:
    values: dict[str, object] = {
        "minimum_observation_count": 4,
        "minimum_field_occurrence_count": 2,
        "minimum_field_recurrence_fraction": 0.5,
        "minimum_signature_occurrence_count": 2,
        "minimum_signature_recurrence_fraction": 0.5,
        "minimum_direction_consistency": 0.75,
        "direction_epsilon": 0.01,
    }
    values.update(changes)
    return TransitionDiscoveryPolicyDTO.create(**values)  # type: ignore[arg-type]


def test_discovers_recurrent_fields_and_exact_cooccurrence_signatures() -> None:
    cohort = _cohort()
    observations = tuple(item[0] for item in cohort)
    fields = cohort[0][3]

    first = discover_recurrent_unexplained_transitions(
        observations,
        _policy(),
    )
    second = discover_recurrent_unexplained_transitions(
        tuple(reversed(observations)),
        _policy(),
    )

    assert first == second
    assert first.status == "candidates_found"
    assert len(first.statistics) == 3
    assert len(first.candidates_for_kind("field")) == 2
    assert len(first.candidates_for_kind("cooccurrence_signature")) == 1

    by_fields = {item.field_ids: item for item in first.statistics}
    primary = by_fields[(fields[1].field_id,)]
    secondary = by_fields[(fields[2].field_id,)]
    signature = by_fields[(fields[1].field_id, fields[2].field_id)]
    assert primary.occurrence_count == 3
    assert primary.recurrence_fraction == 0.75
    assert secondary.occurrence_count == 2
    assert secondary.recurrence_fraction == 0.5
    assert signature.occurrence_count == 2
    assert signature.recurrence_fraction == 0.5

    assert {item.proposed_expected_change for item in first.candidates} == {
        "increase"
    }
    assert {item.hypothesis_status for item in first.candidates} == {
        "candidate_unvalidated"
    }
    assert observations[3].unexplained_fields == ()

    repeated_transition_id = cohort[0][1].transition_evidence_id
    assert first.transition_evidence_ids.count(repeated_transition_id) == 2
    repeated_report_id = cohort[0][2].report_id
    assert first.conformance_report_ids.count(repeated_report_id) == 2


def test_evidence_gate_and_candidate_thresholds_are_separate() -> None:
    observations = tuple(item[0] for item in _cohort())

    insufficient = discover_recurrent_unexplained_transitions(
        observations[:3],
        _policy(minimum_observation_count=4),
    )
    assert insufficient.status == "insufficient_evidence"
    assert insufficient.statistics
    assert insufficient.candidates == ()

    no_candidates = discover_recurrent_unexplained_transitions(
        observations,
        _policy(
            minimum_field_occurrence_count=4,
            minimum_field_recurrence_fraction=1.0,
            minimum_signature_occurrence_count=4,
            minimum_signature_recurrence_fraction=1.0,
        ),
    )
    assert no_candidates.status == "no_candidates"
    assert no_candidates.statistics
    assert no_candidates.candidates == ()


def test_direction_proposal_remains_nondirectional_when_evidence_is_mixed() -> None:
    observations = (
        _observation(
            interaction_id="positive-1",
            before_values=[0, 0, 0, 0, 0, 0],
            after_values=[0, 0, 200, 200, 0, 0],
        )[0],
        _observation(
            interaction_id="positive-2",
            before_values=[0, 0, 0, 0, 0, 0],
            after_values=[0, 0, 180, 180, 0, 0],
        )[0],
        _observation(
            interaction_id="negative-1",
            before_values=[0, 0, 255, 255, 0, 0],
            after_values=[0, 0, 0, 0, 0, 0],
        )[0],
        _observation(
            interaction_id="negative-2",
            before_values=[0, 0, 220, 220, 0, 0],
            after_values=[0, 0, 0, 0, 0, 0],
        )[0],
    )

    report = discover_recurrent_unexplained_transitions(
        observations,
        _policy(
            minimum_field_occurrence_count=4,
            minimum_field_recurrence_fraction=1.0,
        ),
    )

    candidate = report.candidates_for_kind("field")[0]
    assert candidate.dominant_direction == "mixed"
    assert candidate.direction_consistency == 0.5
    assert candidate.proposed_expected_change == "change"


def test_discovery_rejects_lineage_cohort_and_identity_tampering() -> None:
    first, second, third, _ = _cohort()

    with pytest.raises(
        PerceptionTransitionDiscoveryError,
        match="does not reference",
    ):
        TransitionDiscoveryObservationDTO.create(
            interaction_id="mismatch",
            cohort_id="discovery/train",
            transition=first[1],
            conformance=third[2],
        )

    other_cohort = TransitionDiscoveryObservationDTO.create(
        interaction_id="other-cohort",
        cohort_id="discovery/other",
        transition=third[1],
        conformance=third[2],
    )
    with pytest.raises(
        PerceptionTransitionDiscoveryError,
        match="one cohort",
    ):
        discover_recurrent_unexplained_transitions(
            (first[0], other_cohort),
            _policy(minimum_observation_count=2),
        )

    duplicate_interaction = TransitionDiscoveryObservationDTO.create(
        interaction_id=first[0].interaction_id,
        cohort_id="discovery/train",
        transition=second[1],
        conformance=second[2],
    )
    with pytest.raises(
        PerceptionTransitionDiscoveryError,
        match="unique identities",
    ):
        discover_recurrent_unexplained_transitions(
            (first[0], duplicate_interaction),
            _policy(minimum_observation_count=2),
        )

    policy = _policy()
    with pytest.raises(
        PerceptionTransitionDiscoveryError,
        match="policy identity",
    ):
        replace(policy, policy_id="sha256:tampered")

    report = discover_recurrent_unexplained_transitions(
        tuple(item[0] for item in _cohort()),
        policy,
    )
    with pytest.raises(
        PerceptionTransitionDiscoveryError,
        match="report identity",
    ):
        replace(report, report_id="sha256:tampered")
