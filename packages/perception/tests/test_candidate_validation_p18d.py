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
    discover_recurrent_unexplained_transitions,
    encode_source_array,
    evaluate_transition_conformance,
)
from zeromodel.perception.candidate_validation import (
    CandidateValidationPolicyDTO,
    HeldOutTransitionObservationDTO,
    PerceptionCandidateValidationError,
    validate_discovered_transition_candidates,
)
from zeromodel.perception.transition_discovery import (
    TransitionDiscoveryObservationDTO,
    TransitionDiscoveryPolicyDTO,
)


def _fixture():
    encoder = SourceImageEncoderSpecDTO(color_space="L")
    source = encode_source_array(np.zeros((1, 4), dtype=np.uint8), encoder)
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


def _transition(
    before_values: list[int],
    after_values: list[int],
    *,
    include_control_annotation: bool,
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
        annotations=(control,) if include_control_annotation else (),
    )
    return transition, fields, control, expectation


def _discovery_observation(
    *,
    interaction_id: str,
    candidate_value: int,
):
    transition, _, control, expectation = _transition(
        [0, 0, 0, 0],
        [0, 0, candidate_value, candidate_value],
        include_control_annotation=True,
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
        cohort_id="discovery/train",
        transition=transition,
        conformance=conformance,
    )
    return observation, transition


def _discovery_report():
    observations = tuple(
        _discovery_observation(
            interaction_id=f"discovery-{index}",
            candidate_value=value,
        )[0]
        for index, value in enumerate((200, 180, 160, 0), start=1)
    )
    policy = TransitionDiscoveryPolicyDTO.create(
        minimum_observation_count=4,
        minimum_field_occurrence_count=3,
        minimum_field_recurrence_fraction=0.75,
        minimum_signature_occurrence_count=4,
        minimum_signature_recurrence_fraction=1.0,
        minimum_direction_consistency=0.75,
    )
    report = discover_recurrent_unexplained_transitions(observations, policy)
    assert report.status == "candidates_found"
    assert len(report.candidates) == 1
    return report, observations


def _held_out(
    *,
    interaction_id: str,
    before_candidate: tuple[int, int] = (0, 0),
    after_candidate: tuple[int, int],
    cohort_id: str = "validation/held-out",
):
    transition, fields, _, _ = _transition(
        [0, 0, *before_candidate],
        [0, 0, *after_candidate],
        include_control_annotation=False,
    )
    return (
        HeldOutTransitionObservationDTO.create(
            interaction_id=interaction_id,
            cohort_id=cohort_id,
            transition=transition,
        ),
        transition,
        fields,
    )


def _policy(**changes: object) -> CandidateValidationPolicyDTO:
    values: dict[str, object] = {
        "minimum_validation_observation_count": 3,
        "minimum_confirmation_fraction": 2 / 3,
        "minimum_rejection_fraction": 2 / 3,
        "minimum_magnitude_retention_fraction": 0.5,
        "direction_epsilon": 0.01,
    }
    values.update(changes)
    return CandidateValidationPolicyDTO.create(**values)  # type: ignore[arg-type]


def test_validates_candidate_on_disjoint_held_out_evidence() -> None:
    discovery, _ = _discovery_report()
    repeated = _held_out(
        interaction_id="validation-1",
        after_candidate=(170, 170),
    )
    repeated_again = HeldOutTransitionObservationDTO.create(
        interaction_id="validation-2",
        cohort_id="validation/held-out",
        transition=repeated[1],
    )
    third = _held_out(
        interaction_id="validation-3",
        after_candidate=(150, 150),
    )[0]
    observations = (repeated[0], repeated_again, third)

    first = validate_discovered_transition_candidates(
        discovery,
        observations,
        _policy(),
    )
    second = validate_discovered_transition_candidates(
        discovery,
        tuple(reversed(observations)),
        _policy(),
    )

    assert first == second
    assert first.status == "all_validated"
    assert len(first.results) == 1
    result = first.results[0]
    assert result.status == "validated"
    assert result.confirmation_count == 3
    assert result.rejection_count == 0
    assert {item.status for item in result.findings} == {"confirmed"}
    assert first.validation_transition_evidence_ids.count(
        repeated[1].transition_evidence_id
    ) == 2

    source = discovery.statistics[0]
    expectation = result.expectation
    assert expectation.minimum_mean_absolute_change == pytest.approx(
        source.mean_absolute_change * 0.5
    )
    assert expectation.minimum_changed_fraction == pytest.approx(
        source.mean_changed_fraction * 0.5
    )
    assert expectation.expected_change == "increase"


def test_preserves_rejected_and_inconclusive_candidate_outcomes() -> None:
    discovery, _ = _discovery_report()
    rejected_observations = tuple(
        _held_out(
            interaction_id=f"rejected-{index}",
            before_candidate=(value, value),
            after_candidate=(value, value),
        )[0]
        for index, value in enumerate((10, 20, 30), start=1)
    )

    rejected = validate_discovered_transition_candidates(
        discovery,
        rejected_observations,
        _policy(),
    )

    assert rejected.status == "none_validated"
    assert rejected.results[0].status == "rejected"
    assert rejected.results[0].rejection_count == 3
    assert {item.status for item in rejected.results[0].findings} == {
        "missing_change"
    }

    inconclusive_observations = (
        _held_out(
            interaction_id="mixed-confirmed",
            after_candidate=(170, 170),
        )[0],
        _held_out(
            interaction_id="mixed-missing",
            before_candidate=(40, 40),
            after_candidate=(40, 40),
        )[0],
        _held_out(
            interaction_id="mixed-direction",
            before_candidate=(0, 255),
            after_candidate=(255, 0),
        )[0],
    )
    inconclusive = validate_discovered_transition_candidates(
        discovery,
        inconclusive_observations,
        _policy(),
    )

    result = inconclusive.results[0]
    assert inconclusive.status == "none_validated"
    assert result.status == "inconclusive"
    assert result.confirmation_count == 1
    assert result.rejection_count == 1
    assert result.inconclusive_count == 1
    assert {item.status for item in result.findings} == {
        "confirmed",
        "missing_change",
        "inconclusive_direction",
    }


def test_validation_evidence_gate_is_separate_from_findings() -> None:
    discovery, _ = _discovery_report()
    observations = (
        _held_out(
            interaction_id="small-1",
            after_candidate=(170, 170),
        )[0],
        _held_out(
            interaction_id="small-2",
            after_candidate=(150, 150),
        )[0],
    )

    report = validate_discovered_transition_candidates(
        discovery,
        observations,
        _policy(minimum_validation_observation_count=3),
    )

    assert report.status == "insufficient_evidence"
    assert report.results[0].status == "insufficient_validation_evidence"
    assert report.results[0].confirmation_count == 2
    assert len(report.results[0].findings) == 2


def test_rejects_discovery_leakage_and_identity_tampering() -> None:
    discovery, discovery_observations = _discovery_report()
    valid = _held_out(
        interaction_id="valid-held-out",
        after_candidate=(170, 170),
    )[0]

    same_cohort = replace(valid, cohort_id=discovery.cohort_id)
    with pytest.raises(
        PerceptionCandidateValidationError,
        match="cohort must differ",
    ):
        validate_discovered_transition_candidates(
            discovery,
            (same_cohort,),
            _policy(minimum_validation_observation_count=1),
        )

    overlap_interaction = HeldOutTransitionObservationDTO.create(
        interaction_id=discovery.interaction_ids[0],
        cohort_id="validation/held-out",
        transition=_held_out(
            interaction_id="unused",
            after_candidate=(170, 170),
        )[1],
    )
    with pytest.raises(
        PerceptionCandidateValidationError,
        match="interaction identities overlap",
    ):
        validate_discovered_transition_candidates(
            discovery,
            (overlap_interaction,),
            _policy(minimum_validation_observation_count=1),
        )

    leaked_transition = HeldOutTransitionObservationDTO.create(
        interaction_id="leaked-transition",
        cohort_id="validation/held-out",
        transition=_discovery_observation(
            interaction_id="unused-discovery",
            candidate_value=200,
        )[1],
    )
    with pytest.raises(
        PerceptionCandidateValidationError,
        match="transition evidence overlaps",
    ):
        validate_discovered_transition_candidates(
            discovery,
            (leaked_transition,),
            _policy(minimum_validation_observation_count=1),
        )

    policy = _policy()
    with pytest.raises(
        PerceptionCandidateValidationError,
        match="policy identity",
    ):
        replace(policy, policy_id="sha256:tampered")

    report = validate_discovered_transition_candidates(
        discovery,
        (valid,),
        _policy(minimum_validation_observation_count=1),
    )
    with pytest.raises(
        PerceptionCandidateValidationError,
        match="report identity",
    ):
        replace(report, report_id="sha256:tampered")

    assert discovery_observations
