from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    CandidateValidationPolicyDTO,
    HeldOutTransitionObservationDTO,
    PerceptionRegionAnnotationDTO,
    SourceImageEncoderSpecDTO,
    TransitionDiscoveryObservationDTO,
    TransitionDiscoveryPolicyDTO,
    TransitionExpectationDTO,
    build_grid_field_schema,
    build_transition_evidence_vpm,
    discover_recurrent_unexplained_transitions,
    encode_source_array,
    evaluate_transition_conformance,
    validate_discovered_transition_candidates,
)
from zeromodel.perception.candidate_promotion import (
    CandidatePromotionDecisionDTO,
    PerceptionCandidatePromotionError,
    propose_validated_candidate_promotions,
    review_candidate_promotion_proposals,
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


def _transition(
    before_values: list[int],
    after_values: list[int],
    *,
    include_control_annotation: bool,
):
    encoder, schema, fields, control, expectation = _fixture()
    before = encode_source_array(np.array([before_values], dtype=np.uint8), encoder)
    after = encode_source_array(np.array([after_values], dtype=np.uint8), encoder)
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
    first_value: int,
    second_value: int,
    cohort_id: str = "discovery/train",
):
    transition, _, control, expectation = _transition(
        [0, 0, 0, 0, 0, 0],
        [0, 0, first_value, first_value, second_value, second_value],
        include_control_annotation=True,
    )
    conformance = evaluate_transition_conformance(
        transition,
        (expectation,),
        (control,),
        minimum_unexplained_mean_absolute_change=0.05,
        minimum_unexplained_changed_fraction=0.5,
    )
    return TransitionDiscoveryObservationDTO.create(
        interaction_id=interaction_id,
        cohort_id=cohort_id,
        transition=transition,
        conformance=conformance,
    )


def _discovery_report(*, cohort_id: str = "discovery/train"):
    values = ((200, 180), (180, 160), (160, 140), (0, 0))
    observations = tuple(
        _discovery_observation(
            interaction_id=f"{cohort_id}-{index}",
            first_value=first,
            second_value=second,
            cohort_id=cohort_id,
        )
        for index, (first, second) in enumerate(values, start=1)
    )
    policy = TransitionDiscoveryPolicyDTO.create(
        minimum_observation_count=4,
        minimum_field_occurrence_count=3,
        minimum_field_recurrence_fraction=0.75,
        minimum_signature_occurrence_count=3,
        minimum_signature_recurrence_fraction=0.75,
        minimum_direction_consistency=0.75,
    )
    report = discover_recurrent_unexplained_transitions(observations, policy)
    assert report.status == "candidates_found"
    assert len(report.candidates) == 3
    return report


def _held_out(
    *,
    interaction_id: str,
    first_value: int,
    second_value: int,
    cohort_id: str = "validation/held-out",
):
    transition, _, _, _ = _transition(
        [0, 0, 0, 0, 0, 0],
        [0, 0, first_value, first_value, second_value, second_value],
        include_control_annotation=False,
    )
    return HeldOutTransitionObservationDTO.create(
        interaction_id=interaction_id,
        cohort_id=cohort_id,
        transition=transition,
    )


def _validation_report(discovery, *, validate_all: bool):
    second_value = 150 if validate_all else 0
    observations = tuple(
        _held_out(
            interaction_id=f"validation-{validate_all}-{index}",
            first_value=170,
            second_value=second_value,
        )
        for index in range(1, 4)
    )
    policy = CandidateValidationPolicyDTO.create(
        minimum_validation_observation_count=3,
        minimum_confirmation_fraction=2 / 3,
        minimum_rejection_fraction=2 / 3,
        minimum_magnitude_retention_fraction=0.75,
        direction_epsilon=0.01,
    )
    return validate_discovered_transition_candidates(
        discovery,
        observations,
        policy,
    )


def test_proposes_only_validated_candidates_and_binds_complete_lineage() -> None:
    discovery = _discovery_report()
    validation = _validation_report(discovery, validate_all=False)
    validated = validation.results_for_status("validated")
    rejected = validation.results_for_status("rejected")

    first = propose_validated_candidate_promotions(discovery, validation)
    second = propose_validated_candidate_promotions(discovery, validation)

    assert first == second
    assert len(validated) == 1
    assert len(rejected) == 2
    assert len(first.proposals) == 1
    proposal = first.proposals[0]
    assert proposal.candidate_id == validated[0].candidate_id
    assert proposal.validation_result_id == validated[0].result_id
    assert proposal.validation_expectation_id == validated[0].expectation.expectation_id
    assert proposal.materialization_status == "not_materialized"
    assert proposal.status == "pending_review"
    assert proposal.validation_observation_ids == validation.validation_observation_ids
    assert proposal.validation_interaction_ids == validation.validation_interaction_ids
    assert proposal.validation_transition_evidence_ids == (
        validation.validation_transition_evidence_ids
    )
    assert proposal.discovery_occurrence_count == len(
        proposal.supporting_discovery_observation_ids
    )
    assert {item.candidate_id for item in first.proposals}.isdisjoint(
        {item.candidate_id for item in rejected}
    )


def test_review_records_decisions_without_materializing_approved_proposals() -> None:
    discovery = _discovery_report()
    validation = _validation_report(discovery, validate_all=True)
    proposal_set = propose_validated_candidate_promotions(discovery, validation)
    assert len(proposal_set.proposals) == 3

    pending = review_candidate_promotion_proposals(proposal_set)
    assert pending.status == "pending_review"
    assert pending.pending_proposal_ids == proposal_set.proposal_ids

    approved = CandidatePromotionDecisionDTO.create(
        proposal_set.proposals[0],
        reviewer_id="reviewer:alice",
        decision="approved",
        rationale="Held-out evidence is strong and the semantic identity is known.",
        semantic_name="projectile-shadow",
        semantic_type="region_component",
        semantic_role="context",
    )
    partial = review_candidate_promotion_proposals(proposal_set, (approved,))
    assert partial.status == "partially_reviewed"
    assert partial.approved_proposal_ids == (approved.proposal_id,)
    assert approved.materialization_status == "not_materialized"

    rejected = CandidatePromotionDecisionDTO.create(
        proposal_set.proposals[1],
        reviewer_id="reviewer:bob",
        decision="rejected",
        rationale="The spatial signature is not semantically coherent.",
    )
    needs_semantic = CandidatePromotionDecisionDTO.create(
        proposal_set.proposals[2],
        reviewer_id="reviewer:carol",
        decision="needs_semantic_annotation",
        rationale="Evidence is valid but the component cannot yet be named.",
    )
    complete = review_candidate_promotion_proposals(
        proposal_set,
        (needs_semantic, approved, rejected),
    )

    assert complete.status == "review_complete"
    assert complete.pending_proposal_ids == ()
    assert complete.approved_proposal_ids == (approved.proposal_id,)
    assert complete.rejected_proposal_ids == (rejected.proposal_id,)
    assert complete.semantic_annotation_required_proposal_ids == (
        needs_semantic.proposal_id,
    )
    assert complete.decisions_for_status("approved") == (approved,)

    deferred = CandidatePromotionDecisionDTO.create(
        proposal_set.proposals[1],
        reviewer_id="reviewer:dana",
        decision="deferred",
        rationale="Collect a second validation cohort before approval.",
    )
    assert deferred.decision == "deferred"
    assert deferred.materialization_status == "not_materialized"


def test_decision_semantics_and_identity_are_strict() -> None:
    discovery = _discovery_report()
    validation = _validation_report(discovery, validate_all=False)
    proposal = propose_validated_candidate_promotions(
        discovery,
        validation,
    ).proposals[0]

    with pytest.raises(
        PerceptionCandidatePromotionError,
        match="require semantic_name",
    ):
        CandidatePromotionDecisionDTO.create(
            proposal,
            reviewer_id="reviewer:missing-label",
            decision="approved",
            rationale="Approve without an invented label.",
        )

    with pytest.raises(
        PerceptionCandidatePromotionError,
        match="non-approved",
    ):
        CandidatePromotionDecisionDTO.create(
            proposal,
            reviewer_id="reviewer:invalid",
            decision="deferred",
            rationale="This should not carry semantic materialization.",
            semantic_name="invented",
            semantic_type="region_component",
        )

    decision = CandidatePromotionDecisionDTO.create(
        proposal,
        reviewer_id="reviewer:valid",
        decision="needs_semantic_annotation",
        rationale="A label must be established separately.",
    )
    with pytest.raises(
        PerceptionCandidatePromotionError,
        match="decision identity",
    ):
        replace(decision, decision_id="sha256:tampered")

    review = review_candidate_promotion_proposals(
        propose_validated_candidate_promotions(discovery, validation),
        (decision,),
    )
    with pytest.raises(
        PerceptionCandidatePromotionError,
        match="review identity",
    ):
        replace(review, review_id="sha256:tampered")


def test_rejects_nonvalidated_and_mismatched_promotion_lineage() -> None:
    discovery = _discovery_report()
    rejected_observations = tuple(
        _held_out(
            interaction_id=f"rejected-all-{index}",
            first_value=0,
            second_value=0,
        )
        for index in range(1, 4)
    )
    rejected_validation = validate_discovered_transition_candidates(
        discovery,
        rejected_observations,
        CandidateValidationPolicyDTO.create(
            minimum_validation_observation_count=3,
            minimum_confirmation_fraction=2 / 3,
            minimum_rejection_fraction=2 / 3,
            minimum_magnitude_retention_fraction=0.75,
        ),
    )
    assert not rejected_validation.results_for_status("validated")
    with pytest.raises(
        PerceptionCandidatePromotionError,
        match="at least one validated result",
    ):
        propose_validated_candidate_promotions(discovery, rejected_validation)

    other_discovery = _discovery_report(cohort_id="discovery/other")
    other_validation = _validation_report(other_discovery, validate_all=False)
    with pytest.raises(
        PerceptionCandidatePromotionError,
        match="does not reference",
    ):
        propose_validated_candidate_promotions(discovery, other_validation)

    first_set = propose_validated_candidate_promotions(
        discovery,
        _validation_report(discovery, validate_all=False),
    )
    second_set = propose_validated_candidate_promotions(
        other_discovery,
        other_validation,
    )
    foreign_decision = CandidatePromotionDecisionDTO.create(
        second_set.proposals[0],
        reviewer_id="reviewer:foreign",
        decision="deferred",
        rationale="Valid decision for another proposal set.",
    )
    with pytest.raises(
        PerceptionCandidatePromotionError,
        match="unknown proposals",
    ):
        review_candidate_promotion_proposals(first_set, (foreign_decision,))
