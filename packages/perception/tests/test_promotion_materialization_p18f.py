from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    CandidatePromotionDecisionDTO,
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
    propose_validated_candidate_promotions,
    review_candidate_promotion_proposals,
    validate_discovered_transition_candidates,
)
from zeromodel.perception.promotion_materialization import (
    PerceptionPromotionMaterializationError,
    PromotionMaterializationBaselineDTO,
    PromotionMaterializationDirectiveDTO,
    materialize_approved_candidate_promotions,
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
    anchor = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[0].field_id,),
        label="anchor",
        role="reference",
    )
    expectation = TransitionExpectationDTO.create(
        field_schema_id=schema.field_schema_id,
        annotation_ids=(control.annotation_id,),
        expected_change="stable",
        maximum_mean_absolute_change=0.0,
        maximum_changed_fraction=0.0,
    )
    return encoder, schema, fields, control, anchor, expectation


def _transition(
    before_values: list[int],
    after_values: list[int],
    *,
    include_control_annotation: bool,
):
    encoder, schema, fields, control, anchor, expectation = _fixture()
    before = encode_source_array(np.array([before_values], dtype=np.uint8), encoder)
    after = encode_source_array(np.array([after_values], dtype=np.uint8), encoder)
    transition = build_transition_evidence_vpm(
        before,
        after,
        schema,
        annotations=(control,) if include_control_annotation else (),
    )
    return transition, fields, control, anchor, expectation, schema


def _discovery_observation(
    *,
    interaction_id: str,
    first_value: int,
    second_value: int,
):
    transition, _, control, _, expectation, _ = _transition(
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
        cohort_id="discovery/train",
        transition=transition,
        conformance=conformance,
    )


def _discovery_report():
    observations = tuple(
        _discovery_observation(
            interaction_id=f"discovery-{index}",
            first_value=first,
            second_value=second,
        )
        for index, (first, second) in enumerate(
            ((200, 180), (180, 160), (160, 140), (0, 0)),
            start=1,
        )
    )
    policy = TransitionDiscoveryPolicyDTO.create(
        minimum_observation_count=4,
        minimum_field_occurrence_count=3,
        minimum_field_recurrence_fraction=0.75,
        minimum_signature_occurrence_count=3,
        minimum_signature_recurrence_fraction=0.75,
        minimum_direction_consistency=0.75,
    )
    return discover_recurrent_unexplained_transitions(observations, policy)


def _held_out(*, interaction_id: str, first_value: int, second_value: int):
    transition, _, _, _, _, _ = _transition(
        [0, 0, 0, 0, 0, 0],
        [0, 0, first_value, first_value, second_value, second_value],
        include_control_annotation=False,
    )
    return HeldOutTransitionObservationDTO.create(
        interaction_id=interaction_id,
        cohort_id="validation/held-out",
        transition=transition,
    )


def _proposal_set():
    discovery = _discovery_report()
    observations = tuple(
        _held_out(
            interaction_id=f"validation-{index}",
            first_value=170,
            second_value=150,
        )
        for index in range(1, 4)
    )
    validation = validate_discovered_transition_candidates(
        discovery,
        observations,
        CandidateValidationPolicyDTO.create(
            minimum_validation_observation_count=3,
            minimum_confirmation_fraction=2 / 3,
            minimum_rejection_fraction=2 / 3,
            minimum_magnitude_retention_fraction=0.75,
            direction_epsilon=0.01,
        ),
    )
    return propose_validated_candidate_promotions(discovery, validation)


def _review_with_two_approvals():
    proposal_set = _proposal_set()
    approved_region = CandidatePromotionDecisionDTO.create(
        proposal_set.proposals[0],
        reviewer_id="reviewer:region",
        decision="approved",
        rationale="The held-out field is stable enough to stage as a region.",
        semantic_name="projectile-shadow",
        semantic_type="region_component",
        semantic_role="context",
    )
    approved_relation = CandidatePromotionDecisionDTO.create(
        proposal_set.proposals[1],
        reviewer_id="reviewer:relation",
        decision="approved",
        rationale="The held-out signature is suitable for an explicit relation.",
        semantic_name="paired-signal",
        semantic_type="relative_context",
        semantic_role="context",
    )
    rejected = CandidatePromotionDecisionDTO.create(
        proposal_set.proposals[2],
        reviewer_id="reviewer:reject",
        decision="rejected",
        rationale="This candidate remains semantically incoherent.",
    )
    review = review_candidate_promotion_proposals(
        proposal_set,
        (rejected, approved_relation, approved_region),
    )
    return proposal_set, review, approved_region, approved_relation


def _baseline(schema, control, anchor, **changes):
    values = {
        "baseline_version_id": "perception-model/v42",
        "field_schema_id": schema.field_schema_id,
        "existing_annotation_ids": tuple(
            sorted((control.annotation_id, anchor.annotation_id))
        ),
        "existing_relation_ids": (),
        "existing_transition_expectation_ids": (),
    }
    values.update(changes)
    return PromotionMaterializationBaselineDTO.create(**values)


def test_materializes_approved_region_and_relation_as_inactive_reversible_changes() -> (
    None
):
    proposal_set, review, approved_region, approved_relation = (
        _review_with_two_approvals()
    )
    _, schema, _, control, anchor, _ = _fixture()
    proposal_map = {item.proposal_id: item for item in proposal_set.proposals}
    directives = (
        PromotionMaterializationDirectiveDTO.create(
            proposal_map[approved_region.proposal_id],
            target_kind="region_annotation",
            annotation_properties=(("review_class", "held_out_validated"),),
        ),
        PromotionMaterializationDirectiveDTO.create(
            proposal_map[approved_relation.proposal_id],
            target_kind="relation_annotation",
            relation_member_annotation_ids=(
                control.annotation_id,
                anchor.annotation_id,
            ),
        ),
    )
    baseline = _baseline(schema, control, anchor)

    first = materialize_approved_candidate_promotions(
        proposal_set,
        review,
        schema,
        baseline,
        directives,
    )
    second = materialize_approved_candidate_promotions(
        proposal_set,
        review,
        schema,
        baseline,
        tuple(reversed(directives)),
    )

    assert first == second
    assert first.status == "staged_inactive"
    assert first.activation_status == "not_admitted"
    assert len(first.changes) == 2
    assert len(first.operations("forward")) == 4
    assert len(first.operations("inverse")) == 4
    assert {item.pair_id for item in first.operations("forward")} == {
        item.pair_id for item in first.operations("inverse")
    }

    region = next(item for item in first.changes if item.annotation is not None)
    relation = next(item for item in first.changes if item.relation is not None)
    assert region.annotation.label == "projectile-shadow"
    assert region.annotation.role == "context"
    assert dict(region.annotation.properties)["semantic_type"] == "region_component"
    assert region.annotation.provenance_ref == region.decision_id
    assert region.transition_expectation.annotation_ids == (
        region.annotation.annotation_id,
    )
    assert relation.relation.relation_type == "relative_context"
    assert relation.relation.value == "paired-signal"
    assert relation.relation.member_annotation_ids == tuple(
        sorted((control.annotation_id, anchor.annotation_id))
    )
    assert relation.transition_expectation.relation_ids == (
        relation.relation.relation_id,
    )

    forward_actions = tuple(item.action for item in first.operations("forward"))
    inverse_actions = tuple(item.action for item in first.operations("inverse"))
    assert forward_actions[1::2] == (
        "add_transition_expectation",
        "add_transition_expectation",
    )
    assert inverse_actions[0::2] == (
        "remove_transition_expectation",
        "remove_transition_expectation",
    )
    assert baseline.existing_relation_ids == ()
    assert baseline.existing_transition_expectation_ids == ()


def test_complete_review_without_approvals_produces_empty_inactive_change_set() -> None:
    proposal_set = _proposal_set()
    decisions = tuple(
        CandidatePromotionDecisionDTO.create(
            proposal,
            reviewer_id=f"reviewer:{index}",
            decision="rejected" if index % 2 else "deferred",
            rationale="Do not stage this candidate yet.",
        )
        for index, proposal in enumerate(proposal_set.proposals, start=1)
    )
    review = review_candidate_promotion_proposals(proposal_set, decisions)
    _, schema, _, control, anchor, _ = _fixture()

    change_set = materialize_approved_candidate_promotions(
        proposal_set,
        review,
        schema,
        _baseline(schema, control, anchor),
    )

    assert change_set.status == "no_approved_changes"
    assert change_set.changes == ()
    assert change_set.forward_operation_ids == ()
    assert change_set.inverse_operation_ids == ()
    assert change_set.activation_status == "not_admitted"


def test_rejects_partial_review_and_inexact_directive_coverage() -> None:
    proposal_set, complete, approved_region, approved_relation = (
        _review_with_two_approvals()
    )
    partial = review_candidate_promotion_proposals(
        proposal_set,
        (complete.decisions_for_status("approved")[0],),
    )
    _, schema, _, control, anchor, _ = _fixture()
    baseline = _baseline(schema, control, anchor)
    proposal_map = {item.proposal_id: item for item in proposal_set.proposals}
    region_only = (
        PromotionMaterializationDirectiveDTO.create(
            proposal_map[approved_region.proposal_id],
            target_kind="region_annotation",
        ),
    )

    with pytest.raises(
        PerceptionPromotionMaterializationError,
        match="fully reviewed",
    ):
        materialize_approved_candidate_promotions(
            proposal_set,
            partial,
            schema,
            baseline,
            region_only,
        )
    with pytest.raises(
        PerceptionPromotionMaterializationError,
        match="exactly cover",
    ):
        materialize_approved_candidate_promotions(
            proposal_set,
            complete,
            schema,
            baseline,
            region_only,
        )

    relation = PromotionMaterializationDirectiveDTO.create(
        proposal_map[approved_relation.proposal_id],
        target_kind="relation_annotation",
        relation_member_annotation_ids=("sha256:missing-a", "sha256:missing-b"),
    )
    with pytest.raises(
        PerceptionPromotionMaterializationError,
        match="outside the baseline",
    ):
        materialize_approved_candidate_promotions(
            proposal_set,
            complete,
            schema,
            baseline,
            region_only + (relation,),
        )


def test_collision_and_identity_tampering_are_rejected() -> None:
    proposal_set, review, approved_region, approved_relation = (
        _review_with_two_approvals()
    )
    _, schema, _, control, anchor, _ = _fixture()
    proposal_map = {item.proposal_id: item for item in proposal_set.proposals}
    directives = (
        PromotionMaterializationDirectiveDTO.create(
            proposal_map[approved_region.proposal_id],
            target_kind="region_annotation",
        ),
        PromotionMaterializationDirectiveDTO.create(
            proposal_map[approved_relation.proposal_id],
            target_kind="relation_annotation",
            relation_member_annotation_ids=(
                control.annotation_id,
                anchor.annotation_id,
            ),
        ),
    )
    baseline = _baseline(schema, control, anchor)
    change_set = materialize_approved_candidate_promotions(
        proposal_set,
        review,
        schema,
        baseline,
        directives,
    )
    generated_annotations = tuple(
        item.annotation.annotation_id
        for item in change_set.changes
        if item.annotation is not None
    )
    collision_baseline = _baseline(
        schema,
        control,
        anchor,
        existing_annotation_ids=tuple(
            sorted(
                (control.annotation_id, anchor.annotation_id, *generated_annotations)
            )
        ),
    )
    with pytest.raises(
        PerceptionPromotionMaterializationError,
        match="collides",
    ):
        materialize_approved_candidate_promotions(
            proposal_set,
            review,
            schema,
            collision_baseline,
            directives,
        )

    with pytest.raises(
        PerceptionPromotionMaterializationError,
        match="directive identity",
    ):
        replace(directives[0], directive_id="sha256:tampered")
    with pytest.raises(
        PerceptionPromotionMaterializationError,
        match="baseline identity",
    ):
        replace(baseline, baseline_id="sha256:tampered")
    with pytest.raises(
        PerceptionPromotionMaterializationError,
        match="change-set identity",
    ):
        replace(change_set, change_set_id="sha256:tampered")
