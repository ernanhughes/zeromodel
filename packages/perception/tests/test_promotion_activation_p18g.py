from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    CandidatePromotionDecisionDTO,
    CandidateValidationPolicyDTO,
    HeldOutTransitionObservationDTO,
    PerceptionRegionAnnotationDTO,
    PromotionMaterializationBaselineDTO,
    PromotionMaterializationDirectiveDTO,
    SourceImageEncoderSpecDTO,
    TransitionDiscoveryObservationDTO,
    TransitionDiscoveryPolicyDTO,
    TransitionExpectationDTO,
    build_grid_field_schema,
    build_transition_evidence_vpm,
    discover_recurrent_unexplained_transitions,
    encode_source_array,
    evaluate_transition_conformance,
    materialize_approved_candidate_promotions,
    propose_validated_candidate_promotions,
    review_candidate_promotion_proposals,
    validate_discovered_transition_candidates,
)
from zeromodel.perception.promotion_activation import (
    ActivePromotionStateDTO,
    InMemoryPromotionActivationStore,
    PerceptionPromotionActivationError,
    PromotionActivationPolicyDTO,
    audit_promotion_activation,
    build_promotion_activation_bundle,
    execute_promotion_activation,
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
    return transition, schema, fields, control, expectation


def _discovery_observation(
    *,
    interaction_id: str,
    first_value: int,
    second_value: int,
):
    transition, _, _, control, expectation = _transition(
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
    return discover_recurrent_unexplained_transitions(
        observations,
        TransitionDiscoveryPolicyDTO.create(
            minimum_observation_count=4,
            minimum_field_occurrence_count=3,
            minimum_field_recurrence_fraction=0.75,
            minimum_signature_occurrence_count=3,
            minimum_signature_recurrence_fraction=0.75,
            minimum_direction_consistency=0.75,
        ),
    )


def _validation_report(discovery):
    observations = []
    for index in range(1, 4):
        transition, _, _, _, _ = _transition(
            [0, 0, 0, 0, 0, 0],
            [0, 0, 170, 170, 150, 150],
            include_control_annotation=False,
        )
        observations.append(
            HeldOutTransitionObservationDTO.create(
                interaction_id=f"validation-{index}",
                cohort_id="validation/held-out",
                transition=transition,
            )
        )
    return validate_discovered_transition_candidates(
        discovery,
        tuple(observations),
        CandidateValidationPolicyDTO.create(
            minimum_validation_observation_count=3,
            minimum_confirmation_fraction=2 / 3,
            minimum_rejection_fraction=2 / 3,
            minimum_magnitude_retention_fraction=0.75,
        ),
    )


def _materialization(*, target_kind: str = "region_annotation"):
    discovery = _discovery_report()
    validation = _validation_report(discovery)
    proposal_set = propose_validated_candidate_promotions(discovery, validation)
    approved_proposal = proposal_set.proposals[0]
    decisions = [
        CandidatePromotionDecisionDTO.create(
            approved_proposal,
            reviewer_id="reviewer:alice",
            decision="approved",
            rationale="Held-out evidence and semantic review support activation.",
            semantic_name="projectile-shadow",
            semantic_type=(
                "spatial_relation"
                if target_kind == "relation_annotation"
                else "region_component"
            ),
            semantic_role="context",
        )
    ]
    for index, proposal in enumerate(proposal_set.proposals[1:], start=1):
        decisions.append(
            CandidatePromotionDecisionDTO.create(
                proposal,
                reviewer_id=f"reviewer:reject-{index}",
                decision="rejected",
                rationale="This candidate is not selected for materialization.",
            )
        )
    review = review_candidate_promotion_proposals(proposal_set, tuple(decisions))
    _, schema, fields, control, stable_expectation = _fixture()
    member = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[1].field_id,),
        label="known-member",
        role="context",
    )
    initial = ActivePromotionStateDTO.create(
        revision=0,
        baseline_version_id="policy-baseline:v1",
        field_schema_id=schema.field_schema_id,
        annotations=(control, member),
        transition_expectations=(stable_expectation,),
    )
    baseline = initial.baseline()
    assert baseline == PromotionMaterializationBaselineDTO.create(
        baseline_version_id="policy-baseline:v1",
        field_schema_id=schema.field_schema_id,
        existing_annotation_ids=(control.annotation_id, member.annotation_id),
        existing_transition_expectation_ids=(stable_expectation.expectation_id,),
    )
    directive = PromotionMaterializationDirectiveDTO.create(
        approved_proposal,
        target_kind=target_kind,
        relation_member_annotation_ids=(
            (control.annotation_id, member.annotation_id)
            if target_kind == "relation_annotation"
            else ()
        ),
        annotation_properties=(
            (("source", "p18g-test"),) if target_kind == "region_annotation" else ()
        ),
    )
    change_set = materialize_approved_candidate_promotions(
        proposal_set,
        review,
        schema,
        baseline,
        (directive,),
    )
    assert change_set.status == "staged_inactive"
    return initial, change_set


def _no_approved_change_set():
    discovery = _discovery_report()
    validation = _validation_report(discovery)
    proposal_set = propose_validated_candidate_promotions(discovery, validation)
    decisions = tuple(
        CandidatePromotionDecisionDTO.create(
            proposal,
            reviewer_id=f"reviewer:no-{index}",
            decision="rejected",
            rationale="No candidate is approved in this review.",
        )
        for index, proposal in enumerate(proposal_set.proposals, start=1)
    )
    review = review_candidate_promotion_proposals(proposal_set, decisions)
    _, schema, fields, control, stable_expectation = _fixture()
    member = PerceptionRegionAnnotationDTO.create(
        schema,
        (fields[1].field_id,),
        label="known-member",
        role="context",
    )
    initial = ActivePromotionStateDTO.create(
        revision=0,
        baseline_version_id="policy-baseline:v1",
        field_schema_id=schema.field_schema_id,
        annotations=(control, member),
        transition_expectations=(stable_expectation,),
    )
    change_set = materialize_approved_candidate_promotions(
        proposal_set,
        review,
        schema,
        initial.baseline(),
    )
    return initial, change_set


def test_activates_all_operations_and_persists_exact_inverse_plan() -> None:
    initial, change_set = _materialization()
    store = InMemoryPromotionActivationStore(initial)

    bundle = execute_promotion_activation(store, change_set)
    active = store.get_active_state()

    assert bundle.audit_report.status == "admissible"
    assert bundle.admission.status == "admitted"
    assert bundle.receipt.status == "activated"
    assert bundle.rollback_plan.status == "stored_inactive"
    assert active == bundle.resulting_state
    assert active.revision == 1
    assert active.last_change_set_id == change_set.change_set_id
    assert active.state_id != initial.state_id
    assert active.baseline().baseline_id == bundle.receipt.resulting_baseline_id
    assert bundle.rollback_plan.restore_state == initial
    assert (
        bundle.rollback_plan.inverse_operation_ids == change_set.inverse_operation_ids
    )
    assert bundle.receipt.forward_operation_ids == change_set.forward_operation_ids
    assert store.get_activation_bundle(change_set.change_set_id) == bundle
    assert store.list_activation_bundles() == (bundle,)

    change = change_set.changes[0]
    assert change.annotation is not None
    assert change.annotation.annotation_id in {
        item.annotation_id for item in active.annotations
    }
    assert change.transition_expectation.expectation_id in {
        item.expectation_id for item in active.transition_expectations
    }


def test_exact_baseline_drift_blocks_activation_without_mutation() -> None:
    initial, change_set = _materialization()
    drifted = ActivePromotionStateDTO.create(
        revision=initial.revision,
        baseline_version_id="policy-baseline:v2",
        field_schema_id=initial.field_schema_id,
        annotations=initial.annotations,
        relations=initial.relations,
        transition_expectations=initial.transition_expectations,
    )
    report = audit_promotion_activation(drifted, change_set)
    assert report.status == "blocked"
    assert {item.code for item in report.findings} >= {
        "baseline_version_mismatch",
        "baseline_identity_mismatch",
    }

    store = InMemoryPromotionActivationStore(drifted)
    with pytest.raises(
        PerceptionPromotionActivationError,
        match="not admissible",
    ):
        execute_promotion_activation(store, change_set)
    assert store.get_active_state() == drifted
    assert store.list_activation_bundles() == ()


def test_activation_policy_can_forbid_relation_materialization() -> None:
    initial, change_set = _materialization(target_kind="relation_annotation")
    policy = PromotionActivationPolicyDTO.create(
        allowed_target_kinds=("region_annotation",),
    )

    report = audit_promotion_activation(initial, change_set, policy)

    assert report.status == "blocked"
    assert {item.code for item in report.findings} == {"target_kind_disallowed"}


def test_mid_plan_failure_is_atomic() -> None:
    initial, change_set = _materialization()

    class FailingStore(InMemoryPromotionActivationStore):
        def _after_operation_applied(self, operation) -> None:
            if operation.sequence == 2:
                raise RuntimeError("injected operation failure")

    store = FailingStore(initial)
    with pytest.raises(RuntimeError, match="injected operation failure"):
        execute_promotion_activation(store, change_set)

    assert store.get_active_state() == initial
    assert store.list_activation_bundles() == ()


def test_compare_and_swap_rejects_concurrent_state_change() -> None:
    initial, change_set = _materialization()
    bundle = build_promotion_activation_bundle(initial, change_set)
    drifted = ActivePromotionStateDTO.create(
        revision=1,
        baseline_version_id="concurrent:v2",
        field_schema_id=initial.field_schema_id,
        annotations=initial.annotations,
        relations=initial.relations,
        transition_expectations=initial.transition_expectations,
        last_change_set_id="sha256:concurrent",
    )
    store = InMemoryPromotionActivationStore(drifted)

    with pytest.raises(
        PerceptionPromotionActivationError,
        match="active state changed",
    ):
        store.commit_activation(initial, change_set, bundle)

    assert store.get_active_state() == drifted
    assert store.list_activation_bundles() == ()


def test_no_approved_changes_are_explicitly_not_applicable() -> None:
    initial, change_set = _no_approved_change_set()
    assert change_set.status == "no_approved_changes"

    report = audit_promotion_activation(initial, change_set)

    assert report.status == "not_applicable"
    assert {item.code for item in report.findings} == {"no_approved_changes"}
    store = InMemoryPromotionActivationStore(initial)
    with pytest.raises(
        PerceptionPromotionActivationError,
        match="not admissible",
    ):
        execute_promotion_activation(store, change_set)
    assert store.get_active_state() == initial


def test_activation_contract_identities_reject_tampering() -> None:
    initial, change_set = _materialization()
    policy = PromotionActivationPolicyDTO.create()
    report = audit_promotion_activation(initial, change_set, policy)
    bundle = build_promotion_activation_bundle(initial, change_set, policy)

    with pytest.raises(
        PerceptionPromotionActivationError,
        match="policy identity",
    ):
        replace(policy, policy_id="sha256:tampered")
    with pytest.raises(
        PerceptionPromotionActivationError,
        match="audit report identity",
    ):
        replace(report, report_id="sha256:tampered")
    with pytest.raises(
        PerceptionPromotionActivationError,
        match="admission identity",
    ):
        replace(bundle.admission, admission_id="sha256:tampered")
    with pytest.raises(
        PerceptionPromotionActivationError,
        match="rollback plan identity",
    ):
        replace(bundle.rollback_plan, rollback_plan_id="sha256:tampered")
    with pytest.raises(
        PerceptionPromotionActivationError,
        match="receipt identity",
    ):
        replace(bundle.receipt, receipt_id="sha256:tampered")
    with pytest.raises(
        PerceptionPromotionActivationError,
        match="bundle identity",
    ):
        replace(bundle, bundle_id="sha256:tampered")
    with pytest.raises(
        PerceptionPromotionActivationError,
        match="state identity",
    ):
        replace(initial, state_id="sha256:tampered")
