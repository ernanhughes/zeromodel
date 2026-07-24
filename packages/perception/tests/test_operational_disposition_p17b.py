from __future__ import annotations

import pytest

from zeromodel.perception.compatibility import (
    assess_rollback_compatibility,
    build_model_compatibility_contract,
)
from zeromodel.perception.disposition import (
    PerceptionOperationalDispositionError,
    disposition_operational_recommendation,
    execute_approved_rollback,
)
from zeromodel.perception.lifecycle import (
    InMemoryPerceptionModelLifecycleStore,
    activate_promoted_model,
    build_model_lifecycle_snapshot,
    register_promoted_model,
    supersede_active_model,
)
from zeromodel.perception.promotion import PromotedPerceptionModelDTO
from zeromodel.perception.recommendation import OperationalRecommendationDTO


def _promoted(name: str) -> PromotedPerceptionModelDTO:
    return PromotedPerceptionModelDTO(
        promoted_model_id=f"promoted-{name}",
        model_kind="single_frame",
        model_id=f"model-{name}",
        rejection_threshold=0.25,
        calibration_id=f"calibration-{name}",
        promotion_decision_id=f"decision-{name}",
        validation_comparison_report_id=f"validation-{name}",
        training_split="train",
        evaluation_split="validation",
    )


def _contract(model: PromotedPerceptionModelDTO):
    return build_model_compatibility_contract(
        model,
        action_schema_id="actions-v1",
        source_encoder_spec_id="encoder-v1",
        field_schema_id="fields-v1",
        inference_semantics_version="runtime-v1",
        deployment_slot="primary",
    )


def _fixture():
    store = InMemoryPerceptionModelLifecycleStore()
    earlier = _promoted("earlier")
    current = _promoted("current")
    for model in (earlier, current):
        register_promoted_model(
            store,
            model,
            registered_by="test",
            registration_reason="candidate",
        )
    activate_promoted_model(store, earlier.promoted_model_id, actor="test", reason="activate")
    supersede_active_model(store, current.promoted_model_id, actor="test", reason="supersede")
    snapshot = build_model_lifecycle_snapshot(store)
    current_contract = _contract(current)
    target_contract = _contract(earlier)
    assessment = assess_rollback_compatibility(current_contract, target_contract)
    recommendation = OperationalRecommendationDTO(
        recommendation_id="sha256:recommendation",
        health_report_id="sha256:health",
        lifecycle_snapshot_id=snapshot.snapshot_id,
        active_pointer_id=snapshot.active_pointer.pointer_id,
        active_pointer_revision=snapshot.active_pointer.revision,
        active_promoted_model_id=current.promoted_model_id,
        current_contract_id=current_contract.contract_id,
        status="rollback_candidate",
        selected_target_promoted_model_id=earlier.promoted_model_id,
        selected_assessment_id=assessment.assessment_id,
        assessed_candidates=(assessment,),
        rationale="supported drift and compatible historical target",
    )
    return store, earlier, current, current_contract, target_contract, recommendation


def test_approval_is_deterministic_and_non_mutating() -> None:
    store, earlier, current, _, _, recommendation = _fixture()

    first = disposition_operational_recommendation(
        recommendation,
        status="approved",
        reviewed_by="operator",
        reason="restore prior compatible model",
    )
    second = disposition_operational_recommendation(
        recommendation,
        status="approved",
        reviewed_by="operator",
        reason="restore prior compatible model",
    )

    assert first == second
    assert store.get_active_pointer().active_promoted_model_id == current.promoted_model_id
    assert first.selected_target_promoted_model_id == earlier.promoted_model_id


def test_rejection_cannot_execute_or_mutate_lifecycle() -> None:
    store, _, current, current_contract, target_contract, recommendation = _fixture()
    disposition = disposition_operational_recommendation(
        recommendation,
        status="rejected",
        reviewed_by="operator",
        reason="hold current model pending investigation",
    )

    with pytest.raises(PerceptionOperationalDispositionError, match="approved"):
        execute_approved_rollback(
            store,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )

    assert store.get_active_pointer().active_promoted_model_id == current.promoted_model_id


def test_approved_fresh_recommendation_executes_governed_rollback() -> None:
    store, earlier, _, current_contract, target_contract, recommendation = _fixture()
    disposition = disposition_operational_recommendation(
        recommendation,
        status="approved",
        reviewed_by="operator",
        reason="restore prior compatible model",
    )

    assessment, transition, pointer = execute_approved_rollback(
        store,
        recommendation,
        disposition,
        current_contract=current_contract,
        target_contract=target_contract,
    )

    assert assessment.status == "compatible"
    assert transition.transition_kind == "rollback"
    assert pointer.active_promoted_model_id == earlier.promoted_model_id


def test_approved_recommendation_is_rejected_after_pointer_revision_changes() -> None:
    store, _, current, current_contract, target_contract, recommendation = _fixture()
    disposition = disposition_operational_recommendation(
        recommendation,
        status="approved",
        reviewed_by="operator",
        reason="restore prior compatible model",
    )
    replacement = _promoted("replacement")
    register_promoted_model(
        store,
        replacement,
        registered_by="test",
        registration_reason="new candidate",
    )
    supersede_active_model(
        store,
        replacement.promoted_model_id,
        actor="other-operator",
        reason="state changed before execution",
    )

    with pytest.raises(PerceptionOperationalDispositionError, match="pointer revision"):
        execute_approved_rollback(
            store,
            recommendation,
            disposition,
            current_contract=current_contract,
            target_contract=target_contract,
        )

    assert store.get_active_pointer().active_promoted_model_id == replacement.promoted_model_id
    assert current.promoted_model_id != replacement.promoted_model_id


def test_non_rollback_recommendation_cannot_be_approved() -> None:
    _, _, _, _, _, recommendation = _fixture()
    healthy = OperationalRecommendationDTO(
        **{
            **recommendation.__dict__,
            "recommendation_id": "sha256:healthy",
            "status": "no_action",
            "selected_target_promoted_model_id": None,
            "selected_assessment_id": None,
            "assessed_candidates": (),
            "rationale": "healthy evidence",
        }
    )

    with pytest.raises(PerceptionOperationalDispositionError, match="rollback-candidate"):
        disposition_operational_recommendation(
            healthy,
            status="approved",
            reviewed_by="operator",
            reason="invalid approval",
        )
