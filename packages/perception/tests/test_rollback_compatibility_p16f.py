from __future__ import annotations

import pytest

from zeromodel.perception.compatibility import (
    PerceptionModelCompatibilityError,
    assess_rollback_compatibility,
    build_model_compatibility_contract,
    rollback_compatible_model,
)
from zeromodel.perception.lifecycle import (
    InMemoryPerceptionModelLifecycleStore,
    activate_promoted_model,
    register_promoted_model,
    supersede_active_model,
)
from zeromodel.perception.promotion import PromotedPerceptionModelDTO


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


def _contract(model: PromotedPerceptionModelDTO, **overrides: str):
    values = {
        "action_schema_id": "actions-v1",
        "source_encoder_spec_id": "encoder-v1",
        "field_schema_id": "fields-v1",
        "inference_semantics_version": "inference-v1",
        "deployment_slot": "primary",
    }
    values.update(overrides)
    return build_model_compatibility_contract(model, **values)


def test_compatibility_assessment_is_deterministic_and_exact() -> None:
    current = _contract(_promoted("current"))
    target = _contract(_promoted("target"))

    first = assess_rollback_compatibility(current, target)
    second = assess_rollback_compatibility(current, target)

    assert first == second
    assert first.status == "compatible"
    assert first.mismatched_fields == ()


def test_action_schema_mismatch_blocks_rollback_eligibility() -> None:
    current = _contract(_promoted("current"))
    target = _contract(_promoted("target"), action_schema_id="actions-v2")

    assessment = assess_rollback_compatibility(current, target)

    assert assessment.status == "incompatible"
    assert assessment.mismatched_fields == ("action_schema_id",)


def test_compatible_historical_model_can_be_rolled_back() -> None:
    store = InMemoryPerceptionModelLifecycleStore()
    earlier = _promoted("earlier")
    current = _promoted("current")
    register_promoted_model(store, earlier, registered_by="test", registration_reason="candidate")
    register_promoted_model(store, current, registered_by="test", registration_reason="candidate")
    activate_promoted_model(store, earlier.promoted_model_id, actor="test", reason="activate")
    supersede_active_model(store, current.promoted_model_id, actor="test", reason="supersede")

    assessment, transition, pointer = rollback_compatible_model(
        store,
        earlier.promoted_model_id,
        current_contract=_contract(current),
        target_contract=_contract(earlier),
        actor="operator",
        reason="restore compatible prior model",
    )

    assert assessment.status == "compatible"
    assert transition.transition_kind == "rollback"
    assert pointer.active_promoted_model_id == earlier.promoted_model_id


def test_incompatible_historical_model_is_not_rolled_back() -> None:
    store = InMemoryPerceptionModelLifecycleStore()
    earlier = _promoted("earlier")
    current = _promoted("current")
    register_promoted_model(store, earlier, registered_by="test", registration_reason="candidate")
    register_promoted_model(store, current, registered_by="test", registration_reason="candidate")
    activate_promoted_model(store, earlier.promoted_model_id, actor="test", reason="activate")
    supersede_active_model(store, current.promoted_model_id, actor="test", reason="supersede")

    with pytest.raises(PerceptionModelCompatibilityError, match="action_schema_id"):
        rollback_compatible_model(
            store,
            earlier.promoted_model_id,
            current_contract=_contract(current),
            target_contract=_contract(earlier, action_schema_id="actions-v0"),
            actor="operator",
            reason="unsafe rollback",
        )

    assert store.get_active_pointer().active_promoted_model_id == current.promoted_model_id
