from __future__ import annotations

import pytest

from zeromodel.perception import (
    ActiveModelPointerDTO,
    InMemoryPerceptionModelLifecycleStore,
    PerceptionModelLifecycleError,
    PromotedPerceptionModelDTO,
    activate_promoted_model,
    build_model_lifecycle_snapshot,
    deactivate_active_model,
    register_promoted_model,
    resolve_active_promoted_model,
    rollback_active_model,
    supersede_active_model,
)


def _model(name: str) -> PromotedPerceptionModelDTO:
    return PromotedPerceptionModelDTO(
        promoted_model_id=f"sha256:promoted-{name}",
        model_kind="single_frame",
        model_id=f"sha256:translator-{name}",
        rejection_threshold=0.25,
        calibration_id=f"sha256:calibration-{name}",
        promotion_decision_id=f"sha256:decision-{name}",
        validation_comparison_report_id=f"sha256:validation-{name}",
        training_split="train",
        evaluation_split="validation",
    )


def test_registration_is_idempotent_and_deterministic() -> None:
    store = InMemoryPerceptionModelLifecycleStore()
    model = _model("a")

    first = register_promoted_model(
        store,
        model,
        registered_by="operator",
        registration_reason="passed final review",
    )
    second = register_promoted_model(
        store,
        model,
        registered_by="operator",
        registration_reason="passed final review",
    )

    assert first == second
    assert store.list_ledger_entries() == (first,)
    assert store.get_active_pointer().revision == 0


def test_activation_supersession_and_rollback_preserve_history() -> None:
    store = InMemoryPerceptionModelLifecycleStore()
    first_model = _model("a")
    second_model = _model("b")
    register_promoted_model(
        store,
        first_model,
        registered_by="operator",
        registration_reason="initial candidate",
    )
    register_promoted_model(
        store,
        second_model,
        registered_by="operator",
        registration_reason="replacement candidate",
    )

    activation, pointer_one = activate_promoted_model(
        store,
        first_model.promoted_model_id,
        actor="operator",
        reason="initial activation",
    )
    supersession, pointer_two = supersede_active_model(
        store,
        second_model.promoted_model_id,
        actor="operator",
        reason="validated replacement",
    )
    rollback, pointer_three = rollback_active_model(
        store,
        first_model.promoted_model_id,
        actor="operator",
        reason="replacement regression",
    )

    assert activation.transition_kind == "activate"
    assert supersession.transition_kind == "supersede"
    assert rollback.transition_kind == "rollback"
    assert rollback.related_transition_id == activation.transition_id
    assert (pointer_one.revision, pointer_two.revision, pointer_three.revision) == (1, 2, 3)
    assert resolve_active_promoted_model(store) == first_model

    snapshot = build_model_lifecycle_snapshot(store)
    assert snapshot.active_pointer == pointer_three
    assert tuple(item.transition_kind for item in snapshot.transitions) == (
        "activate",
        "supersede",
        "rollback",
    )
    assert len(snapshot.ledger_entries) == 2


def test_rollback_requires_target_to_have_been_active() -> None:
    store = InMemoryPerceptionModelLifecycleStore()
    first_model = _model("a")
    second_model = _model("b")
    register_promoted_model(
        store,
        first_model,
        registered_by="operator",
        registration_reason="initial candidate",
    )
    register_promoted_model(
        store,
        second_model,
        registered_by="operator",
        registration_reason="unactivated candidate",
    )
    activate_promoted_model(
        store,
        first_model.promoted_model_id,
        actor="operator",
        reason="initial activation",
    )

    with pytest.raises(PerceptionModelLifecycleError, match="never previously active"):
        rollback_active_model(
            store,
            second_model.promoted_model_id,
            actor="operator",
            reason="invalid rollback",
        )


def test_deactivation_clears_pointer_without_deleting_model() -> None:
    store = InMemoryPerceptionModelLifecycleStore()
    model = _model("a")
    entry = register_promoted_model(
        store,
        model,
        registered_by="operator",
        registration_reason="candidate",
    )
    activate_promoted_model(
        store,
        model.promoted_model_id,
        actor="operator",
        reason="activate",
    )

    transition, pointer = deactivate_active_model(
        store,
        actor="operator",
        reason="maintenance",
    )

    assert transition.transition_kind == "deactivate"
    assert pointer.active_promoted_model_id is None
    assert store.get_ledger_entry(model.promoted_model_id) == entry
    with pytest.raises(PerceptionModelLifecycleError, match="no promoted model is active"):
        resolve_active_promoted_model(store)


def test_pointer_replacement_rejects_stale_revision() -> None:
    store = InMemoryPerceptionModelLifecycleStore()
    stale_replacement = ActiveModelPointerDTO(
        pointer_id="sha256:pointer",
        revision=1,
        active_promoted_model_id="sha256:model",
        last_transition_id="sha256:transition",
    )

    with pytest.raises(PerceptionModelLifecycleError, match="revision changed"):
        store.replace_active_pointer(stale_replacement, expected_revision=1)
