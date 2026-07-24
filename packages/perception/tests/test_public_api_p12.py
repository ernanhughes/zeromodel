from __future__ import annotations

from zeromodel.perception import (
    ACTIVE_POINTER_SEMANTICS,
    MODEL_LEDGER_SEMANTICS,
    MODEL_TRANSITION_SEMANTICS,
    InMemoryPerceptionModelLifecycleStore,
)


def test_model_lifecycle_public_contract() -> None:
    assert MODEL_LEDGER_SEMANTICS == "append_only_promoted_model_registration"
    assert MODEL_TRANSITION_SEMANTICS == (
        "append_only_model_activation_supersession_and_rollback"
    )
    assert ACTIVE_POINTER_SEMANTICS == (
        "revisioned_active_model_pointer_with_optimistic_concurrency"
    )
    store = InMemoryPerceptionModelLifecycleStore()
    assert store.get_active_pointer().revision == 0
    assert store.list_ledger_entries() == ()
    assert store.list_transitions() == ()
