from __future__ import annotations

import sqlite3

import pytest

from zeromodel.perception import (
    InMemoryPromotionActivationStore,
    PerceptionPromotionActivationError,
    PromotionRollbackPolicyDTO,
    PromotionRollbackRequestDTO,
    SqlitePromotionActivationStore,
    execute_promotion_activation,
    execute_promotion_rollback,
)
from zeromodel.perception.promotion_activation import _dumps

from test_promotion_activation_p18g import _materialization


@pytest.mark.parametrize(
    "store_factory",
    [
        lambda initial, path: InMemoryPromotionActivationStore(initial),
        lambda initial, path: SqlitePromotionActivationStore(str(path), initial),
    ],
)
def test_governed_rollback_restores_exact_state_and_preserves_history(
    tmp_path,
    store_factory,
) -> None:
    initial, change_set = _materialization()
    store = store_factory(initial, tmp_path / "activation.sqlite3")
    activation = execute_promotion_activation(store, change_set)
    request = PromotionRollbackRequestDTO.create(
        rollback_plan_id=activation.rollback_plan.rollback_plan_id,
        expected_active_state_id=activation.resulting_state.state_id,
        requested_by="operator:alice",
        reason="Release validation rollback exercise.",
    )

    admission = store.admit_rollback(request)
    rollback = store.commit_rollback(admission)

    assert rollback.restored_state == initial
    assert store.get_active_state() == initial
    assert store.get_activation_bundle(change_set.change_set_id) == activation
    assert rollback.activation_receipt == activation.receipt
    assert rollback.receipt.prior_state_id == activation.resulting_state.state_id
    assert rollback.receipt.restored_state_id == initial.state_id
    assert rollback.rollback_plan.status == "executed"

    second = store.commit_rollback(admission)
    assert second == rollback


def test_sqlite_activation_and_rollback_survive_restart(tmp_path) -> None:
    initial, change_set = _materialization()
    path = tmp_path / "activation.sqlite3"
    store = SqlitePromotionActivationStore(str(path), initial)
    activation = execute_promotion_activation(store, change_set)
    request = PromotionRollbackRequestDTO.create(
        rollback_plan_id=activation.rollback_plan.rollback_plan_id,
        expected_active_state_id=activation.resulting_state.state_id,
        requested_by="operator:bob",
        reason="Restart-safe rollback validation.",
    )
    admission = store.admit_rollback(request, PromotionRollbackPolicyDTO.create())

    reopened = SqlitePromotionActivationStore(str(path))
    assert reopened.get_active_state() == activation.resulting_state
    assert reopened.get_activation_bundle(change_set.change_set_id) == activation
    assert reopened.get_rollback_plan(activation.rollback_plan.rollback_plan_id) == activation.rollback_plan

    rollback = reopened.commit_rollback(admission)
    again = SqlitePromotionActivationStore(str(path))
    assert again.get_active_state() == initial
    assert again.commit_rollback(admission) == rollback
    assert again.get_rollback_plan(activation.rollback_plan.rollback_plan_id).status == "executed"


def test_stale_rollback_admission_is_rejected_without_mutation(tmp_path) -> None:
    initial, first_change_set = _materialization()
    path = tmp_path / "activation.sqlite3"
    store = SqlitePromotionActivationStore(str(path), initial)
    first = execute_promotion_activation(store, first_change_set)
    request = PromotionRollbackRequestDTO.create(
        rollback_plan_id=first.rollback_plan.rollback_plan_id,
        expected_active_state_id=first.resulting_state.state_id,
        requested_by="operator:alice",
        reason="Stale admission race test.",
    )
    admission = store.admit_rollback(request)
    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE active_promotion_state SET state_id = ?, revision = ?, baseline_version_id = ?, payload = ? WHERE scope = 'default'",
            (
                initial.state_id,
                initial.revision,
                initial.baseline_version_id,
                _dumps(initial),
            ),
        )

    with pytest.raises(PerceptionPromotionActivationError, match="active state changed"):
        store.commit_rollback(admission)

    assert store.get_active_state() == initial
    assert store.get_rollback_plan(first.rollback_plan.rollback_plan_id).status == "stored_inactive"


def test_sqlite_rollback_failure_injection_leaves_no_partial_state(tmp_path) -> None:
    initial, change_set = _materialization()

    class FailingStore(SqlitePromotionActivationStore):
        def _after_rollback_operation_applied(self, operation) -> None:
            if operation.sequence == 1:
                raise RuntimeError("injected rollback failure")

    store = FailingStore(str(tmp_path / "activation.sqlite3"), initial)
    activation = execute_promotion_activation(store, change_set)
    request = PromotionRollbackRequestDTO.create(
        rollback_plan_id=activation.rollback_plan.rollback_plan_id,
        expected_active_state_id=activation.resulting_state.state_id,
        requested_by="operator:alice",
        reason="Fault injection.",
    )
    admission = store.admit_rollback(request)

    with pytest.raises(RuntimeError, match="injected rollback failure"):
        store.commit_rollback(admission)

    reopened = SqlitePromotionActivationStore(str(tmp_path / "activation.sqlite3"))
    assert reopened.get_active_state() == activation.resulting_state
    assert reopened.get_rollback_plan(activation.rollback_plan.rollback_plan_id).status == "stored_inactive"


def test_sqlite_rejects_malformed_json_and_missing_operation_ordinals(tmp_path) -> None:
    initial, change_set = _materialization()
    path = tmp_path / "activation.sqlite3"
    store = SqlitePromotionActivationStore(str(path), initial)
    activation = execute_promotion_activation(store, change_set)

    with sqlite3.connect(path) as conn:
        conn.execute(
            "UPDATE activation_bundles SET payload = ? WHERE change_set_id = ?",
            ("{not json", change_set.change_set_id),
        )
    with pytest.raises(PerceptionPromotionActivationError, match="malformed"):
        SqlitePromotionActivationStore(str(path)).get_activation_bundle(change_set.change_set_id)

    store = SqlitePromotionActivationStore(str(tmp_path / "ordinals.sqlite3"), initial)
    activation = execute_promotion_activation(store, change_set)
    with sqlite3.connect(tmp_path / "ordinals.sqlite3") as conn:
        payload = conn.execute(
            "SELECT payload FROM activation_bundles WHERE change_set_id = ?",
            (change_set.change_set_id,),
        ).fetchone()[0]
        payload = payload.replace('"sequence":1', '"sequence":2', 1)
        conn.execute(
            "UPDATE activation_bundles SET payload = ? WHERE change_set_id = ?",
            (payload, change_set.change_set_id),
        )
    with pytest.raises(PerceptionPromotionActivationError, match="ordinals"):
        SqlitePromotionActivationStore(str(tmp_path / "ordinals.sqlite3")).get_rollback_plan(
            activation.rollback_plan.rollback_plan_id
        )
