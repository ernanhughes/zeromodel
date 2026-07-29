import sqlite3

from zeromodel.observer import (
    SqliteObserverHabitRegistry,
    SqliteObserverHabitRegistryStore,
    activate_observer_habit,
)

from test_observer_habit_registry_store import _admitted, _request, _scope


def test_database_lock_returns_bounded_result(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path, busy_timeout_ms=1)
    source = registry.current_snapshot()
    event = registry.register_admission(
        habit_specification=habit, admission_decision=decision
    )
    assert event is not None
    result = registry.current_snapshot()
    registry.close()

    store = SqliteObserverHabitRegistryStore(path, busy_timeout_ms=1)
    locker = sqlite3.connect(path, isolation_level=None)
    locker.execute("BEGIN IMMEDIATE")
    try:
        commit = store.append_transition(
            expected_source_snapshot_id=source.habit_registry_snapshot_id,
            event=event,
            result_snapshot=result,
        )
        assert commit.disposition in {"database_locked", "stale_source_snapshot"}
    finally:
        locker.execute("ROLLBACK")
        locker.close()
        store.close()


def test_reader_during_writer_sees_committed_snapshot(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path)
    old = registry.current_snapshot()
    writer = sqlite3.connect(path, isolation_level=None)
    writer.execute("BEGIN IMMEDIATE")
    try:
        reader = SqliteObserverHabitRegistryStore(path)
        assert reader.load_current_snapshot() == old
        reader.close()
    finally:
        writer.execute("ROLLBACK")
        writer.close()
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    reader = SqliteObserverHabitRegistryStore(path)
    assert reader.load_current_snapshot().registry_sequence == 1
    reader.close()
    registry.close()


def test_integrity_verification_uses_one_consistent_read_snapshot(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    a = SqliteObserverHabitRegistry.open(path)
    b = SqliteObserverHabitRegistry.open(path)
    a.register_admission(habit_specification=habit, admission_decision=decision)
    source = b.current_snapshot()
    activation_scope = _scope(habit)

    def advance_after_read_begins() -> None:
        activate_observer_habit(
            registry=a,
            activation_scope=activation_scope,
            activation_request=_request(habit, source, activation_scope),
            habit_specification=habit,
        )

    b.store._verification_read_started_hook = advance_after_read_begins  # noqa: SLF001
    verification = b.verify_integrity()
    assert verification.status == "verified"
    assert verification.head_registry_sequence in {1, 2}
    assert a.verify_integrity().status == "verified"
    a.close()
    b.close()
