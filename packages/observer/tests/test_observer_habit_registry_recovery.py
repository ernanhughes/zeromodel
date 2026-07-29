import sqlite3

from zeromodel.observer import (
    SqliteObserverHabitRegistry,
    SqliteObserverHabitRegistryStore,
    empty_observer_habit_registry_snapshot,
)

from test_observer_habit_registry_store import _admitted


def test_missing_head_recovery(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path)
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    expected = registry.current_snapshot()
    registry.close()

    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM observer_habit_registry_head")
    conn.commit()
    conn.close()

    store = SqliteObserverHabitRegistryStore(path)
    recovery = store.recover()
    assert recovery.disposition == "recovered"
    assert recovery.recovered_head_snapshot_id == expected.habit_registry_snapshot_id
    assert store.verify_integrity().status == "verified"
    store.close()


def test_corrupt_history_recovery_refuses(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path)
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    registry.close()

    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM observer_habit_registry_head")
    conn.execute(
        "UPDATE observer_habit_registry_events SET source_snapshot_id = 'wrong'"
    )
    conn.commit()
    conn.close()

    store = SqliteObserverHabitRegistryStore(path)
    recovery = store.recover()
    assert recovery.disposition == "unsafe_to_recover"
    store.close()


def test_missing_head_recovery_does_not_overwrite_raced_head(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path)
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    expected = registry.current_snapshot()
    registry.close()

    conn = sqlite3.connect(path)
    conn.execute("DELETE FROM observer_habit_registry_head")
    conn.commit()
    conn.close()

    first = SqliteObserverHabitRegistryStore(path)
    second = SqliteObserverHabitRegistryStore(path)
    first_recovery = first.recover()
    second_recovery = second.recover()
    assert {first_recovery.disposition, second_recovery.disposition} <= {
        "recovered",
        "not_needed",
    }
    conn = sqlite3.connect(path)
    rows = conn.execute(
        "SELECT current_snapshot_id, current_registry_sequence FROM observer_habit_registry_head"
    ).fetchall()
    conn.close()
    assert rows == [(expected.habit_registry_snapshot_id, expected.registry_sequence)]
    assert first.verify_integrity().status == "verified"
    assert second.verify_integrity().status == "verified"
    first.close()
    second.close()


def test_initialize_completes_tables_only_database(tmp_path) -> None:
    path = tmp_path / "registry.sqlite"
    store = SqliteObserverHabitRegistryStore(path)
    store._create_schema()  # noqa: SLF001
    store.close()

    reopened = SqliteObserverHabitRegistryStore(path)
    reopened.initialize()
    assert reopened.verify_integrity().status == "verified"
    reopened.close()


def test_initialize_completes_metadata_only_database(tmp_path) -> None:
    path = tmp_path / "registry.sqlite"
    store = SqliteObserverHabitRegistryStore(path)
    store._create_schema()  # noqa: SLF001
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO observer_habit_registry_metadata VALUES (?, ?)",
        ("schema_version", "observer-habit-registry-sqlite/1"),
    )
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO observer_habit_registry_metadata VALUES (?, ?)",
        ("store_id", "sqlite-store:test"),
    )
    store.close()

    reopened = SqliteObserverHabitRegistryStore(path)
    reopened.initialize()
    assert reopened.verify_integrity().status == "verified"
    reopened.close()


def test_initialize_completes_empty_snapshot_without_head(tmp_path) -> None:
    path = tmp_path / "registry.sqlite"
    store = SqliteObserverHabitRegistryStore(path)
    store._create_schema()  # noqa: SLF001
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO observer_habit_registry_metadata VALUES (?, ?)",
        ("schema_version", "observer-habit-registry-sqlite/1"),
    )
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO observer_habit_registry_metadata VALUES (?, ?)",
        ("store_id", "sqlite-store:test"),
    )
    store._insert_snapshot(empty_observer_habit_registry_snapshot())  # noqa: SLF001
    store.close()

    reopened = SqliteObserverHabitRegistryStore(path)
    reopened.initialize()
    assert reopened.verify_integrity().status == "verified"
    reopened.close()


def test_initialize_completes_head_without_metadata(tmp_path) -> None:
    path = tmp_path / "registry.sqlite"
    store = SqliteObserverHabitRegistryStore(path)
    store._create_schema()  # noqa: SLF001
    empty = empty_observer_habit_registry_snapshot()
    store._insert_snapshot(empty)  # noqa: SLF001
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO observer_habit_registry_head VALUES (1, ?, 0, 0)",
        (empty.habit_registry_snapshot_id,),
    )
    store.close()

    reopened = SqliteObserverHabitRegistryStore(path)
    reopened.initialize()
    assert reopened.verify_integrity().status == "verified"
    reopened.close()


def test_initialize_rejects_unsupported_schema_metadata(tmp_path) -> None:
    path = tmp_path / "registry.sqlite"
    store = SqliteObserverHabitRegistryStore(path)
    store._create_schema()  # noqa: SLF001
    store._conn.execute(  # noqa: SLF001
        "INSERT INTO observer_habit_registry_metadata VALUES (?, ?)",
        ("schema_version", "observer-habit-registry-sqlite/999"),
    )
    store.close()

    reopened = SqliteObserverHabitRegistryStore(path)
    try:
        reopened.initialize()
    except Exception as exc:
        assert "unsupported registry schema" in str(exc)
    else:
        raise AssertionError("unsupported schema initialized")
    finally:
        reopened.close()
