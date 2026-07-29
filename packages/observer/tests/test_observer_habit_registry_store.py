import sqlite3

import pytest

from zeromodel.observer import (
    InMemoryObserverHabitRegistry,
    ObserverHabitActivationRequestDTO,
    ObserverHabitRegistryEntryDTO,
    ObserverHabitRegistrySnapshotDTO,
    ObserverHabitRollbackRequestDTO,
    SqliteObserverHabitRegistry,
    SqliteObserverHabitRegistryStore,
    activate_observer_habit,
    empty_observer_habit_registry_snapshot,
)
from zeromodel.observer.habit_registry_sqlite import _encode_payload

from test_observer_habit_admission import admitted_evidence, admission_recipe
from zeromodel.observer import ObserverHabitActivationScopeDTO, admit_observer_habit


def _admitted() -> tuple[object, object]:
    habit, historical, episode, audit = admitted_evidence()
    decision = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=historical,
        live_shadow_episodes=(episode,),
        admission_recipe=admission_recipe(habit),
    )
    return habit, decision


def _scope(habit):
    return ObserverHabitActivationScopeDTO.create(
        fixture_id="fixture:habit",
        observation_schema_id=habit.observation_schema_id,
        grouping_recipe_id=habit.grouping_recipe_id,
        allowed_action_names=(habit.recommended_action,),
        maximum_active_habit_count=1,
        allow_overlapping_source_classes=False,
    )


def _request(habit, snapshot, scope):
    return ObserverHabitActivationRequestDTO.create(
        habit_specification_id=habit.habit_specification_id,
        expected_source_registry_snapshot_id=snapshot.habit_registry_snapshot_id,
        activation_scope_id=scope.habit_activation_scope_id,
        reason_codes=("operator_activation",),
    )


def test_initialize_new_database_and_idempotent_reopen(tmp_path) -> None:
    path = tmp_path / "registry.sqlite"
    store = SqliteObserverHabitRegistryStore(path)
    store.initialize()
    first_id = store.store_id()
    first = store.load_current_snapshot()
    assert first.registry_sequence == 0
    assert store.verify_integrity().status == "verified"
    store.close()

    reopened = SqliteObserverHabitRegistryStore(path)
    reopened.initialize()
    assert reopened.store_id() == first_id
    assert reopened.load_current_snapshot() == first
    assert reopened.verify_integrity().status == "verified"
    reopened.close()


def test_register_admission_activation_and_restart(tmp_path) -> None:
    habit, decision = _admitted()
    registry = SqliteObserverHabitRegistry.open(tmp_path / "registry.sqlite")
    event = registry.register_admission(
        habit_specification=habit, admission_decision=decision
    )
    assert event is not None
    registry.close()

    reopened = SqliteObserverHabitRegistry.open(tmp_path / "registry.sqlite")
    entry = reopened.get_entry(habit.habit_specification_id)
    assert entry is not None
    assert entry.status == "admitted_inactive"
    source = reopened.current_snapshot()
    activation_scope = _scope(habit)
    activation = activate_observer_habit(
        registry=reopened,
        activation_scope=activation_scope,
        activation_request=_request(habit, source, activation_scope),
        habit_specification=habit,
    )
    assert activation.disposition == "activated"
    reopened.close()

    final = SqliteObserverHabitRegistry.open(tmp_path / "registry.sqlite")
    active = final.get_entry(habit.habit_specification_id)
    assert active is not None
    assert active.status == "active"
    assert active.activation_generation == 1
    assert final.verify_integrity().status == "verified"
    final.close()


def test_rejected_admission_not_persisted(tmp_path) -> None:
    habit, decision = _admitted()
    rejected = type(decision).create(
        habit_specification_id=decision.habit_specification_id,
        habit_shadow_audit_id=decision.habit_shadow_audit_id,
        habit_admission_recipe_id=decision.habit_admission_recipe_id,
        decision="reject",
        admitted_registry_status=None,
        evidence_replay_ids=decision.evidence_replay_ids,
        reason_codes=("forced_reject",),
    )
    registry = SqliteObserverHabitRegistry.open(tmp_path / "registry.sqlite")
    assert (
        registry.register_admission(
            habit_specification=habit, admission_decision=rejected
        )
        is None
    )
    assert registry.events() == ()
    assert registry.current_snapshot().registry_sequence == 0
    registry.close()


def test_cross_process_current_read_and_stale_activation(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    a = SqliteObserverHabitRegistry.open(path)
    b = SqliteObserverHabitRegistry.open(path)
    a.register_admission(habit_specification=habit, admission_decision=decision)
    assert b.get_entry(habit.habit_specification_id).status == "admitted_inactive"  # type: ignore[union-attr]

    source_a = a.current_snapshot()
    source_b = b.current_snapshot()
    activation_scope = _scope(habit)
    activate_observer_habit(
        registry=a,
        activation_scope=activation_scope,
        activation_request=_request(habit, source_a, activation_scope),
        habit_specification=habit,
    )
    assert b.get_entry(habit.habit_specification_id).status == "active"  # type: ignore[union-attr]
    stale = activate_observer_habit(
        registry=b,
        activation_scope=activation_scope,
        activation_request=_request(habit, source_b, activation_scope),
        habit_specification=habit,
    )
    assert stale.disposition == "stale_registry_snapshot"
    assert [event.event_type for event in b.events()].count("activated") == 1
    a.close()
    b.close()


def test_store_parity_with_in_memory_registry(tmp_path) -> None:
    habit, decision = _admitted()
    mem = InMemoryObserverHabitRegistry()
    sql = SqliteObserverHabitRegistry.open(tmp_path / "registry.sqlite")
    mem.register_admission(habit_specification=habit, admission_decision=decision)
    sql.register_admission(habit_specification=habit, admission_decision=decision)
    for registry in (mem, sql):
        source = registry.current_snapshot()
        activation_scope = _scope(habit)
        activate_observer_habit(
            registry=registry,
            activation_scope=activation_scope,
            activation_request=_request(habit, source, activation_scope),
            habit_specification=habit,
        )
        registry.deactivate(habit_specification_id=habit.habit_specification_id)
        registry.retire(habit_specification_id=habit.habit_specification_id)
    assert mem.events() == sql.events()
    assert mem.snapshots() == sql.snapshots()
    sql.close()


def test_integrity_detects_projection_tampering(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path)
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    registry.close()
    conn = sqlite3.connect(path)
    conn.execute(
        "UPDATE observer_habit_registry_events SET event_type = 'activated' WHERE registry_sequence = 1"
    )
    conn.commit()
    conn.close()

    store = SqliteObserverHabitRegistryStore(path)
    verification = store.verify_integrity()
    assert verification.status == "failed"
    assert "ObserverHabitRegistrySchemaError" in verification.failure_codes
    store.close()


def test_durable_rollback_creates_new_snapshot(tmp_path) -> None:
    habit, decision = _admitted()
    registry = SqliteObserverHabitRegistry.open(tmp_path / "registry.sqlite")
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    target = registry.current_snapshot()
    activation_scope = _scope(habit)
    activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=_request(habit, target, activation_scope),
        habit_specification=habit,
    )
    registry.deactivate(habit_specification_id=habit.habit_specification_id)
    source = registry.current_snapshot()
    result = registry.rollback_to(
        ObserverHabitRollbackRequestDTO.create(
            current_registry_snapshot_id=source.habit_registry_snapshot_id,
            target_registry_snapshot_id=target.habit_registry_snapshot_id,
            reason_codes=("operator_rollback",),
        )
    )
    assert result.disposition == "rolled_back"
    assert registry.current_snapshot().entries == target.entries
    assert (
        registry.current_snapshot().habit_registry_snapshot_id
        != target.habit_registry_snapshot_id
    )
    assert registry.verify_integrity().status == "verified"
    registry.close()


def test_conflicting_next_sequence_snapshot_does_not_advance_head(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path)
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    source = registry.current_snapshot()
    event = registry._event_for_entries(  # noqa: SLF001
        source=source,
        event_type="retired",
        habit_specification_id=habit.habit_specification_id,
        admission_decision_id=decision.habit_admission_decision_id,
        entries=(
            ObserverHabitRegistryEntryDTO.create(
                habit_specification_id=habit.habit_specification_id,
                habit_admission_decision_id=decision.habit_admission_decision_id,
                status="retired",
                activation_generation=0,
                active_since_registry_sequence=None,
                suspended_since_registry_sequence=None,
                retired_since_registry_sequence=source.registry_sequence + 1,
                status_reason_codes=("retired",),
            ),
        ),
        reason_codes=("retired",),
    )
    result = ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=source.registry_sequence + 1,
        entries=(
            ObserverHabitRegistryEntryDTO.create(
                habit_specification_id=habit.habit_specification_id,
                habit_admission_decision_id=decision.habit_admission_decision_id,
                status="retired",
                activation_generation=0,
                active_since_registry_sequence=None,
                suspended_since_registry_sequence=None,
                retired_since_registry_sequence=source.registry_sequence + 1,
                status_reason_codes=("retired",),
            ),
        ),
        previous_registry_snapshot_id=source.habit_registry_snapshot_id,
    )
    conflict = ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=result.registry_sequence,
        entries=source.entries,
        previous_registry_snapshot_id=source.habit_registry_snapshot_id,
    )
    registry.store._insert_snapshot(conflict)  # noqa: SLF001
    commit = registry.store.append_transition(
        expected_source_snapshot_id=source.habit_registry_snapshot_id,
        event=event,
        result_snapshot=result,
    )
    assert commit.disposition == "database_integrity_failure"
    assert registry.current_snapshot() == source
    assert registry.store.load_snapshot(result.habit_registry_snapshot_id) is None
    registry.close()


def test_conflicting_sequence_zero_snapshot_blocks_initialization(tmp_path) -> None:
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
    entry = ObserverHabitRegistryEntryDTO.create(
        habit_specification_id="habit:conflict",
        habit_admission_decision_id="admission:conflict",
        status="admitted_inactive",
        activation_generation=0,
        active_since_registry_sequence=None,
        suspended_since_registry_sequence=None,
        retired_since_registry_sequence=None,
        status_reason_codes=("admitted",),
    )
    conflict = ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=0,
        entries=(entry,),
        previous_registry_snapshot_id=None,
    )
    store._insert_snapshot(conflict)  # noqa: SLF001
    store.close()

    reopened = SqliteObserverHabitRegistryStore(path)
    with pytest.raises(Exception, match="conflicting empty registry snapshot"):
        reopened.initialize()
    reopened.close()


def test_rollback_to_empty_with_conflicting_result_sequence_does_not_advance_head(
    tmp_path,
) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path)
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    target = empty_observer_habit_registry_snapshot()
    source = registry.current_snapshot()
    event = registry._event_for_entries(  # noqa: SLF001
        source=source,
        event_type="rolled_back",
        habit_specification_id="registry",
        admission_decision_id=None,
        entries=target.entries,
        reason_codes=("operator_rollback",),
        previous_registry_snapshot_id=target.habit_registry_snapshot_id,
    )
    result = ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=source.registry_sequence + 1,
        entries=target.entries,
        previous_registry_snapshot_id=source.habit_registry_snapshot_id,
    )
    conflict = ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=result.registry_sequence,
        entries=source.entries,
        previous_registry_snapshot_id=source.habit_registry_snapshot_id,
    )
    registry.store._insert_snapshot(conflict)  # noqa: SLF001
    commit = registry.store.append_transition(
        expected_source_snapshot_id=source.habit_registry_snapshot_id,
        event=event,
        result_snapshot=result,
    )
    assert commit.disposition == "database_integrity_failure"
    assert registry.current_snapshot() == source
    assert registry.store.load_snapshot(result.habit_registry_snapshot_id) is None
    registry.close()


def test_preexisting_exact_result_snapshot_does_not_advance_head(tmp_path) -> None:
    habit, decision = _admitted()
    path = tmp_path / "registry.sqlite"
    registry = SqliteObserverHabitRegistry.open(path)
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    source = registry.current_snapshot()
    event = registry._event_for_entries(  # noqa: SLF001
        source=source,
        event_type="rolled_back",
        habit_specification_id="registry",
        admission_decision_id=None,
        entries=empty_observer_habit_registry_snapshot().entries,
        reason_codes=("operator_rollback",),
        previous_registry_snapshot_id=empty_observer_habit_registry_snapshot().habit_registry_snapshot_id,
    )
    result = ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=source.registry_sequence + 1,
        entries=(),
        previous_registry_snapshot_id=source.habit_registry_snapshot_id,
    )
    registry.store._insert_snapshot(result)  # noqa: SLF001
    commit = registry.store.append_transition(
        expected_source_snapshot_id=source.habit_registry_snapshot_id,
        event=event,
        result_snapshot=result,
    )
    assert commit.disposition == "database_integrity_failure"
    assert registry.current_snapshot() == source
    registry.close()


def test_preexisting_snapshot_with_altered_entry_projection_blocks_init(
    tmp_path,
) -> None:
    path = tmp_path / "registry.sqlite"
    store = SqliteObserverHabitRegistryStore(path)
    store._create_schema()  # noqa: SLF001
    empty = empty_observer_habit_registry_snapshot()
    store._conn.execute(  # noqa: SLF001
        """
        INSERT INTO observer_habit_registry_snapshots
            (registry_sequence, snapshot_id, previous_snapshot_id, canonical_payload_json)
        VALUES (?, ?, ?, ?)
        """,
        (
            empty.registry_sequence,
            empty.habit_registry_snapshot_id,
            empty.previous_registry_snapshot_id,
            _encode_payload(dict(empty.canonical_payload())),
        ),
    )
    entry = ObserverHabitRegistryEntryDTO.create(
        habit_specification_id="habit:orphan",
        habit_admission_decision_id="admission:orphan",
        status="admitted_inactive",
        activation_generation=0,
        active_since_registry_sequence=None,
        suspended_since_registry_sequence=None,
        retired_since_registry_sequence=None,
        status_reason_codes=("admitted",),
    )
    store._conn.execute(  # noqa: SLF001
        """
        INSERT INTO observer_habit_registry_snapshot_entries
            (snapshot_id, habit_specification_id, registry_entry_id, status,
             activation_generation, canonical_payload_json)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            empty.habit_registry_snapshot_id,
            entry.habit_specification_id,
            entry.habit_registry_entry_id,
            entry.status,
            entry.activation_generation,
            _encode_payload(dict(entry.canonical_payload())),
        ),
    )
    store.close()
    reopened = SqliteObserverHabitRegistryStore(path)
    with pytest.raises(Exception, match="snapshot entry mismatch"):
        reopened.initialize()
    reopened.close()
