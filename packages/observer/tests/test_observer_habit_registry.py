import pytest

from zeromodel.observer import (
    InMemoryObserverHabitRegistry,
    ObserverHabitActivationRequestDTO,
    ObserverHabitRegistryEntryDTO,
    ObserverHabitRegistrySnapshotDTO,
    ObserverHabitRollbackRequestDTO,
    activate_observer_habit,
    empty_observer_habit_registry_snapshot,
    replay_observer_habit_registry,
)

from test_observer_habit_admission import admitted_evidence, admission_recipe
from zeromodel.observer import admit_observer_habit, ObserverHabitActivationScopeDTO


def tampered_event(event, **changes):
    values = {
        field_name: getattr(event, field_name)
        for field_name in event.__dataclass_fields__
    }
    values.update(changes)
    tampered = object.__new__(type(event))
    for field_name, value in values.items():
        object.__setattr__(tampered, field_name, value)
    return tampered


def admitted_registry():
    habit, historical, episode, audit = admitted_evidence()
    decision = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=historical,
        live_shadow_episodes=(episode,),
        admission_recipe=admission_recipe(habit),
    )
    registry = InMemoryObserverHabitRegistry()
    event = registry.register_admission(
        habit_specification=habit, admission_decision=decision
    )
    assert event is not None
    return habit, decision, registry


def scope(habit, **overrides):
    values = {
        "fixture_id": "fixture:habit",
        "observation_schema_id": habit.observation_schema_id,
        "grouping_recipe_id": habit.grouping_recipe_id,
        "allowed_action_names": (habit.recommended_action,),
        "maximum_active_habit_count": 1,
        "allow_overlapping_source_classes": False,
    }
    values.update(overrides)
    return ObserverHabitActivationScopeDTO.create(**values)


def request(habit, snapshot, activation_scope):
    return ObserverHabitActivationRequestDTO.create(
        habit_specification_id=habit.habit_specification_id,
        expected_source_registry_snapshot_id=snapshot.habit_registry_snapshot_id,
        activation_scope_id=activation_scope.habit_activation_scope_id,
        reason_codes=("operator_activation",),
    )


def test_register_admission_and_rejected_habit_not_registered() -> None:
    habit, _, registry = admitted_registry()
    snapshot = registry.current_snapshot()
    entry = registry.get_entry(habit.habit_specification_id)
    assert entry is not None
    assert entry.status == "admitted_inactive"
    assert snapshot.registry_sequence == 1
    assert registry.events()[0].event_type == "admitted"


def test_atomic_activation_and_stale_request() -> None:
    habit, _, registry = admitted_registry()
    activation_scope = scope(habit)
    source = registry.current_snapshot()
    activation = activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request(habit, source, activation_scope),
        habit_specification=habit,
    )
    assert activation.disposition == "activated"
    entry = registry.get_entry(habit.habit_specification_id)
    assert entry is not None
    assert entry.status == "active"
    assert entry.activation_generation == 1

    stale = activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request(habit, source, activation_scope),
        habit_specification=habit,
    )
    assert stale.disposition == "stale_registry_snapshot"


def test_activation_scope_mismatch_and_retired_rejected() -> None:
    habit, _, registry = admitted_registry()
    bad_scope = scope(habit, allowed_action_names=("other",))
    result = activate_observer_habit(
        registry=registry,
        activation_scope=bad_scope,
        activation_request=request(habit, registry.current_snapshot(), bad_scope),
        habit_specification=habit,
    )
    assert result.disposition == "scope_mismatch"
    registry.retire(habit_specification_id=habit.habit_specification_id)
    retired = activate_observer_habit(
        registry=registry,
        activation_scope=scope(habit),
        activation_request=request(habit, registry.current_snapshot(), scope(habit)),
        habit_specification=habit,
    )
    assert retired.disposition == "retired"


def test_deactivate_suspend_resume_and_snapshot_immutability() -> None:
    habit, _, registry = admitted_registry()
    activation_scope = scope(habit)
    before = registry.current_snapshot()
    activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request(habit, before, activation_scope),
        habit_specification=habit,
    )
    active = registry.current_snapshot()
    registry.deactivate(habit_specification_id=habit.habit_specification_id)
    inactive = registry.current_snapshot()
    assert before.active_habit_ids == ()
    assert active.active_habit_ids == (habit.habit_specification_id,)
    assert inactive.active_habit_ids == ()

    activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request(habit, inactive, activation_scope),
        habit_specification=habit,
    )
    registry.suspend(habit_specification_id=habit.habit_specification_id)
    assert registry.get_entry(habit.habit_specification_id).status == "suspended"  # type: ignore[union-attr]
    registry.resume(habit_specification_id=habit.habit_specification_id)
    assert (
        registry.get_entry(habit.habit_specification_id).status == "admitted_inactive"
    )  # type: ignore[union-attr]


def test_rollback_to_ancestor_and_replay() -> None:
    habit, _, registry = admitted_registry()
    s1 = registry.current_snapshot()
    activation_scope = scope(habit)
    activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request(habit, s1, activation_scope),
        habit_specification=habit,
    )
    registry.deactivate(habit_specification_id=habit.habit_specification_id)
    s3 = registry.current_snapshot()
    rollback = registry.rollback_to(
        ObserverHabitRollbackRequestDTO.create(
            current_registry_snapshot_id=s3.habit_registry_snapshot_id,
            target_registry_snapshot_id=s1.habit_registry_snapshot_id,
            reason_codes=("operator_rollback",),
        )
    )
    assert rollback.disposition == "rolled_back"
    assert registry.current_snapshot().entries == s1.entries
    assert len(registry.events()) == 4
    assert registry.verify_integrity().status == "verified"


def test_rollback_target_not_ancestor_and_stale_request() -> None:
    habit, _, registry = admitted_registry()
    result = registry.rollback_to(
        ObserverHabitRollbackRequestDTO.create(
            current_registry_snapshot_id=registry.current_snapshot().habit_registry_snapshot_id,
            target_registry_snapshot_id="missing-snapshot",
            reason_codes=("operator_rollback",),
        )
    )
    assert result.disposition == "target_not_found"
    stale = registry.rollback_to(
        ObserverHabitRollbackRequestDTO.create(
            current_registry_snapshot_id="stale",
            target_registry_snapshot_id=registry.current_snapshot().habit_registry_snapshot_id,
            reason_codes=("operator_rollback",),
        )
    )
    assert stale.disposition == "stale_registry_snapshot"


def test_registry_replay_tampering_detects_source_mismatch() -> None:
    habit, _, registry = admitted_registry()
    event = registry.events()[0]
    tampered = type(event).create(
        registry_sequence=event.registry_sequence,
        event_type=event.event_type,
        habit_specification_id=event.habit_specification_id,
        habit_registry_entry_id=event.habit_registry_entry_id,
        admission_decision_id=event.admission_decision_id,
        previous_registry_snapshot_id=event.previous_registry_snapshot_id,
        source_registry_snapshot_id="wrong",
        reason_codes=event.reason_codes,
    )
    replay = replay_observer_habit_registry(
        initial_snapshot=empty_observer_habit_registry_snapshot(),
        events=(tampered,),
        final_snapshot=registry.current_snapshot(),
        snapshots=registry.snapshots(),
    )
    assert replay.status == "failed"
    assert "event_source_snapshot_mismatch" in replay.failure_codes


def test_registry_replay_rejects_semantically_invalid_supplied_snapshot() -> None:
    habit, _, registry = admitted_registry()
    source = empty_observer_habit_registry_snapshot()
    event = registry.events()[0]
    bad_entry = ObserverHabitRegistryEntryDTO.create(
        habit_specification_id=habit.habit_specification_id,
        habit_admission_decision_id=event.admission_decision_id,
        status="admitted_inactive",
        activation_generation=1,
        active_since_registry_sequence=None,
        suspended_since_registry_sequence=None,
        retired_since_registry_sequence=None,
        status_reason_codes=event.reason_codes,
    )
    bad_snapshot = ObserverHabitRegistrySnapshotDTO.create(
        registry_sequence=event.registry_sequence,
        entries=(bad_entry,),
        previous_registry_snapshot_id=source.habit_registry_snapshot_id,
    )
    replay = replay_observer_habit_registry(
        initial_snapshot=source,
        events=(event,),
        final_snapshot=bad_snapshot,
        snapshots=(source, bad_snapshot),
    )
    assert replay.status == "failed"
    assert "event_application_mismatch" in replay.failure_codes
    assert "activation_generation_mismatch" in replay.failure_codes


def test_registry_entry_status_specific_invariants() -> None:
    habit, decision, _ = admitted_registry()

    with pytest.raises(Exception, match="inactive entry"):
        ObserverHabitRegistryEntryDTO.create(
            habit_specification_id=habit.habit_specification_id,
            habit_admission_decision_id=decision.habit_admission_decision_id,
            status="admitted_inactive",
            activation_generation=0,
            active_since_registry_sequence=1,
            suspended_since_registry_sequence=None,
            retired_since_registry_sequence=None,
            status_reason_codes=("bad",),
        )
    with pytest.raises(Exception, match="active entry"):
        ObserverHabitRegistryEntryDTO.create(
            habit_specification_id=habit.habit_specification_id,
            habit_admission_decision_id=decision.habit_admission_decision_id,
            status="active",
            activation_generation=0,
            active_since_registry_sequence=1,
            suspended_since_registry_sequence=None,
            retired_since_registry_sequence=None,
            status_reason_codes=("bad",),
        )


def test_registry_replay_rejects_tampered_event_entry_id() -> None:
    _, _, registry = admitted_registry()
    event = registry.events()[0]
    tampered = tampered_event(event, habit_registry_entry_id="sha256:wrong")
    replay = replay_observer_habit_registry(
        initial_snapshot=empty_observer_habit_registry_snapshot(),
        events=(tampered,),
        final_snapshot=registry.current_snapshot(),
        snapshots=registry.snapshots(),
    )
    assert replay.status == "failed"
    assert "event_entry_id_mismatch" in replay.failure_codes


def test_registry_replay_rejects_tampered_admission_decision_id() -> None:
    habit, _, registry = admitted_registry()
    activation_scope = scope(habit)
    activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request(
            habit, registry.current_snapshot(), activation_scope
        ),
        habit_specification=habit,
    )
    tampered = tampered_event(
        registry.events()[1], admission_decision_id="other-admission"
    )
    replay = replay_observer_habit_registry(
        initial_snapshot=empty_observer_habit_registry_snapshot(),
        events=(registry.events()[0], tampered),
        final_snapshot=registry.current_snapshot(),
        snapshots=registry.snapshots(),
    )
    assert replay.status == "failed"
    assert "admission_decision_mismatch" in replay.failure_codes


def test_registry_replay_rejects_non_rollback_target_on_normal_event() -> None:
    _, _, registry = admitted_registry()
    event = tampered_event(
        registry.events()[0],
        previous_registry_snapshot_id=empty_observer_habit_registry_snapshot().habit_registry_snapshot_id,
    )
    replay = replay_observer_habit_registry(
        initial_snapshot=empty_observer_habit_registry_snapshot(),
        events=(event,),
        final_snapshot=registry.current_snapshot(),
        snapshots=registry.snapshots(),
    )
    assert replay.status == "failed"
    assert "unexpected_rollback_target" in replay.failure_codes


def test_registry_replay_rejects_rollback_with_entry_id() -> None:
    habit, _, registry = admitted_registry()
    s1 = registry.current_snapshot()
    activation_scope = scope(habit)
    activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request(habit, s1, activation_scope),
        habit_specification=habit,
    )
    rollback = registry.rollback_to(
        ObserverHabitRollbackRequestDTO.create(
            current_registry_snapshot_id=registry.current_snapshot().habit_registry_snapshot_id,
            target_registry_snapshot_id=s1.habit_registry_snapshot_id,
            reason_codes=("operator_rollback",),
        )
    )
    assert rollback.registry_event_id is not None
    events = registry.events()
    bad_rollback = tampered_event(
        events[-1],
        habit_registry_entry_id=registry.current_snapshot()
        .entries[0]
        .habit_registry_entry_id,
    )
    replay = replay_observer_habit_registry(
        initial_snapshot=empty_observer_habit_registry_snapshot(),
        events=(*events[:-1], bad_rollback),
        final_snapshot=registry.current_snapshot(),
        snapshots=registry.snapshots(),
    )
    assert replay.status == "failed"
    assert "event_entry_id_mismatch" in replay.failure_codes


def test_registry_replay_rejects_rollback_with_habit_specification_id() -> None:
    habit, _, registry = admitted_registry()
    s1 = registry.current_snapshot()
    activation_scope = scope(habit)
    activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request(habit, s1, activation_scope),
        habit_specification=habit,
    )
    registry.rollback_to(
        ObserverHabitRollbackRequestDTO.create(
            current_registry_snapshot_id=registry.current_snapshot().habit_registry_snapshot_id,
            target_registry_snapshot_id=s1.habit_registry_snapshot_id,
            reason_codes=("operator_rollback",),
        )
    )
    events = registry.events()
    bad_rollback = tampered_event(
        events[-1], habit_specification_id=habit.habit_specification_id
    )
    replay = replay_observer_habit_registry(
        initial_snapshot=empty_observer_habit_registry_snapshot(),
        events=(*events[:-1], bad_rollback),
        final_snapshot=registry.current_snapshot(),
        snapshots=registry.snapshots(),
    )
    assert replay.status == "failed"
    assert "event_application_mismatch" in replay.failure_codes
