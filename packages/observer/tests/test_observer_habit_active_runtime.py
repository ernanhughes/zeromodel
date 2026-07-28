from zeromodel.observer import (
    InMemoryObserverHabitRegistry,
    ObserverHabitAdmissionDecisionDTO,
    ObserverFixtureActionDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    ObserverHabitActivationRequestDTO,
    ObserverHabitRuntimeSafetyRecipeDTO,
    activate_observer_habit,
    build_observer_fixture_comparison_recipe,
    run_observer_fixture_with_active_habit,
    select_observer_active_action,
)
from zeromodel.observer.fixture_predictor import _observation_for_state
import pytest

from test_observer_habit_registry import scope


def safety(**overrides):
    values = {
        "suspend_on_wrong_target": True,
        "suspend_on_contradiction": True,
        "suspend_on_inconclusive": True,
        "suspend_on_invalid_evaluation": True,
        "maximum_consecutive_habit_failures": 1,
        "maximum_total_habit_failures": 1,
        "maximum_consecutive_fallbacks": None,
    }
    values.update(overrides)
    return ObserverHabitRuntimeSafetyRecipeDTO.create(**values)


def active_setup(habit):
    decision = ObserverHabitAdmissionDecisionDTO.create(
        habit_specification_id=habit.habit_specification_id,
        habit_shadow_audit_id="audit",
        habit_admission_recipe_id="recipe",
        decision="admit",
        reason_codes=("admission_requirements_met",),
        admitted_registry_status="admitted_inactive",
        evidence_replay_ids=("replay",),
    )
    registry = InMemoryObserverHabitRegistry()
    registry.register_admission(habit_specification=habit, admission_decision=decision)
    activation_scope = scope(habit)
    request = ObserverHabitActivationRequestDTO.create(
        habit_specification_id=habit.habit_specification_id,
        expected_source_registry_snapshot_id=registry.current_snapshot().habit_registry_snapshot_id,
        activation_scope_id=activation_scope.habit_activation_scope_id,
        reason_codes=("activate",),
    )
    activation = activate_observer_habit(
        registry=registry,
        activation_scope=activation_scope,
        activation_request=request,
        habit_specification=habit,
    )
    assert activation.disposition == "activated"
    return registry, activation_scope


def test_exact_active_habit_fire_and_fallback_preserved() -> None:
    schema, _, group, _, entries, _, _, result = __import__(
        "test_observer_habit"
    ).compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    registry, activation_scope = active_setup(habit)
    observation = _observation_for_state(
        state=entries[0].source_state,
        action_effect="initial",
        observation_schema=schema,
    )
    decision = select_observer_active_action(
        registry_snapshot=registry.current_snapshot(),
        activation_scope=activation_scope,
        observation=observation,
        authoritative_action=ObserverFixtureActionDTO.create(action_name="move_right"),
        habits=(habit,),
        grouping_recipe=group,
        observation_schema=schema,
    )
    assert decision.decision_source == "habit"
    assert decision.selected_action == habit.recommended_action
    assert decision.authoritative_fallback_action == "move_right"


def test_active_habit_abstention_uses_fallback() -> None:
    schema, _, group, _, entries, _, _, result = __import__(
        "test_observer_habit"
    ).compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    registry, activation_scope = active_setup(habit)
    observation = _observation_for_state(
        state=entries[-1].executed_step.actual_state,
        action_effect="moved_right",
        observation_schema=schema,
    )
    decision = select_observer_active_action(
        registry_snapshot=registry.current_snapshot(),
        activation_scope=activation_scope,
        observation=observation,
        authoritative_action=ObserverFixtureActionDTO.create(action_name="move_right"),
        habits=(habit,),
        grouping_recipe=group,
        observation_schema=schema,
    )
    assert decision.decision_source == "authoritative_fallback"
    assert decision.selected_action == "move_right"


def test_active_runner_executes_habit_action_and_fallback_after_suspension() -> None:
    schema, rule, group, _, entries, _, _, result = __import__(
        "test_observer_habit"
    ).compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    registry, activation_scope = active_setup(habit)
    episode, ledger_entries, report = run_observer_fixture_with_active_habit(
        registry=registry,
        activation_scope=activation_scope,
        habits=(habit,),
        initial_state=entries[0].source_state,
        authoritative_actions=(
            ObserverFixtureActionDTO.create(action_name="move_right"),
            ObserverFixtureActionDTO.create(action_name="move_right"),
        ),
        predictor_rule_set=rule,
        environment_rule_schedule=(
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=0, rule_set_id=rule.fixture_rule_set_id
            ),
        ),
        environment_rule_sets=(rule,),
        observation_schema=schema,
        grouping_recipe=group,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        runtime_safety_recipe=safety(),
    )
    assert episode.ledger_snapshot.entry_count == len(ledger_entries)
    assert ledger_entries[0].transition_verification.transition_record.action == "wait"
    assert report.habit_execution_count >= 1
    assert report.active_occurrences[0].decision_source == "habit"


def test_wrong_target_triggers_suspension_and_no_automatic_reactivation() -> None:
    schema, rule, group, _, entries, _, _, result = __import__(
        "test_observer_habit"
    ).compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    wrong_habit = type(habit).create(
        habit_compilation_recipe_id=habit.habit_compilation_recipe_id,
        promotion_candidate_id=habit.promotion_candidate_id,
        promotion_analysis_id=habit.promotion_analysis_id,
        ledger_snapshot_id=habit.ledger_snapshot_id,
        observation_graph_id=habit.observation_graph_id,
        grouping_recipe_id=habit.grouping_recipe_id,
        observation_schema_id=habit.observation_schema_id,
        transition_key_id=habit.transition_key_id,
        source_state_class_id=habit.source_state_class_id,
        recommended_action=habit.recommended_action,
        expected_target_state_class_id="wrong-target-class",
        positive_guards=habit.positive_guards,
        counterexample_guards=habit.counterexample_guards,
        supporting_occurrence_ids=habit.supporting_occurrence_ids,
        supporting_ledger_entry_ids=habit.supporting_ledger_entry_ids,
        status=habit.status,
    )
    registry, activation_scope = active_setup(wrong_habit)
    _, ledger_entries, report = run_observer_fixture_with_active_habit(
        registry=registry,
        activation_scope=activation_scope,
        habits=(wrong_habit,),
        initial_state=entries[0].source_state,
        authoritative_actions=(
            ObserverFixtureActionDTO.create(action_name="move_right"),
            ObserverFixtureActionDTO.create(action_name="move_right"),
        ),
        predictor_rule_set=rule,
        environment_rule_schedule=(
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=0, rule_set_id=rule.fixture_rule_set_id
            ),
        ),
        environment_rule_sets=(rule,),
        observation_schema=schema,
        grouping_recipe=group,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        runtime_safety_recipe=safety(),
    )
    assert report.status == "completed_with_suspension"
    assert report.suspension_event_ids
    assert registry.get_entry(wrong_habit.habit_specification_id).status == "suspended"  # type: ignore[union-attr]
    if len(ledger_entries) > 1:
        assert (
            ledger_entries[1].transition_verification.transition_record.action
            == "move_right"
        )


def test_runtime_safety_recipe_sensitivity_and_occurrence_identity() -> None:
    strict = safety(suspend_on_inconclusive=True)
    loose = safety(suspend_on_inconclusive=False)
    assert strict.habit_runtime_safety_recipe_id != loose.habit_runtime_safety_recipe_id
    schema, _, group, _, entries, _, _, result = __import__(
        "test_observer_habit"
    ).compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    registry, activation_scope = active_setup(habit)
    observation = _observation_for_state(
        state=entries[0].source_state,
        action_effect="initial",
        observation_schema=schema,
    )
    decision = select_observer_active_action(
        registry_snapshot=registry.current_snapshot(),
        activation_scope=activation_scope,
        observation=observation,
        authoritative_action=ObserverFixtureActionDTO.create(action_name="move_right"),
        habits=(habit,),
        grouping_recipe=group,
        observation_schema=schema,
    )
    changed = type(decision).create(
        registry_snapshot_id=decision.registry_snapshot_id,
        habit_specification_id=decision.habit_specification_id,
        observation_artifact_id=decision.observation_artifact_id,
        habit_evaluation_id=decision.habit_evaluation_id,
        decision_source=decision.decision_source,
        selected_action=decision.selected_action,
        authoritative_fallback_action="wait",
        reason_codes=decision.reason_codes,
    )
    assert decision.active_habit_decision_id != changed.active_habit_decision_id


def test_runtime_safety_failure_thresholds_are_positive() -> None:
    with pytest.raises(Exception, match="maximum_consecutive_habit_failures"):
        safety(maximum_consecutive_habit_failures=0)
    with pytest.raises(Exception, match="maximum_total_habit_failures"):
        safety(maximum_total_habit_failures=0)
