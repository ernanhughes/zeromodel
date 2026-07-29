from zeromodel.observer import (
    ObserverHabitArbitrationPlanRecipeDTO,
    compile_observer_habit_arbitration_plan,
    evaluate_observer_habit_arbitration,
)
from zeromodel.observer.habit_overlap import _observation_for_state

from test_observer_habit_overlap import _analysis, _case, _decision, _guard, _variant


def _observation(episode, schema):
    entry = getattr(episode.ledger_snapshot, "entries")[0]
    return _observation_for_state(
        state=entry.source_state,
        action_effect="initial",
        observation_schema=schema,
    )


def _recipe(**values):
    defaults = {
        "allow_different_action_overlap": True,
        "allow_target_conflict": True,
        "allow_inconclusive_pairs": True,
        "require_complete_pair_analysis": False,
    }
    defaults.update(values)
    return ObserverHabitArbitrationPlanRecipeDTO.create(**defaults)


def _plan(strategy, habits, schema, group, episode, graph, scope, **recipe):
    analysis = _analysis(
        habits,
        tuple(_decision(habit) for habit in habits),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    compilation = compile_observer_habit_arbitration_plan(
        overlap_analysis=analysis,
        habit_specifications=habits,
        plan_recipe=_recipe(**recipe),
        requested_strategy=strategy,
        declared_order=tuple(habit.habit_specification_id for habit in habits)
        if strategy == "declared_order"
        else (),
    )
    assert compilation.arbitration_plan is not None
    return compilation.arbitration_plan


def test_plan_compilation_blocks_conservative_action_and_target_conflicts() -> None:
    schema, group, episode, graph, scope, habit = _case()
    action_conflict = _variant(habit, "action", recommended_action="move_left")
    target_conflict = _variant(
        habit,
        "target",
        expected_target_state_class_id="state:other",
    )

    action_analysis = _analysis(
        (habit, action_conflict),
        (_decision(habit), _decision(action_conflict)),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    action_compile = compile_observer_habit_arbitration_plan(
        overlap_analysis=action_analysis,
        habit_specifications=(habit, action_conflict),
        plan_recipe=ObserverHabitArbitrationPlanRecipeDTO.create(),
        requested_strategy="strict_unique_fire",
    )
    assert action_compile.disposition == "blocked_by_action_conflict"
    assert action_compile.arbitration_plan is None

    target_analysis = _analysis(
        (habit, target_conflict),
        (_decision(habit), _decision(target_conflict)),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    target_compile = compile_observer_habit_arbitration_plan(
        overlap_analysis=target_analysis,
        habit_specifications=(habit, target_conflict),
        plan_recipe=ObserverHabitArbitrationPlanRecipeDTO.create(),
        requested_strategy="strict_unique_fire",
    )
    assert target_compile.disposition == "blocked_by_target_conflict"
    assert target_compile.arbitration_plan is None


def test_strict_unique_fire_selects_one_and_falls_back_on_ambiguity_or_no_fire() -> (
    None
):
    schema, group, episode, graph, scope, habit = _case()
    no_fire = _variant(
        habit,
        "no-fire",
        positive_guards=(_guard("visible.agent_x", 99),),
    )
    duplicate = _variant(habit, "duplicate")
    observation = _observation(episode, schema)

    unique_plan = _plan(
        "strict_unique_fire",
        (habit, no_fire),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    unique = evaluate_observer_habit_arbitration(
        arbitration_plan=unique_plan,
        habit_specifications=(habit, no_fire),
        observation=observation,
        grouping_recipe=group,
        observation_schema=schema,
        authoritative_fallback_action="authoritative",
    )
    assert unique.decision == "selected_habit"
    assert unique.selected_habit_id == habit.habit_specification_id
    assert unique.selected_action == habit.recommended_action

    no_fire_plan = _plan(
        "strict_unique_fire",
        (no_fire, _variant(no_fire, "second")),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    no_fire_eval = evaluate_observer_habit_arbitration(
        arbitration_plan=no_fire_plan,
        habit_specifications=(no_fire, _variant(no_fire, "second")),
        observation=observation,
        grouping_recipe=group,
        observation_schema=schema,
        authoritative_fallback_action="authoritative",
    )
    assert no_fire_eval.decision == "fallback_no_fire"
    assert no_fire_eval.selected_action == "authoritative"

    ambiguous_plan = _plan(
        "strict_unique_fire",
        (habit, duplicate),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    ambiguous = evaluate_observer_habit_arbitration(
        arbitration_plan=ambiguous_plan,
        habit_specifications=(habit, duplicate),
        observation=observation,
        grouping_recipe=group,
        observation_schema=schema,
        authoritative_fallback_action="authoritative",
    )
    assert ambiguous.decision == "fallback_ambiguous"
    assert ambiguous.selected_action == "authoritative"


def test_most_specific_guard_and_declared_order_are_deterministic() -> None:
    schema, group, episode, graph, scope, habit = _case()
    broad = _variant(
        habit,
        "broad",
        positive_guards=(_guard("visible.agent_x", 0),),
        recommended_action="move_left",
    )
    narrow = _variant(
        habit,
        "narrow",
        positive_guards=(
            _guard("visible.agent_x", 0),
            _guard("visible.target_x", 6),
        ),
        recommended_action="move_right",
    )
    observation = _observation(episode, schema)

    specific_plan = _plan(
        "most_specific_guard",
        (broad, narrow),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    specific = evaluate_observer_habit_arbitration(
        arbitration_plan=specific_plan,
        habit_specifications=(broad, narrow),
        observation=observation,
        grouping_recipe=group,
        observation_schema=schema,
        authoritative_fallback_action="authoritative",
    )
    assert specific.decision == "selected_habit"
    assert specific.selected_habit_id == narrow.habit_specification_id
    assert specific.selected_action == "move_right"

    declared_plan = _plan(
        "declared_order",
        (broad, narrow),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    declared = evaluate_observer_habit_arbitration(
        arbitration_plan=declared_plan,
        habit_specifications=(broad, narrow),
        observation=observation,
        grouping_recipe=group,
        observation_schema=schema,
        authoritative_fallback_action="authoritative",
    )
    assert declared.decision == "selected_habit"
    assert declared.selected_habit_id == broad.habit_specification_id
    assert declared.selected_action == "move_left"


def test_invalid_lineage_and_identity_sensitivity() -> None:
    schema, group, episode, graph, scope, habit = _case()
    other = _variant(habit, "other")
    plan = _plan(
        "strict_unique_fire", (habit, other), schema, group, episode, graph, scope
    )
    observation = _observation(episode, schema)

    valid = evaluate_observer_habit_arbitration(
        arbitration_plan=plan,
        habit_specifications=(habit, other),
        observation=observation,
        grouping_recipe=group,
        observation_schema=schema,
        authoritative_fallback_action="authoritative",
    )
    changed = evaluate_observer_habit_arbitration(
        arbitration_plan=plan,
        habit_specifications=(habit, other),
        observation=observation,
        grouping_recipe=group,
        observation_schema=schema,
        authoritative_fallback_action="different",
    )
    assert (
        valid.habit_arbitration_evaluation_id != changed.habit_arbitration_evaluation_id
    )

    bad_compile = compile_observer_habit_arbitration_plan(
        overlap_analysis=_analysis(
            (habit, other),
            (_decision(habit), _decision(other)),
            schema,
            group,
            episode,
            graph,
            scope,
        ),
        habit_specifications=(habit,),
        plan_recipe=_recipe(),
        requested_strategy="strict_unique_fire",
    )
    assert bad_compile.disposition == "invalid_lineage"
    assert bad_compile.arbitration_plan is None
