from zeromodel.observer import (
    ObserverHabitArbitrationAuditRecipeDTO,
    ObserverHabitArbitrationShadowEpisodeDTO,
    ObserverHabitActivationScopeDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    audit_observer_habit_arbitration_shadow,
    build_observer_fixture_comparison_recipe,
    run_observer_fixture_arbitration_shadow_episode,
)

from test_observer_habit import action, compile_first
from test_observer_habit_arbitration import _plan
from test_observer_habit_overlap import _analysis, _case, _decision, _guard, _variant
from zeromodel.observer.habit_arbitration_shadow import (
    evaluate_observer_habit_arbitration_over_ledger,
)


def _eligible_recipe():
    return ObserverHabitArbitrationAuditRecipeDTO.create(
        maximum_wrong_action_count=99,
        maximum_wrong_target_count=99,
        maximum_ambiguous_fallback_count=99,
        maximum_missed_opportunity_count=99,
        require_zero_different_action_conflicts=False,
        require_zero_target_conflicts=False,
    )


def test_historical_shadow_replay_and_audit_are_identity_stable() -> None:
    schema, group, episode, graph, scope, habit = _case()
    no_fire = _variant(
        habit,
        "no-fire",
        positive_guards=(_guard("visible.agent_x", 99),),
    )
    plan = _plan(
        "strict_unique_fire",
        (habit, no_fire),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    entries = getattr(episode.ledger_snapshot, "entries")

    replay = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        habit_specifications=(habit, no_fire),
        ledger_entries=entries,
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    replay_again = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        habit_specifications=(habit, no_fire),
        ledger_entries=entries,
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    assert replay.habit_arbitration_shadow_replay_id == (
        replay_again.habit_arbitration_shadow_replay_id
    )
    assert replay.applicable_count == len(entries)
    assert replay.habit_selection_count >= 1

    overlap = _analysis(
        (habit, no_fire),
        (_decision(habit), _decision(no_fire)),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    audit = audit_observer_habit_arbitration_shadow(
        arbitration_plan=plan,
        overlap_analysis=overlap,
        historical_shadow_replay=replay,
        fixture_shadow_episodes=(),
        audit_recipe=_eligible_recipe(),
    )
    assert audit.disposition == "eligible_for_multi_habit_activation_review"
    assert audit.eligible_for_multi_habit_activation_review


def test_audit_rejects_duplicate_shadow_evidence() -> None:
    schema, group, episode, graph, scope, habit = _case()
    other = _variant(habit, "other")
    plan = _plan(
        "strict_unique_fire", (habit, other), schema, group, episode, graph, scope
    )
    replay = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        habit_specifications=(habit, other),
        ledger_entries=getattr(episode.ledger_snapshot, "entries"),
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    overlap = _analysis(
        (habit, other),
        (_decision(habit), _decision(other)),
        schema,
        group,
        episode,
        graph,
        scope,
    )

    audit = audit_observer_habit_arbitration_shadow(
        arbitration_plan=plan,
        overlap_analysis=overlap,
        historical_shadow_replay=replay,
        fixture_shadow_episodes=(),
        audit_recipe=ObserverHabitArbitrationAuditRecipeDTO.create(
            minimum_evaluated_replay_count=2
        ),
    )
    assert audit.disposition == "insufficient_evidence"

    duplicate_episode = ObserverHabitArbitrationShadowEpisodeDTO.create(
        habit_arbitration_plan_id=plan.habit_arbitration_plan_id,
        fixture_episode_result_id="episode:duplicate",
        ledger_snapshot_id=episode.ledger_snapshot.ledger_snapshot_id,
        shadow_replay=replay,
        authoritative_action_ids=(),
        status="completed",
        failure_codes=(),
    )
    duplicate = audit_observer_habit_arbitration_shadow(
        arbitration_plan=plan,
        overlap_analysis=overlap,
        historical_shadow_replay=replay,
        fixture_shadow_episodes=(duplicate_episode,),
        audit_recipe=_eligible_recipe(),
    )
    assert duplicate.disposition == "invalid_evidence"
    assert "duplicate_replay" in duplicate.reason_codes


def test_fixture_shadow_episode_preserves_authoritative_actions() -> None:
    schema, rule, group, episode, entries, graph_build, _, result = compile_first(
        action_name="wait"
    )
    object.__setattr__(episode.ledger_snapshot, "entries", entries)
    graph = graph_build.graph
    assert graph is not None
    habit = result.habit_specification
    assert habit is not None
    scope = ObserverHabitActivationScopeDTO.create(
        fixture_id=episode.ledger_snapshot.fixture_id,
        observation_schema_id=schema.schema_id,
        grouping_recipe_id=group.grouping_recipe_id,
        allowed_action_names=("move_left", "move_right", "wait"),
        maximum_active_habit_count=1,
        allow_overlapping_source_classes=False,
    )
    other = _variant(habit, "move", recommended_action="move_left")
    plan = _plan("declared_order", (habit, other), schema, group, episode, graph, scope)
    authoritative_actions = (action("wait"), action("wait"))

    fixture_episode, entries, shadow_episode = (
        run_observer_fixture_arbitration_shadow_episode(
            arbitration_plan=plan,
            habit_specifications=(habit, other),
            initial_state=entries_initial_state(episode),
            authoritative_actions=authoritative_actions,
            predictor_rule_set=rule,
            environment_rule_schedule=(
                ObserverFixtureRuleScheduleEntryDTO.create(
                    start_step=0,
                    rule_set_id=rule.fixture_rule_set_id,
                ),
            ),
            environment_rule_sets=(rule,),
            observation_schema=schema,
            grouping_recipe=group,
            comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        )
    )
    assert fixture_episode.episode_result_id == shadow_episode.fixture_episode_result_id
    assert shadow_episode.authoritative_action_ids == tuple(
        item.action_id for item in entries
    )
    assert shadow_episode.authoritative_action_ids == tuple(
        item.fixture_action_id for item in authoritative_actions
    )
    assert shadow_episode.shadow_replay.applicable_count == len(entries)


def entries_initial_state(episode):
    return getattr(episode.ledger_snapshot, "entries")[0].source_state
