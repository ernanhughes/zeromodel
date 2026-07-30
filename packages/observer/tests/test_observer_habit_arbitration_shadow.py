from zeromodel.observer import (
    ObserverHabitAdmissionDecisionDTO,
    ObserverHabitArbitrationAuditRecipeDTO,
    ObserverHabitArbitrationShadowEpisodeDTO,
    ObserverHabitArbitrationShadowOccurrenceDTO,
    ObserverHabitArbitrationShadowReplayDTO,
    ObserverHabitActivationScopeDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    audit_observer_habit_arbitration_shadow,
    build_observer_fixture_comparison_recipe,
    run_observer_fixture_arbitration_shadow_episode,
)
from zeromodel.observer._canonical import canonical_id

from test_observer_habit import action, compile_first
from test_observer_habit_arbitration import _plan, _plan_analysis
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


def _alternate_decision(habit):
    return ObserverHabitAdmissionDecisionDTO.create(
        habit_specification_id=habit.habit_specification_id,
        habit_shadow_audit_id=f"audit:alternate:{habit.habit_specification_id}",
        habit_admission_recipe_id=f"recipe:alternate:{habit.habit_specification_id}",
        decision="admit",
        reason_codes=("admit", "alternate"),
        admitted_registry_status="admitted_inactive",
        evidence_replay_ids=(f"replay:alternate:{habit.habit_specification_id}",),
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
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, no_fire),
        ledger_entries=entries,
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    replay_again = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
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
        overlap_analysis=_plan_analysis(plan),
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


def test_shadow_replay_rejects_entries_that_do_not_match_snapshot() -> None:
    schema, group, episode, graph, scope, habit = _case()
    other = _variant(habit, "other")
    plan = _plan(
        "strict_unique_fire", (habit, other), schema, group, episode, graph, scope
    )
    entries = getattr(episode.ledger_snapshot, "entries")

    omitted = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, other),
        ledger_entries=entries[:1],
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    assert omitted.status == "failed"
    assert "ledger_snapshot_entry_mismatch" in omitted.failure_codes

    duplicate = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, other),
        ledger_entries=(entries[0], entries[0]),
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    assert duplicate.status == "failed"
    assert "duplicate_ledger_entry" in duplicate.failure_codes

    reordered = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, other),
        ledger_entries=tuple(reversed(entries)),
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    assert reordered.status == "failed"
    assert "ledger_snapshot_entry_mismatch" in reordered.failure_codes


def test_shadow_replay_records_reconstruction_failure_without_skipping(
    monkeypatch,
) -> None:
    schema, group, episode, graph, scope, habit = _case()
    other = _variant(habit, "other")
    plan = _plan(
        "strict_unique_fire", (habit, other), schema, group, episode, graph, scope
    )

    import zeromodel.observer.habit_arbitration_shadow as shadow

    def fail_once(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(shadow, "_observation_for_state", fail_once)
    replay = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, other),
        ledger_entries=getattr(episode.ledger_snapshot, "entries"),
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    assert replay.status == "completed_with_failures"
    assert "entry_reconstruction_failed" in replay.failure_codes
    assert replay.applicable_count == episode.ledger_snapshot.entry_count
    assert replay.invalid_count == episode.ledger_snapshot.entry_count
    assert all(
        item.outcome == "invalid_evaluation" for item in replay.shadow_occurrences
    )


def test_audit_rejects_falsified_replay_aggregates_and_episode_ledger_mismatch() -> (
    None
):
    schema, group, episode, graph, scope, habit = _case()
    no_fire = _variant(
        habit,
        "no-fire",
        positive_guards=(_guard("visible.agent_x", 99),),
    )
    plan = _plan(
        "strict_unique_fire", (habit, no_fire), schema, group, episode, graph, scope
    )
    replay = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, no_fire),
        ledger_entries=getattr(episode.ledger_snapshot, "entries"),
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    payload = dict(replay.canonical_payload(include_id=False))
    payload["applicable_count"] = 0
    falsified = ObserverHabitArbitrationShadowReplayDTO(
        habit_arbitration_shadow_replay_id=canonical_id(payload),
        habit_arbitration_plan_id=replay.habit_arbitration_plan_id,
        ledger_snapshot_id=replay.ledger_snapshot_id,
        shadow_occurrences=replay.shadow_occurrences,
        evaluated_entry_ids=replay.evaluated_entry_ids,
        applicable_count=0,
        habit_selection_count=replay.habit_selection_count,
        fallback_count=replay.fallback_count,
        correct_selection_count=replay.correct_selection_count,
        wrong_action_count=0,
        wrong_target_count=replay.wrong_target_count,
        ambiguous_fallback_count=replay.ambiguous_fallback_count,
        missed_opportunity_count=replay.missed_opportunity_count,
        invalid_count=replay.invalid_count,
        status=replay.status,
        failure_codes=replay.failure_codes,
    )
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
        historical_shadow_replay=falsified,
        fixture_shadow_episodes=(),
        audit_recipe=_eligible_recipe(),
    )
    assert audit.disposition == "invalid_evidence"
    assert "replay_aggregate_mismatch" in audit.reason_codes

    shadow_episode = ObserverHabitArbitrationShadowEpisodeDTO.create(
        habit_arbitration_plan_id=plan.habit_arbitration_plan_id,
        fixture_episode_result_id="episode:ledger-mismatch",
        ledger_snapshot_id="ledger:foreign",
        shadow_replay=replay,
        authoritative_action_ids=tuple(
            entry.action_id for entry in getattr(episode.ledger_snapshot, "entries")
        ),
        status="completed",
        failure_codes=(),
    )
    episode_audit = audit_observer_habit_arbitration_shadow(
        arbitration_plan=plan,
        overlap_analysis=overlap,
        historical_shadow_replay=replay,
        fixture_shadow_episodes=(shadow_episode,),
        audit_recipe=_eligible_recipe(),
    )
    assert episode_audit.disposition == "invalid_evidence"
    assert "episode_replay_ledger_mismatch" in episode_audit.reason_codes


def test_audit_rejects_overlap_analysis_substitution_for_same_habits() -> None:
    schema, group, episode, graph, scope, habit = _case()
    other = _variant(habit, "other")
    plan = _plan(
        "strict_unique_fire", (habit, other), schema, group, episode, graph, scope
    )
    replay = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, other),
        ledger_entries=getattr(episode.ledger_snapshot, "entries"),
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    substituted = _analysis(
        (habit, other),
        (_alternate_decision(habit), _alternate_decision(other)),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    assert (
        substituted.habit_specification_ids
        == _plan_analysis(plan).habit_specification_ids
    )
    assert substituted.habit_overlap_analysis_id != plan.habit_overlap_analysis_id
    audit = audit_observer_habit_arbitration_shadow(
        arbitration_plan=plan,
        overlap_analysis=substituted,
        historical_shadow_replay=replay,
        fixture_shadow_episodes=(),
        audit_recipe=_eligible_recipe(),
    )
    assert audit.disposition == "invalid_evidence"
    assert "plan_overlap_analysis_mismatch" in audit.reason_codes


def test_audit_rejects_replay_occurrence_plan_mismatch() -> None:
    schema, group, episode, graph, scope, habit = _case()
    other = _variant(habit, "other")
    plan = _plan(
        "strict_unique_fire", (habit, other), schema, group, episode, graph, scope
    )
    replay = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, other),
        ledger_entries=getattr(episode.ledger_snapshot, "entries"),
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    occurrence = replay.shadow_occurrences[0]
    foreign = ObserverHabitArbitrationShadowOccurrenceDTO.create(
        habit_arbitration_plan_id="plan:foreign",
        ledger_entry_id=occurrence.ledger_entry_id,
        observation_artifact_id=occurrence.observation_artifact_id,
        arbitration_evaluation_id=occurrence.arbitration_evaluation_id,
        selected_habit_id=occurrence.selected_habit_id,
        selected_action=occurrence.selected_action,
        authoritative_action=occurrence.authoritative_action,
        actual_target_state_class_id=occurrence.actual_target_state_class_id,
        expected_target_state_class_id=occurrence.expected_target_state_class_id,
        outcome=occurrence.outcome,
        reason_codes=occurrence.reason_codes,
    )
    malformed = ObserverHabitArbitrationShadowReplayDTO.create(
        habit_arbitration_plan_id=plan.habit_arbitration_plan_id,
        ledger_snapshot_id=replay.ledger_snapshot_id,
        shadow_occurrences=(foreign,),
        evaluated_entry_ids=(foreign.ledger_entry_id,),
        status="completed",
        failure_codes=(),
    )
    audit = audit_observer_habit_arbitration_shadow(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        historical_shadow_replay=malformed,
        fixture_shadow_episodes=(),
        audit_recipe=_eligible_recipe(),
    )
    assert audit.disposition == "invalid_evidence"
    assert "occurrence_plan_mismatch" in audit.reason_codes


def test_audit_rejects_replay_occurrence_entry_correspondence_tampering() -> None:
    schema, group, episode, graph, scope, habit = _case()
    other = _variant(habit, "other")
    plan = _plan(
        "strict_unique_fire", (habit, other), schema, group, episode, graph, scope
    )
    replay = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=plan,
        overlap_analysis=_plan_analysis(plan),
        habit_specifications=(habit, other),
        ledger_entries=getattr(episode.ledger_snapshot, "entries"),
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=group,
        observation_schema=schema,
    )
    occurrences = replay.shadow_occurrences
    omitted = ObserverHabitArbitrationShadowReplayDTO.create(
        habit_arbitration_plan_id=plan.habit_arbitration_plan_id,
        ledger_snapshot_id=replay.ledger_snapshot_id,
        shadow_occurrences=occurrences[:1],
        evaluated_entry_ids=(),
        status="completed",
        failure_codes=(),
    )
    extra = ObserverHabitArbitrationShadowReplayDTO.create(
        habit_arbitration_plan_id=plan.habit_arbitration_plan_id,
        ledger_snapshot_id=replay.ledger_snapshot_id,
        shadow_occurrences=occurrences[:1],
        evaluated_entry_ids=(occurrences[0].ledger_entry_id, "entry:extra"),
        status="completed",
        failure_codes=(),
    )
    duplicate = ObserverHabitArbitrationShadowReplayDTO.create(
        habit_arbitration_plan_id=plan.habit_arbitration_plan_id,
        ledger_snapshot_id=replay.ledger_snapshot_id,
        shadow_occurrences=(occurrences[0], occurrences[0]),
        evaluated_entry_ids=(
            occurrences[0].ledger_entry_id,
            occurrences[0].ledger_entry_id,
        ),
        status="completed",
        failure_codes=(),
    )
    reordered = ObserverHabitArbitrationShadowReplayDTO.create(
        habit_arbitration_plan_id=plan.habit_arbitration_plan_id,
        ledger_snapshot_id=replay.ledger_snapshot_id,
        shadow_occurrences=tuple(reversed(occurrences)),
        evaluated_entry_ids=tuple(item.ledger_entry_id for item in occurrences),
        status="completed",
        failure_codes=(),
    )
    cases = (
        (omitted, {"replay_entry_occurrence_mismatch", "occurrence_not_evaluated"}),
        (extra, {"replay_entry_occurrence_mismatch"}),
        (duplicate, {"duplicate_occurrence_entry"}),
        (reordered, {"replay_entry_occurrence_mismatch"}),
    )
    for malformed, expected in cases:
        audit = audit_observer_habit_arbitration_shadow(
            arbitration_plan=plan,
            overlap_analysis=_plan_analysis(plan),
            historical_shadow_replay=malformed,
            fixture_shadow_episodes=(),
            audit_recipe=_eligible_recipe(),
        )
        assert audit.disposition == "invalid_evidence"
        assert expected <= set(audit.reason_codes)


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
            overlap_analysis=_plan_analysis(plan),
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
