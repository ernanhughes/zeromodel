import pytest

from zeromodel.observer import (
    ObserverCategoryMappingDTO,
    ObserverFixtureActionDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
    ObserverGroupingFeatureDTO,
    ObserverHabitCompilationRecipeDTO,
    ObserverHabitError,
    ObserverHabitShadowEpisodeDTO,
    ObserverHabitShadowOccurrenceDTO,
    ObserverHabitGuardDTO,
    ObserverHabitShadowAuditRecipeDTO,
    ObserverHabitShadowReplayDTO,
    ObserverStateGroupingRecipeDTO,
    analyze_observer_promotion_candidates,
    audit_observer_habit_shadow,
    build_observer_fixture_comparison_recipe,
    build_observer_fixture_observation_schema,
    build_observer_observation_graph,
    compile_observer_habit_specification,
    evaluate_observer_habit,
    evaluate_observer_habit_over_ledger,
    run_observer_fixture_episode,
    run_observer_fixture_habit_shadow_episode,
)


def action(name: str) -> ObserverFixtureActionDTO:
    return ObserverFixtureActionDTO.create(action_name=name)


def rules():
    schema = build_observer_fixture_observation_schema()
    rule = ObserverFixtureRuleSetDTO.create(
        fixture_id="fixture:habit",
        rule_version="fixture-rule/1",
        minimum_position=-8,
        maximum_position=12,
        cooldown_period=1,
        cooldown_effect="block",
        observation_schema_id=schema.schema_id,
    )
    return schema, rule


def grouping(schema_id: str, *, bucket: bool = False, ignored_hidden: bool = True):
    return ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema_id,
        type_mismatch_policy="separate_class",
        feature_groupings=(
            ObserverGroupingFeatureDTO.create(
                feature_key="hidden.cooldown_remaining",
                mode="ignored" if ignored_hidden else "exact",
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="history.previous_action", mode="ignored"
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="visible.action_effect",
                mode="categorical",
                category_mapping=(
                    ObserverCategoryMappingDTO.create(
                        raw_value="blocked_by_cooldown", mapped_value="blocked"
                    ),
                    ObserverCategoryMappingDTO.create(
                        raw_value="initial", mapped_value="other"
                    ),
                    ObserverCategoryMappingDTO.create(
                        raw_value="moved_right", mapped_value="moved"
                    ),
                    ObserverCategoryMappingDTO.create(
                        raw_value="waited", mapped_value="other"
                    ),
                ),
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="visible.agent_x",
                mode="numeric_bucket" if bucket else "exact",
                bucket_size=2.0 if bucket else None,
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="visible.target_x", mode="exact"
            ),
        ),
    )


def build_case(
    actions,
    *,
    initial_x=0,
    bucket=False,
    hidden=True,
    ignored_hidden=True,
    episode_id="episode:habit",
):
    schema, rule = rules()
    episode, entries = run_observer_fixture_episode(
        initial_state=ObserverFixtureStateDTO.create(
            fixture_id=rule.fixture_id,
            rule_set_id=rule.fixture_rule_set_id,
            episode_id=episode_id,
            step_index=0,
            agent_x=initial_x,
            target_x=6,
        ),
        actions=tuple(action(item) for item in actions),
        predictor_rule_set=rule,
        environment_rule_schedule=(
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=0, rule_set_id=rule.fixture_rule_set_id
            ),
        ),
        environment_rule_sets=(rule,),
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        supply_hidden_evidence=hidden,
    )
    group = grouping(schema.schema_id, bucket=bucket, ignored_hidden=ignored_hidden)
    graph_build = build_observer_observation_graph(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        grouping_recipe=group,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(rule,),
        environment_rule_sets=(rule,),
    )
    assert graph_build.status == "built"
    promotion_recipe = __import__(
        "zeromodel.observer", fromlist=["ObserverPromotionEvidenceRecipeDTO"]
    ).ObserverPromotionEvidenceRecipeDTO.create(
        observation_graph_id=graph_build.graph.observation_graph_id,
        grouping_recipe_id=group.grouping_recipe_id,
        minimum_traversal_count=1,
        minimum_confirmed_count=1,
        minimum_independent_episode_count=1,
        minimum_distinct_source_state_count=1,
        minimum_distinct_rule_regime_count=1,
        maximum_contradicted_count=0,
        maximum_inconclusive_count=0,
        minimum_confirmation_ratio_numerator=1,
        minimum_confirmation_ratio_denominator=1,
    )
    analysis = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=graph_build,
        grouping_recipe=group,
        promotion_recipe=promotion_recipe,
        observation_schema=schema,
    )
    assert analysis.status == "built"
    return (
        schema,
        rule,
        group,
        episode,
        entries,
        graph_build,
        promotion_recipe,
        analysis,
    )


def recipe(promotion_recipe, group, schema, **overrides):
    values = {
        "promotion_recipe_id": promotion_recipe.promotion_recipe_id,
        "grouping_recipe_id": group.grouping_recipe_id,
        "observation_schema_id": schema.schema_id,
        "allowed_guard_feature_keys": (
            "hidden.cooldown_remaining",
            "visible.action_effect",
            "visible.agent_x",
            "visible.target_x",
        ),
        "required_guard_feature_keys": (),
        "forbidden_guard_feature_keys": (),
        "maximum_guard_count": 8,
        "maximum_counterexample_guard_count": 4,
        "allow_exact_guards": True,
        "allow_categorical_guards": True,
        "allow_numeric_range_guards": True,
        "require_counterexample_guards": False,
    }
    values.update(overrides)
    return ObserverHabitCompilationRecipeDTO.create(**values)


def compile_first(
    actions=("wait", "wait"), *, action_name: str | None = None, **kwargs
):
    schema, rule, group, episode, entries, graph_build, promotion_recipe, analysis = (
        build_case(actions, **kwargs)
    )
    candidate = next(
        item
        for item in analysis.promotion_candidates
        if item.disposition == "eligible"
        and (
            action_name is None
            or next(
                edge
                for edge in graph_build.graph.edges
                if edge.transition_key.transition_key_id == item.transition_key_id
            ).transition_key.action
            == action_name
        )
    )
    result = compile_observer_habit_specification(
        promotion_analysis=analysis,
        promotion_candidate=candidate,
        graph_build=graph_build,
        grouping_recipe=group,
        observation_schema=schema,
        compilation_recipe=recipe(promotion_recipe, group, schema),
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
    )
    assert result.disposition == "compiled_for_shadow"
    assert result.habit_specification is not None
    return schema, rule, group, episode, entries, graph_build, analysis, result


def test_compile_eligible_habit_lineage_and_no_activation_status() -> None:
    _, _, _, episode, _, graph_build, analysis, result = compile_first()
    habit = result.habit_specification
    assert habit is not None
    edge = next(
        item
        for item in graph_build.graph.edges
        if item.transition_key.transition_key_id == habit.transition_key_id
    )
    assert habit.status == "shadow_candidate"
    assert habit.source_state_class_id == edge.transition_key.source_state_class_id
    assert habit.recommended_action == edge.transition_key.action
    assert (
        habit.expected_target_state_class_id
        == edge.transition_key.target_state_class_id
    )
    assert habit.ledger_snapshot_id == episode.ledger_snapshot.ledger_snapshot_id
    assert habit.promotion_analysis_id == analysis.promotion_analysis_id
    assert habit.status not in {"active", "admitted", "certified", "retired"}


@pytest.mark.parametrize(
    "disposition",
    [
        "insufficient_evidence",
        "unstable",
        "contradicted",
        "not_independent",
        "not_rule_change_tested",
        "unsupported",
    ],
)
def test_reject_non_eligible_candidate(disposition: str) -> None:
    schema, _, group, episode, entries, graph_build, promotion_recipe, analysis = (
        build_case(("wait",))
    )
    candidate = analysis.promotion_candidates[0]
    bad = type(candidate).create(
        promotion_recipe_id=candidate.promotion_recipe_id,
        ledger_snapshot_id=candidate.ledger_snapshot_id,
        observation_graph_id=candidate.observation_graph_id,
        transition_key_id=candidate.transition_key_id,
        graph_edge_id=candidate.graph_edge_id,
        recurrence_id=candidate.recurrence_id,
        stability_id=candidate.stability_id,
        independence_id=candidate.independence_id,
        rule_change_survival_id=candidate.rule_change_survival_id,
        disposition=disposition,
        eligible_for_compilation=False,
        reason_codes=("stability_not_met",),
        supporting_occurrence_ids=candidate.supporting_occurrence_ids,
        supporting_ledger_entry_ids=candidate.supporting_ledger_entry_ids,
    )
    result = compile_observer_habit_specification(
        promotion_analysis=analysis,
        promotion_candidate=bad,
        graph_build=graph_build,
        grouping_recipe=group,
        observation_schema=schema,
        compilation_recipe=recipe(promotion_recipe, group, schema),
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
    )
    assert result.disposition == "invalid_candidate"
    assert result.habit_specification is None


def test_exact_guard_strict_type_semantics() -> None:
    guard = ObserverHabitGuardDTO.create(
        feature_key="visible.agent_x",
        operator="equals",
        expected_type="int",
        expected_value=1,
        minimum_value=None,
        maximum_value=None,
        guard_role="positive",
        source_evidence_ids=("evidence",),
    )
    with pytest.raises(Exception):
        ObserverHabitGuardDTO.create(
            feature_key="visible.agent_x",
            operator="equals",
            expected_type="int",
            expected_value=1.0,
            minimum_value=None,
            maximum_value=None,
            guard_role="positive",
            source_evidence_ids=("evidence",),
        )
    assert guard.expected_value == 1


def test_numeric_bucket_guard_boundaries_and_ignored_feature_excluded() -> None:
    _, _, _, _, _, _, _, result = compile_first(("wait",), initial_x=4, bucket=True)
    habit = result.habit_specification
    assert habit is not None
    agent_guard = next(
        item for item in habit.positive_guards if item.feature_key == "visible.agent_x"
    )
    assert agent_guard.operator == "in_closed_range"
    assert (agent_guard.minimum_value, agent_guard.maximum_value) == (4, 5)
    assert "hidden.cooldown_remaining" not in {
        item.feature_key for item in habit.positive_guards
    }


def test_categorical_guard_uses_raw_category() -> None:
    _, _, _, _, _, _, _, result = compile_first(("move_right",))
    habit = result.habit_specification
    assert habit is not None
    guard = next(
        item
        for item in habit.positive_guards
        if item.feature_key == "visible.action_effect"
    )
    assert guard.operator in {"equals", "is_present"}
    if guard.operator == "equals":
        assert guard.expected_value in {"initial", "moved_right", "waited", "other"}


def test_known_distinguishable_counterexample_produces_safe_guard() -> None:
    schema, _, group, episode, entries, graph_build, promotion_recipe, analysis = (
        build_case(("move_right", "move_right"), ignored_hidden=False)
    )
    candidate = next(
        item for item in analysis.promotion_candidates if item.disposition == "eligible"
    )
    result = compile_observer_habit_specification(
        promotion_analysis=analysis,
        promotion_candidate=candidate,
        graph_build=graph_build,
        grouping_recipe=group,
        observation_schema=schema,
        compilation_recipe=recipe(promotion_recipe, group, schema),
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
    )
    if result.counterexamples:
        assert result.habit_specification is not None
        assert result.habit_specification.counterexample_guards
    else:
        assert result.disposition == "compiled_for_shadow"


def test_guard_limits_forbidden_and_required_features_block() -> None:
    schema, _, group, episode, entries, graph_build, promotion_recipe, analysis = (
        build_case(("wait",))
    )
    candidate = analysis.promotion_candidates[0]
    for custom_recipe, expected in (
        (
            recipe(promotion_recipe, group, schema, maximum_guard_count=1),
            "guard_limit_exceeded",
        ),
        (
            recipe(
                promotion_recipe,
                group,
                schema,
                required_guard_feature_keys=("hidden.cooldown_remaining",),
            ),
            "insufficient_guard_evidence",
        ),
    ):
        result = compile_observer_habit_specification(
            promotion_analysis=analysis,
            promotion_candidate=candidate,
            graph_build=graph_build,
            grouping_recipe=group,
            observation_schema=schema,
            compilation_recipe=custom_recipe,
            ledger_snapshot=episode.ledger_snapshot,
            entries=entries,
        )
        assert result.disposition == expected
        assert result.habit_specification is None


def test_fire_abstain_invalid_schema_and_historical_replay_determinism() -> None:
    schema, _, group, episode, entries, graph_build, _, result = compile_first(
        action_name="wait"
    )
    habit = result.habit_specification
    assert habit is not None
    replay = evaluate_observer_habit_over_ledger(
        habit_specification=habit,
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=graph_build,
        grouping_recipe=group,
        observation_schema=schema,
    )
    replay2 = evaluate_observer_habit_over_ledger(
        habit_specification=habit,
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=graph_build,
        grouping_recipe=group,
        observation_schema=schema,
    )
    assert replay.habit_shadow_replay_id == replay2.habit_shadow_replay_id
    assert any(item.outcome == "correct_fire" for item in replay.shadow_occurrences)
    other_schema = type(schema).create(schema_name="other", features=schema.features)
    invalid = evaluate_observer_habit(
        habit_specification=habit,
        observation=entries[0].source_state
        and __import__(
            "zeromodel.observer._observation_replay",
            fromlist=["observation_for_fixture_state"],
        ).observation_for_fixture_state(
            state=entries[0].source_state,
            action_effect="initial",
            observation_schema=schema,
        ),
        grouping_recipe=group,
        observation_schema=other_schema,
    )
    assert invalid.decision == "invalid"
    assert invalid.recommended_action is None


def test_shadow_execution_does_not_control_environment_and_audit() -> None:
    schema, rule, group, episode, entries, graph_build, _, result = compile_first(
        action_name="wait"
    )
    habit = result.habit_specification
    assert habit is not None
    shadow_episode = run_observer_fixture_habit_shadow_episode(
        habit_specification=habit,
        initial_state=ObserverFixtureStateDTO.create(
            fixture_id=rule.fixture_id,
            rule_set_id=rule.fixture_rule_set_id,
            episode_id="episode:shadow-live",
            step_index=0,
            agent_x=0,
            target_x=6,
        ),
        actions=(action("move_right"),),
        predictor_rule_set=rule,
        environment_rule_schedule=(
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=0, rule_set_id=rule.fixture_rule_set_id
            ),
        ),
        environment_rule_sets=(rule,),
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        grouping_recipe=group,
        graph_build=graph_build,
    )
    assert shadow_episode.authoritative_action_ids == (
        action("move_right").fixture_action_id,
    )
    assert shadow_episode.shadow_replay.status == "failed"
    assert "habit_ledger_mismatch" in shadow_episode.shadow_replay.failure_codes
    replay = evaluate_observer_habit_over_ledger(
        habit_specification=habit,
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=graph_build,
        grouping_recipe=group,
        observation_schema=schema,
    )
    audit_recipe = ObserverHabitShadowAuditRecipeDTO.create(
        minimum_applicable_count=1,
        minimum_episode_count=1,
        minimum_correct_fire_count=1,
        maximum_wrong_action_fire_count=0,
        maximum_wrong_target_fire_count=0,
        maximum_missed_opportunity_count=0,
        maximum_invalid_count=0,
        require_zero_false_fires=True,
        require_counterexample_coverage=False,
    )
    audit = audit_observer_habit_shadow(
        habit_specification=habit,
        shadow_audit_recipe=audit_recipe,
        historical_shadow_replay=replay,
        live_shadow_episodes=(),
    )
    assert audit.eligible_for_admission_review is True


def shadow_occurrence(habit, *, outcome: str, episode_index: int = 0):
    decision = "fire" if "fire" in outcome else "abstain"
    return ObserverHabitShadowOccurrenceDTO.create(
        habit_specification_id=habit.habit_specification_id,
        ledger_entry_id=f"ledger:{outcome}:{episode_index}",
        source_observation_artifact_id=f"observation:{outcome}:{episode_index}",
        source_state_class_id=habit.source_state_class_id,
        habit_evaluation_id=f"evaluation:{outcome}:{episode_index}",
        habit_decision=decision,
        habit_recommended_action=habit.recommended_action
        if decision == "fire"
        else None,
        authoritative_action=habit.recommended_action,
        actual_target_state_class_id=habit.expected_target_state_class_id
        if outcome != "wrong_target_fire"
        else "other-target",
        expected_target_state_class_id=habit.expected_target_state_class_id,
        outcome=outcome,
        reason_codes=(outcome,),
    )


def shadow_replay(habit, *, outcomes, episode_id: str):
    return ObserverHabitShadowReplayDTO.create(
        habit_specification_id=habit.habit_specification_id,
        ledger_snapshot_id=habit.ledger_snapshot_id,
        shadow_occurrences=tuple(
            shadow_occurrence(habit, outcome=outcome, episode_index=index)
            for index, outcome in enumerate(outcomes)
        ),
        episode_ids=(episode_id,),
        status="failed"
        if any("wrong" in outcome for outcome in outcomes)
        else "verified",
        failure_codes=("false_fire_detected",)
        if any("wrong" in outcome for outcome in outcomes)
        else (),
    )


def shadow_episode(habit, replay, episode_id: str):
    return ObserverHabitShadowEpisodeDTO.create(
        habit_specification_id=habit.habit_specification_id,
        fixture_episode_result_id=episode_id,
        ledger_snapshot_id=replay.ledger_snapshot_id,
        shadow_replay=replay,
        authoritative_action_ids=("action",),
        habit_fire_sequences=(0,),
        habit_abstain_sequences=(),
        status="shadow_recorded",
    )


def shadow_audit_recipe():
    return ObserverHabitShadowAuditRecipeDTO.create(
        minimum_applicable_count=1,
        minimum_episode_count=1,
        minimum_correct_fire_count=1,
        maximum_wrong_action_fire_count=0,
        maximum_wrong_target_fire_count=0,
        maximum_missed_opportunity_count=0,
        maximum_invalid_count=0,
        require_zero_false_fires=True,
        require_counterexample_coverage=False,
    )


def test_audit_counts_live_false_fires() -> None:
    _, _, _, _, _, _, _, result = compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    historical = shadow_replay(
        habit,
        outcomes=(
            "correct_fire",
            "correct_fire",
            "correct_fire",
            "correct_fire",
            "correct_fire",
        ),
        episode_id="historical",
    )
    live = shadow_replay(habit, outcomes=("wrong_target_fire",), episode_id="live")
    audit_recipe = ObserverHabitShadowAuditRecipeDTO.create(
        minimum_applicable_count=5,
        minimum_episode_count=1,
        minimum_correct_fire_count=5,
        maximum_wrong_action_fire_count=0,
        maximum_wrong_target_fire_count=0,
        maximum_missed_opportunity_count=0,
        maximum_invalid_count=0,
        require_zero_false_fires=True,
        require_counterexample_coverage=False,
    )
    audit = audit_observer_habit_shadow(
        habit_specification=habit,
        shadow_audit_recipe=audit_recipe,
        historical_shadow_replay=historical,
        live_shadow_episodes=(shadow_episode(habit, live, "live-episode"),),
    )
    assert audit.eligible_for_admission_review is False
    assert audit.disposition in {"false_fire_detected", "target_instability_detected"}
    assert audit.evaluated_shadow_replay_ids == tuple(
        sorted((historical.habit_shadow_replay_id, live.habit_shadow_replay_id))
    )


def test_audit_counts_live_evidence_toward_minimums() -> None:
    _, _, _, _, _, _, _, result = compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    historical = shadow_replay(
        habit, outcomes=("correct_fire",), episode_id="historical"
    )
    live = shadow_replay(habit, outcomes=("correct_fire",), episode_id="live")
    audit_recipe = ObserverHabitShadowAuditRecipeDTO.create(
        minimum_applicable_count=2,
        minimum_episode_count=2,
        minimum_correct_fire_count=2,
        maximum_wrong_action_fire_count=0,
        maximum_wrong_target_fire_count=0,
        maximum_missed_opportunity_count=0,
        maximum_invalid_count=0,
        require_zero_false_fires=True,
        require_counterexample_coverage=False,
    )
    audit = audit_observer_habit_shadow(
        habit_specification=habit,
        shadow_audit_recipe=audit_recipe,
        historical_shadow_replay=historical,
        live_shadow_episodes=(shadow_episode(habit, live, "live-episode"),),
    )
    assert audit.eligible_for_admission_review is True
    assert audit.evaluated_shadow_replay_ids == tuple(
        sorted((historical.habit_shadow_replay_id, live.habit_shadow_replay_id))
    )


def test_replay_lineage_mismatch_fails_without_occurrences() -> None:
    schema, _, group, episode, entries, graph_build, _, result = compile_first(
        action_name="wait"
    )
    habit = result.habit_specification
    assert habit is not None
    other_schema = type(schema).create(schema_name="other", features=schema.features)
    replay = evaluate_observer_habit_over_ledger(
        habit_specification=habit,
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=graph_build,
        grouping_recipe=group,
        observation_schema=other_schema,
    )
    assert replay.status == "failed"
    assert replay.shadow_occurrences == ()
    assert "habit_schema_mismatch" in replay.failure_codes


def test_audit_rejects_foreign_historical_replay() -> None:
    _, _, _, _, _, _, _, result_a = compile_first(action_name="wait")
    _, _, _, _, _, _, _, result_b = compile_first(("move_right",))
    habit_a = result_a.habit_specification
    habit_b = result_b.habit_specification
    assert habit_a is not None
    assert habit_b is not None
    foreign = shadow_replay(habit_b, outcomes=("correct_fire",), episode_id="foreign")

    with pytest.raises(ObserverHabitError, match="historical shadow replay habit"):
        audit_observer_habit_shadow(
            habit_specification=habit_a,
            shadow_audit_recipe=shadow_audit_recipe(),
            historical_shadow_replay=foreign,
        )


def test_audit_rejects_foreign_live_episode() -> None:
    _, _, _, _, _, _, _, result_a = compile_first(action_name="wait")
    _, _, _, _, _, _, _, result_b = compile_first(("move_right",))
    habit_a = result_a.habit_specification
    habit_b = result_b.habit_specification
    assert habit_a is not None
    assert habit_b is not None
    historical = shadow_replay(
        habit_a, outcomes=("correct_fire",), episode_id="historical"
    )
    foreign = shadow_replay(habit_b, outcomes=("correct_fire",), episode_id="foreign")

    with pytest.raises(ObserverHabitError, match="live shadow episode habit"):
        audit_observer_habit_shadow(
            habit_specification=habit_a,
            shadow_audit_recipe=shadow_audit_recipe(),
            historical_shadow_replay=historical,
            live_shadow_episodes=(shadow_episode(habit_b, foreign, "foreign-live"),),
        )


def test_audit_rejects_duplicate_shadow_replay_evidence() -> None:
    _, _, _, _, _, _, _, result = compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    replay = shadow_replay(habit, outcomes=("correct_fire",), episode_id="same")

    with pytest.raises(ObserverHabitError, match="duplicate shadow replay evidence"):
        audit_observer_habit_shadow(
            habit_specification=habit,
            shadow_audit_recipe=shadow_audit_recipe(),
            historical_shadow_replay=replay,
            live_shadow_episodes=(shadow_episode(habit, replay, "same-live"),),
        )


def test_audit_rejects_live_episode_replay_snapshot_mismatch() -> None:
    _, _, _, _, _, _, _, result = compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    historical = shadow_replay(
        habit, outcomes=("correct_fire",), episode_id="historical"
    )
    live = shadow_replay(habit, outcomes=("correct_fire",), episode_id="live")
    mismatched_episode = ObserverHabitShadowEpisodeDTO.create(
        habit_specification_id=habit.habit_specification_id,
        fixture_episode_result_id="live-episode",
        ledger_snapshot_id="other-ledger-snapshot",
        shadow_replay=live,
        authoritative_action_ids=("action",),
        habit_fire_sequences=(0,),
        habit_abstain_sequences=(),
        status="shadow_recorded",
    )

    with pytest.raises(ObserverHabitError, match="replay snapshot mismatch"):
        audit_observer_habit_shadow(
            habit_specification=habit,
            shadow_audit_recipe=shadow_audit_recipe(),
            historical_shadow_replay=historical,
            live_shadow_episodes=(mismatched_episode,),
        )


def test_public_api_exports() -> None:
    import zeromodel.observer as observer

    for name in (
        "ObserverHabitCompilationRecipeDTO",
        "ObserverHabitSpecificationDTO",
        "ObserverHabitShadowReplayDTO",
        "compile_observer_habit_specification",
        "evaluate_observer_habit",
        "evaluate_observer_habit_over_ledger",
        "run_observer_fixture_habit_shadow_episode",
        "audit_observer_habit_shadow",
    ):
        assert name in observer.__all__
