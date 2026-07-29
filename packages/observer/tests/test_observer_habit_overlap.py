from zeromodel.observer import (
    ObserverHabitAdmissionDecisionDTO,
    ObserverHabitGuardDTO,
    ObserverHabitOverlapAnalysisRecipeDTO,
    ObserverHabitSpecificationDTO,
    ObserverHabitActivationScopeDTO,
    analyze_observer_habit_overlap,
)

from test_observer_habit import compile_first, grouping


def _case():
    schema, _, group, episode, entries, graph_build, _, result = compile_first(
        action_name="wait"
    )
    object.__setattr__(episode.ledger_snapshot, "entries", entries)
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
    return schema, group, episode, graph_build.graph, scope, habit


def _decision(habit, *, decision="admit"):
    return ObserverHabitAdmissionDecisionDTO.create(
        habit_specification_id=habit.habit_specification_id,
        habit_shadow_audit_id=f"audit:{habit.habit_specification_id}",
        habit_admission_recipe_id=f"recipe:{habit.habit_specification_id}",
        decision=decision,
        reason_codes=(decision,),
        admitted_registry_status="admitted_inactive" if decision == "admit" else None,
        evidence_replay_ids=(f"replay:{habit.habit_specification_id}",),
    )


def _guard(key, value, expected_type="int"):
    return ObserverHabitGuardDTO.create(
        feature_key=key,
        operator="equals",
        expected_type=expected_type,
        expected_value=value,
        minimum_value=None,
        maximum_value=None,
        guard_role="positive",
        source_evidence_ids=("test:evidence",),
    )


def _variant(habit, suffix, **changes):
    values = {
        field: getattr(habit, field)
        for field in habit.__dataclass_fields__
        if field not in {"habit_specification_id", "version"}
    }
    values.update(changes)
    values["promotion_candidate_id"] = f"{values['promotion_candidate_id']}:{suffix}"
    values["transition_key_id"] = f"{values['transition_key_id']}:{suffix}"
    return ObserverHabitSpecificationDTO.create(**values)


def _analysis(habits, decisions, schema, group, episode, graph, scope, **recipe):
    return analyze_observer_habit_overlap(
        habit_specifications=tuple(habits),
        admission_decisions=tuple(decisions),
        activation_scope=scope,
        observation_schema=schema,
        grouping_recipe=group,
        ledger_snapshot=episode.ledger_snapshot,
        observation_graph=graph,
        analysis_recipe=ObserverHabitOverlapAnalysisRecipeDTO.create(**recipe),
    )


def _assert_subsumes(pair, *, broader, narrower) -> None:
    if pair.left_habit_specification_id == broader.habit_specification_id:
        assert pair.right_habit_specification_id == narrower.habit_specification_id
        assert pair.guard_relation == "left_subsumes_right"
    else:
        assert pair.left_habit_specification_id == narrower.habit_specification_id
        assert pair.right_habit_specification_id == broader.habit_specification_id
        assert pair.guard_relation == "right_subsumes_left"


def test_two_admitted_habits_produce_canonical_pair_and_identity_ordering() -> None:
    schema, group, episode, graph, scope, left = _case()
    right = _variant(left, "right")
    a = _analysis(
        (right, left),
        (_decision(right), _decision(left)),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    b = _analysis(
        (left, right),
        (_decision(left), _decision(right)),
        schema,
        group,
        episode,
        graph,
        scope,
    )
    assert a.status in {"completed", "completed_with_conflicts"}
    assert a.pair_count == 1
    assert (
        a.pair_overlaps[0].left_habit_specification_id
        < a.pair_overlaps[0].right_habit_specification_id
    )
    assert a.habit_overlap_analysis_id == b.habit_overlap_analysis_id


def test_input_validation_failures() -> None:
    schema, group, episode, graph, scope, habit = _case()
    assert (
        _analysis(
            (habit,), (_decision(habit),), schema, group, episode, graph, scope
        ).status
        == "failed"
    )
    assert (
        _analysis(
            (habit, habit), (_decision(habit),), schema, group, episode, graph, scope
        ).status
        == "failed"
    )
    other = _variant(habit, "other")
    assert (
        _analysis(
            (habit, other), (_decision(habit),), schema, group, episode, graph, scope
        ).status
        == "failed"
    )
    assert (
        _analysis(
            (habit, other),
            (_decision(habit), _decision(other, decision="reject")),
            schema,
            group,
            episode,
            graph,
            scope,
        ).status
        == "failed"
    )


def test_schema_grouping_scope_and_pair_bound_failures() -> None:
    schema, group, episode, graph, scope, habit = _case()
    other = _variant(habit, "other")
    bad_group = grouping(schema.schema_id, bucket=True)
    assert (
        _analysis(
            (habit, other),
            (_decision(habit), _decision(other)),
            schema,
            bad_group,
            episode,
            graph,
            scope,
        ).status
        == "failed"
    )
    bad_scope = ObserverHabitActivationScopeDTO.create(
        fixture_id="fixture:other",
        observation_schema_id=schema.schema_id,
        grouping_recipe_id=group.grouping_recipe_id,
        allowed_action_names=("wait",),
        maximum_active_habit_count=1,
        allow_overlapping_source_classes=False,
    )
    assert (
        _analysis(
            (habit, other),
            (_decision(habit), _decision(other)),
            schema,
            group,
            episode,
            graph,
            bad_scope,
        ).status
        == "failed"
    )
    third = _variant(habit, "third")
    assert (
        _analysis(
            (habit, other, third),
            (_decision(habit), _decision(other), _decision(third)),
            schema,
            group,
            episode,
            graph,
            scope,
            maximum_pair_count=1,
        ).status
        == "failed"
    )


def test_guard_relations_and_strict_type_distinction() -> None:
    schema, group, episode, graph, scope, habit = _case()
    broad = _variant(habit, "broad", positive_guards=(_guard("visible.agent_x", 0),))
    narrow = _variant(
        habit,
        "narrow",
        positive_guards=(
            _guard("visible.agent_x", 0),
            _guard("visible.target_x", 6),
        ),
    )
    disjoint = _variant(
        habit, "disjoint", positive_guards=(_guard("visible.agent_x", 1),)
    )
    typed = _variant(
        habit, "typed", positive_guards=(_guard("visible.agent_x", True, "bool"),)
    )
    unknown = _variant(
        habit,
        "unknown",
        positive_guards=(
            ObserverHabitGuardDTO.create(
                feature_key="visible.agent_x",
                operator="not_equals",
                expected_type="int",
                expected_value=9,
                minimum_value=None,
                maximum_value=None,
                guard_role="positive",
                source_evidence_ids=("test:evidence",),
            ),
        ),
    )
    same = _variant(broad, "same")
    assert (
        _analysis(
            (broad, same),
            (_decision(broad), _decision(same)),
            schema,
            group,
            episode,
            graph,
            scope,
        )
        .pair_overlaps[0]
        .guard_relation
        == "equivalent"
    )
    _assert_subsumes(
        _analysis(
            (broad, narrow),
            (_decision(broad), _decision(narrow)),
            schema,
            group,
            episode,
            graph,
            scope,
        ).pair_overlaps[0],
        broader=broad,
        narrower=narrow,
    )
    assert (
        _analysis(
            (broad, disjoint),
            (_decision(broad), _decision(disjoint)),
            schema,
            group,
            episode,
            graph,
            scope,
        )
        .pair_overlaps[0]
        .guard_relation
        == "disjoint"
    )
    assert (
        _analysis(
            (broad, typed),
            (_decision(broad), _decision(typed)),
            schema,
            group,
            episode,
            graph,
            scope,
        )
        .pair_overlaps[0]
        .guard_relation
        == "disjoint"
    )
    assert (
        _analysis(
            (broad, unknown),
            (_decision(broad), _decision(unknown)),
            schema,
            group,
            episode,
            graph,
            scope,
        )
        .pair_overlaps[0]
        .guard_relation
        == "unknown"
    )


def test_action_target_theoretical_and_evidenced_overlap() -> None:
    schema, group, episode, graph, scope, habit = _case()
    same = _variant(habit, "same")
    target = _variant(habit, "target", expected_target_state_class_id="state:other")
    action = _variant(habit, "action", recommended_action="move_left")
    same_pair = _analysis(
        (habit, same),
        (_decision(habit), _decision(same)),
        schema,
        group,
        episode,
        graph,
        scope,
    ).pair_overlaps[0]
    target_pair = _analysis(
        (habit, target),
        (_decision(habit), _decision(target)),
        schema,
        group,
        episode,
        graph,
        scope,
    ).pair_overlaps[0]
    action_pair = _analysis(
        (habit, action),
        (_decision(habit), _decision(action)),
        schema,
        group,
        episode,
        graph,
        scope,
    ).pair_overlaps[0]
    assert same_pair.theoretical_overlap
    assert same_pair.evidenced_overlap
    assert "same_action_target_agreement" in same_pair.conflict_classifications
    assert "same_action_target_conflict" in target_pair.conflict_classifications
    assert "different_action_conflict" in action_pair.conflict_classifications


def test_different_source_class_is_disjoint_and_unobserved_theoretical_overlap() -> (
    None
):
    schema, group, episode, graph, scope, habit = _case()
    different_source = _variant(habit, "source", source_state_class_id="state:other")
    source_pair = _analysis(
        (habit, different_source),
        (_decision(habit), _decision(different_source)),
        schema,
        group,
        episode,
        graph,
        scope,
    ).pair_overlaps[0]
    assert source_pair.source_state_relation == "different_source_state_class"
    assert "source_class_disjoint" in source_pair.conflict_classifications
    no_entries = episode.ledger_snapshot
    object.__setattr__(no_entries, "entries", ())
    other = _variant(habit, "unobserved")
    unobserved = _analysis(
        (habit, other),
        (_decision(habit), _decision(other)),
        schema,
        group,
        episode,
        graph,
        scope,
    ).pair_overlaps[0]
    assert unobserved.theoretical_overlap
    assert not unobserved.evidenced_overlap
    assert "theoretical_overlap_unobserved" in unobserved.conflict_classifications
