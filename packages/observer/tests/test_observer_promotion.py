import pytest

from zeromodel.observer import (
    ObserverCategoryMappingDTO,
    ObserverFixtureActionDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    ObserverFixtureRuleSetDTO,
    ObserverGroupingFeatureDTO,
    ObserverObservationGraphDTO,
    ObserverPromotionEvidenceRecipeDTO,
    ObserverStateGroupingRecipeDTO,
    ObserverTransitionRecurrenceDTO,
    ObserverTransitionOccurrenceDTO,
    analyze_observer_promotion_candidates,
    build_observer_fixture_comparison_recipe,
    build_observer_fixture_observation_schema,
    build_observer_observation_graph,
    run_observer_fixture_episode,
)
from zeromodel.observer.fixture import ObserverFixtureStateDTO
from zeromodel.observer.graph import ObserverObservationGraphEdgeDTO
from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.promotion_service import _build_stability


def action(name: str) -> ObserverFixtureActionDTO:
    return ObserverFixtureActionDTO.create(action_name=name)


def rules():
    schema = build_observer_fixture_observation_schema()
    rule1 = ObserverFixtureRuleSetDTO.create(
        fixture_id="fixture:promotion",
        rule_version="fixture-rule/1",
        minimum_position=-8,
        maximum_position=12,
        cooldown_period=1,
        cooldown_effect="block",
        observation_schema_id=schema.schema_id,
    )
    rule2 = ObserverFixtureRuleSetDTO.create(
        fixture_id="fixture:promotion",
        rule_version="fixture-rule/2",
        minimum_position=-8,
        maximum_position=12,
        cooldown_period=1,
        cooldown_effect="reverse",
        observation_schema_id=schema.schema_id,
    )
    return schema, rule1, rule2


def grouping(schema_id: str, *, position_only: bool = False):
    return ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema_id,
        type_mismatch_policy="separate_class",
        feature_groupings=(
            ObserverGroupingFeatureDTO.create(
                feature_key="hidden.cooldown_remaining", mode="ignored"
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="history.previous_action", mode="ignored"
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="visible.action_effect",
                mode="ignored" if position_only else "categorical",
                category_mapping=()
                if position_only
                else (
                    ObserverCategoryMappingDTO.create(
                        raw_value="blocked_by_cooldown", mapped_value="blocked"
                    ),
                    ObserverCategoryMappingDTO.create(
                        raw_value="initial", mapped_value="other"
                    ),
                    ObserverCategoryMappingDTO.create(
                        raw_value="moved_left", mapped_value="moved"
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
                feature_key="visible.agent_x", mode="exact"
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="visible.target_x", mode="exact"
            ),
        ),
    )


def build_case(
    actions,
    *,
    episode_id="episode:promotion",
    initial_x=0,
    rule_switch=False,
    hidden=True,
    position_only=False,
):
    schema, rule1, rule2 = rules()
    schedule = (
        ObserverFixtureRuleScheduleEntryDTO.create(
            start_step=0, rule_set_id=rule1.fixture_rule_set_id
        ),
    )
    env_rules = (rule1,)
    if rule_switch:
        schedule = (
            schedule[0],
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=1, rule_set_id=rule2.fixture_rule_set_id
            ),
        )
        env_rules = (rule1, rule2)
    episode, entries = run_observer_fixture_episode(
        initial_state=ObserverFixtureStateDTO.create(
            fixture_id=rule1.fixture_id,
            rule_set_id=rule1.fixture_rule_set_id,
            episode_id=episode_id,
            step_index=0,
            agent_x=initial_x,
            target_x=6,
        ),
        actions=tuple(action(item) for item in actions),
        predictor_rule_set=rule1,
        environment_rule_schedule=schedule,
        environment_rule_sets=env_rules,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        supply_hidden_evidence=hidden,
    )
    group = grouping(schema.schema_id, position_only=position_only)
    build = build_observer_observation_graph(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        grouping_recipe=group,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(rule1,),
        environment_rule_sets=(rule1, rule2),
    )
    assert build.status == "built"
    assert build.graph is not None
    return schema, group, episode, entries, build


def recipe_for(build, group, **overrides):
    values = {
        "observation_graph_id": build.graph.observation_graph_id,
        "grouping_recipe_id": group.grouping_recipe_id,
        "minimum_traversal_count": 1,
        "minimum_confirmed_count": 1,
        "minimum_independent_episode_count": 1,
        "minimum_distinct_source_state_count": 1,
        "minimum_distinct_rule_regime_count": 1,
        "maximum_contradicted_count": 0,
        "maximum_inconclusive_count": 1,
        "minimum_confirmation_ratio_numerator": 1,
        "minimum_confirmation_ratio_denominator": 1,
        "require_post_rule_change_confirmation": False,
    }
    values.update(overrides)
    return ObserverPromotionEvidenceRecipeDTO.create(**values)


def analyze(actions, **kwargs):
    schema, group, episode, entries, build = build_case(actions, **kwargs)
    promotion_recipe = recipe_for(build, group, **kwargs.get("recipe_overrides", {}))
    analysis = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=promotion_recipe,
    )
    assert analysis.status == "built"
    return schema, group, episode, entries, build, promotion_recipe, analysis


def test_novelty_chronology_and_exact_observation_deduplication() -> None:
    *_, analysis = analyze(("wait", "wait"), position_only=True)
    statuses = {item.novelty_status for item in analysis.novelty_evidence}
    assert "novel" in statuses
    assert "recurrent" in statuses
    first = sorted(
        analysis.novelty_evidence,
        key=lambda item: (
            item.first_ledger_sequence,
            item.first_observation_artifact_id,
        ),
    )[0]
    assert first.novelty_status == "novel"
    assert first.prior_observation_count == 0
    observation_ids = [
        item.first_observation_artifact_id for item in analysis.novelty_evidence
    ]
    assert len(observation_ids) == len(set(observation_ids))


def test_transition_recurrence_aggregates_occurrences() -> None:
    *_, analysis = analyze(("wait", "wait", "wait"), position_only=True)
    recurrence = max(analysis.recurrences, key=lambda item: item.traversal_count)
    assert recurrence.traversal_count == 3
    assert len(recurrence.occurrence_ids) == 3
    assert len(analysis.occurrences) == 3


def test_different_actions_and_targets_remain_separate_patterns() -> None:
    *_, mixed = analyze(("move_right", "wait"), position_only=True)
    assert len({item.transition_key_id for item in mixed.recurrences}) == 2
    *_, targets = analyze(("move_right", "move_right"))
    assert len({item.transition_key_id for item in targets.recurrences}) == 2


def test_stable_candidate_and_insufficient_traversal() -> None:
    schema, group, episode, entries, build = build_case(("wait", "wait"))
    eligible_recipe = recipe_for(build, group, minimum_traversal_count=2)
    eligible = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=eligible_recipe,
    )
    assert any(item.status == "stable" for item in eligible.stabilities)
    assert any(item.disposition == "eligible" for item in eligible.promotion_candidates)
    assert eligible.eligible_candidate_ids

    strict_recipe = recipe_for(build, group, minimum_traversal_count=3)
    strict = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=strict_recipe,
    )
    assert any(
        item.disposition == "insufficient_evidence"
        for item in strict.promotion_candidates
    )


def test_episode_independence_and_source_diversity_are_declared_proxies() -> None:
    *_, build, _, _ = analyze(("wait", "wait"), position_only=True)
    assert build.graph is not None
    schema, group, episode, entries, build = build_case(
        ("wait", "wait"), position_only=True
    )
    promotion_recipe = recipe_for(
        build,
        group,
        minimum_independent_episode_count=2,
        minimum_distinct_source_state_count=2,
    )
    analysis = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=promotion_recipe,
    )
    assert any(item.status == "insufficient" for item in analysis.independence_results)
    assert any(
        item.disposition == "not_independent" for item in analysis.promotion_candidates
    )


def test_contradiction_ratio_and_inconclusive_limits() -> None:
    recurrence = ObserverTransitionRecurrenceDTO.create(
        transition_key_id="transition:key",
        observation_graph_id="graph:id",
        occurrence_ids=("occurrence:1", "occurrence:2", "occurrence:3"),
        supporting_ledger_entry_ids=("ledger:1", "ledger:2", "ledger:3"),
        episode_ids=("episode:1",),
        source_observation_artifact_ids=("source:1", "source:2", "source:3"),
        target_observation_artifact_ids=("target:1",),
        predictor_rule_set_ids=("rule:1",),
        environment_rule_set_ids=("rule:1",),
        rule_regime_ids=("regime:1",),
        traversal_count=3,
        independent_episode_count=1,
        distinct_source_observation_count=3,
        distinct_target_observation_count=1,
        distinct_rule_regime_count=1,
        first_ledger_sequence=0,
        last_ledger_sequence=2,
    )
    occurrences = [
        type(
            "Occurrence",
            (),
            {"occurrence_id": "occurrence:1", "verification_status": "confirmed"},
        )(),
        type(
            "Occurrence",
            (),
            {"occurrence_id": "occurrence:2", "verification_status": "contradicted"},
        )(),
        type(
            "Occurrence",
            (),
            {"occurrence_id": "occurrence:3", "verification_status": "confirmed"},
        )(),
    ]
    ratio_recipe = ObserverPromotionEvidenceRecipeDTO.create(
        observation_graph_id="graph:id",
        grouping_recipe_id="group:id",
        maximum_contradicted_count=0,
        maximum_inconclusive_count=3,
        minimum_confirmation_ratio_numerator=2,
        minimum_confirmation_ratio_denominator=3,
    )
    on_boundary = _build_stability(
        recurrence=recurrence, occurrences=occurrences, recipe=ratio_recipe
    )
    assert on_boundary.status == "unstable"
    assert on_boundary.confirmation_ratio_numerator == 2
    assert on_boundary.confirmation_ratio_denominator == 3
    assert "confirmation_ratio_met" in on_boundary.reason_codes
    assert "contradiction_limit_exceeded" in on_boundary.reason_codes

    above_boundary_recipe = ObserverPromotionEvidenceRecipeDTO.create(
        observation_graph_id="graph:id",
        grouping_recipe_id="group:id",
        maximum_contradicted_count=3,
        maximum_inconclusive_count=3,
        minimum_confirmation_ratio_numerator=3,
        minimum_confirmation_ratio_denominator=4,
    )
    below = _build_stability(
        recurrence=recurrence,
        occurrences=occurrences,
        recipe=above_boundary_recipe,
    )
    assert below.status == "unstable"
    assert "confirmation_ratio_not_met" in below.reason_codes

    _, group2, episode2, entries2, build2 = build_case(
        ("move_right", "move_right"), rule_switch=True, hidden=False
    )
    inconclusive_recipe = recipe_for(build2, group2, maximum_inconclusive_count=0)
    inconclusive = analyze_observer_promotion_candidates(
        ledger_snapshot=episode2.ledger_snapshot,
        entries=entries2,
        graph_build=build2,
        grouping_recipe=group2,
        promotion_recipe=inconclusive_recipe,
    )
    assert not inconclusive.eligible_candidate_ids


def test_rule_change_survival_modes() -> None:
    _, group, episode, entries, build = build_case(
        ("wait", "wait"), rule_switch=True, position_only=True
    )
    required = recipe_for(build, group, require_post_rule_change_confirmation=True)
    survived = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=required,
    )
    assert any(item.status == "survived" for item in survived.rule_change_results)

    _, group2, episode2, entries2, build2 = build_case(
        ("move_right", "move_right"), rule_switch=True
    )
    failed_recipe = recipe_for(
        build2, group2, require_post_rule_change_confirmation=True
    )
    failed = analyze_observer_promotion_candidates(
        ledger_snapshot=episode2.ledger_snapshot,
        entries=entries2,
        graph_build=build2,
        grouping_recipe=group2,
        promotion_recipe=failed_recipe,
    )
    assert any(item.status == "failed" for item in failed.rule_change_results)
    assert not failed.eligible_candidate_ids

    _, group3, episode3, entries3, build3 = build_case(("wait", "wait"))
    not_tested_recipe = recipe_for(
        build3, group3, require_post_rule_change_confirmation=True
    )
    not_tested = analyze_observer_promotion_candidates(
        ledger_snapshot=episode3.ledger_snapshot,
        entries=entries3,
        graph_build=build3,
        grouping_recipe=group3,
        promotion_recipe=not_tested_recipe,
    )
    assert any(item.status == "not_tested" for item in not_tested.rule_change_results)
    assert not not_tested.eligible_candidate_ids
    optional_recipe = recipe_for(
        build3, group3, require_post_rule_change_confirmation=False
    )
    optional = analyze_observer_promotion_candidates(
        ledger_snapshot=episode3.ledger_snapshot,
        entries=entries3,
        graph_build=build3,
        grouping_recipe=group3,
        promotion_recipe=optional_recipe,
    )
    assert optional.eligible_candidate_ids


def replace_graph(graph: ObserverObservationGraphDTO, edges):
    payload = {
        "assignment_ids": list(graph.assignment_ids),
        "edge_ids": [item.graph_edge_id for item in edges],
        "edges": [item.canonical_payload() for item in edges],
        "grouping_recipe_id": graph.grouping_recipe_id,
        "ledger_snapshot_id": graph.ledger_snapshot_id,
        "node_ids": list(graph.node_ids),
        "nodes": [item.canonical_payload() for item in graph.nodes],
        "observation_schema_id": graph.observation_schema_id,
        "rejected_ledger_entry_ids": list(graph.rejected_ledger_entry_ids),
        "version": graph.version,
    }
    return ObserverObservationGraphDTO(
        observation_graph_id=canonical_id(payload),
        ledger_snapshot_id=graph.ledger_snapshot_id,
        grouping_recipe_id=graph.grouping_recipe_id,
        observation_schema_id=graph.observation_schema_id,
        node_ids=graph.node_ids,
        edge_ids=tuple(item.graph_edge_id for item in edges),
        nodes=graph.nodes,
        edges=tuple(edges),
        assignment_ids=graph.assignment_ids,
        rejected_ledger_entry_ids=graph.rejected_ledger_entry_ids,
    )


def test_graph_count_tampering_missing_support_and_duplicate_occurrence_fail() -> None:
    _, group, episode, entries, build = build_case(
        ("move_right", "move_right"), rule_switch=True, hidden=False
    )
    assert build.graph is not None
    edge = next(item for item in build.graph.edges if item.inconclusive_count)
    tampered_edge = ObserverObservationGraphEdgeDTO.create(
        transition_key=edge.transition_key,
        supporting_ledger_entry_ids=edge.supporting_ledger_entry_ids,
        transition_verification_ids=edge.transition_verification_ids,
        comparison_result_ids=edge.comparison_result_ids,
        confirmed_count=edge.confirmed_count + 1,
        contradicted_count=edge.contradicted_count,
        inconclusive_count=edge.inconclusive_count - 1,
        predictor_rule_set_ids=edge.predictor_rule_set_ids,
        environment_rule_set_ids=edge.environment_rule_set_ids,
    )
    tampered_graph = replace_graph(
        build.graph,
        tuple(tampered_edge if item == edge else item for item in build.graph.edges),
    )
    build_payload = {
        "assignments": [item.canonical_payload() for item in build.assignments],
        "failure_codes": list(build.failure_codes),
        "graph": tampered_graph.canonical_payload(),
        "grouping_recipe_id": build.grouping_recipe_id,
        "ledger_integrity_result_id": build.ledger_integrity_result_id,
        "ledger_semantic_replay_result_id": build.ledger_semantic_replay_result_id,
        "ledger_snapshot_id": build.ledger_snapshot_id,
        "state_classes": [item.canonical_payload() for item in build.state_classes],
        "status": build.status,
        "version": build.version,
    }
    tampered_build = build.__class__(
        graph_build_id=canonical_id(build_payload),
        ledger_snapshot_id=build.ledger_snapshot_id,
        ledger_integrity_result_id=build.ledger_integrity_result_id,
        ledger_semantic_replay_result_id=build.ledger_semantic_replay_result_id,
        grouping_recipe_id=build.grouping_recipe_id,
        assignments=build.assignments,
        state_classes=build.state_classes,
        graph=tampered_graph,
        status=build.status,
        failure_codes=build.failure_codes,
    )
    promotion_recipe = recipe_for(tampered_build, group)
    tampered = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=tampered_build,
        grouping_recipe=group,
        promotion_recipe=promotion_recipe,
    )
    assert tampered.status == "failed"
    assert "edge_count_mismatch" in tampered.failure_codes

    missing = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries[:1],
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=recipe_for(build, group),
    )
    assert missing.status == "failed"
    assert "edge_support_missing" in missing.failure_codes

    with pytest.raises(Exception):
        ObserverTransitionOccurrenceDTO(
            occurrence_id="bad",
            transition_key_id="key",
            graph_edge_id="edge",
            ledger_entry_id="ledger",
            ledger_sequence=0,
            episode_id="episode",
            source_state_class_id="source",
            source_observation_artifact_id="obs",
            target_state_class_id="target",
            target_observation_artifact_id="obs2",
            action="wait",
            verification_status="confirmed",
            comparison_result_id="comparison",
            predictor_rule_set_id="rule",
            environment_rule_set_id="rule",
            rule_regime_id="regime",
        )


def test_recipe_sensitivity_snapshot_immutability_empty_graph_and_replay() -> None:
    _, group, episode, entries, build = build_case(("wait",))
    loose = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=recipe_for(build, group),
    )
    strict_recipe = recipe_for(build, group, minimum_traversal_count=2)
    strict = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=strict_recipe,
    )
    assert loose.promotion_analysis_id != strict.promotion_analysis_id
    assert (
        loose.promotion_candidates[0].promotion_candidate_id
        != strict.promotion_candidates[0].promotion_candidate_id
    )

    replay = analyze_observer_promotion_candidates(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=build,
        grouping_recipe=group,
        promotion_recipe=recipe_for(build, group),
    )
    assert replay.promotion_analysis_id == loose.promotion_analysis_id
    assert [item.occurrence_id for item in replay.occurrences] == [
        item.occurrence_id for item in loose.occurrences
    ]

    _, group2, episode2, entries2, build2 = build_case(("wait", "wait"))
    newer = analyze_observer_promotion_candidates(
        ledger_snapshot=episode2.ledger_snapshot,
        entries=entries2,
        graph_build=build2,
        grouping_recipe=group2,
        promotion_recipe=recipe_for(build2, group2),
    )
    assert (
        loose.promotion_candidates[0].promotion_candidate_id
        != newer.promotion_candidates[0].promotion_candidate_id
    )

    _, empty_group, empty_episode, empty_entries, empty_build = build_case(())
    empty = analyze_observer_promotion_candidates(
        ledger_snapshot=empty_episode.ledger_snapshot,
        entries=empty_entries,
        graph_build=empty_build,
        grouping_recipe=empty_group,
        promotion_recipe=recipe_for(empty_build, empty_group),
    )
    assert empty.status == "built"
    assert empty.promotion_candidates == ()
    assert empty.eligible_candidate_ids == ()


def test_public_api_exports() -> None:
    import zeromodel.observer as observer

    assert "ObserverPromotionEvidenceRecipeDTO" in observer.__all__
    assert "ObserverPromotionCandidateDTO" in observer.__all__
    assert "analyze_observer_promotion_candidates" in observer.__all__
