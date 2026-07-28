import pytest

from zeromodel.observer import (
    ObserverCategoryMappingDTO,
    ObserverFeatureDefinitionDTO,
    ObserverFixtureActionDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
    ObserverGroupingFeatureDTO,
    ObserverObservationArtifactDTO,
    ObserverObservationGraphDTO,
    ObserverObservationGraphError,
    ObserverObservationSchemaDTO,
    ObserverStateGroupingRecipeDTO,
    assign_observation_to_state_class,
    build_observer_fixture_comparison_recipe,
    build_observer_fixture_observation_schema,
    build_observer_observation_graph,
    build_observer_transition_ledger_snapshot,
    run_observer_fixture_episode,
    verify_observer_graph_rebuild,
)
from zeromodel.observer._canonical import canonical_id


def schema_and_rules():
    schema = build_observer_fixture_observation_schema()
    rule1 = ObserverFixtureRuleSetDTO.create(
        fixture_id="fixture:graph",
        rule_version="fixture-rule/1",
        minimum_position=-4,
        maximum_position=8,
        cooldown_period=1,
        cooldown_effect="block",
        observation_schema_id=schema.schema_id,
    )
    rule2 = ObserverFixtureRuleSetDTO.create(
        fixture_id="fixture:graph",
        rule_version="fixture-rule/2",
        minimum_position=-4,
        maximum_position=8,
        cooldown_period=1,
        cooldown_effect="reverse",
        observation_schema_id=schema.schema_id,
    )
    return schema, rule1, rule2


def state(rule: ObserverFixtureRuleSetDTO, *, x: int = 0) -> ObserverFixtureStateDTO:
    return ObserverFixtureStateDTO.create(
        fixture_id=rule.fixture_id,
        rule_set_id=rule.fixture_rule_set_id,
        episode_id="episode:graph",
        step_index=0,
        agent_x=x,
        target_x=6,
    )


def action(name: str) -> ObserverFixtureActionDTO:
    return ObserverFixtureActionDTO.create(action_name=name)


def run_graph_episode(
    *,
    actions: tuple[ObserverFixtureActionDTO, ...],
    initial_x: int = 0,
    rule_switch: bool = False,
    supply_hidden_evidence: bool = True,
):
    schema, rule1, rule2 = schema_and_rules()
    schedule = (
        ObserverFixtureRuleScheduleEntryDTO.create(
            start_step=0, rule_set_id=rule1.fixture_rule_set_id
        ),
    )
    rules = (rule1,)
    if rule_switch:
        schedule = (
            schedule[0],
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=1, rule_set_id=rule2.fixture_rule_set_id
            ),
        )
        rules = (rule1, rule2)
    episode, entries = run_observer_fixture_episode(
        initial_state=state(rule1, x=initial_x),
        actions=actions,
        predictor_rule_set=rule1,
        environment_rule_schedule=schedule,
        environment_rule_sets=rules,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        supply_hidden_evidence=supply_hidden_evidence,
    )
    return schema, rule1, rule2, episode, entries


def recipe(
    schema_id: str,
    *,
    bucket: bool = False,
    map_effect: bool = False,
    include_hidden: bool = False,
    missing: str = "reject",
) -> ObserverStateGroupingRecipeDTO:
    effect_feature = ObserverGroupingFeatureDTO.create(
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
                raw_value="moved_left", mapped_value="moved"
            ),
            ObserverCategoryMappingDTO.create(
                raw_value="moved_right", mapped_value="moved"
            ),
            ObserverCategoryMappingDTO.create(raw_value="waited", mapped_value="other"),
        )
        if map_effect
        else (),
    )
    return ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema_id,
        missing_feature_policy=missing,
        type_mismatch_policy="separate_class",
        feature_groupings=(
            ObserverGroupingFeatureDTO.create(
                feature_key="hidden.cooldown_remaining",
                mode="exact" if include_hidden else "ignored",
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="history.previous_action", mode="ignored"
            ),
            effect_feature,
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


def build_graph(actions, *, grouping_recipe=None, rule_switch=False, hidden=True):
    schema, rule1, rule2, episode, entries = run_graph_episode(
        actions=actions, rule_switch=rule_switch, supply_hidden_evidence=hidden
    )
    grouping_recipe = grouping_recipe or recipe(schema.schema_id)
    build = build_observer_observation_graph(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        grouping_recipe=grouping_recipe,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(rule1,),
        environment_rule_sets=(rule1, rule2),
    )
    return schema, rule1, rule2, episode, entries, build


def observation(schema, **features):
    return ObserverObservationArtifactDTO.create(
        observation_schema=schema,
        visible_state_features={
            "action_effect": features.get("action_effect", "moved_right"),
            "agent_x": features.get("agent_x", 1),
            "target_x": features.get("target_x", 6),
        },
        recent_history_features=features.get("history", {}),
        hidden_state_uncertainty=features.get("hidden", {"cooldown_remaining": 0}),
        provenance=features.get("provenance", {}),
        sequence_index=features.get("sequence", 1),
    )


def float_schema() -> ObserverObservationSchemaDTO:
    return ObserverObservationSchemaDTO.create(
        schema_name="graph-float",
        features=(
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.agent_x", value_type="float", required=True
            ),
        ),
    )


def raw_observation(
    schema_id: str, *, visible_state_features: dict[str, object]
) -> ObserverObservationArtifactDTO:
    payload = {
        "hidden_state_uncertainty": {},
        "observation_schema_id": schema_id,
        "provenance": {},
        "recent_history_features": {},
        "sequence_index": 0,
        "version": "observer-observation-artifact/2",
        "visible_state_features": visible_state_features,
    }
    return ObserverObservationArtifactDTO(
        observation_artifact_id=canonical_id(payload),
        observation_schema_id=schema_id,
        visible_state_features=visible_state_features,
        recent_history_features={},
        hidden_state_uncertainty={},
        provenance={},
        sequence_index=0,
    )


def test_exact_grouping_singletons_and_equivalent_observations_share_class() -> None:
    schema, *_ = schema_and_rules()
    position_only = ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema.schema_id,
        feature_groupings=(
            ObserverGroupingFeatureDTO.create(
                feature_key="hidden.cooldown_remaining", mode="ignored"
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="history.previous_action", mode="ignored"
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="visible.action_effect", mode="ignored"
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="visible.agent_x", mode="exact"
            ),
            ObserverGroupingFeatureDTO.create(
                feature_key="visible.target_x", mode="exact"
            ),
        ),
    )
    schema, _, _, _, _, wait_build = build_graph(
        (action("wait"),), grouping_recipe=position_only
    )
    assert wait_build.status == "built"
    shared_nodes = [
        node
        for node in wait_build.graph.nodes
        if len(node.observation_artifact_ids) == 2
    ]
    assert shared_nodes

    _, _, _, _, _, move_build = build_graph((action("move_right"),))
    assert len(move_build.graph.nodes) == 2

    first, second = shared_nodes[0].observation_artifact_ids
    assert first != second


def test_numeric_bucket_and_negative_values_are_deterministic() -> None:
    schema = float_schema()
    grouping = ObserverGroupingFeatureDTO.create(
        feature_key="visible.agent_x", mode="numeric_bucket", bucket_size=1.0
    )
    grouping_recipe = ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema.schema_id,
        feature_groupings=(grouping,),
    )
    assignments = [
        assign_observation_to_state_class(
            observation=raw_observation(
                schema.schema_id, visible_state_features={"agent_x": value}
            ),
            grouping_recipe=grouping_recipe,
            observation_schema=schema,
        )
        for value in (0.1, 0.9, 1.1, -0.1)
    ]
    buckets = [
        item.state_class_key[0].grouped_value["bucket_index"] for item in assignments
    ]
    assert buckets == [0, 0, 1, -1]


def test_strict_type_distinction_and_ignored_feature() -> None:
    schema, *_ = schema_and_rules()
    exact = ObserverGroupingFeatureDTO.create(
        feature_key="visible.agent_x", mode="exact"
    )
    grouping_recipe = ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema.schema_id,
        feature_groupings=(exact,),
        type_mismatch_policy="separate_class",
    )
    bool_obs = raw_observation(
        schema.schema_id,
        visible_state_features={
            "action_effect": "moved_right",
            "agent_x": True,
            "target_x": 6,
        },
    )
    assigned_bool = assign_observation_to_state_class(
        observation=bool_obs,
        grouping_recipe=grouping_recipe,
        observation_schema=schema,
    )
    assigned_int = assign_observation_to_state_class(
        observation=observation(schema, agent_x=1),
        grouping_recipe=grouping_recipe,
        observation_schema=schema,
    )
    assert assigned_bool.state_class_id != assigned_int.state_class_id
    assert assigned_bool.state_class_key[0].grouped_kind == "type_mismatch"

    _, _, _, _, _, hidden_ignored = build_graph((action("move_right"),))
    hidden_recipe = recipe(schema.schema_id, include_hidden=True)
    _, _, _, _, _, hidden_included = build_graph(
        (action("move_right"),), grouping_recipe=hidden_recipe
    )
    assert (
        hidden_ignored.graph.observation_graph_id
        != hidden_included.graph.observation_graph_id
    )


def test_categorical_mapping_changes_graph_identity() -> None:
    schema, _, _, _, entries, exact_build = build_graph(
        (action("move_right"), action("move_right")), rule_switch=True
    )
    mapped_recipe = recipe(schema.schema_id, map_effect=True)
    mapped = build_observer_observation_graph(
        ledger_snapshot=build_observer_transition_ledger_snapshot(entries=entries),
        entries=entries,
        grouping_recipe=mapped_recipe,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(schema_and_rules()[1],),
        environment_rule_sets=(schema_and_rules()[1], schema_and_rules()[2]),
    )
    assert mapped.graph.observation_graph_id != exact_build.graph.observation_graph_id


def test_missing_feature_policies_and_schema_mismatch() -> None:
    schema, *_ = schema_and_rules()
    missing_feature = ObserverGroupingFeatureDTO.create(
        feature_key="history.previous_action", mode="exact"
    )
    reject_recipe = ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema.schema_id,
        feature_groupings=(missing_feature,),
        missing_feature_policy="reject",
    )
    separate_recipe = ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema.schema_id,
        feature_groupings=(missing_feature,),
        missing_feature_policy="separate_class",
    )
    obs = observation(schema, history={})
    rejected = assign_observation_to_state_class(
        observation=obs, grouping_recipe=reject_recipe, observation_schema=schema
    )
    separated = assign_observation_to_state_class(
        observation=obs, grouping_recipe=separate_recipe, observation_schema=schema
    )
    assert rejected.status == "rejected"
    assert separated.status == "assigned"
    assert separated.state_class_key[0].grouped_kind == "missing"

    bad_recipe = ObserverStateGroupingRecipeDTO.create(
        observation_schema_id="schema:other",
        feature_groupings=(missing_feature,),
    )
    bad = assign_observation_to_state_class(
        observation=obs, grouping_recipe=bad_recipe, observation_schema=schema
    )
    assert bad.status == "rejected"
    assert "schema_mismatch" in bad.reason_codes


def test_action_labelled_edges_aggregation_status_counts_and_rule_evidence() -> None:
    schema, rule1, rule2, _, entries, build = build_graph(
        (action("move_right"), action("move_right"), action("wait")),
        rule_switch=True,
        hidden=False,
    )
    assert build.status == "built"
    assert all(
        edge.traversal_count
        == edge.confirmed_count + edge.contradicted_count + edge.inconclusive_count
        for edge in build.graph.edges
    )
    assert {edge.transition_key.action for edge in build.graph.edges} >= {
        "move_right",
        "wait",
    }
    assert any(
        rule2.fixture_rule_set_id in edge.environment_rule_set_ids
        for edge in build.graph.edges
    )
    assert any(edge.inconclusive_count for edge in build.graph.edges)
    assert all(
        rule1.fixture_rule_set_id in edge.predictor_rule_set_ids
        for edge in build.graph.edges
    )
    assert all(edge.supporting_ledger_entry_ids for edge in build.graph.edges)
    assert set().union(
        *(set(edge.supporting_ledger_entry_ids) for edge in build.graph.edges)
    ) <= {entry.ledger_entry_id for entry in entries}


def test_predicted_observations_are_not_traversed_nodes_and_source_chains() -> None:
    _, _, _, _, entries, build = build_graph(
        (action("move_right"), action("move_right")), rule_switch=True
    )
    predicted_ids = {
        entry.predicted_transition.predicted_observation.observation_artifact_id
        for entry in entries
    }
    node_observation_ids = {
        observation_id
        for node in build.graph.nodes
        for observation_id in node.observation_artifact_ids
    }
    actual_ids = {entry.executed_step.actual_observation_id for entry in entries}
    assert actual_ids <= node_observation_ids
    assert not (predicted_ids - actual_ids) & node_observation_ids


def test_graph_rebuild_verification_and_tamper_detection() -> None:
    schema, rule1, rule2, episode, entries, build = build_graph(
        (action("move_right"), action("wait")), rule_switch=True
    )
    verification = verify_observer_graph_rebuild(
        expected_graph=build.graph,
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        grouping_recipe=recipe(schema.schema_id),
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(rule1,),
        environment_rule_sets=(rule1, rule2),
    )
    assert verification.status == "verified"
    tampered = replace_graph_id(
        ObserverObservationGraphDTO(
            observation_graph_id=canonical_id(
                {
                    "assignment_ids": list(build.graph.assignment_ids),
                    "edge_ids": list(build.graph.edge_ids),
                    "edges": [item.canonical_payload() for item in build.graph.edges],
                    "grouping_recipe_id": build.graph.grouping_recipe_id,
                    "ledger_snapshot_id": build.graph.ledger_snapshot_id,
                    "node_ids": list(build.graph.node_ids[:-1]),
                    "nodes": [
                        item.canonical_payload() for item in build.graph.nodes[:-1]
                    ],
                    "observation_schema_id": build.graph.observation_schema_id,
                    "rejected_ledger_entry_ids": list(
                        build.graph.rejected_ledger_entry_ids
                    ),
                    "version": "observer-observation-graph/1",
                }
            ),
            ledger_snapshot_id=build.graph.ledger_snapshot_id,
            grouping_recipe_id=build.graph.grouping_recipe_id,
            observation_schema_id=build.graph.observation_schema_id,
            node_ids=build.graph.node_ids[:-1],
            edge_ids=build.graph.edge_ids,
            nodes=build.graph.nodes[:-1],
            edges=build.graph.edges,
            assignment_ids=build.graph.assignment_ids,
            rejected_ledger_entry_ids=build.graph.rejected_ledger_entry_ids,
        )
    )
    failed = verify_observer_graph_rebuild(
        expected_graph=tampered,
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        grouping_recipe=recipe(schema.schema_id),
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(rule1,),
        environment_rule_sets=(rule1, rule2),
    )
    assert failed.status == "failed"
    assert "graph_id_mismatch" in failed.failure_codes


def replace_graph_id(graph: ObserverObservationGraphDTO) -> ObserverObservationGraphDTO:
    from zeromodel.observer._canonical import canonical_id

    return ObserverObservationGraphDTO(
        observation_graph_id=canonical_id(graph.canonical_payload(include_id=False)),
        ledger_snapshot_id=graph.ledger_snapshot_id,
        grouping_recipe_id=graph.grouping_recipe_id,
        observation_schema_id=graph.observation_schema_id,
        node_ids=graph.node_ids,
        edge_ids=graph.edge_ids,
        nodes=graph.nodes,
        edges=graph.edges,
        assignment_ids=graph.assignment_ids,
        rejected_ledger_entry_ids=graph.rejected_ledger_entry_ids,
    )


def test_recipe_ledger_empty_and_public_api_sensitivity() -> None:
    schema, _, _, episode, entries, build = build_graph((action("wait"),))
    bucket_recipe = recipe(schema.schema_id, bucket=True)
    bucket_build = build_observer_observation_graph(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        grouping_recipe=bucket_recipe,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(schema_and_rules()[1],),
        environment_rule_sets=(schema_and_rules()[1], schema_and_rules()[2]),
    )
    assert build.graph.observation_graph_id != bucket_build.graph.observation_graph_id

    empty = build_observer_observation_graph(
        ledger_snapshot=build_observer_transition_ledger_snapshot(entries=()),
        entries=(),
        grouping_recipe=recipe(schema.schema_id),
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(),
        environment_rule_sets=(),
    )
    assert empty.status == "built"
    assert empty.graph.nodes == ()
    assert empty.graph.edges == ()

    import zeromodel.observer as observer

    assert "ObserverStateGroupingRecipeDTO" in observer.__all__
    assert "build_observer_observation_graph" in observer.__all__
    assert "verify_observer_graph_rebuild" in observer.__all__


def test_rejected_transition_reporting_and_duplicate_membership_guard() -> None:
    schema, _, _, episode, entries, _ = build_graph((action("move_right"),))
    reject_recipe = ObserverStateGroupingRecipeDTO.create(
        observation_schema_id=schema.schema_id,
        feature_groupings=(
            ObserverGroupingFeatureDTO.create(
                feature_key="history.previous_action", mode="exact"
            ),
        ),
        missing_feature_policy="reject",
    )
    rejected = build_observer_observation_graph(
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        grouping_recipe=reject_recipe,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets=(schema_and_rules()[1],),
        environment_rule_sets=(schema_and_rules()[1],),
    )
    assert rejected.status == "failed"
    assert entries[0].ledger_entry_id in rejected.graph.rejected_ledger_entry_ids
    with pytest.raises(ObserverObservationGraphError):
        rejected.graph.nodes[0].__class__(
            graph_node_id=rejected.graph.nodes[0].graph_node_id,
            state_class_id=rejected.graph.nodes[0].state_class_id,
            grouping_recipe_id=rejected.graph.nodes[0].grouping_recipe_id,
            observation_schema_id=rejected.graph.nodes[0].observation_schema_id,
            observation_artifact_ids=(
                rejected.graph.nodes[0].observation_artifact_ids
                + rejected.graph.nodes[0].observation_artifact_ids[:1]
            ),
            assignment_ids=rejected.graph.nodes[0].assignment_ids,
            supporting_ledger_entry_ids=rejected.graph.nodes[
                0
            ].supporting_ledger_entry_ids,
            first_ledger_sequence=rejected.graph.nodes[0].first_ledger_sequence,
            last_ledger_sequence=rejected.graph.nodes[0].last_ledger_sequence,
            visit_count=rejected.graph.nodes[0].visit_count + 1,
        )
