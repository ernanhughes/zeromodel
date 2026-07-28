import pytest

from zeromodel.observer import (
    InMemoryObserverTransitionLedger,
    ObserverFixtureActionDTO,
    ObserverFixtureError,
    ObserverFixtureRuleScheduleEntryDTO,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
    ObserverTransitionLedgerEntryDTO,
    ObserverWakePolicyDTO,
    build_observer_fixture_comparison_recipe,
    build_observer_fixture_observation_schema,
    build_observer_transition_ledger_snapshot,
    build_wake_policy_ablation,
    evaluate_wake_policy_over_ledger,
    execute_observer_fixture_step,
    predict_observer_fixture_transition,
    replay_observer_fixture_ledger,
    run_observer_fixture_episode,
    verify_observer_transition_ledger_integrity,
    verify_observer_transition,
)


def schema_and_rules():
    schema = build_observer_fixture_observation_schema()
    rule1 = ObserverFixtureRuleSetDTO.create(
        fixture_id="fixture:line",
        rule_version="fixture-rule/1",
        minimum_position=0,
        maximum_position=6,
        cooldown_period=1,
        cooldown_effect="block",
        observation_schema_id=schema.schema_id,
    )
    rule2 = ObserverFixtureRuleSetDTO.create(
        fixture_id="fixture:line",
        rule_version="fixture-rule/2",
        minimum_position=0,
        maximum_position=6,
        cooldown_period=1,
        cooldown_effect="reverse",
        observation_schema_id=schema.schema_id,
    )
    return schema, rule1, rule2


def initial_state(rule: ObserverFixtureRuleSetDTO) -> ObserverFixtureStateDTO:
    return ObserverFixtureStateDTO.create(
        fixture_id=rule.fixture_id,
        rule_set_id=rule.fixture_rule_set_id,
        episode_id="episode:1",
        step_index=0,
        agent_x=1,
        target_x=6,
        cooldown_remaining=0,
    )


def action(name: str = "move_right") -> ObserverFixtureActionDTO:
    return ObserverFixtureActionDTO.create(action_name=name)


def one_entry(
    *,
    sequence: int = 0,
    previous_id: str | None = None,
    environment_rule: ObserverFixtureRuleSetDTO | None = None,
):
    schema, rule1, rule2 = schema_and_rules()
    environment_rule = environment_rule or rule1
    state = initial_state(rule1)
    move = action()
    prediction = predict_observer_fixture_transition(
        source_state=state,
        action=move,
        predictor_rule_set=rule1,
        observation_schema=schema,
    )
    executed, actual_observation = execute_observer_fixture_step(
        source_state=state,
        action=move,
        environment_rule_set=environment_rule,
        observation_schema=schema,
    )
    verification = verify_observer_transition(
        recipe=build_observer_fixture_comparison_recipe(schema),
        predicted_observation=prediction.predicted_observation,
        observed_observation=actual_observation,
        policy_artifact_id="policy:fixture",
        state_before_id=state.fixture_state_id,
        action=move.action_name,
        affected_policy_row_id="row:1",
        hidden_state_hypothesis_set=prediction.hidden_state_hypothesis_set,
        reproduction={"episode_id": state.episode_id, "step_index": sequence},
        relevant_context_keys=("hidden.cooldown_remaining",),
    )
    return ObserverTransitionLedgerEntryDTO.create(
        ledger_sequence=sequence,
        episode_id=state.episode_id,
        fixture_id=state.fixture_id,
        source_state=state,
        source_state_id=state.fixture_state_id,
        action_id=move.fixture_action_id,
        predictor_rule_set_id=rule1.fixture_rule_set_id,
        environment_rule_set_id=environment_rule.fixture_rule_set_id,
        predicted_transition=prediction,
        executed_step=executed,
        transition_verification=verification,
        previous_ledger_entry_id=previous_id,
        recorded_at_logical_step=sequence,
    )


def test_deterministic_prediction_replay_and_rule_sensitivity() -> None:
    schema, rule1, rule2 = schema_and_rules()
    state = initial_state(rule1)
    move = action()

    first = predict_observer_fixture_transition(
        source_state=state,
        action=move,
        predictor_rule_set=rule1,
        observation_schema=schema,
    )
    replay = predict_observer_fixture_transition(
        source_state=state,
        action=move,
        predictor_rule_set=rule1,
        observation_schema=schema,
    )
    switched = predict_observer_fixture_transition(
        source_state=ObserverFixtureStateDTO.create(
            fixture_id=rule2.fixture_id,
            rule_set_id=rule2.fixture_rule_set_id,
            episode_id="episode:1",
            step_index=0,
            agent_x=1,
            target_x=6,
            cooldown_remaining=0,
        ),
        action=move,
        predictor_rule_set=rule2,
        observation_schema=schema,
    )

    assert first.predicted_transition_id == replay.predicted_transition_id
    assert (
        first.predicted_state.fixture_state_id
        == replay.predicted_state.fixture_state_id
    )
    assert (
        first.predicted_observation.observation_artifact_id
        == replay.predicted_observation.observation_artifact_id
    )
    assert first.predicted_transition_id != switched.predicted_transition_id


def test_environment_rule_sensitivity_and_transition_statuses() -> None:
    schema, rule1, rule2 = schema_and_rules()
    state = ObserverFixtureStateDTO.create(
        fixture_id=rule1.fixture_id,
        rule_set_id=rule1.fixture_rule_set_id,
        episode_id="episode:1",
        step_index=1,
        agent_x=2,
        target_x=6,
        previous_action="move_right",
        cooldown_remaining=1,
    )
    move = action()
    predicted = predict_observer_fixture_transition(
        source_state=state,
        action=move,
        predictor_rule_set=rule1,
        observation_schema=schema,
    )
    same, same_observation = execute_observer_fixture_step(
        source_state=state,
        action=move,
        environment_rule_set=rule1,
        observation_schema=schema,
    )
    changed, changed_observation = execute_observer_fixture_step(
        source_state=state,
        action=move,
        environment_rule_set=rule2,
        observation_schema=schema,
    )
    confirmed = verify_observer_transition(
        recipe=build_observer_fixture_comparison_recipe(schema),
        predicted_observation=predicted.predicted_observation,
        observed_observation=same_observation,
        policy_artifact_id="policy:fixture",
        state_before_id=state.fixture_state_id,
        action=move.action_name,
        affected_policy_row_id="row:2",
        hidden_state_hypothesis_set=predicted.hidden_state_hypothesis_set,
    )
    contradicted = verify_observer_transition(
        recipe=build_observer_fixture_comparison_recipe(schema),
        predicted_observation=predicted.predicted_observation,
        observed_observation=changed_observation,
        policy_artifact_id="policy:fixture",
        state_before_id=state.fixture_state_id,
        action=move.action_name,
        affected_policy_row_id="row:2",
        hidden_state_hypothesis_set=predicted.hidden_state_hypothesis_set,
    )

    assert same.executed_step_id != changed.executed_step_id
    assert (
        same_observation.observation_artifact_id
        != changed_observation.observation_artifact_id
    )
    assert confirmed.verification_status == "confirmed"
    assert contradicted.verification_status == "contradicted"
    assert contradicted.contradiction_artifact is not None


def test_inconclusive_transition_when_hidden_evidence_missing() -> None:
    schema, rule1, _ = schema_and_rules()
    state = initial_state(rule1)
    move = action()
    prediction = predict_observer_fixture_transition(
        source_state=state,
        action=move,
        predictor_rule_set=rule1,
        observation_schema=schema,
    )
    executed, actual_observation = execute_observer_fixture_step(
        source_state=state,
        action=move,
        environment_rule_set=rule1,
        observation_schema=schema,
    )

    result = verify_observer_transition(
        recipe=build_observer_fixture_comparison_recipe(schema),
        predicted_observation=prediction.predicted_observation,
        observed_observation=actual_observation,
        policy_artifact_id="policy:fixture",
        state_before_id=state.fixture_state_id,
        action=move.action_name,
        affected_policy_row_id="row:1",
    )

    assert executed.action_effect == "moved_right"
    assert result.verification_status == "inconclusive"
    assert result.comparison_result.inconclusive_reasons == (
        "missing_hidden_state_hypothesis_set",
    )


def test_ledger_append_sequence_immutability_and_snapshot_identity() -> None:
    first = one_entry(sequence=0)
    second = one_entry(sequence=1, previous_id=first.ledger_entry_id)
    third = one_entry(sequence=2, previous_id=second.ledger_entry_id)
    ledger = InMemoryObserverTransitionLedger(
        fixture_id="fixture:line", episode_id="episode:1"
    )
    ledger.append(first)
    ledger.append(second)
    ledger.append(third)

    assert ledger.entries() == (first, second, third)
    copied = ledger.entries()
    copied += (first,)
    assert ledger.entries() == (first, second, third)
    assert ledger.verify_integrity().status == "verified"
    assert (
        ledger.snapshot().ledger_snapshot_id
        == build_observer_transition_ledger_snapshot(
            entries=(first, second, third)
        ).ledger_snapshot_id
    )
    with pytest.raises(ObserverFixtureError, match="sequence"):
        ledger.append(second)
    with pytest.raises(ObserverFixtureError, match="sequence"):
        ledger.append(one_entry(sequence=4, previous_id=third.ledger_entry_id))
    with pytest.raises(ObserverFixtureError, match="previous"):
        InMemoryObserverTransitionLedger(
            fixture_id="fixture:line", episode_id="episode:1"
        ).append(one_entry(sequence=0, previous_id="wrong"))


def test_ledger_replay_detects_tampering() -> None:
    entry = one_entry()
    snapshot = build_observer_transition_ledger_snapshot(entries=(entry,))
    result = verify_observer_transition_ledger_integrity(
        ledger_snapshot=snapshot, entries=(entry,)
    )

    assert result.status == "verified"
    with pytest.raises(ObserverFixtureError, match="source_state_id"):
        ObserverTransitionLedgerEntryDTO.create(
            ledger_sequence=entry.ledger_sequence,
            episode_id=entry.episode_id,
            fixture_id=entry.fixture_id,
            source_state=entry.source_state,
            source_state_id="state:other",
            action_id=entry.action_id,
            predictor_rule_set_id=entry.predictor_rule_set_id,
            environment_rule_set_id=entry.environment_rule_set_id,
            predicted_transition=entry.predicted_transition,
            executed_step=entry.executed_step,
            transition_verification=entry.transition_verification,
            previous_ledger_entry_id=entry.previous_ledger_entry_id,
            recorded_at_logical_step=entry.recorded_at_logical_step,
        )


def test_semantic_replay_rebuilds_fixture_predictions_and_verifications() -> None:
    schema, rule1, rule2 = schema_and_rules()
    result, entries = run_observer_fixture_episode(
        initial_state=initial_state(rule1),
        actions=(action(), action()),
        predictor_rule_set=rule1,
        environment_rule_schedule=(
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=0, rule_set_id=rule1.fixture_rule_set_id
            ),
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=1, rule_set_id=rule2.fixture_rule_set_id
            ),
        ),
        environment_rule_sets=(rule1, rule2),
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
    )

    replay = replay_observer_fixture_ledger(
        ledger_snapshot=result.ledger_snapshot,
        entries=entries,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets={rule1.fixture_rule_set_id: rule1},
        environment_rule_sets={
            rule1.fixture_rule_set_id: rule1,
            rule2.fixture_rule_set_id: rule2,
        },
    )
    missing = replay_observer_fixture_ledger(
        ledger_snapshot=result.ledger_snapshot,
        entries=entries,
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
        predictor_rule_sets={rule1.fixture_rule_set_id: rule1},
        environment_rule_sets={rule1.fixture_rule_set_id: rule1},
    )

    assert replay.status == "verified"
    assert missing.status == "failed"
    assert "missing_environment_rule_set" in missing.failure_codes


def test_episode_runner_rule_schedule_and_wake_replay() -> None:
    schema, rule1, rule2 = schema_and_rules()
    result, entries = run_observer_fixture_episode(
        initial_state=initial_state(rule1),
        actions=(action(), action(), action()),
        predictor_rule_set=rule1,
        environment_rule_schedule=(
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=0, rule_set_id=rule1.fixture_rule_set_id
            ),
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=1, rule_set_id=rule2.fixture_rule_set_id
            ),
        ),
        environment_rule_sets=(rule1, rule2),
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
    )

    always = ObserverWakePolicyDTO.create(
        policy_name="always", wake_on_every_transition=True
    )
    contradiction_only = ObserverWakePolicyDTO.create(
        policy_name="contradiction-only", wake_on_contradiction=True
    )
    conservative = ObserverWakePolicyDTO.create(
        policy_name="conservative",
        wake_on_contradiction=True,
        wake_on_inconclusive=True,
    )
    never = ObserverWakePolicyDTO.create(policy_name="never")
    always_replay = evaluate_wake_policy_over_ledger(
        ledger_snapshot=result.ledger_snapshot,
        entries=entries,
        wake_policy=always,
    )
    contradiction_replay = evaluate_wake_policy_over_ledger(
        ledger_snapshot=result.ledger_snapshot,
        entries=entries,
        wake_policy=contradiction_only,
    )
    conservative_replay = evaluate_wake_policy_over_ledger(
        ledger_snapshot=result.ledger_snapshot,
        entries=entries,
        wake_policy=conservative,
    )
    never_replay = evaluate_wake_policy_over_ledger(
        ledger_snapshot=result.ledger_snapshot,
        entries=entries,
        wake_policy=never,
    )
    ablation = build_wake_policy_ablation(
        ledger_snapshot=result.ledger_snapshot,
        wake_policy_replays=(
            always_replay,
            contradiction_replay,
            conservative_replay,
            never_replay,
        ),
        baseline_policy_id=always.wake_policy_id,
    )

    assert result.rule_change_steps == (1,)
    assert result.ledger_snapshot.entry_count == len(entries)
    assert result.contradicted_entry_ids
    assert always_replay.wake_count == len(entries)
    assert contradiction_replay.wake_count == len(result.contradicted_entry_ids)
    assert conservative_replay.wake_count == contradiction_replay.wake_count
    assert never_replay.missed_contradiction_entry_ids == result.contradicted_entry_ids
    assert never.wake_policy_id in ablation.policy_ids_missing_contradictions


def test_terminal_state_prevents_same_episode_append_and_public_api() -> None:
    schema, rule1, _ = schema_and_rules()
    terminal_state = ObserverFixtureStateDTO.create(
        fixture_id=rule1.fixture_id,
        rule_set_id=rule1.fixture_rule_set_id,
        episode_id="episode:1",
        step_index=0,
        agent_x=5,
        target_x=6,
    )
    result, entries = run_observer_fixture_episode(
        initial_state=terminal_state,
        actions=(action(), action()),
        predictor_rule_set=rule1,
        environment_rule_schedule=(
            ObserverFixtureRuleScheduleEntryDTO.create(
                start_step=0, rule_set_id=rule1.fixture_rule_set_id
            ),
        ),
        environment_rule_sets=(rule1,),
        observation_schema=schema,
        comparison_recipe=build_observer_fixture_comparison_recipe(schema),
    )
    ledger = InMemoryObserverTransitionLedger(
        fixture_id="fixture:line", episode_id="episode:1"
    )
    first = entries[0]
    ledger.append(first)
    with pytest.raises(ObserverFixtureError, match="terminal"):
        ledger.append(one_entry(sequence=1, previous_id=first.ledger_entry_id))

    import zeromodel.observer as observer

    assert result.final_state_id == first.executed_step.actual_state.fixture_state_id
    assert "ObserverTransitionLedgerEntryDTO" in observer.__all__
    assert "run_observer_fixture_episode" in observer.__all__


def test_canonical_mutations_change_ids() -> None:
    schema, rule1, rule2 = schema_and_rules()
    state = initial_state(rule1)
    move = action()
    other_action = action("wait")
    prediction = predict_observer_fixture_transition(
        source_state=state,
        action=move,
        predictor_rule_set=rule1,
        observation_schema=schema,
    )
    other_prediction = predict_observer_fixture_transition(
        source_state=state,
        action=other_action,
        predictor_rule_set=rule1,
        observation_schema=schema,
    )
    entry = one_entry()
    other_entry = one_entry(environment_rule=rule2)
    policy = ObserverWakePolicyDTO.create(policy_name="a", wake_on_contradiction=True)
    other_policy = ObserverWakePolicyDTO.create(
        policy_name="a", wake_on_inconclusive=True
    )

    assert rule1.fixture_rule_set_id != rule2.fixture_rule_set_id
    assert move.fixture_action_id != other_action.fixture_action_id
    assert (
        prediction.predicted_transition_id != other_prediction.predicted_transition_id
    )
    assert entry.ledger_entry_id != other_entry.ledger_entry_id
    assert policy.wake_policy_id != other_policy.wake_policy_id
