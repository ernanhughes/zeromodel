"""Inactive shadow replay and audit for Observer habit specifications."""

from __future__ import annotations

from zeromodel.observer._observation_replay import (
    source_observation_for_entry,
    target_observation_for_entry,
)
from zeromodel.observer.artifacts import ObserverObservationSchemaDTO
from zeromodel.observer.comparison import ObserverComparisonRecipeDTO
from zeromodel.observer.fixture import (
    FIXTURE_ACTIONS,
    ObserverFixtureActionDTO,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
)
from zeromodel.observer.fixture_runtime import (
    ObserverFixtureRuleScheduleEntryDTO,
    active_rule_for_step,
    run_observer_fixture_episode,
)
from zeromodel.observer.graph import ObserverObservationGraphBuildDTO
from zeromodel.observer.grouping import (
    ObserverStateGroupingRecipeDTO,
    assign_observation_to_state_class,
)
from zeromodel.observer.habit import (
    ObserverHabitCounterexampleCoverageDTO,
    ObserverHabitCounterexampleDTO,
    ObserverHabitError,
    ObserverHabitShadowAuditDTO,
    ObserverHabitShadowAuditRecipeDTO,
    ObserverHabitShadowEpisodeDTO,
    ObserverHabitShadowOccurrenceDTO,
    ObserverHabitShadowReplayDTO,
    ObserverHabitSpecificationDTO,
)
from zeromodel.observer.habit_service import evaluate_observer_habit
from zeromodel.observer.ledger import (
    ObserverTransitionLedgerEntryDTO,
    ObserverTransitionLedgerSnapshotDTO,
)


def evaluate_observer_habit_over_ledger(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    graph_build: ObserverObservationGraphBuildDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverHabitShadowReplayDTO:
    """Evaluate an inactive habit over immutable historical source observations."""

    lineage_failures = _replay_lineage_failures(
        habit_specification=habit_specification,
        ledger_snapshot=ledger_snapshot,
        graph_build=graph_build,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
    )
    if lineage_failures:
        return ObserverHabitShadowReplayDTO.create(
            habit_specification_id=habit_specification.habit_specification_id,
            ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
            shadow_occurrences=(),
            episode_ids=tuple(sorted({entry.episode_id for entry in entries})),
            status="failed",
            failure_codes=lineage_failures,
        )
    occurrences: list[ObserverHabitShadowOccurrenceDTO] = []
    previous_target = None
    previous_effect = None
    assignment_by_observation = {
        item.observation_artifact_id: item
        for item in graph_build.assignments
        if item.status == "assigned"
    }
    for entry in entries:
        source = source_observation_for_entry(
            entry=entry,
            observation_schema=observation_schema,
            previous_target_observation=previous_target,
            previous_target_action_effect=previous_effect,
        )
        target = target_observation_for_entry(
            entry=entry, observation_schema=observation_schema
        )
        previous_target = target
        previous_effect = None if target is None else entry.executed_step.action_effect
        authoritative_action = _action_name_for_id(entry.action_id)
        if source is None:
            occurrences.append(
                _invalid_shadow_occurrence(
                    habit_specification=habit_specification,
                    entry=entry,
                    authoritative_action=authoritative_action,
                    source_observation_artifact_id=f"unavailable:{entry.ledger_entry_id}:source",
                    reason_code="source_observation_unavailable",
                )
            )
            continue
        if target is None:
            occurrences.append(
                _invalid_shadow_occurrence(
                    habit_specification=habit_specification,
                    entry=entry,
                    authoritative_action=authoritative_action,
                    source_observation_artifact_id=source.observation_artifact_id,
                    reason_code="target_observation_unavailable",
                )
            )
            continue
        evaluation = evaluate_observer_habit(
            habit_specification=habit_specification,
            observation=source,
            grouping_recipe=grouping_recipe,
            observation_schema=observation_schema,
        )
        source_assignment = assignment_by_observation.get(
            source.observation_artifact_id
        )
        if source_assignment is None:
            source_assignment = assign_observation_to_state_class(
                observation=source,
                grouping_recipe=grouping_recipe,
                observation_schema=observation_schema,
            )
        target_assignment = assignment_by_observation.get(
            target.observation_artifact_id
        )
        if target_assignment is None:
            target_assignment = assign_observation_to_state_class(
                observation=target,
                grouping_recipe=grouping_recipe,
                observation_schema=observation_schema,
            )
        source_class_id = source_assignment.state_class_id or "unassigned"
        target_class_id = target_assignment.state_class_id or "unassigned"
        outcome, reasons = _classify_shadow_outcome(
            habit_specification=habit_specification,
            habit_decision=evaluation.decision,
            recommended_action=evaluation.recommended_action,
            source_class_id=source_assignment.state_class_id,
            authoritative_action=authoritative_action,
            actual_target_state_class_id=target_class_id,
        )
        occurrences.append(
            ObserverHabitShadowOccurrenceDTO.create(
                habit_specification_id=habit_specification.habit_specification_id,
                ledger_entry_id=entry.ledger_entry_id,
                source_observation_artifact_id=source.observation_artifact_id,
                source_state_class_id=source_class_id,
                habit_evaluation_id=evaluation.habit_evaluation_id,
                habit_decision=evaluation.decision,
                habit_recommended_action=evaluation.recommended_action,
                authoritative_action=authoritative_action,
                actual_target_state_class_id=target_class_id,
                expected_target_state_class_id=(
                    habit_specification.expected_target_state_class_id
                ),
                outcome=outcome,
                reason_codes=reasons,
            )
        )
    wrong = any(
        item.outcome in {"wrong_action_fire", "wrong_target_fire"}
        for item in occurrences
    )
    invalid = any(item.outcome == "invalid_evaluation" for item in occurrences)
    return ObserverHabitShadowReplayDTO.create(
        habit_specification_id=habit_specification.habit_specification_id,
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        shadow_occurrences=tuple(occurrences),
        episode_ids=tuple(sorted({entry.episode_id for entry in entries})),
        status="failed" if wrong else "inconclusive" if invalid else "verified",
        failure_codes=(
            ("false_fire_detected",)
            if wrong
            else ("invalid_evaluation",)
            if invalid
            else ()
        ),
    )


def run_observer_fixture_habit_shadow_episode(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    initial_state: ObserverFixtureStateDTO,
    actions: tuple[ObserverFixtureActionDTO, ...],
    predictor_rule_set: ObserverFixtureRuleSetDTO,
    environment_rule_schedule: tuple[ObserverFixtureRuleScheduleEntryDTO, ...],
    environment_rule_sets: tuple[ObserverFixtureRuleSetDTO, ...],
    observation_schema: ObserverObservationSchemaDTO,
    comparison_recipe: ObserverComparisonRecipeDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    graph_build: ObserverObservationGraphBuildDTO,
    supply_hidden_evidence: bool = True,
) -> ObserverHabitShadowEpisodeDTO:
    """Run a fixture episode while the habit only observes and records shadow output."""

    episode, entries = run_observer_fixture_episode(
        initial_state=initial_state,
        actions=actions,
        predictor_rule_set=predictor_rule_set,
        environment_rule_schedule=environment_rule_schedule,
        environment_rule_sets=environment_rule_sets,
        observation_schema=observation_schema,
        comparison_recipe=comparison_recipe,
        supply_hidden_evidence=supply_hidden_evidence,
    )
    _ = active_rule_for_step
    replay = evaluate_observer_habit_over_ledger(
        habit_specification=habit_specification,
        ledger_snapshot=episode.ledger_snapshot,
        entries=entries,
        graph_build=graph_build,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
    )
    sequence_by_entry_id = {
        entry.ledger_entry_id: entry.ledger_sequence for entry in entries
    }
    return ObserverHabitShadowEpisodeDTO.create(
        habit_specification_id=habit_specification.habit_specification_id,
        fixture_episode_result_id=episode.episode_result_id,
        ledger_snapshot_id=episode.ledger_snapshot.ledger_snapshot_id,
        shadow_replay=replay,
        authoritative_action_ids=tuple(entry.action_id for entry in entries),
        habit_fire_sequences=tuple(
            sequence_by_entry_id[item.ledger_entry_id]
            for item in replay.shadow_occurrences
            if item.habit_decision == "fire"
            and item.ledger_entry_id in sequence_by_entry_id
        ),
        habit_abstain_sequences=tuple(
            sequence_by_entry_id[item.ledger_entry_id]
            for item in replay.shadow_occurrences
            if item.habit_decision == "abstain"
            and item.ledger_entry_id in sequence_by_entry_id
        ),
        status="shadow_recorded",
    )


def audit_observer_habit_shadow(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    shadow_audit_recipe: ObserverHabitShadowAuditRecipeDTO,
    historical_shadow_replay: ObserverHabitShadowReplayDTO,
    live_shadow_episodes: tuple[ObserverHabitShadowEpisodeDTO, ...] = (),
    counterexamples: tuple[ObserverHabitCounterexampleDTO, ...] = (),
) -> ObserverHabitShadowAuditDTO:
    """Apply explicit threshold checks to shadow evidence."""

    coverage = _coverage(
        habit_specification=habit_specification,
        counterexamples=counterexamples,
    )
    all_replays = _validated_audit_replays(
        habit_specification=habit_specification,
        historical_shadow_replay=historical_shadow_replay,
        live_shadow_episodes=live_shadow_episodes,
    )
    applicable_count = sum(replay.applicable_count for replay in all_replays)
    correct_fire_count = sum(replay.correct_fire_count for replay in all_replays)
    wrong_action_fire_count = sum(
        replay.wrong_action_fire_count for replay in all_replays
    )
    wrong_target_fire_count = sum(
        replay.wrong_target_fire_count for replay in all_replays
    )
    missed_opportunity_count = sum(
        replay.missed_opportunity_count for replay in all_replays
    )
    invalid_count = sum(replay.invalid_count for replay in all_replays)
    episode_ids = tuple(
        sorted(
            {episode_id for replay in all_replays for episode_id in replay.episode_ids}
        )
    )
    reasons: set[str] = set()
    if applicable_count < shadow_audit_recipe.minimum_applicable_count:
        reasons.add("minimum_applicable_count_not_met")
    if len(episode_ids) < shadow_audit_recipe.minimum_episode_count:
        reasons.add("minimum_episode_count_not_met")
    if correct_fire_count < shadow_audit_recipe.minimum_correct_fire_count:
        reasons.add("minimum_correct_fire_count_not_met")
    false_fires = wrong_action_fire_count + wrong_target_fire_count
    if shadow_audit_recipe.require_zero_false_fires and false_fires:
        reasons.add("false_fire_detected")
    if wrong_action_fire_count > shadow_audit_recipe.maximum_wrong_action_fire_count:
        reasons.add("wrong_action_fire_limit_exceeded")
    if wrong_target_fire_count > shadow_audit_recipe.maximum_wrong_target_fire_count:
        reasons.add("wrong_target_fire_limit_exceeded")
    if missed_opportunity_count > shadow_audit_recipe.maximum_missed_opportunity_count:
        reasons.add("missed_opportunity_limit_exceeded")
    if invalid_count > shadow_audit_recipe.maximum_invalid_count:
        reasons.add("invalid_limit_exceeded")
    if (
        shadow_audit_recipe.require_counterexample_coverage
        and not coverage.coverage_complete
    ):
        reasons.add("counterexample_coverage_incomplete")
    disposition = _audit_disposition(reasons)
    return ObserverHabitShadowAuditDTO.create(
        habit_specification_id=habit_specification.habit_specification_id,
        shadow_audit_recipe_id=shadow_audit_recipe.shadow_audit_recipe_id,
        historical_shadow_replay_id=historical_shadow_replay.habit_shadow_replay_id,
        evaluated_shadow_replay_ids=tuple(
            replay.habit_shadow_replay_id for replay in all_replays
        ),
        live_shadow_episode_ids=tuple(
            item.habit_shadow_episode_id for item in live_shadow_episodes
        ),
        counterexample_coverage=coverage,
        disposition=disposition,
        reason_codes=tuple(sorted(reasons or {"shadow_thresholds_met"})),
    )


def _validated_audit_replays(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    historical_shadow_replay: ObserverHabitShadowReplayDTO,
    live_shadow_episodes: tuple[ObserverHabitShadowEpisodeDTO, ...],
) -> tuple[ObserverHabitShadowReplayDTO, ...]:
    habit_id = habit_specification.habit_specification_id
    if historical_shadow_replay.habit_specification_id != habit_id:
        raise ObserverHabitError("historical shadow replay habit mismatch")

    replays_by_id = {
        historical_shadow_replay.habit_shadow_replay_id: historical_shadow_replay
    }
    for episode in live_shadow_episodes:
        if episode.habit_specification_id != habit_id:
            raise ObserverHabitError("live shadow episode habit mismatch")
        replay = episode.shadow_replay
        if replay.habit_specification_id != habit_id:
            raise ObserverHabitError("live shadow replay habit mismatch")
        if replay.ledger_snapshot_id != episode.ledger_snapshot_id:
            raise ObserverHabitError("live shadow episode replay snapshot mismatch")
        if replay.habit_shadow_replay_id in replays_by_id:
            raise ObserverHabitError("duplicate shadow replay evidence")
        replays_by_id[replay.habit_shadow_replay_id] = replay
    return tuple(replays_by_id[key] for key in sorted(replays_by_id))


def _classify_shadow_outcome(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    habit_decision: str,
    recommended_action: str | None,
    source_class_id: str | None,
    authoritative_action: str,
    actual_target_state_class_id: str,
) -> tuple[str, tuple[str, ...]]:
    opportunity = (
        source_class_id == habit_specification.source_state_class_id
        and authoritative_action == habit_specification.recommended_action
        and actual_target_state_class_id
        == habit_specification.expected_target_state_class_id
    )
    if source_class_id != habit_specification.source_state_class_id:
        return "not_applicable", ("source_class_mismatch",)
    if habit_decision == "invalid":
        return "invalid_evaluation", ("invalid_evaluation",)
    if habit_decision == "fire":
        if recommended_action != authoritative_action:
            return "wrong_action_fire", ("authoritative_action_disagreed",)
        if (
            actual_target_state_class_id
            != habit_specification.expected_target_state_class_id
        ):
            return "wrong_target_fire", ("expected_target_not_observed",)
        return "correct_fire", ("authoritative_action_and_target_matched",)
    if opportunity:
        return "missed_opportunity", ("guards_abstained_on_matching_pattern",)
    return "safe_abstention", ("guards_abstained_outside_supported_pattern",)


def _replay_lineage_failures(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    graph_build: ObserverObservationGraphBuildDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> tuple[str, ...]:
    failures: set[str] = set()
    graph = graph_build.graph
    if habit_specification.ledger_snapshot_id != ledger_snapshot.ledger_snapshot_id:
        failures.add("habit_ledger_mismatch")
    if graph is None:
        failures.add("habit_graph_mismatch")
    else:
        if habit_specification.observation_graph_id != graph.observation_graph_id:
            failures.add("habit_graph_mismatch")
        if graph.ledger_snapshot_id != ledger_snapshot.ledger_snapshot_id:
            failures.add("graph_ledger_mismatch")
    if habit_specification.grouping_recipe_id != grouping_recipe.grouping_recipe_id:
        failures.add("habit_grouping_mismatch")
    if habit_specification.observation_schema_id != observation_schema.schema_id:
        failures.add("habit_schema_mismatch")
    return tuple(sorted(failures))


def _invalid_shadow_occurrence(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    entry: ObserverTransitionLedgerEntryDTO,
    authoritative_action: str,
    source_observation_artifact_id: str,
    reason_code: str,
) -> ObserverHabitShadowOccurrenceDTO:
    return ObserverHabitShadowOccurrenceDTO.create(
        habit_specification_id=habit_specification.habit_specification_id,
        ledger_entry_id=entry.ledger_entry_id,
        source_observation_artifact_id=source_observation_artifact_id,
        source_state_class_id="unassigned",
        habit_evaluation_id=f"invalid:{entry.ledger_entry_id}",
        habit_decision="invalid",
        habit_recommended_action=None,
        authoritative_action=authoritative_action,
        actual_target_state_class_id="unassigned",
        expected_target_state_class_id=habit_specification.expected_target_state_class_id,
        outcome="invalid_evaluation",
        reason_codes=(reason_code,),
    )


def _action_name_for_id(action_id: str) -> str:
    for action_name in sorted(FIXTURE_ACTIONS):
        action = ObserverFixtureActionDTO.create(action_name=action_name)
        if action.fixture_action_id == action_id:
            return action.action_name
    return action_id


def _coverage(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    counterexamples: tuple[ObserverHabitCounterexampleDTO, ...],
) -> ObserverHabitCounterexampleCoverageDTO:
    relevant = tuple(
        item
        for item in counterexamples
        if item.habit_specification_id == habit_specification.habit_specification_id
        and item.transition_key_id == habit_specification.transition_key_id
        and item.expected_target_state_class_id
        == habit_specification.expected_target_state_class_id
    )
    counterexample_ids = tuple(item.counterexample_id for item in relevant)
    guarded = tuple(
        item.counterexample_id
        for item in relevant
        if any(
            item.counterexample_id in guard.source_evidence_ids
            for guard in habit_specification.counterexample_guards
        )
    )
    unguarded = tuple(sorted(set(counterexample_ids) - set(guarded)))
    return ObserverHabitCounterexampleCoverageDTO.create(
        habit_specification_id=habit_specification.habit_specification_id,
        counterexample_ids=counterexample_ids,
        guarded_counterexample_ids=guarded,
        unguarded_counterexample_ids=unguarded,
    )


def _audit_disposition(reasons: set[str]) -> str:
    if not reasons:
        return "eligible_for_admission_review"
    if (
        "false_fire_detected" in reasons
        or "wrong_action_fire_limit_exceeded" in reasons
    ):
        return "false_fire_detected"
    if "wrong_target_fire_limit_exceeded" in reasons:
        return "target_instability_detected"
    if "missed_opportunity_limit_exceeded" in reasons:
        return "too_many_missed_opportunities"
    if "invalid_limit_exceeded" in reasons:
        return "invalid_evidence"
    if "counterexample_coverage_incomplete" in reasons:
        return "counterexample_coverage_incomplete"
    if any("minimum" in reason for reason in reasons):
        return "insufficient_shadow_evidence"
    return "unsupported"
