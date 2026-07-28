"""Bounded fixture runtime for active Observer habits with fallback."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import ObserverObservationSchemaDTO
from zeromodel.observer.comparison import ObserverComparisonRecipeDTO
from zeromodel.observer.fixture import (
    ObserverFixtureActionDTO,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
)
from zeromodel.observer.fixture_predictor import (
    _observation_for_state,
    execute_observer_fixture_step,
    predict_observer_fixture_transition,
)
from zeromodel.observer.fixture_runtime import (
    ObserverFixtureEpisodeResultDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    active_rule_for_step,
)
from zeromodel.observer.grouping import (
    ObserverStateGroupingRecipeDTO,
    assign_observation_to_state_class,
)
from zeromodel.observer.habit import ObserverHabitError, ObserverHabitSpecificationDTO
from zeromodel.observer.habit_activation import (
    ObserverActiveHabitDecisionDTO,
    ObserverHabitActivationScopeDTO,
    select_observer_active_action,
)
from zeromodel.observer.habit_registry import InMemoryObserverHabitRegistry
from zeromodel.observer.ledger import (
    InMemoryObserverTransitionLedger,
    ObserverTransitionLedgerEntryDTO,
)
from zeromodel.observer.transition_service import verify_observer_transition

OBSERVER_HABIT_RUNTIME_SAFETY_RECIPE_VERSION: Final = (
    "observer-habit-runtime-safety-recipe/1"
)
OBSERVER_ACTIVE_HABIT_OCCURRENCE_VERSION: Final = "observer-active-habit-occurrence/1"
OBSERVER_ACTIVE_HABIT_EXECUTION_REPORT_VERSION: Final = (
    "observer-active-habit-execution-report/1"
)

ACTIVE_OCCURRENCE_OUTCOMES: Final = frozenset(
    {
        "habit_success",
        "habit_wrong_target",
        "habit_contradicted",
        "habit_inconclusive",
        "fallback_executed",
        "fallback_after_invalid_habit",
        "fallback_after_abstention",
        "fallback_after_ambiguity",
    }
)
ACTIVE_REPORT_STATUSES: Final = frozenset(
    {"completed", "completed_with_suspension", "failed", "inconclusive"}
)


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverHabitError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverHabitError(f"{field_name} must be unique and sorted")


@dataclass(frozen=True)
class ObserverHabitRuntimeSafetyRecipeDTO:
    habit_runtime_safety_recipe_id: str
    suspend_on_wrong_target: bool
    suspend_on_contradiction: bool
    suspend_on_inconclusive: bool
    suspend_on_invalid_evaluation: bool
    maximum_consecutive_habit_failures: int
    maximum_total_habit_failures: int
    maximum_consecutive_fallbacks: int | None
    version: str = OBSERVER_HABIT_RUNTIME_SAFETY_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_RUNTIME_SAFETY_RECIPE_VERSION:
            raise ObserverHabitError("unsupported runtime safety recipe version")
        if self.maximum_consecutive_habit_failures < 1:
            raise ObserverHabitError(
                "maximum_consecutive_habit_failures must be positive"
            )
        if self.maximum_total_habit_failures < 1:
            raise ObserverHabitError("maximum_total_habit_failures must be positive")
        if (
            self.maximum_consecutive_fallbacks is not None
            and self.maximum_consecutive_fallbacks < 0
        ):
            raise ObserverHabitError(
                "maximum_consecutive_fallbacks must be non-negative"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_runtime_safety_recipe_id != expected_id:
            raise ObserverHabitError("habit_runtime_safety_recipe_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "maximum_consecutive_fallbacks": self.maximum_consecutive_fallbacks,
            "maximum_consecutive_habit_failures": self.maximum_consecutive_habit_failures,
            "maximum_total_habit_failures": self.maximum_total_habit_failures,
            "suspend_on_contradiction": self.suspend_on_contradiction,
            "suspend_on_inconclusive": self.suspend_on_inconclusive,
            "suspend_on_invalid_evaluation": self.suspend_on_invalid_evaluation,
            "suspend_on_wrong_target": self.suspend_on_wrong_target,
            "version": self.version,
        }
        if include_id:
            payload["habit_runtime_safety_recipe_id"] = (
                self.habit_runtime_safety_recipe_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitRuntimeSafetyRecipeDTO":
        payload = {**values, "version": OBSERVER_HABIT_RUNTIME_SAFETY_RECIPE_VERSION}
        return cls(
            habit_runtime_safety_recipe_id=canonical_id(payload),
            version=OBSERVER_HABIT_RUNTIME_SAFETY_RECIPE_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverActiveHabitOccurrenceDTO:
    active_habit_occurrence_id: str
    registry_snapshot_id: str
    habit_specification_id: str | None
    active_habit_decision_id: str
    ledger_entry_id: str
    source_observation_artifact_id: str
    selected_action: str
    authoritative_fallback_action: str
    decision_source: str
    actual_target_state_class_id: str
    expected_target_state_class_id: str | None
    verification_status: str
    outcome: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_ACTIVE_HABIT_OCCURRENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_ACTIVE_HABIT_OCCURRENCE_VERSION:
            raise ObserverHabitError("unsupported active habit occurrence version")
        for field_name in (
            "registry_snapshot_id",
            "active_habit_decision_id",
            "ledger_entry_id",
            "source_observation_artifact_id",
            "selected_action",
            "authoritative_fallback_action",
            "decision_source",
            "actual_target_state_class_id",
            "verification_status",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.outcome not in ACTIVE_OCCURRENCE_OUTCOMES:
            raise ObserverHabitError("unsupported active habit outcome")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.active_habit_occurrence_id != expected_id:
            raise ObserverHabitError("active_habit_occurrence_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "active_habit_decision_id": self.active_habit_decision_id,
            "actual_target_state_class_id": self.actual_target_state_class_id,
            "authoritative_fallback_action": self.authoritative_fallback_action,
            "decision_source": self.decision_source,
            "expected_target_state_class_id": self.expected_target_state_class_id,
            "habit_specification_id": self.habit_specification_id,
            "ledger_entry_id": self.ledger_entry_id,
            "outcome": self.outcome,
            "reason_codes": list(self.reason_codes),
            "registry_snapshot_id": self.registry_snapshot_id,
            "selected_action": self.selected_action,
            "source_observation_artifact_id": self.source_observation_artifact_id,
            "verification_status": self.verification_status,
            "version": self.version,
        }
        if include_id:
            payload["active_habit_occurrence_id"] = self.active_habit_occurrence_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverActiveHabitOccurrenceDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        payload = {
            **values,
            "reason_codes": list(cast(tuple[str, ...], values["reason_codes"])),
            "version": OBSERVER_ACTIVE_HABIT_OCCURRENCE_VERSION,
        }
        return cls(
            active_habit_occurrence_id=canonical_id(payload),
            version=OBSERVER_ACTIVE_HABIT_OCCURRENCE_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverActiveHabitExecutionReportDTO:
    active_execution_report_id: str
    fixture_episode_result_id: str
    initial_registry_snapshot_id: str
    final_registry_snapshot_id: str
    active_occurrences: tuple[ObserverActiveHabitOccurrenceDTO, ...]
    habit_execution_count: int
    fallback_execution_count: int
    habit_success_count: int
    habit_failure_count: int
    suspension_event_ids: tuple[str, ...]
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_ACTIVE_HABIT_EXECUTION_REPORT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_ACTIVE_HABIT_EXECUTION_REPORT_VERSION:
            raise ObserverHabitError("unsupported active execution report version")
        if self.status not in ACTIVE_REPORT_STATUSES:
            raise ObserverHabitError("unsupported active execution report status")
        if self.habit_execution_count != sum(
            1 for item in self.active_occurrences if item.decision_source == "habit"
        ):
            raise ObserverHabitError("habit_execution_count mismatch")
        if self.fallback_execution_count != sum(
            1
            for item in self.active_occurrences
            if item.decision_source == "authoritative_fallback"
        ):
            raise ObserverHabitError("fallback_execution_count mismatch")
        if self.habit_success_count != sum(
            1 for item in self.active_occurrences if item.outcome == "habit_success"
        ):
            raise ObserverHabitError("habit_success_count mismatch")
        if self.habit_failure_count != sum(
            1
            for item in self.active_occurrences
            if item.outcome
            in {"habit_wrong_target", "habit_contradicted", "habit_inconclusive"}
        ):
            raise ObserverHabitError("habit_failure_count mismatch")
        _ensure_sorted_unique(self.suspension_event_ids, "suspension_event_ids")
        _ensure_sorted_unique(self.failure_codes, "failure_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.active_execution_report_id != expected_id:
            raise ObserverHabitError("active_execution_report_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "active_occurrences": [
                item.canonical_payload() for item in self.active_occurrences
            ],
            "failure_codes": list(self.failure_codes),
            "fallback_execution_count": self.fallback_execution_count,
            "final_registry_snapshot_id": self.final_registry_snapshot_id,
            "fixture_episode_result_id": self.fixture_episode_result_id,
            "habit_execution_count": self.habit_execution_count,
            "habit_failure_count": self.habit_failure_count,
            "habit_success_count": self.habit_success_count,
            "initial_registry_snapshot_id": self.initial_registry_snapshot_id,
            "status": self.status,
            "suspension_event_ids": list(self.suspension_event_ids),
            "version": self.version,
        }
        if include_id:
            payload["active_execution_report_id"] = self.active_execution_report_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverActiveHabitExecutionReportDTO":
        occurrences = cast(
            tuple[ObserverActiveHabitOccurrenceDTO, ...], values["active_occurrences"]
        )
        values["habit_execution_count"] = sum(
            1 for item in occurrences if item.decision_source == "habit"
        )
        values["fallback_execution_count"] = sum(
            1
            for item in occurrences
            if item.decision_source == "authoritative_fallback"
        )
        values["habit_success_count"] = sum(
            1 for item in occurrences if item.outcome == "habit_success"
        )
        values["habit_failure_count"] = sum(
            1
            for item in occurrences
            if item.outcome
            in {"habit_wrong_target", "habit_contradicted", "habit_inconclusive"}
        )
        for key in ("suspension_event_ids", "failure_codes"):
            values[key] = tuple(sorted(set(cast(Sequence[str], values.get(key, ())))))
        payload = {
            **values,
            "active_occurrences": [item.canonical_payload() for item in occurrences],
            "failure_codes": list(cast(tuple[str, ...], values["failure_codes"])),
            "suspension_event_ids": list(
                cast(tuple[str, ...], values["suspension_event_ids"])
            ),
            "version": OBSERVER_ACTIVE_HABIT_EXECUTION_REPORT_VERSION,
        }
        return cls(
            active_execution_report_id=canonical_id(payload),
            version=OBSERVER_ACTIVE_HABIT_EXECUTION_REPORT_VERSION,
            **values,  # type: ignore[arg-type]
        )


def run_observer_fixture_with_active_habit(
    *,
    registry: InMemoryObserverHabitRegistry,
    activation_scope: ObserverHabitActivationScopeDTO,
    habits: tuple[ObserverHabitSpecificationDTO, ...],
    initial_state: ObserverFixtureStateDTO,
    authoritative_actions: tuple[ObserverFixtureActionDTO, ...],
    predictor_rule_set: ObserverFixtureRuleSetDTO,
    environment_rule_schedule: tuple[ObserverFixtureRuleScheduleEntryDTO, ...],
    environment_rule_sets: tuple[ObserverFixtureRuleSetDTO, ...],
    observation_schema: ObserverObservationSchemaDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    comparison_recipe: ObserverComparisonRecipeDTO,
    runtime_safety_recipe: ObserverHabitRuntimeSafetyRecipeDTO,
    policy_artifact_id: str = "observer-fixture-policy",
) -> tuple[
    ObserverFixtureEpisodeResultDTO,
    tuple[ObserverTransitionLedgerEntryDTO, ...],
    ObserverActiveHabitExecutionReportDTO,
]:
    rule_map = {rule.fixture_rule_set_id: rule for rule in environment_rule_sets}
    ledger = InMemoryObserverTransitionLedger(
        fixture_id=initial_state.fixture_id, episode_id=initial_state.episode_id
    )
    state = initial_state
    occurrences: list[ObserverActiveHabitOccurrenceDTO] = []
    suspension_events: list[str] = []
    consecutive_failures = 0
    total_failures = 0
    consecutive_fallbacks = 0
    initial_registry_snapshot_id = (
        registry.current_snapshot().habit_registry_snapshot_id
    )
    for sequence, authoritative_action in enumerate(authoritative_actions):
        if state.terminal:
            break
        environment_rule_set = active_rule_for_step(
            step_index=sequence,
            schedule=environment_rule_schedule,
            rule_sets=rule_map,
        )
        source_observation = _observation_for_state(
            state=state,
            action_effect="initial" if state.step_index == 0 else "source",
            observation_schema=observation_schema,
        )
        snapshot_before = registry.current_snapshot()
        decision = select_observer_active_action(
            registry_snapshot=snapshot_before,
            activation_scope=activation_scope,
            observation=source_observation,
            authoritative_action=authoritative_action,
            habits=habits,
            grouping_recipe=grouping_recipe,
            observation_schema=observation_schema,
        )
        selected_action = ObserverFixtureActionDTO.create(
            action_name=decision.selected_action
        )
        prediction = predict_observer_fixture_transition(
            source_state=state,
            action=selected_action,
            predictor_rule_set=predictor_rule_set,
            observation_schema=observation_schema,
        )
        executed, actual_observation = execute_observer_fixture_step(
            source_state=state,
            action=selected_action,
            environment_rule_set=environment_rule_set,
            observation_schema=observation_schema,
        )
        verification = verify_observer_transition(
            recipe=comparison_recipe,
            predicted_observation=prediction.predicted_observation,
            observed_observation=actual_observation,
            policy_artifact_id=policy_artifact_id,
            state_before_id=state.fixture_state_id,
            action=selected_action.action_name,
            affected_policy_row_id=f"row:{state.agent_x}",
            hidden_state_hypothesis_set=prediction.hidden_state_hypothesis_set,
            reproduction={
                "episode_id": state.episode_id,
                "fixture_id": state.fixture_id,
                "step_index": sequence,
            },
            relevant_context_keys=("hidden.cooldown_remaining",),
        )
        previous_entry = ledger.head()
        entry = ObserverTransitionLedgerEntryDTO.create(
            ledger_sequence=sequence,
            episode_id=state.episode_id,
            fixture_id=state.fixture_id,
            source_state=state,
            source_state_id=state.fixture_state_id,
            action_id=selected_action.fixture_action_id,
            predictor_rule_set_id=predictor_rule_set.fixture_rule_set_id,
            environment_rule_set_id=environment_rule_set.fixture_rule_set_id,
            predicted_transition=prediction,
            executed_step=executed,
            transition_verification=verification,
            previous_ledger_entry_id=(
                None if previous_entry is None else previous_entry.ledger_entry_id
            ),
            recorded_at_logical_step=sequence,
        )
        ledger.append(entry)
        occurrence = _active_occurrence(
            decision=decision,
            ledger_entry=entry,
            actual_observation=actual_observation,
            grouping_recipe=grouping_recipe,
            observation_schema=observation_schema,
            habits=habits,
        )
        occurrences.append(occurrence)
        should_suspend = False
        if occurrence.decision_source == "habit":
            failed = occurrence.outcome != "habit_success"
            consecutive_fallbacks = 0
            if failed:
                total_failures += 1
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            should_suspend = _should_suspend(
                occurrence=occurrence,
                safety=runtime_safety_recipe,
                consecutive_failures=consecutive_failures,
                total_failures=total_failures,
            )
        else:
            consecutive_failures = 0
            consecutive_fallbacks += 1
            should_suspend = (
                runtime_safety_recipe.suspend_on_invalid_evaluation
                and "invalid_active_habit" in occurrence.reason_codes
            )
            if (
                runtime_safety_recipe.maximum_consecutive_fallbacks is not None
                and consecutive_fallbacks
                > runtime_safety_recipe.maximum_consecutive_fallbacks
            ):
                should_suspend = True
        if should_suspend and decision.habit_specification_id is not None:
            event = registry.suspend(
                habit_specification_id=decision.habit_specification_id,
                reason_codes=("runtime_safety_suspension", occurrence.outcome),
            )
            suspension_events.append(event.habit_registry_event_id)
        state = executed.actual_state
    entries = ledger.entries()
    confirmed = tuple(
        entry.ledger_entry_id
        for entry in entries
        if entry.transition_verification.verification_status == "confirmed"
    )
    contradicted = tuple(
        entry.ledger_entry_id
        for entry in entries
        if entry.transition_verification.verification_status == "contradicted"
    )
    inconclusive = tuple(
        entry.ledger_entry_id
        for entry in entries
        if entry.transition_verification.verification_status == "inconclusive"
    )
    snapshot = ledger.snapshot()
    episode_payload = {
        "confirmed_entry_ids": list(confirmed),
        "contradicted_entry_ids": list(contradicted),
        "episode_id": initial_state.episode_id,
        "final_state_id": state.fixture_state_id,
        "inconclusive_entry_ids": list(inconclusive),
        "initial_state_id": initial_state.fixture_state_id,
        "ledger_snapshot": snapshot.canonical_payload(),
        "rule_change_steps": [
            item.start_step for item in environment_rule_schedule if item.start_step
        ],
        "version": "observer-fixture-episode-result/1",
    }
    episode = ObserverFixtureEpisodeResultDTO(
        episode_result_id=canonical_id(episode_payload),
        episode_id=initial_state.episode_id,
        initial_state_id=initial_state.fixture_state_id,
        final_state_id=state.fixture_state_id,
        ledger_snapshot=snapshot,
        confirmed_entry_ids=confirmed,
        contradicted_entry_ids=contradicted,
        inconclusive_entry_ids=inconclusive,
        rule_change_steps=tuple(
            sorted(
                item.start_step for item in environment_rule_schedule if item.start_step
            )
        ),
    )
    status = "completed_with_suspension" if suspension_events else "completed"
    report = ObserverActiveHabitExecutionReportDTO.create(
        fixture_episode_result_id=episode.episode_result_id,
        initial_registry_snapshot_id=initial_registry_snapshot_id,
        final_registry_snapshot_id=registry.current_snapshot().habit_registry_snapshot_id,
        active_occurrences=tuple(occurrences),
        suspension_event_ids=tuple(suspension_events),
        status=status,
        failure_codes=(),
    )
    return episode, entries, report


def _active_occurrence(
    *,
    decision: ObserverActiveHabitDecisionDTO,
    ledger_entry: ObserverTransitionLedgerEntryDTO,
    actual_observation,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    habits: tuple[ObserverHabitSpecificationDTO, ...],
) -> ObserverActiveHabitOccurrenceDTO:
    habit = next(
        (
            item
            for item in habits
            if item.habit_specification_id == decision.habit_specification_id
        ),
        None,
    )
    assignment = assign_observation_to_state_class(
        observation=actual_observation,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
    )
    expected = None if habit is None else habit.expected_target_state_class_id
    if decision.decision_source == "habit":
        if ledger_entry.transition_verification.verification_status == "contradicted":
            outcome = "habit_contradicted"
        elif ledger_entry.transition_verification.verification_status == "inconclusive":
            outcome = "habit_inconclusive"
        elif assignment.state_class_id == expected:
            outcome = "habit_success"
        else:
            outcome = "habit_wrong_target"
    elif "invalid_active_habit" in decision.reason_codes:
        outcome = "fallback_after_invalid_habit"
    elif "ambiguous_active_habits" in decision.reason_codes:
        outcome = "fallback_after_ambiguity"
    elif "active_habit_abstained" in decision.reason_codes:
        outcome = "fallback_after_abstention"
    else:
        outcome = "fallback_executed"
    return ObserverActiveHabitOccurrenceDTO.create(
        registry_snapshot_id=decision.registry_snapshot_id,
        habit_specification_id=decision.habit_specification_id,
        active_habit_decision_id=decision.active_habit_decision_id,
        ledger_entry_id=ledger_entry.ledger_entry_id,
        source_observation_artifact_id=decision.observation_artifact_id,
        selected_action=decision.selected_action,
        authoritative_fallback_action=decision.authoritative_fallback_action,
        decision_source=decision.decision_source,
        actual_target_state_class_id=assignment.state_class_id,
        expected_target_state_class_id=expected,
        verification_status=ledger_entry.transition_verification.verification_status,
        outcome=outcome,
        reason_codes=(outcome,),
    )


def _should_suspend(
    *,
    occurrence: ObserverActiveHabitOccurrenceDTO,
    safety: ObserverHabitRuntimeSafetyRecipeDTO,
    consecutive_failures: int,
    total_failures: int,
) -> bool:
    if occurrence.outcome == "habit_wrong_target" and safety.suspend_on_wrong_target:
        return True
    if occurrence.outcome == "habit_contradicted" and safety.suspend_on_contradiction:
        return True
    if occurrence.outcome == "habit_inconclusive" and safety.suspend_on_inconclusive:
        return True
    if consecutive_failures >= safety.maximum_consecutive_habit_failures:
        return True
    if total_failures >= safety.maximum_total_habit_failures:
        return True
    return False
