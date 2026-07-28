"""Bounded deterministic Observer fixture episode runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import ObserverObservationSchemaDTO
from zeromodel.observer.comparison import ObserverComparisonRecipeDTO
from zeromodel.observer.fixture import (
    ObserverFixtureActionDTO,
    ObserverFixtureError,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
)
from zeromodel.observer.fixture_predictor import (
    execute_observer_fixture_step,
    predict_observer_fixture_transition,
)
from zeromodel.observer.ledger import (
    InMemoryObserverTransitionLedger,
    ObserverTransitionLedgerEntryDTO,
    ObserverTransitionLedgerSnapshotDTO,
)
from zeromodel.observer.transition_service import verify_observer_transition

OBSERVER_FIXTURE_RULE_SCHEDULE_ENTRY_VERSION: Final = (
    "observer-fixture-rule-schedule-entry/1"
)
OBSERVER_FIXTURE_EPISODE_RESULT_VERSION: Final = "observer-fixture-episode-result/1"


@dataclass(frozen=True)
class ObserverFixtureRuleScheduleEntryDTO:
    """Declared environment rule active from a logical step."""

    rule_schedule_entry_id: str
    start_step: int
    rule_set_id: str
    version: str = OBSERVER_FIXTURE_RULE_SCHEDULE_ENTRY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_FIXTURE_RULE_SCHEDULE_ENTRY_VERSION:
            raise ObserverFixtureError("unsupported rule schedule entry version")
        if self.start_step < 0:
            raise ObserverFixtureError("start_step must be non-negative")
        if not self.rule_set_id:
            raise ObserverFixtureError("rule_set_id must be non-empty")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.rule_schedule_entry_id != expected_id:
            raise ObserverFixtureError(
                "rule_schedule_entry_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "rule_set_id": self.rule_set_id,
            "start_step": self.start_step,
            "version": self.version,
        }
        if include_id:
            payload["rule_schedule_entry_id"] = self.rule_schedule_entry_id
        return payload

    @classmethod
    def create(
        cls, *, start_step: int, rule_set_id: str
    ) -> "ObserverFixtureRuleScheduleEntryDTO":
        payload = {
            "rule_set_id": rule_set_id,
            "start_step": start_step,
            "version": OBSERVER_FIXTURE_RULE_SCHEDULE_ENTRY_VERSION,
        }
        return cls(
            rule_schedule_entry_id=canonical_id(payload),
            start_step=start_step,
            rule_set_id=rule_set_id,
        )


@dataclass(frozen=True)
class ObserverFixtureEpisodeResultDTO:
    """Summary of one deterministic fixture episode run."""

    episode_result_id: str
    episode_id: str
    initial_state_id: str
    final_state_id: str
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO
    confirmed_entry_ids: tuple[str, ...]
    contradicted_entry_ids: tuple[str, ...]
    inconclusive_entry_ids: tuple[str, ...]
    rule_change_steps: tuple[int, ...]
    version: str = OBSERVER_FIXTURE_EPISODE_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_FIXTURE_EPISODE_RESULT_VERSION:
            raise ObserverFixtureError("unsupported fixture episode result version")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.episode_result_id != expected_id:
            raise ObserverFixtureError(
                "episode_result_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "confirmed_entry_ids": list(self.confirmed_entry_ids),
            "contradicted_entry_ids": list(self.contradicted_entry_ids),
            "episode_id": self.episode_id,
            "final_state_id": self.final_state_id,
            "inconclusive_entry_ids": list(self.inconclusive_entry_ids),
            "initial_state_id": self.initial_state_id,
            "ledger_snapshot": self.ledger_snapshot.canonical_payload(),
            "rule_change_steps": list(self.rule_change_steps),
            "version": self.version,
        }
        if include_id:
            payload["episode_result_id"] = self.episode_result_id
        return payload


def active_rule_for_step(
    *,
    step_index: int,
    schedule: tuple[ObserverFixtureRuleScheduleEntryDTO, ...],
    rule_sets: Mapping[str, ObserverFixtureRuleSetDTO],
) -> ObserverFixtureRuleSetDTO:
    active = None
    for item in sorted(schedule, key=lambda entry: entry.start_step):
        if item.rule_set_id not in rule_sets:
            raise ObserverFixtureError("schedule references unknown rule set")
        if item.start_step <= step_index:
            active = item
    if active is None:
        raise ObserverFixtureError("rule schedule has no active rule")
    return rule_sets[active.rule_set_id]


def run_observer_fixture_episode(
    *,
    initial_state: ObserverFixtureStateDTO,
    actions: tuple[ObserverFixtureActionDTO, ...],
    predictor_rule_set: ObserverFixtureRuleSetDTO,
    environment_rule_schedule: tuple[ObserverFixtureRuleScheduleEntryDTO, ...],
    environment_rule_sets: tuple[ObserverFixtureRuleSetDTO, ...],
    observation_schema: ObserverObservationSchemaDTO,
    comparison_recipe: ObserverComparisonRecipeDTO,
    policy_artifact_id: str = "observer-fixture-policy",
    supply_hidden_evidence: bool = True,
) -> tuple[
    ObserverFixtureEpisodeResultDTO, tuple[ObserverTransitionLedgerEntryDTO, ...]
]:
    """Run one bounded deterministic fixture episode and return its ledger."""

    rule_map = {rule.fixture_rule_set_id: rule for rule in environment_rule_sets}
    ledger = InMemoryObserverTransitionLedger(
        fixture_id=initial_state.fixture_id, episode_id=initial_state.episode_id
    )
    state = initial_state
    rule_change_steps = tuple(
        sorted(item.start_step for item in environment_rule_schedule if item.start_step)
    )
    for sequence, action in enumerate(actions):
        if state.terminal:
            break
        environment_rule_set = active_rule_for_step(
            step_index=sequence,
            schedule=environment_rule_schedule,
            rule_sets=rule_map,
        )
        prediction = predict_observer_fixture_transition(
            source_state=state,
            action=action,
            predictor_rule_set=predictor_rule_set,
            observation_schema=observation_schema,
        )
        executed, actual_observation = execute_observer_fixture_step(
            source_state=state,
            action=action,
            environment_rule_set=environment_rule_set,
            observation_schema=observation_schema,
        )
        verification = verify_observer_transition(
            recipe=comparison_recipe,
            predicted_observation=prediction.predicted_observation,
            observed_observation=actual_observation,
            policy_artifact_id=policy_artifact_id,
            state_before_id=state.fixture_state_id,
            action=action.action_name,
            affected_policy_row_id=f"row:{state.agent_x}",
            hidden_state_hypothesis_set=(
                prediction.hidden_state_hypothesis_set
                if supply_hidden_evidence
                else None
            ),
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
            source_state_id=state.fixture_state_id,
            action_id=action.fixture_action_id,
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
    payload = {
        "confirmed_entry_ids": list(confirmed),
        "contradicted_entry_ids": list(contradicted),
        "episode_id": initial_state.episode_id,
        "final_state_id": state.fixture_state_id,
        "inconclusive_entry_ids": list(inconclusive),
        "initial_state_id": initial_state.fixture_state_id,
        "ledger_snapshot": snapshot.canonical_payload(),
        "rule_change_steps": list(rule_change_steps),
        "version": OBSERVER_FIXTURE_EPISODE_RESULT_VERSION,
    }
    return (
        ObserverFixtureEpisodeResultDTO(
            episode_result_id=canonical_id(payload),
            episode_id=initial_state.episode_id,
            initial_state_id=initial_state.fixture_state_id,
            final_state_id=state.fixture_state_id,
            ledger_snapshot=snapshot,
            confirmed_entry_ids=confirmed,
            contradicted_entry_ids=contradicted,
            inconclusive_entry_ids=inconclusive,
            rule_change_steps=rule_change_steps,
        ),
        entries,
    )
