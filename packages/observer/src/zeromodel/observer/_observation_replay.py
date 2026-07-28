"""Shared fixture-observation reconstruction for Observer ledger replay."""

from __future__ import annotations

from zeromodel.observer.artifacts import (
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)
from zeromodel.observer.fixture import ObserverFixtureStateDTO
from zeromodel.observer.ledger import ObserverTransitionLedgerEntryDTO


def source_observation_for_entry(
    *,
    entry: ObserverTransitionLedgerEntryDTO,
    observation_schema: ObserverObservationSchemaDTO,
    previous_target_observation: ObserverObservationArtifactDTO | None,
    previous_target_action_effect: str | None,
) -> ObserverObservationArtifactDTO | None:
    if previous_target_observation is not None:
        if previous_target_action_effect is None:
            return None
        expected_source = observation_for_fixture_state(
            state=entry.source_state,
            action_effect=previous_target_action_effect,
            observation_schema=observation_schema,
        )
        if (
            previous_target_observation.observation_artifact_id
            != expected_source.observation_artifact_id
        ):
            return None
        return previous_target_observation
    return observation_for_fixture_state(
        state=entry.source_state,
        action_effect="initial",
        observation_schema=observation_schema,
    )


def target_observation_for_entry(
    *,
    entry: ObserverTransitionLedgerEntryDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverObservationArtifactDTO | None:
    target = observation_for_fixture_state(
        state=entry.executed_step.actual_state,
        action_effect=entry.executed_step.action_effect,
        observation_schema=observation_schema,
    )
    if target.observation_artifact_id != entry.executed_step.actual_observation_id:
        return None
    return target


def observation_for_fixture_state(
    *,
    state: ObserverFixtureStateDTO,
    action_effect: str,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverObservationArtifactDTO:
    history = {}
    if state.previous_action is not None:
        history["previous_action"] = state.previous_action
    return ObserverObservationArtifactDTO.create(
        observation_schema=observation_schema,
        visible_state_features={
            "action_effect": action_effect,
            "agent_x": state.agent_x,
            "target_x": state.target_x,
        },
        recent_history_features=history,
        hidden_state_uncertainty={"cooldown_remaining": state.cooldown_remaining},
        provenance={"fixture_id": state.fixture_id, "rule_set_id": state.rule_set_id},
        sequence_index=state.step_index,
    )
