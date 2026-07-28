"""Deterministic fixture prediction and execution for Stage O3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import (
    ObserverFeatureDefinitionDTO,
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)
from zeromodel.observer.comparison import (
    ObserverFeatureComparisonDTO,
    ObserverHiddenStateHypothesisDTO,
    ObserverHiddenStateHypothesisSetDTO,
)
from zeromodel.observer.fixture import (
    ObserverExecutedFixtureStepDTO,
    ObserverFixtureActionDTO,
    ObserverFixtureError,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
)

OBSERVER_PREDICTED_TRANSITION_VERSION: Final = "observer-predicted-transition/1"


@dataclass(frozen=True)
class ObserverPredictedTransitionDTO:
    """Canonical deterministic fixture prediction."""

    predicted_transition_id: str
    source_state_id: str
    action_id: str
    predictor_rule_set_id: str
    predicted_state: ObserverFixtureStateDTO
    predicted_observation: ObserverObservationArtifactDTO
    hidden_state_hypothesis_set: ObserverHiddenStateHypothesisSetDTO | None
    version: str = OBSERVER_PREDICTED_TRANSITION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_PREDICTED_TRANSITION_VERSION:
            raise ObserverFixtureError("unsupported predicted transition version")
        for field_name in ("source_state_id", "action_id", "predictor_rule_set_id"):
            if not getattr(self, field_name):
                raise ObserverFixtureError(f"{field_name} must be non-empty")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.predicted_transition_id != expected_id:
            raise ObserverFixtureError(
                "predicted_transition_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "action_id": self.action_id,
            "hidden_state_hypothesis_set": (
                None
                if self.hidden_state_hypothesis_set is None
                else self.hidden_state_hypothesis_set.canonical_payload()
            ),
            "predicted_observation": self.predicted_observation.canonical_payload(),
            "predicted_state": self.predicted_state.canonical_payload(),
            "predictor_rule_set_id": self.predictor_rule_set_id,
            "source_state_id": self.source_state_id,
            "version": self.version,
        }
        if include_id:
            payload["predicted_transition_id"] = self.predicted_transition_id
        return payload


def build_observer_fixture_observation_schema() -> ObserverObservationSchemaDTO:
    """Build the canonical Stage O3.1 fixture observation schema."""

    return ObserverObservationSchemaDTO.create(
        schema_name="observer-fixture-o3.1",
        features=(
            ObserverFeatureDefinitionDTO.create(
                qualified_key="hidden.cooldown_remaining",
                value_type="int",
                required=False,
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="history.previous_action",
                value_type="str",
                required=False,
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.action_effect",
                value_type="str",
                required=True,
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.agent_x",
                value_type="int",
                required=True,
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.target_x",
                value_type="int",
                required=True,
            ),
        ),
    )


def build_observer_fixture_comparison_recipe(
    observation_schema: ObserverObservationSchemaDTO,
):
    from zeromodel.observer.comparison import ObserverComparisonRecipeDTO

    comparisons = (
        ObserverFeatureComparisonDTO.create(
            feature_key="hidden.cooldown_remaining", mode="exact", expected_type="int"
        ),
        ObserverFeatureComparisonDTO.create(
            feature_key="history.previous_action",
            mode="categorical",
            expected_type="str",
        ),
        ObserverFeatureComparisonDTO.create(
            feature_key="visible.action_effect",
            mode="categorical",
            expected_type="str",
        ),
        ObserverFeatureComparisonDTO.create(
            feature_key="visible.agent_x", mode="exact", expected_type="int"
        ),
        ObserverFeatureComparisonDTO.create(
            feature_key="visible.target_x", mode="exact", expected_type="int"
        ),
    )
    return ObserverComparisonRecipeDTO.create(
        feature_comparisons=comparisons,
        observable_feature_keys=(
            "history.previous_action",
            "visible.agent_x",
            "visible.target_x",
        ),
        action_effect_keys=("visible.action_effect",),
        hidden_state_keys=("hidden.cooldown_remaining",),
    )


def _validate_state_rule_action(
    *,
    state: ObserverFixtureStateDTO,
    action: ObserverFixtureActionDTO,
    rule_set: ObserverFixtureRuleSetDTO,
    observation_schema: ObserverObservationSchemaDTO,
    require_state_rule_match: bool = True,
) -> None:
    if state.fixture_id != rule_set.fixture_id:
        raise ObserverFixtureError("state and rule set fixture_id mismatch")
    if require_state_rule_match and state.rule_set_id != rule_set.fixture_rule_set_id:
        raise ObserverFixtureError("state rule_set_id does not match rule set")
    if rule_set.observation_schema_id != observation_schema.schema_id:
        raise ObserverFixtureError("rule set observation_schema_id mismatch")
    if action.action_name not in rule_set.allowed_actions:
        raise ObserverFixtureError("action is not allowed by rule set")
    if not rule_set.minimum_position <= state.agent_x <= rule_set.maximum_position:
        raise ObserverFixtureError("agent_x is outside rule-set bounds")
    if not rule_set.minimum_position <= state.target_x <= rule_set.maximum_position:
        raise ObserverFixtureError("target_x is outside rule-set bounds")


def _advance_state(
    *,
    source_state: ObserverFixtureStateDTO,
    action: ObserverFixtureActionDTO,
    rule_set: ObserverFixtureRuleSetDTO,
) -> tuple[ObserverFixtureStateDTO, str]:
    if source_state.terminal:
        return source_state, "terminal"
    agent_x = source_state.agent_x
    effect = "waited"
    blocked = source_state.cooldown_remaining > 0
    if action.action_name == "move_left":
        agent_x = max(rule_set.minimum_position, agent_x - 1)
        effect = "moved_left"
    elif action.action_name == "move_right":
        if blocked and rule_set.cooldown_effect == "block":
            effect = "blocked_by_cooldown"
        elif blocked and rule_set.cooldown_effect == "reverse":
            agent_x = max(rule_set.minimum_position, agent_x - 1)
            effect = "reversed_by_cooldown"
        else:
            agent_x = min(rule_set.maximum_position, agent_x + 1)
            effect = "moved_right"
    cooldown_remaining = max(0, source_state.cooldown_remaining - 1)
    if action.action_name in {"move_left", "move_right"} and not blocked:
        cooldown_remaining = rule_set.cooldown_period
    terminal = agent_x == source_state.target_x
    next_state = ObserverFixtureStateDTO.create(
        fixture_id=source_state.fixture_id,
        rule_set_id=rule_set.fixture_rule_set_id,
        episode_id=source_state.episode_id,
        step_index=source_state.step_index + 1,
        agent_x=agent_x,
        target_x=source_state.target_x,
        previous_action=action.action_name,
        cooldown_remaining=cooldown_remaining,
        terminal=terminal,
    )
    return next_state, effect


def _observation_for_state(
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


def _hypothesis_set(
    *,
    state: ObserverFixtureStateDTO,
    observation_schema: ObserverObservationSchemaDTO,
    evidence_id: str,
) -> ObserverHiddenStateHypothesisSetDTO:
    return ObserverHiddenStateHypothesisSetDTO.create(
        observation_schema_id=observation_schema.schema_id,
        hypotheses=(
            ObserverHiddenStateHypothesisDTO.create(
                state_key="hidden.cooldown_remaining",
                state_value=state.cooldown_remaining,
                evidence_ids=(evidence_id,),
                status="possible",
            ),
        ),
        derivation_evidence_ids=(evidence_id,),
    )


def predict_observer_fixture_transition(
    *,
    source_state: ObserverFixtureStateDTO,
    action: ObserverFixtureActionDTO,
    predictor_rule_set: ObserverFixtureRuleSetDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverPredictedTransitionDTO:
    """Predict one fixture transition without executing the environment."""

    _validate_state_rule_action(
        state=source_state,
        action=action,
        rule_set=predictor_rule_set,
        observation_schema=observation_schema,
        require_state_rule_match=False,
    )
    predicted_state, action_effect = _advance_state(
        source_state=source_state, action=action, rule_set=predictor_rule_set
    )
    predicted_observation = _observation_for_state(
        state=predicted_state,
        action_effect=action_effect,
        observation_schema=observation_schema,
    )
    hypotheses = _hypothesis_set(
        state=predicted_state,
        observation_schema=observation_schema,
        evidence_id=predicted_observation.observation_artifact_id,
    )
    payload = {
        "action_id": action.fixture_action_id,
        "hidden_state_hypothesis_set": hypotheses.canonical_payload(),
        "predicted_observation": predicted_observation.canonical_payload(),
        "predicted_state": predicted_state.canonical_payload(),
        "predictor_rule_set_id": predictor_rule_set.fixture_rule_set_id,
        "source_state_id": source_state.fixture_state_id,
        "version": OBSERVER_PREDICTED_TRANSITION_VERSION,
    }
    return ObserverPredictedTransitionDTO(
        predicted_transition_id=canonical_id(payload),
        source_state_id=source_state.fixture_state_id,
        action_id=action.fixture_action_id,
        predictor_rule_set_id=predictor_rule_set.fixture_rule_set_id,
        predicted_state=predicted_state,
        predicted_observation=predicted_observation,
        hidden_state_hypothesis_set=hypotheses,
    )


def execute_observer_fixture_step(
    *,
    source_state: ObserverFixtureStateDTO,
    action: ObserverFixtureActionDTO,
    environment_rule_set: ObserverFixtureRuleSetDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> tuple[ObserverExecutedFixtureStepDTO, ObserverObservationArtifactDTO]:
    """Execute one deterministic environment fixture step."""

    _validate_state_rule_action(
        state=source_state,
        action=action,
        rule_set=environment_rule_set,
        observation_schema=observation_schema,
        require_state_rule_match=False,
    )
    actual_state, action_effect = _advance_state(
        source_state=source_state, action=action, rule_set=environment_rule_set
    )
    actual_observation = _observation_for_state(
        state=actual_state,
        action_effect=action_effect,
        observation_schema=observation_schema,
    )
    payload = {
        "action_effect": action_effect,
        "action_id": action.fixture_action_id,
        "actual_observation_id": actual_observation.observation_artifact_id,
        "actual_state": actual_state.canonical_payload(),
        "environment_rule_set_id": environment_rule_set.fixture_rule_set_id,
        "source_state_id": source_state.fixture_state_id,
        "version": "observer-executed-fixture-step/1",
    }
    executed = ObserverExecutedFixtureStepDTO(
        executed_step_id=canonical_id(payload),
        source_state_id=source_state.fixture_state_id,
        action_id=action.fixture_action_id,
        environment_rule_set_id=environment_rule_set.fixture_rule_set_id,
        actual_state=actual_state,
        actual_observation_id=actual_observation.observation_artifact_id,
        action_effect=action_effect,
    )
    return executed, actual_observation
