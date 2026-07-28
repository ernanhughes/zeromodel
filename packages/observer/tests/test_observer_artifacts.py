import pytest

from zeromodel.observer import (
    ObserverFeatureComparisonDTO,
    ObserverFeatureDefinitionDTO,
    ObserverHiddenStateHypothesisDTO,
    ObserverHiddenStateHypothesisSetDTO,
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
    ObserverPolicyConsequenceEvidenceDTO,
    ObserverComparisonRecipeDTO,
    build_contradiction_artifact,
    build_replacement_policy_artifact,
    build_transition_record,
    compare_observer_transition,
)
from zeromodel.observer.artifacts import ObserverArtifactError


def schema() -> ObserverObservationSchemaDTO:
    return ObserverObservationSchemaDTO.create(
        schema_name="artifact-test",
        features=(
            ObserverFeatureDefinitionDTO.create(
                qualified_key="hidden.cooldown", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="history.previous_action",
                value_type="str",
                required=False,
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.action_effect",
                value_type="str",
                required=False,
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.agent_x", value_type="int", required=True
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.target_x", value_type="int", required=False
            ),
        ),
    )


def specs(*keys: str) -> tuple[ObserverFeatureComparisonDTO, ...]:
    return tuple(
        ObserverFeatureComparisonDTO.create(feature_key=key, mode="exact")
        for key in sorted(keys)
    )


def test_observation_artifact_identity_is_canonical() -> None:
    observation_schema = schema()
    left = ObserverObservationArtifactDTO.create(
        observation_schema=observation_schema,
        visible_state_features={"target_x": 9, "agent_x": 4},
        recent_history_features={"previous_action": "move_right"},
        hidden_state_uncertainty={"cooldown": "maybe_active"},
        provenance={"fixture": "hidden-cooldown-v0"},
        sequence_index=3,
    )
    right = ObserverObservationArtifactDTO.create(
        observation_schema=observation_schema,
        visible_state_features={"agent_x": 4, "target_x": 9},
        recent_history_features={"previous_action": "move_right"},
        hidden_state_uncertainty={"cooldown": "maybe_active"},
        provenance={"fixture": "hidden-cooldown-v0"},
        sequence_index=3,
    )

    assert left.observation_artifact_id == right.observation_artifact_id


def test_comparison_result_is_structured_and_replayable() -> None:
    observation_schema = schema()
    predicted = ObserverObservationArtifactDTO.create(
        observation_schema=observation_schema,
        visible_state_features={
            "agent_x": 5,
            "target_x": 9,
            "action_effect": "moved_right",
        },
        hidden_state_uncertainty={"cooldown": "clear"},
    )
    observed = ObserverObservationArtifactDTO.create(
        observation_schema=observation_schema,
        visible_state_features={
            "agent_x": 4,
            "target_x": 9,
            "action_effect": "blocked_by_cooldown",
        },
        hidden_state_uncertainty={"cooldown": "active"},
    )
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=specs(
            "hidden.cooldown",
            "visible.action_effect",
            "visible.agent_x",
            "visible.target_x",
        ),
        observable_feature_keys=("visible.agent_x", "visible.target_x"),
        action_effect_keys=("visible.action_effect",),
        hidden_state_keys=("hidden.cooldown",),
    )
    hypotheses = ObserverHiddenStateHypothesisSetDTO.create(
        observation_schema_id=observation_schema.schema_id,
        hypotheses=(
            ObserverHiddenStateHypothesisDTO.create(
                state_key="hidden.cooldown", state_value="clear"
            ),
        ),
    )
    policy_evidence = ObserverPolicyConsequenceEvidenceDTO.create(
        policy_artifact_id="policy:A",
        predicted_state_artifact_id=predicted.observation_artifact_id,
        observed_state_artifact_id=observed.observation_artifact_id,
        predicted_selected_action="move_right",
        observed_selected_action="wait",
        predicted_decision_trace_id="trace:predicted",
        observed_decision_trace_id="trace:observed",
        reader_contract_id="reader:v1",
    )

    result = compare_observer_transition(
        recipe=recipe,
        predicted_observation_artifact_id=predicted.observation_artifact_id,
        observed_observation_artifact_id=observed.observation_artifact_id,
        predicted_observation_schema_id=predicted.observation_schema_id,
        observed_observation_schema_id=observed.observation_schema_id,
        predicted_features={
            "hidden.cooldown": "clear",
            "visible.agent_x": 5,
            "visible.target_x": 9,
            "visible.action_effect": "moved_right",
        },
        observed_features={
            "hidden.cooldown": "active",
            "visible.agent_x": 4,
            "visible.target_x": 9,
            "visible.action_effect": "blocked_by_cooldown",
        },
        hidden_state_hypothesis_set=hypotheses,
        policy_consequence_evidence=policy_evidence,
    )

    replay = compare_observer_transition(
        recipe=recipe,
        predicted_observation_artifact_id=predicted.observation_artifact_id,
        observed_observation_artifact_id=observed.observation_artifact_id,
        predicted_observation_schema_id=predicted.observation_schema_id,
        observed_observation_schema_id=observed.observation_schema_id,
        predicted_features={
            "hidden.cooldown": "clear",
            "visible.agent_x": 5,
            "visible.target_x": 9,
            "visible.action_effect": "moved_right",
        },
        observed_features={
            "hidden.cooldown": "active",
            "visible.agent_x": 4,
            "visible.target_x": 9,
            "visible.action_effect": "blocked_by_cooldown",
        },
        hidden_state_hypothesis_set=hypotheses,
        policy_consequence_evidence=policy_evidence,
    )

    assert result.observable_feature_match is False
    assert result.action_effect_match is False
    assert result.policy_consequence_match is False
    assert result.wake_required is True
    assert result.contradiction is True
    assert result.mismatched_feature_keys == (
        "hidden.cooldown",
        "visible.action_effect",
        "visible.agent_x",
    )
    assert result.comparison_result_id == replay.comparison_result_id


def test_contradiction_requires_failing_comparison() -> None:
    observation_schema = schema()
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=specs("visible.agent_x"),
        observable_feature_keys=("visible.agent_x",),
    )
    result = compare_observer_transition(
        recipe=recipe,
        predicted_observation_artifact_id="state:predicted",
        observed_observation_artifact_id="state:observed",
        predicted_observation_schema_id=observation_schema.schema_id,
        observed_observation_schema_id=observation_schema.schema_id,
        predicted_features={"visible.agent_x": 4},
        observed_features={"visible.agent_x": 4},
    )
    transition = build_transition_record(
        policy_artifact_id="policy:A",
        state_before_id="state:before",
        action="wait",
        predicted_state_after_id="state:predicted",
        observed_state_after_id="state:observed",
        comparison_recipe_id=recipe.recipe_id,
        comparison_result_id=result.comparison_result_id,
        verification_status="confirmed",
        affected_policy_row_id="row:state-before",
    )

    with pytest.raises(ObserverArtifactError, match="not a contradiction"):
        build_contradiction_artifact(transition=transition, comparison=result)


def test_contradiction_and_replacement_lineage_are_content_addressed() -> None:
    observation_schema = schema()
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=specs("visible.action_effect", "visible.agent_x"),
        observable_feature_keys=("visible.agent_x",),
        action_effect_keys=("visible.action_effect",),
    )
    result = compare_observer_transition(
        recipe=recipe,
        predicted_observation_artifact_id="state:x5",
        observed_observation_artifact_id="state:x4",
        predicted_observation_schema_id=observation_schema.schema_id,
        observed_observation_schema_id=observation_schema.schema_id,
        predicted_features={
            "visible.agent_x": 5,
            "visible.action_effect": "moved_right",
        },
        observed_features={
            "visible.agent_x": 4,
            "visible.action_effect": "blocked_by_cooldown",
        },
    )
    transition = build_transition_record(
        policy_artifact_id="policy:A",
        state_before_id="state:cooldown-visible",
        action="move_right",
        predicted_state_after_id="state:x5",
        observed_state_after_id="state:x4",
        comparison_recipe_id=recipe.recipe_id,
        comparison_result_id=result.comparison_result_id,
        verification_status="contradicted",
        affected_policy_row_id="row:cooldown-visible",
    )
    contradiction = build_contradiction_artifact(
        transition=transition,
        comparison=result,
        reproduction={"episode_id": "episode:1", "step": 7},
        relevant_context_keys=("history.previous_action",),
    )
    replacement = build_replacement_policy_artifact(
        parent_policy_artifact_id="policy:A",
        replacement_policy_artifact_id="policy:B",
        contradiction_artifact_id=contradiction.contradiction_artifact_id,
        changed_row_ids=("row:cooldown-visible",),
        changed_cell_ids=("row:cooldown-visible/action:move_right",),
        verified_result_ids=(result.comparison_result_id,),
        unchanged_region_result_id="unchanged:all-other-rows",
    )

    assert contradiction.source_policy_artifact_id == "policy:A"
    assert replacement.parent_policy_artifact_id == "policy:A"
    assert replacement.replacement_policy_artifact_id == "policy:B"
    assert replacement.relation == "repairs"
