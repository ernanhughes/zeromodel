import pytest

from zeromodel.observer import (
    ObserverFeatureComparisonDTO,
    ObserverFeatureDefinitionDTO,
    ObserverHiddenStateHypothesisDTO,
    ObserverHiddenStateHypothesisSetDTO,
    ObserverComparisonRecipeDTO,
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
    ObserverPolicyConsequenceEvidenceDTO,
    ObserverTransitionVerificationDTO,
    ObserverTransitionVerificationError,
    verify_observer_transition,
)


def schema(*, name: str = "stage-o3") -> ObserverObservationSchemaDTO:
    return ObserverObservationSchemaDTO.create(
        schema_name=name,
        features=(
            ObserverFeatureDefinitionDTO.create(
                qualified_key="hidden.cooldown", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="hidden.mode", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="history.mode", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.action_effect", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.agent_x", value_type="int", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.mode", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.next_action", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.target_x", value_type="int", required=False
            ),
        ),
    )


SCHEMA = schema()


def observation(
    *,
    sequence_index: int,
    visible: dict[str, object],
    history: dict[str, object] | None = None,
    hidden: dict[str, object] | None = None,
) -> ObserverObservationArtifactDTO:
    return ObserverObservationArtifactDTO.create(
        observation_schema=SCHEMA,
        visible_state_features=visible,
        recent_history_features=history or {},
        hidden_state_uncertainty=hidden or {},
        provenance={"fixture": "stage-o1"},
        sequence_index=sequence_index,
    )


def verify(
    *,
    recipe: ObserverComparisonRecipeDTO,
    predicted: ObserverObservationArtifactDTO,
    observed: ObserverObservationArtifactDTO,
    predicted_margin: float = 0.3,
    observed_margin: float = 0.3,
    hidden_remaining: int = 1,
    policy_evidence: ObserverPolicyConsequenceEvidenceDTO | None = None,
    reproduction: dict[str, object] | None = None,
    relevant_context_keys: tuple[str, ...] = (),
):
    return verify_observer_transition(
        recipe=recipe,
        predicted_observation=predicted,
        observed_observation=observed,
        policy_artifact_id="policy:A",
        state_before_id="state:before",
        action="move_right",
        affected_policy_row_id="row:before",
        predicted_decision_margin=predicted_margin,
        observed_decision_margin=observed_margin,
        hidden_state_hypothesis_set=ObserverHiddenStateHypothesisSetDTO.create(
            observation_schema_id=SCHEMA.schema_id,
            hypotheses=(
                ObserverHiddenStateHypothesisDTO.create(
                    state_key="hidden.cooldown",
                    state_value="clear",
                    status="possible" if hidden_remaining else "eliminated",
                ),
            ),
        ),
        policy_consequence_evidence=policy_evidence,
        reproduction=reproduction,
        relevant_context_keys=relevant_context_keys,
    )


def feature_specs(*keys: str) -> tuple[ObserverFeatureComparisonDTO, ...]:
    return tuple(
        ObserverFeatureComparisonDTO.create(feature_key=key, mode="exact")
        for key in sorted(keys)
    )


def test_confirmed_transition_replays_to_identical_ids() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs(
            "hidden.cooldown",
            "visible.action_effect",
            "visible.agent_x",
            "visible.target_x",
        ),
        observable_feature_keys=("visible.agent_x", "visible.target_x"),
        action_effect_keys=("visible.action_effect",),
        hidden_state_keys=("hidden.cooldown",),
    )
    predicted = observation(
        sequence_index=2,
        visible={
            "agent_x": 5,
            "target_x": 9,
            "action_effect": "moved_right",
            "next_action": "move_right",
        },
        hidden={"cooldown": "clear"},
    )
    observed = observation(
        sequence_index=2,
        visible={
            "agent_x": 5,
            "target_x": 9,
            "action_effect": "moved_right",
            "next_action": "move_right",
        },
        hidden={"cooldown": "clear"},
    )

    first = verify(recipe=recipe, predicted=predicted, observed=observed)
    second = verify(recipe=recipe, predicted=predicted, observed=observed)

    assert first.verification_status == "confirmed"
    assert first.comparison_result.contradiction is False
    assert first.comparison_result.wake_required is False
    assert first.contradiction_artifact is None
    assert (
        first.transition_record.predicted_state_after_id
        == predicted.observation_artifact_id
    )
    assert first.transition_record.observed_state_after_id == (
        observed.observation_artifact_id
    )
    assert first.comparison_result.comparison_result_id == (
        second.comparison_result.comparison_result_id
    )
    assert first.transition_record.transition_record_id == (
        second.transition_record.transition_record_id
    )
    assert first.verification_id == second.verification_id


def test_hidden_cooldown_contradiction_builds_artifact() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs(
            "hidden.cooldown",
            "visible.action_effect",
            "visible.agent_x",
            "visible.target_x",
        ),
        observable_feature_keys=("visible.agent_x", "visible.target_x"),
        action_effect_keys=("visible.action_effect",),
        hidden_state_keys=("hidden.cooldown",),
        wake_on_policy_consequence_mismatch=True,
    )
    predicted = observation(
        sequence_index=1,
        visible={
            "agent_x": 5,
            "target_x": 9,
            "action_effect": "moved_right",
            "next_action": "move_right",
        },
        hidden={"cooldown": "clear"},
    )
    observed = observation(
        sequence_index=1,
        visible={
            "agent_x": 4,
            "target_x": 9,
            "action_effect": "blocked_by_cooldown",
            "next_action": "wait",
        },
        hidden={"cooldown": "active"},
    )

    result = verify(
        recipe=recipe,
        predicted=predicted,
        observed=observed,
        observed_margin=0.12,
        reproduction={"episode_id": "episode:1", "step": 7},
        relevant_context_keys=("history.previous_action",),
    )

    assert result.verification_status == "contradicted"
    assert result.comparison_result.wake_required is True
    assert result.contradiction_artifact is not None
    assert result.contradiction_artifact.affected_policy_row_id == "row:before"
    assert result.contradiction_artifact.reproduction == {
        "episode_id": "episode:1",
        "step": 7,
    }
    assert result.contradiction_artifact.relevant_context_keys == (
        "history.previous_action",
    )


def test_missing_required_feature_is_inconclusive_not_match_or_contradiction() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs(
            "visible.action_effect",
            "visible.agent_x",
            "visible.target_x",
        ),
        observable_feature_keys=("visible.agent_x", "visible.target_x"),
        action_effect_keys=("visible.action_effect",),
    )
    predicted = observation(
        sequence_index=1,
        visible={"agent_x": 5, "target_x": 9, "action_effect": "moved_right"},
    )
    observed = observation(
        sequence_index=1,
        visible={"agent_x": 5, "target_x": 9},
    )

    result = verify(recipe=recipe, predicted=predicted, observed=observed)

    assert result.verification_status == "inconclusive"
    assert result.comparison_result.missing_observed_feature_keys == (
        "visible.action_effect",
    )
    assert result.comparison_result.action_effect_match is False
    assert result.comparison_result.contradiction is False
    assert result.contradiction_artifact is None


def test_feature_projection_namespaces_visible_history_and_hidden_keys() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs(
            "hidden.mode", "history.mode", "visible.mode"
        ),
        observable_feature_keys=("visible.mode",),
        action_effect_keys=("history.mode",),
        hidden_state_keys=("hidden.mode",),
    )
    predicted = observation(
        sequence_index=1,
        visible={"mode": "visible-ready"},
        history={"mode": "history-right"},
        hidden={"mode": "hidden-clear"},
    )
    observed = observation(
        sequence_index=1,
        visible={"mode": "visible-ready"},
        history={"mode": "history-right"},
        hidden={"mode": "hidden-clear"},
    )

    result = verify(recipe=recipe, predicted=predicted, observed=observed)

    assert result.verification_status == "confirmed"
    assert result.comparison_result.missing_predicted_feature_keys == ()
    assert result.comparison_result.missing_observed_feature_keys == ()


def test_invalid_sequence_order_is_rejected() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs("visible.agent_x"),
        observable_feature_keys=("visible.agent_x",),
    )
    predicted = observation(sequence_index=2, visible={"agent_x": 4})
    observed = observation(sequence_index=4, visible={"agent_x": 4})

    with pytest.raises(
        ObserverTransitionVerificationError,
        match="same target sequence position",
    ):
        verify(recipe=recipe, predicted=predicted, observed=observed)


def test_recipe_sensitivity_changes_wake_and_ids() -> None:
    predicted = observation(
        sequence_index=1,
        visible={"agent_x": 4, "next_action": "move_right"},
    )
    observed = observation(
        sequence_index=1,
        visible={"agent_x": 4, "next_action": "wait"},
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
    passive = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs("visible.agent_x"),
        observable_feature_keys=("visible.agent_x",),
        require_policy_consequence_evidence=True,
        wake_on_policy_consequence_mismatch=False,
    )
    waking = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs("visible.agent_x"),
        observable_feature_keys=("visible.agent_x",),
        require_policy_consequence_evidence=True,
        wake_on_policy_consequence_mismatch=True,
    )

    passive_result = verify(
        recipe=passive,
        predicted=predicted,
        observed=observed,
        policy_evidence=policy_evidence,
        predicted_margin=0.3,
        observed_margin=0.29,
    )
    waking_result = verify(
        recipe=waking,
        predicted=predicted,
        observed=observed,
        policy_evidence=policy_evidence,
        predicted_margin=0.3,
        observed_margin=0.29,
    )

    assert passive_result.comparison_result.wake_required is False
    assert waking_result.comparison_result.wake_required is True
    assert passive_result.comparison_result.comparison_result_id != (
        waking_result.comparison_result.comparison_result_id
    )
    assert passive_result.verification_id != waking_result.verification_id
    assert passive_result.predicted_observation_artifact_id == (
        waking_result.predicted_observation_artifact_id
    )
    assert passive_result.observed_observation_artifact_id == (
        waking_result.observed_observation_artifact_id
    )


def test_contradiction_can_be_recorded_without_wake() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs("visible.agent_x"),
        observable_feature_keys=("visible.agent_x",),
        wake_on_observable_mismatch=False,
    )
    predicted = observation(sequence_index=1, visible={"agent_x": 5})
    observed = observation(sequence_index=1, visible={"agent_x": 4})

    result = verify(recipe=recipe, predicted=predicted, observed=observed)

    assert result.comparison_result.contradiction is True
    assert result.comparison_result.wake_required is False
    assert result.verification_status == "contradicted"
    assert result.contradiction_artifact is not None


def test_verification_identity_changes_with_reproduction_evidence() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs("visible.agent_x"),
        observable_feature_keys=("visible.agent_x",),
    )
    predicted = observation(sequence_index=1, visible={"agent_x": 5})
    observed = observation(sequence_index=1, visible={"agent_x": 4})

    first = verify(
        recipe=recipe,
        predicted=predicted,
        observed=observed,
        reproduction={"step": 1},
    )
    second = verify(
        recipe=recipe,
        predicted=predicted,
        observed=observed,
        reproduction={"step": 2},
    )

    assert first.verification_id != second.verification_id
    assert first.contradiction_artifact is not None
    assert second.contradiction_artifact is not None
    assert first.contradiction_artifact.contradiction_artifact_id != (
        second.contradiction_artifact.contradiction_artifact_id
    )


def test_transition_service_public_api() -> None:
    import zeromodel.observer as observer

    assert "ObserverTransitionVerificationDTO" in observer.__all__
    assert "ObserverTransitionVerificationError" in observer.__all__
    assert "verify_observer_transition" in observer.__all__
    assert hasattr(observer, "ObserverTransitionVerificationDTO")
    assert hasattr(observer, "ObserverTransitionVerificationError")
    assert hasattr(observer, "verify_observer_transition")


def test_canonical_payload_reconstruction_preserves_identity() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=feature_specs("visible.agent_x"),
        observable_feature_keys=("visible.agent_x",),
    )
    predicted = observation(sequence_index=1, visible={"agent_x": 4})
    observed = observation(sequence_index=1, visible={"agent_x": 4})
    result = verify(recipe=recipe, predicted=predicted, observed=observed)

    reconstructed = ObserverTransitionVerificationDTO(
        verification_id=result.verification_id,
        recipe_id=result.recipe_id,
        predicted_observation_artifact_id=(result.predicted_observation_artifact_id),
        observed_observation_artifact_id=result.observed_observation_artifact_id,
        comparison_result=result.comparison_result,
        transition_record=result.transition_record,
        contradiction_artifact=result.contradiction_artifact,
        verification_status=result.verification_status,
    )

    assert reconstructed.verification_id == result.verification_id
    assert reconstructed.canonical_payload() == result.canonical_payload()
