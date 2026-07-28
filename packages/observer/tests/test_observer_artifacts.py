import pytest

from zeromodel.observer import (
    ObserverComparisonRecipeDTO,
    ObserverObservationArtifactDTO,
    build_contradiction_artifact,
    build_replacement_policy_artifact,
    build_transition_record,
    compare_observer_transition,
)
from zeromodel.observer.artifacts import ObserverArtifactError


def test_observation_artifact_identity_is_canonical() -> None:
    left = ObserverObservationArtifactDTO.create(
        visible_state_features={"target_x": 9, "agent_x": 4},
        recent_history_features={"previous_action": "move_right"},
        hidden_state_uncertainty={"cooldown": ("unknown", "maybe_active")},
        provenance={"fixture": "hidden-cooldown-v0"},
        sequence_index=3,
    )
    right = ObserverObservationArtifactDTO.create(
        visible_state_features={"agent_x": 4, "target_x": 9},
        recent_history_features={"previous_action": "move_right"},
        hidden_state_uncertainty={"cooldown": ("unknown", "maybe_active")},
        provenance={"fixture": "hidden-cooldown-v0"},
        sequence_index=3,
    )

    assert left.observation_artifact_id == right.observation_artifact_id


def test_comparison_result_is_structured_and_replayable() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        observable_feature_keys=("agent_x", "target_x"),
        action_effect_keys=("action_effect",),
        policy_consequence_key="next_action",
        hidden_state_keys=("cooldown",),
        wake_on_policy_consequence_mismatch=True,
    )

    result = compare_observer_transition(
        recipe=recipe,
        predicted_features={
            "agent_x": 5,
            "target_x": 9,
            "action_effect": "moved_right",
            "next_action": "move_right",
        },
        observed_features={
            "agent_x": 4,
            "target_x": 9,
            "action_effect": "blocked_by_cooldown",
            "next_action": "wait",
        },
        predicted_decision_margin=0.30,
        observed_decision_margin=0.12,
        hidden_state_hypotheses_remaining=1,
    )

    assert result.observable_feature_match is False
    assert result.action_effect_match is False
    assert result.next_action_equivalent is False
    assert result.wake_required is True
    assert result.contradiction is True
    assert result.mismatched_feature_keys == (
        "action_effect",
        "agent_x",
        "next_action",
    )


def test_contradiction_requires_failing_comparison() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        observable_feature_keys=("agent_x",),
    )
    result = compare_observer_transition(
        recipe=recipe,
        predicted_features={"agent_x": 4},
        observed_features={"agent_x": 4},
        predicted_decision_margin=0.1,
        observed_decision_margin=0.1,
        hidden_state_hypotheses_remaining=0,
    )
    transition = build_transition_record(
        policy_artifact_id="policy:A",
        state_before_id="state:before",
        action="wait",
        predicted_state_after_id="state:after",
        observed_state_after_id="state:after",
        comparison_recipe_id=recipe.recipe_id,
        comparison_result_id=result.comparison_result_id,
        verification_status="accepted",
        affected_policy_row_id="row:state-before",
    )

    with pytest.raises(ObserverArtifactError, match="not a contradiction"):
        build_contradiction_artifact(
            transition=transition,
            comparison=result,
        )


def test_contradiction_and_replacement_lineage_are_content_addressed() -> None:
    recipe = ObserverComparisonRecipeDTO.create(
        observable_feature_keys=("agent_x",),
        action_effect_keys=("action_effect",),
    )
    result = compare_observer_transition(
        recipe=recipe,
        predicted_features={"agent_x": 5, "action_effect": "moved_right"},
        observed_features={"agent_x": 4, "action_effect": "blocked_by_cooldown"},
        predicted_decision_margin=0.3,
        observed_decision_margin=0.2,
        hidden_state_hypotheses_remaining=1,
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
        relevant_context_keys=("previous_action",),
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
