import math

import pytest

from zeromodel.observer import (
    ObserverComparisonRecipeDTO,
    ObserverFeatureComparisonDTO,
    ObserverFeatureDefinitionDTO,
    ObserverHiddenStateHypothesisDTO,
    ObserverHiddenStateHypothesisSetDTO,
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
    ObserverPolicyConsequenceEvidenceDTO,
    ObserverTransitionVerificationError,
    compare_observer_transition,
    verify_observer_transition,
)
from zeromodel.observer.artifacts import ObserverArtifactError
from zeromodel.observer.comparison import ObserverComparisonError


def schema(*, name: str = "contract") -> ObserverObservationSchemaDTO:
    return ObserverObservationSchemaDTO.create(
        schema_name=name,
        features=(
            ObserverFeatureDefinitionDTO.create(
                qualified_key="hidden.cooldown", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.label", value_type="str", required=False
            ),
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.score", value_type="number", required=False
            ),
        ),
    )


def comparison(
    key: str,
    *,
    mode: str = "exact",
    expected_type: str | None = None,
    absolute_tolerance: float | None = None,
    relative_tolerance: float | None = None,
) -> ObserverFeatureComparisonDTO:
    return ObserverFeatureComparisonDTO.create(
        feature_key=key,
        mode=mode,
        expected_type=expected_type,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )


def recipe(
    spec: ObserverFeatureComparisonDTO,
    *,
    require_policy: bool = False,
    wake_policy: bool = False,
    hidden: bool = False,
    wake_observable: bool = True,
) -> ObserverComparisonRecipeDTO:
    return ObserverComparisonRecipeDTO.create(
        feature_comparisons=(spec,),
        observable_feature_keys=(spec.feature_key,),
        hidden_state_keys=(spec.feature_key,) if hidden else (),
        require_policy_consequence_evidence=require_policy,
        wake_on_policy_consequence_mismatch=wake_policy,
        wake_on_observable_mismatch=wake_observable,
    )


def compare(
    spec: ObserverFeatureComparisonDTO,
    predicted: object,
    observed: object,
):
    observation_schema = schema()
    return compare_observer_transition(
        recipe=recipe(spec),
        predicted_observation_artifact_id="obs:predicted",
        observed_observation_artifact_id="obs:observed",
        predicted_observation_schema_id=observation_schema.schema_id,
        observed_observation_schema_id=observation_schema.schema_id,
        predicted_features={spec.feature_key: predicted},
        observed_features={spec.feature_key: observed},
    ).feature_results[0]


@pytest.mark.parametrize(
    ("predicted", "observed", "status"),
    [
        (True, 1, "type_mismatch"),
        (1, 1.0, "type_mismatch"),
        ("1", 1, "type_mismatch"),
        (1, 1, "match"),
    ],
)
def test_strict_exact_comparison(
    predicted: object, observed: object, status: str
) -> None:
    result = compare(comparison("visible.score", mode="exact"), predicted, observed)

    assert result.status == status


def test_numeric_tolerance_absolute_and_relative() -> None:
    tolerant = comparison(
        "visible.score",
        mode="numeric_tolerance",
        expected_type="number",
        absolute_tolerance=1e-12,
        relative_tolerance=0.0,
    )
    strict = comparison(
        "visible.score",
        mode="numeric_tolerance",
        expected_type="number",
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )
    relative = comparison(
        "visible.score",
        mode="numeric_tolerance",
        expected_type="number",
        absolute_tolerance=0.0,
        relative_tolerance=0.1,
    )

    assert compare(tolerant, 0.30000000000000004, 0.3).status == "match"
    assert compare(strict, 0.30000000000000004, 0.3).status == "mismatch"
    assert compare(relative, 100.0, 105.0).status == "match"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf, True])
def test_invalid_numeric_values(value: object) -> None:
    spec = comparison(
        "visible.score",
        mode="numeric_tolerance",
        expected_type="number",
        absolute_tolerance=0.0,
        relative_tolerance=0.0,
    )

    assert compare(spec, value, 1.0).status == "invalid_value"


def test_recipe_identity_changes_with_comparison_semantics() -> None:
    baseline = recipe(comparison("visible.score", mode="exact"))

    variants = (
        recipe(comparison("visible.score", mode="categorical", expected_type="int")),
        recipe(comparison("visible.score", mode="exact", expected_type="int")),
        recipe(
            comparison(
                "visible.score",
                mode="numeric_tolerance",
                expected_type="number",
                absolute_tolerance=0.1,
                relative_tolerance=0.0,
            )
        ),
        recipe(
            comparison(
                "visible.score",
                mode="numeric_tolerance",
                expected_type="number",
                absolute_tolerance=0.0,
                relative_tolerance=0.1,
            )
        ),
    )

    assert all(item.recipe_id != baseline.recipe_id for item in variants)


def test_recipe_rejects_unused_comparison_specs() -> None:
    required = comparison("visible.score", mode="exact")
    extra = comparison("visible.label", mode="exact")

    with pytest.raises(ObserverComparisonError, match="exactly match"):
        ObserverComparisonRecipeDTO.create(
            feature_comparisons=(extra, required),
            observable_feature_keys=("visible.score",),
        )


def test_per_feature_evidence_statuses() -> None:
    observation_schema = schema()
    spec = comparison("visible.label", mode="exact")
    base = {
        "recipe": recipe(spec),
        "predicted_observation_artifact_id": "obs:p",
        "observed_observation_artifact_id": "obs:o",
        "predicted_observation_schema_id": observation_schema.schema_id,
        "observed_observation_schema_id": observation_schema.schema_id,
    }

    assert compare(spec, "a", "a").status == "match"
    assert compare(spec, "a", "b").status == "mismatch"
    assert (
        compare_observer_transition(
            **base, predicted_features={}, observed_features={"visible.label": "a"}
        )
        .feature_results[0]
        .status
        == "missing_predicted"
    )
    assert (
        compare_observer_transition(
            **base, predicted_features={"visible.label": "a"}, observed_features={}
        )
        .feature_results[0]
        .status
        == "missing_observed"
    )
    assert compare(spec, "1", 1).status == "type_mismatch"


def test_observation_schema_enforcement_and_identity() -> None:
    observation_schema = schema(name="one")
    other_schema = schema(name="two")

    with pytest.raises(ObserverArtifactError, match="undeclared"):
        ObserverObservationArtifactDTO.create(
            observation_schema=observation_schema,
            visible_state_features={"unknown": 1},
        )
    required_schema = ObserverObservationSchemaDTO.create(
        schema_name="required",
        features=(
            ObserverFeatureDefinitionDTO.create(
                qualified_key="visible.score", value_type="number", required=True
            ),
        ),
    )
    with pytest.raises(ObserverArtifactError, match="missing required"):
        ObserverObservationArtifactDTO.create(
            observation_schema=required_schema,
            visible_state_features={},
        )
    with pytest.raises(ObserverArtifactError, match="expected number"):
        ObserverObservationArtifactDTO.create(
            observation_schema=observation_schema,
            visible_state_features={"score": True},
        )

    left = ObserverObservationArtifactDTO.create(
        observation_schema=observation_schema,
        visible_state_features={"score": 1},
    )
    right = ObserverObservationArtifactDTO.create(
        observation_schema=other_schema,
        visible_state_features={"score": 1},
    )
    assert left.observation_artifact_id != right.observation_artifact_id


def test_schema_mismatch_is_inconclusive_not_contradiction() -> None:
    left_schema = schema(name="left")
    right_schema = schema(name="right")
    result = compare_observer_transition(
        recipe=recipe(comparison("visible.score", mode="exact")),
        predicted_observation_artifact_id="obs:p",
        observed_observation_artifact_id="obs:o",
        predicted_observation_schema_id=left_schema.schema_id,
        observed_observation_schema_id=right_schema.schema_id,
        predicted_features={"visible.score": 1},
        observed_features={"visible.score": 2},
    )

    assert result.contradiction is False
    assert result.inconclusive_reasons == ("schema_mismatch",)


def test_hypothesis_set_evidence_controls_exhaustion_and_identity() -> None:
    observation_schema = schema()
    spec = comparison("hidden.cooldown", mode="exact")
    possible = ObserverHiddenStateHypothesisSetDTO.create(
        observation_schema_id=observation_schema.schema_id,
        hypotheses=(
            ObserverHiddenStateHypothesisDTO.create(
                state_key="hidden.cooldown",
                state_value="clear",
                status="possible",
                evidence_ids=("evidence:a",),
            ),
        ),
    )
    eliminated = ObserverHiddenStateHypothesisSetDTO.create(
        observation_schema_id=observation_schema.schema_id,
        hypotheses=(
            ObserverHiddenStateHypothesisDTO.create(
                state_key="hidden.cooldown",
                state_value="clear",
                status="eliminated",
                evidence_ids=("evidence:b",),
            ),
        ),
    )

    common = {
        "recipe": recipe(spec, hidden=True),
        "predicted_observation_artifact_id": "obs:p",
        "observed_observation_artifact_id": "obs:o",
        "predicted_observation_schema_id": observation_schema.schema_id,
        "observed_observation_schema_id": observation_schema.schema_id,
        "predicted_features": {"hidden.cooldown": "clear"},
        "observed_features": {"hidden.cooldown": "clear"},
    }
    missing = compare_observer_transition(**common)
    still_possible = compare_observer_transition(
        **common, hidden_state_hypothesis_set=possible
    )
    exhausted = compare_observer_transition(
        **common, hidden_state_hypothesis_set=eliminated
    )

    assert missing.inconclusive_reasons == ("missing_hidden_state_hypothesis_set",)
    assert still_possible.hidden_state_exhausted is False
    assert exhausted.hidden_state_exhausted is True
    assert still_possible.comparison_result_id != exhausted.comparison_result_id


def test_transition_rejects_hypothesis_set_schema_mismatch_with_observed() -> None:
    predicted_schema = schema(name="predicted")
    observed_schema = schema(name="observed")
    predicted = ObserverObservationArtifactDTO.create(
        observation_schema=predicted_schema,
        visible_state_features={"score": 1},
    )
    observed = ObserverObservationArtifactDTO.create(
        observation_schema=observed_schema,
        visible_state_features={"score": 1},
    )
    hypotheses = ObserverHiddenStateHypothesisSetDTO.create(
        observation_schema_id=predicted_schema.schema_id,
        hypotheses=(
            ObserverHiddenStateHypothesisDTO.create(
                state_key="hidden.cooldown", state_value="clear"
            ),
        ),
    )

    visible_score_recipe = ObserverComparisonRecipeDTO.create(
        feature_comparisons=(comparison("visible.score", mode="exact"),),
        observable_feature_keys=("visible.score",),
    )

    with pytest.raises(
        ObserverTransitionVerificationError, match="predicted and observed"
    ):
        verify_observer_transition(
            recipe=visible_score_recipe,
            predicted_observation=predicted,
            observed_observation=observed,
            policy_artifact_id="policy:A",
            state_before_id="state:before",
            action="wait",
            affected_policy_row_id="row:before",
            hidden_state_hypothesis_set=hypotheses,
        )


def test_policy_consequence_evidence_is_required_and_validated() -> None:
    observation_schema = schema()
    spec = comparison("visible.label", mode="exact")
    required = recipe(spec, require_policy=True)
    absent = compare_observer_transition(
        recipe=required,
        predicted_observation_artifact_id="obs:p",
        observed_observation_artifact_id="obs:o",
        predicted_observation_schema_id=observation_schema.schema_id,
        observed_observation_schema_id=observation_schema.schema_id,
        predicted_features={"visible.label": "same", "visible.next_action": "go"},
        observed_features={"visible.label": "same", "visible.next_action": "go"},
    )
    equal = ObserverPolicyConsequenceEvidenceDTO.create(
        policy_artifact_id="policy:A",
        predicted_state_artifact_id="obs:p",
        observed_state_artifact_id="obs:o",
        predicted_selected_action="go",
        observed_selected_action="go",
        predicted_decision_trace_id="trace:p",
        observed_decision_trace_id="trace:o",
        reader_contract_id="reader:v1",
    )
    different = ObserverPolicyConsequenceEvidenceDTO.create(
        policy_artifact_id="policy:A",
        predicted_state_artifact_id="obs:p",
        observed_state_artifact_id="obs:o",
        predicted_selected_action="go",
        observed_selected_action="wait",
        predicted_decision_trace_id="trace:p",
        observed_decision_trace_id="trace:o",
        reader_contract_id="reader:v1",
    )

    assert absent.inconclusive_reasons == ("missing_policy_consequence_evidence",)
    assert absent.contradiction is False
    assert equal.equivalent is True
    assert different.equivalent is False
    with pytest.raises(ObserverComparisonError, match="equivalent"):
        ObserverPolicyConsequenceEvidenceDTO(
            policy_consequence_evidence_id=different.policy_consequence_evidence_id,
            policy_artifact_id="policy:A",
            predicted_state_artifact_id="obs:p",
            observed_state_artifact_id="obs:o",
            predicted_selected_action="go",
            observed_selected_action="wait",
            predicted_decision_trace_id="trace:p",
            observed_decision_trace_id="trace:o",
            equivalent=True,
            reader_contract_id="reader:v1",
        )


def test_contradiction_and_wake_independence_and_replay() -> None:
    observation_schema = schema()
    spec = comparison("visible.score", mode="exact")
    no_wake = compare_observer_transition(
        recipe=recipe(spec, wake_observable=False),
        predicted_observation_artifact_id="obs:p",
        observed_observation_artifact_id="obs:o",
        predicted_observation_schema_id=observation_schema.schema_id,
        observed_observation_schema_id=observation_schema.schema_id,
        predicted_features={"visible.score": 1},
        observed_features={"visible.score": 2},
    )
    missing = compare_observer_transition(
        recipe=recipe(spec),
        predicted_observation_artifact_id="obs:p",
        observed_observation_artifact_id="obs:o",
        predicted_observation_schema_id=observation_schema.schema_id,
        observed_observation_schema_id=observation_schema.schema_id,
        predicted_features={},
        observed_features={},
    )
    replay = compare_observer_transition(
        recipe=recipe(spec, wake_observable=False),
        predicted_observation_artifact_id="obs:p",
        observed_observation_artifact_id="obs:o",
        predicted_observation_schema_id=observation_schema.schema_id,
        observed_observation_schema_id=observation_schema.schema_id,
        predicted_features={"visible.score": 1},
        observed_features={"visible.score": 2},
    )

    assert no_wake.contradiction is True
    assert no_wake.wake_required is False
    assert missing.contradiction is False
    assert missing.wake_required is True
    assert no_wake.comparison_result_id == replay.comparison_result_id


def test_legacy_recipe_version_is_rejected() -> None:
    spec = comparison("visible.score", mode="exact")
    payload = recipe(spec).canonical_payload(include_id=False)
    payload["version"] = "observer-comparison-recipe/1"

    with pytest.raises(ObserverComparisonError, match="legacy comparison recipes"):
        ObserverComparisonRecipeDTO(
            recipe_id=recipe(spec).recipe_id,
            feature_comparisons=(spec,),
            observable_feature_keys=("visible.score",),
            version="observer-comparison-recipe/1",
        )
