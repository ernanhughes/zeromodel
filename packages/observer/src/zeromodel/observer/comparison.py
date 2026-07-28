"""Structured comparison recipes for Observer transition checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id

OBSERVER_COMPARISON_RECIPE_VERSION: Final = "observer-comparison-recipe/1"
OBSERVER_COMPARISON_RESULT_VERSION: Final = "observer-comparison-result/1"


class ObserverComparisonError(ValueError):
    """Raised when an Observer comparison DTO is invalid."""


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverComparisonError(f"{field_name} must be unique and sorted")


@dataclass(frozen=True)
class ObserverComparisonRecipeDTO:
    """Declared hypothesis for comparing predicted and observed transitions."""

    recipe_id: str
    observable_feature_keys: tuple[str, ...]
    action_effect_keys: tuple[str, ...] = ()
    policy_consequence_key: str | None = None
    hidden_state_keys: tuple[str, ...] = ()
    decision_margin_tolerance: float = 0.0
    wake_on_observable_mismatch: bool = True
    wake_on_action_effect_mismatch: bool = True
    wake_on_policy_consequence_mismatch: bool = False
    wake_on_hidden_state_exhausted: bool = True
    version: str = OBSERVER_COMPARISON_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_COMPARISON_RECIPE_VERSION:
            raise ObserverComparisonError("unsupported comparison recipe version")
        if not self.observable_feature_keys:
            raise ObserverComparisonError("observable_feature_keys must be non-empty")
        _ensure_sorted_unique(self.observable_feature_keys, "observable_feature_keys")
        _ensure_sorted_unique(self.action_effect_keys, "action_effect_keys")
        _ensure_sorted_unique(self.hidden_state_keys, "hidden_state_keys")
        if self.policy_consequence_key == "":
            raise ObserverComparisonError("policy_consequence_key cannot be empty")
        if self.decision_margin_tolerance < 0.0:
            raise ObserverComparisonError(
                "decision_margin_tolerance must be non-negative"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.recipe_id != expected_id:
            raise ObserverComparisonError("recipe_id disagrees with canonical payload")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "action_effect_keys": list(self.action_effect_keys),
            "decision_margin_tolerance": self.decision_margin_tolerance,
            "hidden_state_keys": list(self.hidden_state_keys),
            "observable_feature_keys": list(self.observable_feature_keys),
            "policy_consequence_key": self.policy_consequence_key,
            "version": self.version,
            "wake_on_action_effect_mismatch": self.wake_on_action_effect_mismatch,
            "wake_on_hidden_state_exhausted": self.wake_on_hidden_state_exhausted,
            "wake_on_observable_mismatch": self.wake_on_observable_mismatch,
            "wake_on_policy_consequence_mismatch": (
                self.wake_on_policy_consequence_mismatch
            ),
        }
        if include_id:
            payload["recipe_id"] = self.recipe_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        observable_feature_keys: tuple[str, ...],
        action_effect_keys: tuple[str, ...] = (),
        policy_consequence_key: str | None = None,
        hidden_state_keys: tuple[str, ...] = (),
        decision_margin_tolerance: float = 0.0,
        wake_on_observable_mismatch: bool = True,
        wake_on_action_effect_mismatch: bool = True,
        wake_on_policy_consequence_mismatch: bool = False,
        wake_on_hidden_state_exhausted: bool = True,
    ) -> "ObserverComparisonRecipeDTO":
        payload = {
            "action_effect_keys": list(action_effect_keys),
            "decision_margin_tolerance": decision_margin_tolerance,
            "hidden_state_keys": list(hidden_state_keys),
            "observable_feature_keys": list(observable_feature_keys),
            "policy_consequence_key": policy_consequence_key,
            "version": OBSERVER_COMPARISON_RECIPE_VERSION,
            "wake_on_action_effect_mismatch": wake_on_action_effect_mismatch,
            "wake_on_hidden_state_exhausted": wake_on_hidden_state_exhausted,
            "wake_on_observable_mismatch": wake_on_observable_mismatch,
            "wake_on_policy_consequence_mismatch": (
                wake_on_policy_consequence_mismatch
            ),
        }
        return cls(
            recipe_id=canonical_id(payload),
            observable_feature_keys=observable_feature_keys,
            action_effect_keys=action_effect_keys,
            policy_consequence_key=policy_consequence_key,
            hidden_state_keys=hidden_state_keys,
            decision_margin_tolerance=decision_margin_tolerance,
            wake_on_observable_mismatch=wake_on_observable_mismatch,
            wake_on_action_effect_mismatch=wake_on_action_effect_mismatch,
            wake_on_policy_consequence_mismatch=(wake_on_policy_consequence_mismatch),
            wake_on_hidden_state_exhausted=wake_on_hidden_state_exhausted,
        )


@dataclass(frozen=True)
class ObserverComparisonResultDTO:
    """Structured transition comparison result."""

    comparison_result_id: str
    recipe_id: str
    observable_feature_match: bool
    action_effect_match: bool
    next_action_equivalent: bool
    decision_margin_delta: float
    hidden_state_hypotheses_remaining: int
    mismatched_feature_keys: tuple[str, ...]
    wake_required: bool
    contradiction: bool
    missing_predicted_feature_keys: tuple[str, ...] = ()
    missing_observed_feature_keys: tuple[str, ...] = ()
    version: str = OBSERVER_COMPARISON_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_COMPARISON_RESULT_VERSION:
            raise ObserverComparisonError("unsupported comparison result version")
        if not self.recipe_id:
            raise ObserverComparisonError("recipe_id must be non-empty")
        if self.decision_margin_delta < 0.0:
            raise ObserverComparisonError("decision_margin_delta must be non-negative")
        if self.hidden_state_hypotheses_remaining < 0:
            raise ObserverComparisonError(
                "hidden_state_hypotheses_remaining must be non-negative"
            )
        _ensure_sorted_unique(self.mismatched_feature_keys, "mismatched_feature_keys")
        _ensure_sorted_unique(
            self.missing_predicted_feature_keys, "missing_predicted_feature_keys"
        )
        _ensure_sorted_unique(
            self.missing_observed_feature_keys, "missing_observed_feature_keys"
        )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.comparison_result_id != expected_id:
            raise ObserverComparisonError(
                "comparison_result_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "action_effect_match": self.action_effect_match,
            "contradiction": self.contradiction,
            "decision_margin_delta": self.decision_margin_delta,
            "hidden_state_hypotheses_remaining": (
                self.hidden_state_hypotheses_remaining
            ),
            "mismatched_feature_keys": list(self.mismatched_feature_keys),
            "missing_observed_feature_keys": list(self.missing_observed_feature_keys),
            "missing_predicted_feature_keys": list(self.missing_predicted_feature_keys),
            "next_action_equivalent": self.next_action_equivalent,
            "observable_feature_match": self.observable_feature_match,
            "recipe_id": self.recipe_id,
            "version": self.version,
            "wake_required": self.wake_required,
        }
        if include_id:
            payload["comparison_result_id"] = self.comparison_result_id
        return payload


def compare_observer_transition(
    *,
    recipe: ObserverComparisonRecipeDTO,
    predicted_features: Mapping[str, object],
    observed_features: Mapping[str, object],
    predicted_decision_margin: float,
    observed_decision_margin: float,
    hidden_state_hypotheses_remaining: int,
) -> ObserverComparisonResultDTO:
    """Compare predicted and observed state without collapsing to one scalar."""

    if predicted_decision_margin < 0.0 or observed_decision_margin < 0.0:
        raise ObserverComparisonError("decision margins must be non-negative")
    if hidden_state_hypotheses_remaining < 0:
        raise ObserverComparisonError(
            "hidden_state_hypotheses_remaining must be non-negative"
        )

    required_keys = tuple(
        sorted(
            set(recipe.observable_feature_keys)
            | set(recipe.action_effect_keys)
            | set(recipe.hidden_state_keys)
            | (
                {recipe.policy_consequence_key}
                if recipe.policy_consequence_key is not None
                else set()
            )
        )
    )
    missing_predicted = tuple(
        key for key in required_keys if key not in predicted_features
    )
    missing_observed = tuple(
        key for key in required_keys if key not in observed_features
    )
    missing_evidence = bool(missing_predicted or missing_observed)

    observable_mismatches = tuple(
        key
        for key in recipe.observable_feature_keys
        if key not in missing_predicted
        and key not in missing_observed
        and predicted_features[key] != observed_features[key]
    )
    action_effect_mismatches = tuple(
        key
        for key in recipe.action_effect_keys
        if key not in missing_predicted
        and key not in missing_observed
        and predicted_features[key] != observed_features[key]
    )
    policy_mismatch = (
        recipe.policy_consequence_key is not None
        and recipe.policy_consequence_key not in missing_predicted
        and recipe.policy_consequence_key not in missing_observed
        and predicted_features[recipe.policy_consequence_key]
        != observed_features[recipe.policy_consequence_key]
    )
    mismatches = tuple(
        sorted(
            set(observable_mismatches)
            | set(action_effect_mismatches)
            | (
                {recipe.policy_consequence_key}
                if policy_mismatch and recipe.policy_consequence_key is not None
                else set()
            )
        )
    )
    observable_feature_match = not observable_mismatches and not any(
        key in missing_predicted or key in missing_observed
        for key in recipe.observable_feature_keys
    )
    action_effect_match = not action_effect_mismatches and not any(
        key in missing_predicted or key in missing_observed
        for key in recipe.action_effect_keys
    )
    next_action_equivalent = not policy_mismatch
    if recipe.policy_consequence_key is not None and (
        recipe.policy_consequence_key in missing_predicted
        or recipe.policy_consequence_key in missing_observed
    ):
        next_action_equivalent = False
    hidden_state_exhausted = (
        bool(recipe.hidden_state_keys) and hidden_state_hypotheses_remaining == 0
    )
    decision_margin_delta = abs(predicted_decision_margin - observed_decision_margin)
    contradiction = False
    if not missing_evidence:
        contradiction = (
            not observable_feature_match
            or not action_effect_match
            or hidden_state_exhausted
            or (
                not next_action_equivalent
                and decision_margin_delta > recipe.decision_margin_tolerance
            )
        )
    wake_required = (
        missing_evidence
        or (recipe.wake_on_observable_mismatch and not observable_feature_match)
        or (recipe.wake_on_action_effect_mismatch and not action_effect_match)
        or (recipe.wake_on_policy_consequence_mismatch and not next_action_equivalent)
        or (recipe.wake_on_hidden_state_exhausted and hidden_state_exhausted)
    )
    payload = {
        "action_effect_match": action_effect_match,
        "contradiction": contradiction,
        "decision_margin_delta": decision_margin_delta,
        "hidden_state_hypotheses_remaining": hidden_state_hypotheses_remaining,
        "mismatched_feature_keys": list(mismatches),
        "missing_observed_feature_keys": list(missing_observed),
        "missing_predicted_feature_keys": list(missing_predicted),
        "next_action_equivalent": next_action_equivalent,
        "observable_feature_match": observable_feature_match,
        "recipe_id": recipe.recipe_id,
        "version": OBSERVER_COMPARISON_RESULT_VERSION,
        "wake_required": wake_required,
    }
    return ObserverComparisonResultDTO(
        comparison_result_id=canonical_id(payload),
        recipe_id=recipe.recipe_id,
        observable_feature_match=observable_feature_match,
        action_effect_match=action_effect_match,
        next_action_equivalent=next_action_equivalent,
        decision_margin_delta=decision_margin_delta,
        hidden_state_hypotheses_remaining=hidden_state_hypotheses_remaining,
        mismatched_feature_keys=mismatches,
        wake_required=wake_required,
        contradiction=contradiction,
        missing_predicted_feature_keys=missing_predicted,
        missing_observed_feature_keys=missing_observed,
    )
