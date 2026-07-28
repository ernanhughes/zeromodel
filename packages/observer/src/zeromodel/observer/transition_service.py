"""Application service for Observer transition verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import (
    ObserverArtifactError,
    ObserverContradictionArtifactDTO,
    ObserverObservationArtifactDTO,
    ObserverTransitionRecordDTO,
    build_contradiction_artifact,
    build_transition_record,
)
from zeromodel.observer.comparison import (
    ObserverComparisonError,
    ObserverComparisonRecipeDTO,
    ObserverComparisonResultDTO,
    compare_observer_transition,
)

OBSERVER_TRANSITION_VERIFICATION_VERSION: Final = "observer-transition-verification/1"
OBSERVER_VERIFICATION_CONFIRMED: Final = "confirmed"
OBSERVER_VERIFICATION_CONTRADICTED: Final = "contradicted"
OBSERVER_VERIFICATION_INCONCLUSIVE: Final = "inconclusive"
OBSERVER_VERIFICATION_STATUSES: Final = frozenset(
    {
        OBSERVER_VERIFICATION_CONFIRMED,
        OBSERVER_VERIFICATION_CONTRADICTED,
        OBSERVER_VERIFICATION_INCONCLUSIVE,
    }
)


class ObserverTransitionVerificationError(ValueError):
    """Raised when Observer transition verification inputs are inconsistent."""


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverTransitionVerificationError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverTransitionVerificationError(
            f"{field_name} must be unique and sorted"
        )


def _project_feature_surface(
    observation: ObserverObservationArtifactDTO,
) -> Mapping[str, object]:
    projected: dict[str, object] = {}
    for prefix, source in (
        ("visible", observation.visible_state_features),
        ("history", observation.recent_history_features),
        ("hidden", observation.hidden_state_uncertainty),
    ):
        for key, value in source.items():
            projected[f"{prefix}.{key}"] = value
    return projected


def _derive_status(comparison: ObserverComparisonResultDTO) -> str:
    if comparison.contradiction:
        return OBSERVER_VERIFICATION_CONTRADICTED
    if (
        comparison.missing_predicted_feature_keys
        or comparison.missing_observed_feature_keys
        or comparison.wake_required
    ):
        return OBSERVER_VERIFICATION_INCONCLUSIVE
    return OBSERVER_VERIFICATION_CONFIRMED


@dataclass(frozen=True)
class ObserverTransitionVerificationDTO:
    """Complete canonical outcome of one Observer transition verification."""

    verification_id: str
    recipe_id: str
    predicted_observation_artifact_id: str
    observed_observation_artifact_id: str
    comparison_result: ObserverComparisonResultDTO
    transition_record: ObserverTransitionRecordDTO
    contradiction_artifact: ObserverContradictionArtifactDTO | None
    verification_status: str
    version: str = OBSERVER_TRANSITION_VERIFICATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_TRANSITION_VERIFICATION_VERSION:
            raise ObserverTransitionVerificationError(
                "unsupported transition verification version"
            )
        if self.verification_status not in OBSERVER_VERIFICATION_STATUSES:
            raise ObserverTransitionVerificationError(
                f"unsupported verification_status: {self.verification_status!r}"
            )
        for name in (
            "recipe_id",
            "predicted_observation_artifact_id",
            "observed_observation_artifact_id",
        ):
            _require_non_empty(getattr(self, name), name)
        if self.comparison_result.recipe_id != self.recipe_id:
            raise ObserverTransitionVerificationError(
                "recipe_id does not match comparison_result.recipe_id"
            )
        if self.transition_record.comparison_recipe_id != self.recipe_id:
            raise ObserverTransitionVerificationError(
                "transition record does not reference the verification recipe"
            )
        if (
            self.transition_record.comparison_result_id
            != self.comparison_result.comparison_result_id
        ):
            raise ObserverTransitionVerificationError(
                "transition record does not reference the comparison result"
            )
        if (
            self.transition_record.predicted_state_after_id
            != self.predicted_observation_artifact_id
        ):
            raise ObserverTransitionVerificationError(
                "transition record predicted_state_after_id does not reference "
                "the predicted observation artifact"
            )
        if (
            self.transition_record.observed_state_after_id
            != self.observed_observation_artifact_id
        ):
            raise ObserverTransitionVerificationError(
                "transition record observed_state_after_id does not reference "
                "the observed observation artifact"
            )
        if self.verification_status == OBSERVER_VERIFICATION_CONTRADICTED:
            if self.contradiction_artifact is None:
                raise ObserverTransitionVerificationError(
                    "contradicted verification requires a contradiction artifact"
                )
            if (
                self.contradiction_artifact.comparison_result_id
                != self.comparison_result.comparison_result_id
            ):
                raise ObserverTransitionVerificationError(
                    "contradiction artifact does not reference the comparison result"
                )
            if (
                self.contradiction_artifact.transition_record_id
                != self.transition_record.transition_record_id
            ):
                raise ObserverTransitionVerificationError(
                    "contradiction artifact does not reference the transition record"
                )
        elif self.contradiction_artifact is not None:
            raise ObserverTransitionVerificationError(
                "confirmed or inconclusive verification cannot carry a "
                "contradiction artifact"
            )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.verification_id != expected_id:
            raise ObserverTransitionVerificationError(
                "verification_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "comparison_result": self.comparison_result.canonical_payload(),
            "contradiction_artifact": (
                None
                if self.contradiction_artifact is None
                else self.contradiction_artifact.canonical_payload()
            ),
            "observed_observation_artifact_id": (self.observed_observation_artifact_id),
            "predicted_observation_artifact_id": (
                self.predicted_observation_artifact_id
            ),
            "recipe_id": self.recipe_id,
            "transition_record": self.transition_record.canonical_payload(),
            "verification_status": self.verification_status,
            "version": self.version,
        }
        if include_id:
            payload["verification_id"] = self.verification_id
        return payload


def verify_observer_transition(
    *,
    recipe: ObserverComparisonRecipeDTO,
    predicted_observation: ObserverObservationArtifactDTO,
    observed_observation: ObserverObservationArtifactDTO,
    policy_artifact_id: str,
    state_before_id: str,
    action: str,
    affected_policy_row_id: str,
    predicted_decision_margin: float,
    observed_decision_margin: float,
    hidden_state_hypotheses_remaining: int,
    reproduction: Mapping[str, object] | None = None,
    relevant_context_keys: tuple[str, ...] = (),
) -> ObserverTransitionVerificationDTO:
    """Verify one predicted observation against the next observed observation."""

    for field_name, value in (
        ("policy_artifact_id", policy_artifact_id),
        ("state_before_id", state_before_id),
        ("action", action),
        ("affected_policy_row_id", affected_policy_row_id),
    ):
        _require_non_empty(value, field_name)
    _ensure_sorted_unique(relevant_context_keys, "relevant_context_keys")
    if observed_observation.sequence_index != predicted_observation.sequence_index:
        raise ObserverTransitionVerificationError(
            "predicted and observed observations must describe the same "
            "target sequence position "
            f"(predicted {predicted_observation.sequence_index}, "
            f"observed {observed_observation.sequence_index})"
        )

    try:
        comparison = compare_observer_transition(
            recipe=recipe,
            predicted_features=_project_feature_surface(predicted_observation),
            observed_features=_project_feature_surface(observed_observation),
            predicted_decision_margin=predicted_decision_margin,
            observed_decision_margin=observed_decision_margin,
            hidden_state_hypotheses_remaining=hidden_state_hypotheses_remaining,
        )
        verification_status = _derive_status(comparison)
        transition = build_transition_record(
            policy_artifact_id=policy_artifact_id,
            state_before_id=state_before_id,
            action=action,
            predicted_state_after_id=(predicted_observation.observation_artifact_id),
            observed_state_after_id=observed_observation.observation_artifact_id,
            comparison_recipe_id=recipe.recipe_id,
            comparison_result_id=comparison.comparison_result_id,
            verification_status=verification_status,
            affected_policy_row_id=affected_policy_row_id,
        )
        contradiction = None
        if verification_status == OBSERVER_VERIFICATION_CONTRADICTED:
            contradiction = build_contradiction_artifact(
                transition=transition,
                comparison=comparison,
                reproduction=reproduction,
                relevant_context_keys=relevant_context_keys,
            )
    except (ObserverArtifactError, ObserverComparisonError) as exc:
        raise ObserverTransitionVerificationError(str(exc)) from exc

    payload = {
        "comparison_result": comparison.canonical_payload(),
        "contradiction_artifact": (
            None if contradiction is None else contradiction.canonical_payload()
        ),
        "observed_observation_artifact_id": (
            observed_observation.observation_artifact_id
        ),
        "predicted_observation_artifact_id": (
            predicted_observation.observation_artifact_id
        ),
        "recipe_id": recipe.recipe_id,
        "transition_record": transition.canonical_payload(),
        "verification_status": verification_status,
        "version": OBSERVER_TRANSITION_VERIFICATION_VERSION,
    }
    return ObserverTransitionVerificationDTO(
        verification_id=canonical_id(payload),
        recipe_id=recipe.recipe_id,
        predicted_observation_artifact_id=(
            predicted_observation.observation_artifact_id
        ),
        observed_observation_artifact_id=observed_observation.observation_artifact_id,
        comparison_result=comparison,
        transition_record=transition,
        contradiction_artifact=contradiction,
        verification_status=verification_status,
    )
