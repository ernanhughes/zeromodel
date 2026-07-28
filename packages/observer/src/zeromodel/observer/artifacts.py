"""Canonical Observer artifacts and lineage records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.comparison import ObserverComparisonResultDTO

OBSERVER_OBSERVATION_ARTIFACT_VERSION: Final = "observer-observation-artifact/1"
OBSERVER_TRANSITION_RECORD_VERSION: Final = "observer-transition-record/1"
OBSERVER_CONTRADICTION_ARTIFACT_VERSION: Final = "observer-contradiction-artifact/1"
OBSERVER_REPLACEMENT_POLICY_VERSION: Final = "observer-replacement-policy/1"


class ObserverArtifactError(ValueError):
    """Raised when an Observer artifact DTO is invalid."""


def _validate_mapping(value: Mapping[str, object], field_name: str) -> None:
    if not isinstance(value, Mapping):
        raise ObserverArtifactError(f"{field_name} must be a mapping")
    for key in value:
        if not isinstance(key, str) or not key:
            raise ObserverArtifactError(f"{field_name} keys must be non-empty strings")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverArtifactError(f"{field_name} must be unique and sorted")


@dataclass(frozen=True)
class ObserverObservationArtifactDTO:
    """Encoded observation state separated from policy action scores."""

    observation_artifact_id: str
    visible_state_features: Mapping[str, object]
    recent_history_features: Mapping[str, object]
    hidden_state_uncertainty: Mapping[str, object]
    provenance: Mapping[str, object]
    sequence_index: int
    version: str = OBSERVER_OBSERVATION_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_OBSERVATION_ARTIFACT_VERSION:
            raise ObserverArtifactError("unsupported observation artifact version")
        if self.sequence_index < 0:
            raise ObserverArtifactError("sequence_index must be non-negative")
        _validate_mapping(self.visible_state_features, "visible_state_features")
        _validate_mapping(self.recent_history_features, "recent_history_features")
        _validate_mapping(self.hidden_state_uncertainty, "hidden_state_uncertainty")
        _validate_mapping(self.provenance, "provenance")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.observation_artifact_id != expected_id:
            raise ObserverArtifactError(
                "observation_artifact_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "hidden_state_uncertainty": dict(self.hidden_state_uncertainty),
            "provenance": dict(self.provenance),
            "recent_history_features": dict(self.recent_history_features),
            "sequence_index": self.sequence_index,
            "version": self.version,
            "visible_state_features": dict(self.visible_state_features),
        }
        if include_id:
            payload["observation_artifact_id"] = self.observation_artifact_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        visible_state_features: Mapping[str, object],
        recent_history_features: Mapping[str, object] | None = None,
        hidden_state_uncertainty: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
        sequence_index: int = 0,
    ) -> "ObserverObservationArtifactDTO":
        payload = {
            "hidden_state_uncertainty": dict(hidden_state_uncertainty or {}),
            "provenance": dict(provenance or {}),
            "recent_history_features": dict(recent_history_features or {}),
            "sequence_index": sequence_index,
            "version": OBSERVER_OBSERVATION_ARTIFACT_VERSION,
            "visible_state_features": dict(visible_state_features),
        }
        return cls(observation_artifact_id=canonical_id(payload), **payload)


@dataclass(frozen=True)
class ObserverTransitionRecordDTO:
    """Immutable record for one predicted-versus-observed transition."""

    transition_record_id: str
    policy_artifact_id: str
    state_before_id: str
    action: str
    predicted_state_after_id: str
    observed_state_after_id: str
    comparison_recipe_id: str
    comparison_result_id: str
    verification_status: str
    affected_policy_row_id: str
    version: str = OBSERVER_TRANSITION_RECORD_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_TRANSITION_RECORD_VERSION:
            raise ObserverArtifactError("unsupported transition record version")
        for name in (
            "policy_artifact_id",
            "state_before_id",
            "action",
            "predicted_state_after_id",
            "observed_state_after_id",
            "comparison_recipe_id",
            "comparison_result_id",
            "verification_status",
            "affected_policy_row_id",
        ):
            if not getattr(self, name):
                raise ObserverArtifactError(f"{name} must be non-empty")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.transition_record_id != expected_id:
            raise ObserverArtifactError(
                "transition_record_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = {
            "action": self.action,
            "affected_policy_row_id": self.affected_policy_row_id,
            "comparison_recipe_id": self.comparison_recipe_id,
            "comparison_result_id": self.comparison_result_id,
            "observed_state_after_id": self.observed_state_after_id,
            "policy_artifact_id": self.policy_artifact_id,
            "predicted_state_after_id": self.predicted_state_after_id,
            "state_before_id": self.state_before_id,
            "verification_status": self.verification_status,
            "version": self.version,
        }
        if include_id:
            payload["transition_record_id"] = self.transition_record_id
        return payload


def build_transition_record(
    *,
    policy_artifact_id: str,
    state_before_id: str,
    action: str,
    predicted_state_after_id: str,
    observed_state_after_id: str,
    comparison_recipe_id: str,
    comparison_result_id: str,
    verification_status: str,
    affected_policy_row_id: str,
) -> ObserverTransitionRecordDTO:
    payload = {
        "action": action,
        "affected_policy_row_id": affected_policy_row_id,
        "comparison_recipe_id": comparison_recipe_id,
        "comparison_result_id": comparison_result_id,
        "observed_state_after_id": observed_state_after_id,
        "policy_artifact_id": policy_artifact_id,
        "predicted_state_after_id": predicted_state_after_id,
        "state_before_id": state_before_id,
        "verification_status": verification_status,
        "version": OBSERVER_TRANSITION_RECORD_VERSION,
    }
    return ObserverTransitionRecordDTO(
        transition_record_id=canonical_id(payload), **payload
    )


@dataclass(frozen=True)
class ObserverContradictionArtifactDTO:
    """Content-addressed evidence that a policy prediction failed."""

    contradiction_artifact_id: str
    transition_record_id: str
    source_policy_artifact_id: str
    state_before_id: str
    action: str
    predicted_state_after_id: str
    observed_state_after_id: str
    comparison_result_id: str
    affected_policy_row_id: str
    reproduction: Mapping[str, object]
    relevant_context_keys: tuple[str, ...] = ()
    version: str = OBSERVER_CONTRADICTION_ARTIFACT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_CONTRADICTION_ARTIFACT_VERSION:
            raise ObserverArtifactError("unsupported contradiction artifact version")
        for name in (
            "transition_record_id",
            "source_policy_artifact_id",
            "state_before_id",
            "action",
            "predicted_state_after_id",
            "observed_state_after_id",
            "comparison_result_id",
            "affected_policy_row_id",
        ):
            if not getattr(self, name):
                raise ObserverArtifactError(f"{name} must be non-empty")
        _validate_mapping(self.reproduction, "reproduction")
        _ensure_sorted_unique(self.relevant_context_keys, "relevant_context_keys")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.contradiction_artifact_id != expected_id:
            raise ObserverArtifactError(
                "contradiction_artifact_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "action": self.action,
            "affected_policy_row_id": self.affected_policy_row_id,
            "comparison_result_id": self.comparison_result_id,
            "observed_state_after_id": self.observed_state_after_id,
            "predicted_state_after_id": self.predicted_state_after_id,
            "relevant_context_keys": list(self.relevant_context_keys),
            "reproduction": dict(self.reproduction),
            "source_policy_artifact_id": self.source_policy_artifact_id,
            "state_before_id": self.state_before_id,
            "transition_record_id": self.transition_record_id,
            "version": self.version,
        }
        if include_id:
            payload["contradiction_artifact_id"] = self.contradiction_artifact_id
        return payload


def build_contradiction_artifact(
    *,
    transition: ObserverTransitionRecordDTO,
    comparison: ObserverComparisonResultDTO,
    reproduction: Mapping[str, object] | None = None,
    relevant_context_keys: tuple[str, ...] = (),
) -> ObserverContradictionArtifactDTO:
    if not comparison.contradiction:
        raise ObserverArtifactError("comparison result is not a contradiction")
    if transition.comparison_result_id != comparison.comparison_result_id:
        raise ObserverArtifactError("transition and comparison result do not match")
    payload = {
        "action": transition.action,
        "affected_policy_row_id": transition.affected_policy_row_id,
        "comparison_result_id": transition.comparison_result_id,
        "observed_state_after_id": transition.observed_state_after_id,
        "predicted_state_after_id": transition.predicted_state_after_id,
        "relevant_context_keys": list(relevant_context_keys),
        "reproduction": dict(reproduction or {}),
        "source_policy_artifact_id": transition.policy_artifact_id,
        "state_before_id": transition.state_before_id,
        "transition_record_id": transition.transition_record_id,
        "version": OBSERVER_CONTRADICTION_ARTIFACT_VERSION,
    }
    return ObserverContradictionArtifactDTO(
        contradiction_artifact_id=canonical_id(payload),
        transition_record_id=transition.transition_record_id,
        source_policy_artifact_id=transition.policy_artifact_id,
        state_before_id=transition.state_before_id,
        action=transition.action,
        predicted_state_after_id=transition.predicted_state_after_id,
        observed_state_after_id=transition.observed_state_after_id,
        comparison_result_id=transition.comparison_result_id,
        affected_policy_row_id=transition.affected_policy_row_id,
        reproduction=dict(reproduction or {}),
        relevant_context_keys=relevant_context_keys,
    )


@dataclass(frozen=True)
class ObserverReplacementPolicyArtifactDTO:
    """Explicit lineage for a policy artifact replacing its parent."""

    replacement_record_id: str
    parent_policy_artifact_id: str
    replacement_policy_artifact_id: str
    relation: str
    contradiction_artifact_id: str
    changed_row_ids: tuple[str, ...]
    changed_cell_ids: tuple[str, ...]
    verified_result_ids: tuple[str, ...]
    unchanged_region_result_id: str
    status: str = "active"
    version: str = OBSERVER_REPLACEMENT_POLICY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_REPLACEMENT_POLICY_VERSION:
            raise ObserverArtifactError("unsupported replacement policy version")
        if self.relation not in {"repairs", "replaces", "refines"}:
            raise ObserverArtifactError("unsupported replacement relation")
        if self.status not in {"candidate", "active", "rejected"}:
            raise ObserverArtifactError("unsupported replacement status")
        for name in (
            "parent_policy_artifact_id",
            "replacement_policy_artifact_id",
            "contradiction_artifact_id",
            "unchanged_region_result_id",
        ):
            if not getattr(self, name):
                raise ObserverArtifactError(f"{name} must be non-empty")
        _ensure_sorted_unique(self.changed_row_ids, "changed_row_ids")
        _ensure_sorted_unique(self.changed_cell_ids, "changed_cell_ids")
        _ensure_sorted_unique(self.verified_result_ids, "verified_result_ids")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.replacement_record_id != expected_id:
            raise ObserverArtifactError(
                "replacement_record_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "changed_cell_ids": list(self.changed_cell_ids),
            "changed_row_ids": list(self.changed_row_ids),
            "contradiction_artifact_id": self.contradiction_artifact_id,
            "parent_policy_artifact_id": self.parent_policy_artifact_id,
            "relation": self.relation,
            "replacement_policy_artifact_id": self.replacement_policy_artifact_id,
            "status": self.status,
            "unchanged_region_result_id": self.unchanged_region_result_id,
            "verified_result_ids": list(self.verified_result_ids),
            "version": self.version,
        }
        if include_id:
            payload["replacement_record_id"] = self.replacement_record_id
        return payload


def build_replacement_policy_artifact(
    *,
    parent_policy_artifact_id: str,
    replacement_policy_artifact_id: str,
    contradiction_artifact_id: str,
    changed_row_ids: tuple[str, ...],
    changed_cell_ids: tuple[str, ...],
    verified_result_ids: tuple[str, ...],
    unchanged_region_result_id: str,
    relation: str = "repairs",
    status: str = "active",
) -> ObserverReplacementPolicyArtifactDTO:
    payload = {
        "changed_cell_ids": list(changed_cell_ids),
        "changed_row_ids": list(changed_row_ids),
        "contradiction_artifact_id": contradiction_artifact_id,
        "parent_policy_artifact_id": parent_policy_artifact_id,
        "relation": relation,
        "replacement_policy_artifact_id": replacement_policy_artifact_id,
        "status": status,
        "unchanged_region_result_id": unchanged_region_result_id,
        "verified_result_ids": list(verified_result_ids),
        "version": OBSERVER_REPLACEMENT_POLICY_VERSION,
    }
    return ObserverReplacementPolicyArtifactDTO(
        replacement_record_id=canonical_id(payload),
        parent_policy_artifact_id=parent_policy_artifact_id,
        replacement_policy_artifact_id=replacement_policy_artifact_id,
        relation=relation,
        contradiction_artifact_id=contradiction_artifact_id,
        changed_row_ids=changed_row_ids,
        changed_cell_ids=changed_cell_ids,
        verified_result_ids=verified_result_ids,
        unchanged_region_result_id=unchanged_region_result_id,
        status=status,
    )
