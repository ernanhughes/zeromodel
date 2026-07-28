"""Canonical Observer artifacts and lineage records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.comparison import ObserverComparisonResultDTO

OBSERVER_FEATURE_DEFINITION_VERSION: Final = "observer-feature-definition/1"
OBSERVER_OBSERVATION_SCHEMA_VERSION: Final = "observer-observation-schema/1"
OBSERVER_OBSERVATION_ARTIFACT_VERSION: Final = "observer-observation-artifact/2"
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


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverArtifactError(f"{field_name} must be non-empty")


def _validate_qualified_key(value: str) -> str:
    if "." not in value:
        raise ObserverArtifactError(f"qualified feature key is malformed: {value!r}")
    namespace, local = value.split(".", 1)
    if namespace not in {"visible", "history", "hidden"} or not local:
        raise ObserverArtifactError(f"qualified feature key is malformed: {value!r}")
    return namespace


def _type_name(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    return type(value).__name__


def _is_value_type(value: object, value_type: str) -> bool:
    if value_type == "bool":
        return isinstance(value, bool)
    if value_type == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if value_type == "float":
        return isinstance(value, float) and math.isfinite(value)
    if value_type == "number":
        return (
            isinstance(value, Real)
            and not isinstance(value, bool)
            and math.isfinite(value)
        )
    if value_type == "str":
        return isinstance(value, str)
    if value_type == "none":
        return value is None
    return False


def _ensure_canonical_value(value: object, field_name: str) -> None:
    from zeromodel.observer._canonical import canonical_json

    try:
        canonical_json({"value": value})
    except (TypeError, ValueError) as exc:
        raise ObserverArtifactError(
            f"{field_name} must be canonical JSON compatible"
        ) from exc


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverArtifactError(f"{field_name} must be unique and sorted")


@dataclass(frozen=True)
class ObserverFeatureDefinitionDTO:
    """Declared schema entry for one observation feature."""

    feature_definition_id: str
    qualified_key: str
    namespace: str
    value_type: str
    required: bool
    nullable: bool = False
    comparison_id: str | None = None
    description_code: str | None = None
    version: str = OBSERVER_FEATURE_DEFINITION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_FEATURE_DEFINITION_VERSION:
            raise ObserverArtifactError("unsupported feature definition version")
        namespace = _validate_qualified_key(self.qualified_key)
        if self.namespace != namespace:
            raise ObserverArtifactError("namespace must match qualified_key prefix")
        if self.value_type not in {"bool", "int", "float", "number", "str", "none"}:
            raise ObserverArtifactError(
                f"unsupported feature value_type: {self.value_type!r}"
            )
        if self.comparison_id == "":
            raise ObserverArtifactError("comparison_id cannot be empty")
        if self.description_code == "":
            raise ObserverArtifactError("description_code cannot be empty")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.feature_definition_id != expected_id:
            raise ObserverArtifactError(
                "feature_definition_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "comparison_id": self.comparison_id,
            "description_code": self.description_code,
            "namespace": self.namespace,
            "nullable": self.nullable,
            "qualified_key": self.qualified_key,
            "required": self.required,
            "value_type": self.value_type,
            "version": self.version,
        }
        if include_id:
            payload["feature_definition_id"] = self.feature_definition_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        qualified_key: str,
        value_type: str,
        required: bool,
        nullable: bool = False,
        comparison_id: str | None = None,
        description_code: str | None = None,
    ) -> "ObserverFeatureDefinitionDTO":
        namespace = _validate_qualified_key(qualified_key)
        payload = {
            "comparison_id": comparison_id,
            "description_code": description_code,
            "namespace": namespace,
            "nullable": nullable,
            "qualified_key": qualified_key,
            "required": required,
            "value_type": value_type,
            "version": OBSERVER_FEATURE_DEFINITION_VERSION,
        }
        return cls(
            feature_definition_id=canonical_id(payload),
            qualified_key=qualified_key,
            namespace=namespace,
            value_type=value_type,
            required=required,
            nullable=nullable,
            comparison_id=comparison_id,
            description_code=description_code,
        )


@dataclass(frozen=True)
class ObserverObservationSchemaDTO:
    """Versioned feature vocabulary for observation artifacts."""

    schema_id: str
    schema_name: str
    features: tuple[ObserverFeatureDefinitionDTO, ...]
    version: str = OBSERVER_OBSERVATION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_OBSERVATION_SCHEMA_VERSION:
            raise ObserverArtifactError("unsupported observation schema version")
        _require_non_empty(self.schema_name, "schema_name")
        keys = tuple(item.qualified_key for item in self.features)
        if keys != tuple(sorted(set(keys))):
            raise ObserverArtifactError("features must be unique and sorted by key")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.schema_id != expected_id:
            raise ObserverArtifactError("schema_id disagrees with canonical payload")

    def definition_map(self) -> Mapping[str, ObserverFeatureDefinitionDTO]:
        return {item.qualified_key: item for item in self.features}

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "features": [item.canonical_payload() for item in self.features],
            "schema_name": self.schema_name,
            "version": self.version,
        }
        if include_id:
            payload["schema_id"] = self.schema_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        schema_name: str,
        features: tuple[ObserverFeatureDefinitionDTO, ...],
    ) -> "ObserverObservationSchemaDTO":
        features = tuple(sorted(features, key=lambda item: item.qualified_key))
        payload = {
            "features": [item.canonical_payload() for item in features],
            "schema_name": schema_name,
            "version": OBSERVER_OBSERVATION_SCHEMA_VERSION,
        }
        return cls(
            schema_id=canonical_id(payload), schema_name=schema_name, features=features
        )


@dataclass(frozen=True)
class ObserverObservationArtifactDTO:
    """Encoded observation state separated from policy action scores."""

    observation_artifact_id: str
    observation_schema_id: str
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
        _require_non_empty(self.observation_schema_id, "observation_schema_id")
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
            "observation_schema_id": self.observation_schema_id,
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
        observation_schema: ObserverObservationSchemaDTO,
        visible_state_features: Mapping[str, object],
        recent_history_features: Mapping[str, object] | None = None,
        hidden_state_uncertainty: Mapping[str, object] | None = None,
        provenance: Mapping[str, object] | None = None,
        sequence_index: int = 0,
    ) -> "ObserverObservationArtifactDTO":
        visible = dict(visible_state_features)
        history = dict(recent_history_features or {})
        hidden = dict(hidden_state_uncertainty or {})
        projected = {
            **{f"visible.{key}": value for key, value in visible.items()},
            **{f"history.{key}": value for key, value in history.items()},
            **{f"hidden.{key}": value for key, value in hidden.items()},
        }
        definitions = observation_schema.definition_map()
        undeclared = set(projected) - set(definitions)
        if undeclared:
            raise ObserverArtifactError(
                f"observation contains undeclared feature keys: {sorted(undeclared)}"
            )
        missing_required = tuple(
            item.qualified_key
            for item in observation_schema.features
            if item.required and item.qualified_key not in projected
        )
        if missing_required:
            raise ObserverArtifactError(
                f"observation is missing required feature keys: {list(missing_required)}"
            )
        for qualified_key, value in projected.items():
            definition = definitions[qualified_key]
            _ensure_canonical_value(value, qualified_key)
            if value is None and definition.nullable:
                continue
            if not _is_value_type(value, definition.value_type):
                raise ObserverArtifactError(
                    f"{qualified_key} expected {definition.value_type}, got {_type_name(value)}"
                )
        payload = {
            "hidden_state_uncertainty": hidden,
            "observation_schema_id": observation_schema.schema_id,
            "provenance": dict(provenance or {}),
            "recent_history_features": history,
            "sequence_index": sequence_index,
            "version": OBSERVER_OBSERVATION_ARTIFACT_VERSION,
            "visible_state_features": visible,
        }
        return cls(
            observation_artifact_id=canonical_id(payload),
            observation_schema_id=observation_schema.schema_id,
            visible_state_features=visible,
            recent_history_features=history,
            hidden_state_uncertainty=hidden,
            provenance=dict(provenance or {}),
            sequence_index=sequence_index,
        )


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
