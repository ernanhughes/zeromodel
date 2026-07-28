"""Declared observation grouping contracts for Observer graphs."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import (
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)

OBSERVER_CATEGORY_MAPPING_VERSION: Final = "observer-category-mapping/1"
OBSERVER_GROUPING_FEATURE_VERSION: Final = "observer-grouping-feature/1"
OBSERVER_STATE_GROUPING_RECIPE_VERSION: Final = "observer-state-grouping-recipe/1"
OBSERVER_GROUPED_FEATURE_VALUE_VERSION: Final = "observer-grouped-feature-value/1"
OBSERVER_STATE_CLASS_ASSIGNMENT_VERSION: Final = "observer-state-class-assignment/1"
OBSERVER_STATE_CLASS_VERSION: Final = "observer-state-class/1"

GROUPING_MODES: Final = frozenset({"exact", "numeric_bucket", "categorical", "ignored"})
MISSING_POLICIES: Final = frozenset({"reject", "separate_class"})
TYPE_MISMATCH_POLICIES: Final = frozenset({"reject", "separate_class"})
UNMAPPED_CATEGORY_POLICIES: Final = frozenset({"preserve", "reject", "other"})
GROUPED_KINDS: Final = frozenset(
    {"exact", "numeric_bucket", "category", "missing", "type_mismatch"}
)
ASSIGNMENT_STATUSES: Final = frozenset({"assigned", "rejected"})
ASSIGNMENT_REASON_CODES: Final = frozenset(
    {
        "exact_grouping",
        "numeric_bucket_grouping",
        "categorical_mapping",
        "missing_feature_separate_class",
        "type_mismatch_separate_class",
        "schema_mismatch",
        "invalid_feature_value",
    }
)


class ObserverObservationGraphError(ValueError):
    """Raised when Observer observation graph contracts are invalid."""


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverObservationGraphError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverObservationGraphError(f"{field_name} must be unique and sorted")


def _type_name(value: object) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if value is None:
        return "null"
    return type(value).__name__


def _schema_type_for_feature(
    *, observation_schema: ObserverObservationSchemaDTO, feature_key: str
) -> str:
    definitions = observation_schema.definition_map()
    if feature_key not in definitions:
        raise ObserverObservationGraphError("grouping feature not declared in schema")
    return definitions[feature_key].value_type


def _project_observation(
    observation: ObserverObservationArtifactDTO,
) -> dict[str, object]:
    return {
        **{
            f"visible.{key}": value
            for key, value in observation.visible_state_features.items()
        },
        **{
            f"history.{key}": value
            for key, value in observation.recent_history_features.items()
        },
        **{
            f"hidden.{key}": value
            for key, value in observation.hidden_state_uncertainty.items()
        },
    }


@dataclass(frozen=True)
class ObserverCategoryMappingDTO:
    category_mapping_id: str
    raw_value: str
    mapped_value: str
    version: str = OBSERVER_CATEGORY_MAPPING_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_CATEGORY_MAPPING_VERSION:
            raise ObserverObservationGraphError("unsupported category mapping version")
        _require_non_empty(self.raw_value, "raw_value")
        _require_non_empty(self.mapped_value, "mapped_value")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.category_mapping_id != expected_id:
            raise ObserverObservationGraphError(
                "category_mapping_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = {
            "mapped_value": self.mapped_value,
            "raw_value": self.raw_value,
            "version": self.version,
        }
        if include_id:
            payload["category_mapping_id"] = self.category_mapping_id
        return payload

    @classmethod
    def create(
        cls, *, raw_value: str, mapped_value: str
    ) -> "ObserverCategoryMappingDTO":
        payload = {
            "mapped_value": mapped_value,
            "raw_value": raw_value,
            "version": OBSERVER_CATEGORY_MAPPING_VERSION,
        }
        return cls(
            category_mapping_id=canonical_id(payload),
            raw_value=raw_value,
            mapped_value=mapped_value,
        )


@dataclass(frozen=True)
class ObserverGroupingFeatureDTO:
    grouping_feature_id: str
    feature_key: str
    mode: str
    bucket_size: float | None
    absolute_tolerance: float | None
    category_mapping: tuple[ObserverCategoryMappingDTO, ...]
    unmapped_category_policy: str
    include_in_class_key: bool
    version: str = OBSERVER_GROUPING_FEATURE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_GROUPING_FEATURE_VERSION:
            raise ObserverObservationGraphError("unsupported grouping feature version")
        _require_non_empty(self.feature_key, "feature_key")
        if self.mode not in GROUPING_MODES:
            raise ObserverObservationGraphError("unsupported grouping mode")
        if self.unmapped_category_policy not in UNMAPPED_CATEGORY_POLICIES:
            raise ObserverObservationGraphError("unsupported unmapped category policy")
        if self.mode == "numeric_bucket" and (
            self.bucket_size is None
            or not math.isfinite(self.bucket_size)
            or self.bucket_size <= 0
        ):
            raise ObserverObservationGraphError(
                "numeric_bucket requires a positive finite bucket_size"
            )
        if self.mode != "numeric_bucket" and self.bucket_size is not None:
            raise ObserverObservationGraphError(
                "bucket_size is only valid for numeric_bucket"
            )
        if self.absolute_tolerance is not None:
            raise ObserverObservationGraphError(
                "absolute_tolerance is reserved and must be None"
            )
        if self.category_mapping != tuple(
            sorted(self.category_mapping, key=lambda item: item.raw_value)
        ):
            raise ObserverObservationGraphError("category_mapping must be sorted")
        raw_values = tuple(item.raw_value for item in self.category_mapping)
        _ensure_sorted_unique(raw_values, "category_mapping raw values")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.grouping_feature_id != expected_id:
            raise ObserverObservationGraphError(
                "grouping_feature_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "absolute_tolerance": self.absolute_tolerance,
            "bucket_size": self.bucket_size,
            "category_mapping": [
                item.canonical_payload() for item in self.category_mapping
            ],
            "feature_key": self.feature_key,
            "include_in_class_key": self.include_in_class_key,
            "mode": self.mode,
            "unmapped_category_policy": self.unmapped_category_policy,
            "version": self.version,
        }
        if include_id:
            payload["grouping_feature_id"] = self.grouping_feature_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        feature_key: str,
        mode: str,
        bucket_size: float | None = None,
        category_mapping: tuple[ObserverCategoryMappingDTO, ...] = (),
        unmapped_category_policy: str = "preserve",
        include_in_class_key: bool = True,
    ) -> "ObserverGroupingFeatureDTO":
        category_mapping = tuple(
            sorted(category_mapping, key=lambda item: item.raw_value)
        )
        payload = {
            "absolute_tolerance": None,
            "bucket_size": bucket_size,
            "category_mapping": [item.canonical_payload() for item in category_mapping],
            "feature_key": feature_key,
            "include_in_class_key": include_in_class_key,
            "mode": mode,
            "unmapped_category_policy": unmapped_category_policy,
            "version": OBSERVER_GROUPING_FEATURE_VERSION,
        }
        return cls(
            grouping_feature_id=canonical_id(payload),
            feature_key=feature_key,
            mode=mode,
            bucket_size=bucket_size,
            absolute_tolerance=None,
            category_mapping=category_mapping,
            unmapped_category_policy=unmapped_category_policy,
            include_in_class_key=include_in_class_key,
        )


@dataclass(frozen=True)
class ObserverStateGroupingRecipeDTO:
    grouping_recipe_id: str
    observation_schema_id: str
    feature_groupings: tuple[ObserverGroupingFeatureDTO, ...]
    missing_feature_policy: str
    type_mismatch_policy: str
    version: str = OBSERVER_STATE_GROUPING_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_STATE_GROUPING_RECIPE_VERSION:
            raise ObserverObservationGraphError("unsupported grouping recipe version")
        _require_non_empty(self.observation_schema_id, "observation_schema_id")
        if self.missing_feature_policy not in MISSING_POLICIES:
            raise ObserverObservationGraphError("unsupported missing feature policy")
        if self.type_mismatch_policy not in TYPE_MISMATCH_POLICIES:
            raise ObserverObservationGraphError("unsupported type mismatch policy")
        if self.feature_groupings != tuple(
            sorted(self.feature_groupings, key=lambda item: item.feature_key)
        ):
            raise ObserverObservationGraphError("feature_groupings must be sorted")
        keys = tuple(item.feature_key for item in self.feature_groupings)
        _ensure_sorted_unique(keys, "feature_groupings")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.grouping_recipe_id != expected_id:
            raise ObserverObservationGraphError(
                "grouping_recipe_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "feature_groupings": [
                item.canonical_payload() for item in self.feature_groupings
            ],
            "missing_feature_policy": self.missing_feature_policy,
            "observation_schema_id": self.observation_schema_id,
            "type_mismatch_policy": self.type_mismatch_policy,
            "version": self.version,
        }
        if include_id:
            payload["grouping_recipe_id"] = self.grouping_recipe_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        observation_schema_id: str,
        feature_groupings: tuple[ObserverGroupingFeatureDTO, ...],
        missing_feature_policy: str = "reject",
        type_mismatch_policy: str = "reject",
    ) -> "ObserverStateGroupingRecipeDTO":
        feature_groupings = tuple(
            sorted(feature_groupings, key=lambda item: item.feature_key)
        )
        payload = {
            "feature_groupings": [
                item.canonical_payload() for item in feature_groupings
            ],
            "missing_feature_policy": missing_feature_policy,
            "observation_schema_id": observation_schema_id,
            "type_mismatch_policy": type_mismatch_policy,
            "version": OBSERVER_STATE_GROUPING_RECIPE_VERSION,
        }
        return cls(
            grouping_recipe_id=canonical_id(payload),
            observation_schema_id=observation_schema_id,
            feature_groupings=feature_groupings,
            missing_feature_policy=missing_feature_policy,
            type_mismatch_policy=type_mismatch_policy,
        )


@dataclass(frozen=True)
class ObserverGroupedFeatureValueDTO:
    grouped_value_id: str
    feature_key: str
    grouping_feature_id: str
    source_type: str | None
    grouped_kind: str
    grouped_value: object
    version: str = OBSERVER_GROUPED_FEATURE_VALUE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_GROUPED_FEATURE_VALUE_VERSION:
            raise ObserverObservationGraphError("unsupported grouped value version")
        _require_non_empty(self.feature_key, "feature_key")
        _require_non_empty(self.grouping_feature_id, "grouping_feature_id")
        if self.grouped_kind not in GROUPED_KINDS:
            raise ObserverObservationGraphError("unsupported grouped kind")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.grouped_value_id != expected_id:
            raise ObserverObservationGraphError(
                "grouped_value_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "feature_key": self.feature_key,
            "grouped_kind": self.grouped_kind,
            "grouped_value": self.grouped_value,
            "grouping_feature_id": self.grouping_feature_id,
            "source_type": self.source_type,
            "version": self.version,
        }
        if include_id:
            payload["grouped_value_id"] = self.grouped_value_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        feature_key: str,
        grouping_feature_id: str,
        source_type: str | None,
        grouped_kind: str,
        grouped_value: object,
    ) -> "ObserverGroupedFeatureValueDTO":
        payload = {
            "feature_key": feature_key,
            "grouped_kind": grouped_kind,
            "grouped_value": grouped_value,
            "grouping_feature_id": grouping_feature_id,
            "source_type": source_type,
            "version": OBSERVER_GROUPED_FEATURE_VALUE_VERSION,
        }
        return cls(
            grouped_value_id=canonical_id(payload),
            feature_key=feature_key,
            grouping_feature_id=grouping_feature_id,
            source_type=source_type,
            grouped_kind=grouped_kind,
            grouped_value=grouped_value,
        )


@dataclass(frozen=True)
class ObserverStateClassAssignmentDTO:
    assignment_id: str
    grouping_recipe_id: str
    observation_artifact_id: str
    observation_schema_id: str
    state_class_key: tuple[ObserverGroupedFeatureValueDTO, ...]
    state_class_id: str | None
    status: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_STATE_CLASS_ASSIGNMENT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_STATE_CLASS_ASSIGNMENT_VERSION:
            raise ObserverObservationGraphError("unsupported assignment version")
        for field_name in (
            "grouping_recipe_id",
            "observation_artifact_id",
            "observation_schema_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.status not in ASSIGNMENT_STATUSES:
            raise ObserverObservationGraphError("unsupported assignment status")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        if set(self.reason_codes) - ASSIGNMENT_REASON_CODES:
            raise ObserverObservationGraphError("unsupported assignment reason code")
        if self.state_class_key != tuple(
            sorted(self.state_class_key, key=lambda item: item.feature_key)
        ):
            raise ObserverObservationGraphError("state_class_key must be sorted")
        if self.status == "assigned" and self.state_class_id is None:
            raise ObserverObservationGraphError(
                "assigned observation needs state_class_id"
            )
        if self.status == "rejected" and self.state_class_id is not None:
            raise ObserverObservationGraphError("rejected assignment cannot have class")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.assignment_id != expected_id:
            raise ObserverObservationGraphError(
                "assignment_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "grouping_recipe_id": self.grouping_recipe_id,
            "observation_artifact_id": self.observation_artifact_id,
            "observation_schema_id": self.observation_schema_id,
            "reason_codes": list(self.reason_codes),
            "state_class_id": self.state_class_id,
            "state_class_key": [
                item.canonical_payload() for item in self.state_class_key
            ],
            "status": self.status,
            "version": self.version,
        }
        if include_id:
            payload["assignment_id"] = self.assignment_id
        return payload


@dataclass(frozen=True)
class ObserverStateClassDTO:
    state_class_id: str
    grouping_recipe_id: str
    observation_schema_id: str
    state_class_key: tuple[ObserverGroupedFeatureValueDTO, ...]
    version: str = OBSERVER_STATE_CLASS_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_STATE_CLASS_VERSION:
            raise ObserverObservationGraphError("unsupported state class version")
        _require_non_empty(self.grouping_recipe_id, "grouping_recipe_id")
        _require_non_empty(self.observation_schema_id, "observation_schema_id")
        if self.state_class_key != tuple(
            sorted(self.state_class_key, key=lambda item: item.feature_key)
        ):
            raise ObserverObservationGraphError("state_class_key must be sorted")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.state_class_id != expected_id:
            raise ObserverObservationGraphError(
                "state_class_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "grouping_recipe_id": self.grouping_recipe_id,
            "observation_schema_id": self.observation_schema_id,
            "state_class_key": [
                item.canonical_payload() for item in self.state_class_key
            ],
            "version": self.version,
        }
        if include_id:
            payload["state_class_id"] = self.state_class_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        grouping_recipe_id: str,
        observation_schema_id: str,
        state_class_key: tuple[ObserverGroupedFeatureValueDTO, ...],
    ) -> "ObserverStateClassDTO":
        state_class_key = tuple(
            sorted(state_class_key, key=lambda item: item.feature_key)
        )
        payload = {
            "grouping_recipe_id": grouping_recipe_id,
            "observation_schema_id": observation_schema_id,
            "state_class_key": [item.canonical_payload() for item in state_class_key],
            "version": OBSERVER_STATE_CLASS_VERSION,
        }
        return cls(
            state_class_id=canonical_id(payload),
            grouping_recipe_id=grouping_recipe_id,
            observation_schema_id=observation_schema_id,
            state_class_key=state_class_key,
        )


def assign_observation_to_state_class(
    *,
    observation: ObserverObservationArtifactDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverStateClassAssignmentDTO:
    if grouping_recipe.observation_schema_id != observation_schema.schema_id:
        return _rejected_assignment(
            observation=observation,
            grouping_recipe=grouping_recipe,
            reason_codes=("schema_mismatch",),
        )
    if observation.observation_schema_id != observation_schema.schema_id:
        return _rejected_assignment(
            observation=observation,
            grouping_recipe=grouping_recipe,
            reason_codes=("schema_mismatch",),
        )
    projected = _project_observation(observation)
    values: list[ObserverGroupedFeatureValueDTO] = []
    reasons: set[str] = set()
    rejected = False
    for feature in grouping_recipe.feature_groupings:
        if feature.mode == "ignored":
            continue
        if feature.feature_key not in projected:
            if grouping_recipe.missing_feature_policy == "reject":
                rejected = True
                reasons.add("invalid_feature_value")
                continue
            values.append(
                ObserverGroupedFeatureValueDTO.create(
                    feature_key=feature.feature_key,
                    grouping_feature_id=feature.grouping_feature_id,
                    source_type=None,
                    grouped_kind="missing",
                    grouped_value="missing",
                )
            )
            reasons.add("missing_feature_separate_class")
            continue
        raw_value = projected[feature.feature_key]
        schema_type = _schema_type_for_feature(
            observation_schema=observation_schema, feature_key=feature.feature_key
        )
        actual_type = _type_name(raw_value)
        if actual_type != schema_type:
            if grouping_recipe.type_mismatch_policy == "reject":
                rejected = True
                reasons.add("invalid_feature_value")
                continue
            values.append(
                ObserverGroupedFeatureValueDTO.create(
                    feature_key=feature.feature_key,
                    grouping_feature_id=feature.grouping_feature_id,
                    source_type=actual_type,
                    grouped_kind="type_mismatch",
                    grouped_value=f"type_mismatch:{actual_type}",
                )
            )
            reasons.add("type_mismatch_separate_class")
            continue
        grouped = _group_value(feature=feature, raw_value=raw_value)
        if grouped is None:
            rejected = True
            reasons.add("invalid_feature_value")
            continue
        grouped_value, reason_code = grouped
        if feature.include_in_class_key:
            values.append(grouped_value)
        reasons.add(reason_code)
    if rejected:
        return _rejected_assignment(
            observation=observation,
            grouping_recipe=grouping_recipe,
            reason_codes=tuple(sorted(reasons)),
        )
    state_class_key = tuple(sorted(values, key=lambda item: item.feature_key))
    state_class = ObserverStateClassDTO.create(
        grouping_recipe_id=grouping_recipe.grouping_recipe_id,
        observation_schema_id=observation_schema.schema_id,
        state_class_key=state_class_key,
    )
    payload = _assignment_payload(
        grouping_recipe=grouping_recipe,
        observation=observation,
        state_class_key=state_class_key,
        state_class_id=state_class.state_class_id,
        status="assigned",
        reason_codes=tuple(sorted(reasons)),
    )
    return ObserverStateClassAssignmentDTO(
        assignment_id=canonical_id(payload),
        grouping_recipe_id=grouping_recipe.grouping_recipe_id,
        observation_artifact_id=observation.observation_artifact_id,
        observation_schema_id=observation.observation_schema_id,
        state_class_key=state_class_key,
        state_class_id=state_class.state_class_id,
        status="assigned",
        reason_codes=tuple(sorted(reasons)),
    )


def _group_value(
    *, feature: ObserverGroupingFeatureDTO, raw_value: object
) -> tuple[ObserverGroupedFeatureValueDTO, str] | None:
    source_type = _type_name(raw_value)
    if feature.mode == "exact":
        return (
            ObserverGroupedFeatureValueDTO.create(
                feature_key=feature.feature_key,
                grouping_feature_id=feature.grouping_feature_id,
                source_type=source_type,
                grouped_kind="exact",
                grouped_value={"type": source_type, "value": raw_value},
            ),
            "exact_grouping",
        )
    if feature.mode == "numeric_bucket":
        if isinstance(raw_value, bool) or not isinstance(raw_value, int | float):
            return None
        value = float(raw_value)
        if not math.isfinite(value) or feature.bucket_size is None:
            return None
        return (
            ObserverGroupedFeatureValueDTO.create(
                feature_key=feature.feature_key,
                grouping_feature_id=feature.grouping_feature_id,
                source_type=source_type,
                grouped_kind="numeric_bucket",
                grouped_value={
                    "bucket_index": math.floor(value / feature.bucket_size),
                    "bucket_size": feature.bucket_size,
                },
            ),
            "numeric_bucket_grouping",
        )
    if feature.mode == "categorical":
        if not isinstance(raw_value, str):
            return None
        mapping = {
            item.raw_value: item.mapped_value for item in feature.category_mapping
        }
        if raw_value in mapping:
            category = mapping[raw_value]
        elif feature.unmapped_category_policy == "preserve":
            category = raw_value
        elif feature.unmapped_category_policy == "other":
            category = "other"
        else:
            return None
        return (
            ObserverGroupedFeatureValueDTO.create(
                feature_key=feature.feature_key,
                grouping_feature_id=feature.grouping_feature_id,
                source_type=source_type,
                grouped_kind="category",
                grouped_value=category,
            ),
            "categorical_mapping",
        )
    return None


def _assignment_payload(
    *,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation: ObserverObservationArtifactDTO,
    state_class_key: tuple[ObserverGroupedFeatureValueDTO, ...],
    state_class_id: str | None,
    status: str,
    reason_codes: tuple[str, ...],
) -> dict[str, object]:
    return {
        "grouping_recipe_id": grouping_recipe.grouping_recipe_id,
        "observation_artifact_id": observation.observation_artifact_id,
        "observation_schema_id": observation.observation_schema_id,
        "reason_codes": list(reason_codes),
        "state_class_id": state_class_id,
        "state_class_key": [item.canonical_payload() for item in state_class_key],
        "status": status,
        "version": OBSERVER_STATE_CLASS_ASSIGNMENT_VERSION,
    }


def _rejected_assignment(
    *,
    observation: ObserverObservationArtifactDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    reason_codes: tuple[str, ...],
) -> ObserverStateClassAssignmentDTO:
    reason_codes = tuple(sorted(set(reason_codes)))
    payload = _assignment_payload(
        grouping_recipe=grouping_recipe,
        observation=observation,
        state_class_key=(),
        state_class_id=None,
        status="rejected",
        reason_codes=reason_codes,
    )
    return ObserverStateClassAssignmentDTO(
        assignment_id=canonical_id(payload),
        grouping_recipe_id=grouping_recipe.grouping_recipe_id,
        observation_artifact_id=observation.observation_artifact_id,
        observation_schema_id=observation.observation_schema_id,
        state_class_key=(),
        state_class_id=None,
        status="rejected",
        reason_codes=reason_codes,
    )
