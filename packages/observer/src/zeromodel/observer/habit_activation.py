"""Scoped activation and active action selection for Observer habits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import (
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)
from zeromodel.observer.fixture import ObserverFixtureActionDTO
from zeromodel.observer.grouping import ObserverStateGroupingRecipeDTO
from zeromodel.observer.habit import ObserverHabitError, ObserverHabitSpecificationDTO
from zeromodel.observer.habit_registry import (
    InMemoryObserverHabitRegistry,
    ObserverHabitRegistrySnapshotDTO,
)
from zeromodel.observer.habit_service import evaluate_observer_habit

OBSERVER_HABIT_ACTIVATION_SCOPE_VERSION: Final = "observer-habit-activation-scope/1"
OBSERVER_HABIT_ACTIVATION_REQUEST_VERSION: Final = "observer-habit-activation-request/1"
OBSERVER_HABIT_ACTIVATION_RESULT_VERSION: Final = "observer-habit-activation-result/1"
OBSERVER_ACTIVE_HABIT_DECISION_VERSION: Final = "observer-active-habit-decision/1"

ACTIVATION_DISPOSITIONS: Final = frozenset(
    {
        "activated",
        "already_active",
        "not_admitted",
        "suspended",
        "retired",
        "scope_mismatch",
        "activation_conflict",
        "stale_registry_snapshot",
        "registry_invalid",
        "unsupported",
    }
)
DECISION_SOURCES: Final = frozenset({"habit", "authoritative_fallback"})


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverHabitError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverHabitError(f"{field_name} must be unique and sorted")


@dataclass(frozen=True)
class ObserverHabitActivationScopeDTO:
    habit_activation_scope_id: str
    fixture_id: str
    observation_schema_id: str
    grouping_recipe_id: str
    allowed_action_names: tuple[str, ...]
    maximum_active_habit_count: int
    allow_overlapping_source_classes: bool
    version: str = OBSERVER_HABIT_ACTIVATION_SCOPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ACTIVATION_SCOPE_VERSION:
            raise ObserverHabitError("unsupported habit activation scope version")
        for field_name in ("fixture_id", "observation_schema_id", "grouping_recipe_id"):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_sorted_unique(self.allowed_action_names, "allowed_action_names")
        if self.maximum_active_habit_count < 1:
            raise ObserverHabitError("maximum_active_habit_count must be positive")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_activation_scope_id != expected_id:
            raise ObserverHabitError("habit_activation_scope_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "allow_overlapping_source_classes": self.allow_overlapping_source_classes,
            "allowed_action_names": list(self.allowed_action_names),
            "fixture_id": self.fixture_id,
            "grouping_recipe_id": self.grouping_recipe_id,
            "maximum_active_habit_count": self.maximum_active_habit_count,
            "observation_schema_id": self.observation_schema_id,
            "version": self.version,
        }
        if include_id:
            payload["habit_activation_scope_id"] = self.habit_activation_scope_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitActivationScopeDTO":
        values["allowed_action_names"] = tuple(
            sorted(set(cast(Sequence[str], values.get("allowed_action_names", ()))))
        )
        payload = {
            **values,
            "allowed_action_names": list(
                cast(tuple[str, ...], values["allowed_action_names"])
            ),
            "version": OBSERVER_HABIT_ACTIVATION_SCOPE_VERSION,
        }
        return cls(
            habit_activation_scope_id=canonical_id(payload),
            version=OBSERVER_HABIT_ACTIVATION_SCOPE_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitActivationRequestDTO:
    habit_activation_request_id: str
    habit_specification_id: str
    expected_source_registry_snapshot_id: str
    activation_scope_id: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_ACTIVATION_REQUEST_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ACTIVATION_REQUEST_VERSION:
            raise ObserverHabitError("unsupported habit activation request version")
        for field_name in (
            "habit_specification_id",
            "expected_source_registry_snapshot_id",
            "activation_scope_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_activation_request_id != expected_id:
            raise ObserverHabitError("habit_activation_request_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "activation_scope_id": self.activation_scope_id,
            "expected_source_registry_snapshot_id": (
                self.expected_source_registry_snapshot_id
            ),
            "habit_specification_id": self.habit_specification_id,
            "reason_codes": list(self.reason_codes),
            "version": self.version,
        }
        if include_id:
            payload["habit_activation_request_id"] = self.habit_activation_request_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitActivationRequestDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        payload = {
            **values,
            "reason_codes": list(cast(tuple[str, ...], values["reason_codes"])),
            "version": OBSERVER_HABIT_ACTIVATION_REQUEST_VERSION,
        }
        return cls(
            habit_activation_request_id=canonical_id(payload),
            version=OBSERVER_HABIT_ACTIVATION_REQUEST_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitActivationResultDTO:
    habit_activation_result_id: str
    habit_activation_request_id: str
    source_registry_snapshot_id: str
    result_registry_snapshot_id: str | None
    registry_event_id: str | None
    disposition: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_ACTIVATION_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ACTIVATION_RESULT_VERSION:
            raise ObserverHabitError("unsupported habit activation result version")
        if self.disposition not in ACTIVATION_DISPOSITIONS:
            raise ObserverHabitError("unsupported activation disposition")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        if self.disposition == "activated":
            if not self.result_registry_snapshot_id or not self.registry_event_id:
                raise ObserverHabitError("activated result requires snapshot and event")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_activation_result_id != expected_id:
            raise ObserverHabitError("habit_activation_result_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "disposition": self.disposition,
            "habit_activation_request_id": self.habit_activation_request_id,
            "reason_codes": list(self.reason_codes),
            "registry_event_id": self.registry_event_id,
            "result_registry_snapshot_id": self.result_registry_snapshot_id,
            "source_registry_snapshot_id": self.source_registry_snapshot_id,
            "version": self.version,
        }
        if include_id:
            payload["habit_activation_result_id"] = self.habit_activation_result_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitActivationResultDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        payload = {
            **values,
            "reason_codes": list(cast(tuple[str, ...], values["reason_codes"])),
            "version": OBSERVER_HABIT_ACTIVATION_RESULT_VERSION,
        }
        return cls(
            habit_activation_result_id=canonical_id(payload),
            version=OBSERVER_HABIT_ACTIVATION_RESULT_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverActiveHabitDecisionDTO:
    active_habit_decision_id: str
    registry_snapshot_id: str
    habit_specification_id: str | None
    observation_artifact_id: str
    habit_evaluation_id: str | None
    decision_source: str
    selected_action: str
    authoritative_fallback_action: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_ACTIVE_HABIT_DECISION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_ACTIVE_HABIT_DECISION_VERSION:
            raise ObserverHabitError("unsupported active habit decision version")
        if self.decision_source not in DECISION_SOURCES:
            raise ObserverHabitError("unsupported active decision source")
        for field_name in (
            "registry_snapshot_id",
            "observation_artifact_id",
            "selected_action",
            "authoritative_fallback_action",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        if self.decision_source == "authoritative_fallback":
            if self.selected_action != self.authoritative_fallback_action:
                raise ObserverHabitError(
                    "fallback decision must select fallback action"
                )
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.active_habit_decision_id != expected_id:
            raise ObserverHabitError("active_habit_decision_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "authoritative_fallback_action": self.authoritative_fallback_action,
            "decision_source": self.decision_source,
            "habit_evaluation_id": self.habit_evaluation_id,
            "habit_specification_id": self.habit_specification_id,
            "observation_artifact_id": self.observation_artifact_id,
            "reason_codes": list(self.reason_codes),
            "registry_snapshot_id": self.registry_snapshot_id,
            "selected_action": self.selected_action,
            "version": self.version,
        }
        if include_id:
            payload["active_habit_decision_id"] = self.active_habit_decision_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverActiveHabitDecisionDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        payload = {
            **values,
            "reason_codes": list(cast(tuple[str, ...], values["reason_codes"])),
            "version": OBSERVER_ACTIVE_HABIT_DECISION_VERSION,
        }
        return cls(
            active_habit_decision_id=canonical_id(payload),
            version=OBSERVER_ACTIVE_HABIT_DECISION_VERSION,
            **values,  # type: ignore[arg-type]
        )


def activate_observer_habit(
    *,
    registry: InMemoryObserverHabitRegistry,
    activation_scope: ObserverHabitActivationScopeDTO,
    activation_request: ObserverHabitActivationRequestDTO,
    habit_specification: ObserverHabitSpecificationDTO,
) -> ObserverHabitActivationResultDTO:
    source = registry.current_snapshot()
    if (
        source.habit_registry_snapshot_id
        != activation_request.expected_source_registry_snapshot_id
    ):
        return _activation_result(
            activation_request,
            source,
            "stale_registry_snapshot",
            ("stale_registry_snapshot",),
        )
    if (
        activation_request.activation_scope_id
        != activation_scope.habit_activation_scope_id
    ):
        return _activation_result(
            activation_request, source, "scope_mismatch", ("scope_mismatch",)
        )
    entry = registry.get_entry(activation_request.habit_specification_id)
    if entry is None:
        return _activation_result(
            activation_request, source, "not_admitted", ("not_admitted",)
        )
    if entry.status == "active":
        return _activation_result(
            activation_request, source, "already_active", ("already_active",)
        )
    if entry.status == "suspended":
        return _activation_result(
            activation_request, source, "suspended", ("suspended",)
        )
    if entry.status == "retired":
        return _activation_result(activation_request, source, "retired", ("retired",))
    scope_reasons = _scope_mismatch_reasons(activation_scope, habit_specification)
    if scope_reasons:
        return _activation_result(
            activation_request, source, "scope_mismatch", scope_reasons
        )
    if len(source.active_habit_ids) >= activation_scope.maximum_active_habit_count:
        return _activation_result(
            activation_request,
            source,
            "activation_conflict",
            ("maximum_active_habit_count_exceeded",),
        )
    if not activation_scope.allow_overlapping_source_classes:
        # Stage O3.5 allows only one active habit, so overlap is covered by active count.
        pass
    try:
        event = registry.activate(
            habit_specification_id=habit_specification.habit_specification_id,
            expected_source_registry_snapshot_id=source.habit_registry_snapshot_id,
            reason_codes=activation_request.reason_codes or ("activated",),
        )
    except ObserverHabitError:
        return _activation_result(
            activation_request, source, "activation_conflict", ("activation_conflict",)
        )
    return ObserverHabitActivationResultDTO.create(
        habit_activation_request_id=activation_request.habit_activation_request_id,
        source_registry_snapshot_id=source.habit_registry_snapshot_id,
        result_registry_snapshot_id=registry.current_snapshot().habit_registry_snapshot_id,
        registry_event_id=event.habit_registry_event_id,
        disposition="activated",
        reason_codes=("activated",),
    )


def select_observer_active_action(
    *,
    registry_snapshot: ObserverHabitRegistrySnapshotDTO,
    activation_scope: ObserverHabitActivationScopeDTO,
    observation: ObserverObservationArtifactDTO,
    authoritative_action: ObserverFixtureActionDTO,
    habits: tuple[ObserverHabitSpecificationDTO, ...],
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverActiveHabitDecisionDTO:
    fallback = authoritative_action.action_name
    if registry_snapshot.active_habit_ids != tuple(
        sorted(
            entry.habit_specification_id
            for entry in registry_snapshot.entries
            if entry.status == "active"
        )
    ):
        return _fallback_decision(
            registry_snapshot, observation, fallback, ("registry_invalid",)
        )
    if (
        len(registry_snapshot.active_habit_ids)
        > activation_scope.maximum_active_habit_count
    ):
        return _fallback_decision(
            registry_snapshot, observation, fallback, ("ambiguous_active_habits",)
        )
    habit_by_id = {habit.habit_specification_id: habit for habit in habits}
    firing: list[tuple[ObserverHabitSpecificationDTO, str]] = []
    invalid = False
    for active_id in registry_snapshot.active_habit_ids:
        habit = habit_by_id.get(active_id)
        if habit is None:
            invalid = True
            continue
        scope_reasons = _scope_mismatch_reasons(activation_scope, habit)
        if scope_reasons:
            invalid = True
            continue
        evaluation = evaluate_observer_habit(
            habit_specification=habit,
            observation=observation,
            grouping_recipe=grouping_recipe,
            observation_schema=observation_schema,
        )
        if evaluation.decision == "fire" and evaluation.recommended_action is not None:
            firing.append((habit, evaluation.habit_evaluation_id))
        elif evaluation.decision == "invalid":
            invalid = True
    if len(firing) == 1:
        habit, evaluation_id = firing[0]
        return ObserverActiveHabitDecisionDTO.create(
            registry_snapshot_id=registry_snapshot.habit_registry_snapshot_id,
            habit_specification_id=habit.habit_specification_id,
            observation_artifact_id=observation.observation_artifact_id,
            habit_evaluation_id=evaluation_id,
            decision_source="habit",
            selected_action=habit.recommended_action,
            authoritative_fallback_action=fallback,
            reason_codes=("active_habit_fired",),
        )
    if len(firing) > 1:
        return _fallback_decision(
            registry_snapshot,
            observation,
            fallback,
            ("ambiguous_active_habits",),
        )
    return _fallback_decision(
        registry_snapshot,
        observation,
        fallback,
        ("invalid_active_habit" if invalid else "active_habit_abstained",),
    )


def _scope_mismatch_reasons(
    scope: ObserverHabitActivationScopeDTO, habit: ObserverHabitSpecificationDTO
) -> tuple[str, ...]:
    reasons: set[str] = set()
    if habit.observation_schema_id != scope.observation_schema_id:
        reasons.add("observation_schema_mismatch")
    if habit.grouping_recipe_id != scope.grouping_recipe_id:
        reasons.add("grouping_recipe_mismatch")
    if habit.recommended_action not in scope.allowed_action_names:
        reasons.add("action_not_allowed")
    return tuple(sorted(reasons))


def _activation_result(
    request: ObserverHabitActivationRequestDTO,
    source: ObserverHabitRegistrySnapshotDTO,
    disposition: str,
    reasons: tuple[str, ...],
) -> ObserverHabitActivationResultDTO:
    return ObserverHabitActivationResultDTO.create(
        habit_activation_request_id=request.habit_activation_request_id,
        source_registry_snapshot_id=source.habit_registry_snapshot_id,
        result_registry_snapshot_id=None,
        registry_event_id=None,
        disposition=disposition,
        reason_codes=reasons,
    )


def _fallback_decision(
    registry_snapshot: ObserverHabitRegistrySnapshotDTO,
    observation: ObserverObservationArtifactDTO,
    fallback: str,
    reasons: tuple[str, ...],
) -> ObserverActiveHabitDecisionDTO:
    return ObserverActiveHabitDecisionDTO.create(
        registry_snapshot_id=registry_snapshot.habit_registry_snapshot_id,
        habit_specification_id=None,
        observation_artifact_id=observation.observation_artifact_id,
        habit_evaluation_id=None,
        decision_source="authoritative_fallback",
        selected_action=fallback,
        authoritative_fallback_action=fallback,
        reason_codes=reasons,
    )
