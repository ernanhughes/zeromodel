"""Canonical DTOs for bounded Observer habit shadow evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id

OBSERVER_HABIT_COMPILATION_RECIPE_VERSION: Final = "observer-habit-compilation-recipe/1"
OBSERVER_HABIT_GUARD_VERSION: Final = "observer-habit-guard/1"
OBSERVER_HABIT_COUNTEREXAMPLE_VERSION: Final = "observer-habit-counterexample/1"
OBSERVER_HABIT_SPECIFICATION_VERSION: Final = "observer-habit-specification/1"
OBSERVER_HABIT_COMPILATION_RESULT_VERSION: Final = "observer-habit-compilation-result/1"
OBSERVER_HABIT_GUARD_EVALUATION_VERSION: Final = "observer-habit-guard-evaluation/1"
OBSERVER_HABIT_EVALUATION_VERSION: Final = "observer-habit-evaluation/1"
OBSERVER_HABIT_SHADOW_OCCURRENCE_VERSION: Final = "observer-habit-shadow-occurrence/1"
OBSERVER_HABIT_SHADOW_REPLAY_VERSION: Final = "observer-habit-shadow-replay/1"
OBSERVER_HABIT_SHADOW_EPISODE_VERSION: Final = "observer-habit-shadow-episode/1"
OBSERVER_HABIT_SHADOW_AUDIT_RECIPE_VERSION: Final = (
    "observer-habit-shadow-audit-recipe/1"
)
OBSERVER_HABIT_COUNTEREXAMPLE_COVERAGE_VERSION: Final = (
    "observer-habit-counterexample-coverage/1"
)
OBSERVER_HABIT_SHADOW_AUDIT_VERSION: Final = "observer-habit-shadow-audit/1"

GUARD_OPERATORS: Final = frozenset(
    {"equals", "not_equals", "in_closed_range", "is_missing", "is_present"}
)
GUARD_ROLES: Final = frozenset({"positive", "counterexample"})
GUARD_VALUE_TYPES: Final = frozenset({"bool", "int", "float", "number", "str", "none"})
HABIT_SPECIFICATION_STATUSES: Final = frozenset({"shadow_candidate"})
HABIT_COMPILATION_DISPOSITIONS: Final = frozenset(
    {
        "compiled_for_shadow",
        "insufficient_guard_evidence",
        "counterexample_conflict",
        "guard_limit_exceeded",
        "unsupported_grouping_mode",
        "schema_mismatch",
        "invalid_candidate",
        "unsupported",
    }
)
HABIT_GUARD_EVALUATION_STATUSES: Final = frozenset(
    {"matched", "not_matched", "invalid"}
)
HABIT_EVALUATION_DECISIONS: Final = frozenset({"fire", "abstain", "invalid"})
HABIT_SHADOW_OUTCOMES: Final = frozenset(
    {
        "correct_fire",
        "wrong_action_fire",
        "wrong_target_fire",
        "safe_abstention",
        "missed_opportunity",
        "invalid_evaluation",
        "not_applicable",
    }
)
HABIT_SHADOW_REPLAY_STATUSES: Final = frozenset({"verified", "failed", "inconclusive"})
HABIT_SHADOW_AUDIT_DISPOSITIONS: Final = frozenset(
    {
        "eligible_for_admission_review",
        "insufficient_shadow_evidence",
        "false_fire_detected",
        "target_instability_detected",
        "too_many_missed_opportunities",
        "invalid_evidence",
        "counterexample_coverage_incomplete",
        "unsupported",
    }
)


class ObserverHabitError(ValueError):
    """Raised when bounded habit contracts are malformed."""


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverHabitError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverHabitError(f"{field_name} must be unique and sorted")


def _ensure_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ObserverHabitError(f"{field_name} must be non-negative")


def _canonical_tuple(values: tuple[object, ...]) -> list[Mapping[str, object]]:
    return [item.canonical_payload() for item in values]  # type: ignore[attr-defined]


def _payload_with_version(version: str, **values: object) -> dict[str, object]:
    payload = dict(values)
    payload["version"] = version
    return payload


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


def _is_finite_number(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int | float)
        and math.isfinite(float(value))
    )


@dataclass(frozen=True)
class ObserverHabitCompilationRecipeDTO:
    habit_compilation_recipe_id: str
    promotion_recipe_id: str
    grouping_recipe_id: str
    observation_schema_id: str
    allowed_guard_feature_keys: tuple[str, ...]
    required_guard_feature_keys: tuple[str, ...]
    forbidden_guard_feature_keys: tuple[str, ...]
    maximum_guard_count: int
    maximum_counterexample_guard_count: int
    allow_exact_guards: bool
    allow_categorical_guards: bool
    allow_numeric_range_guards: bool
    require_counterexample_guards: bool
    version: str = OBSERVER_HABIT_COMPILATION_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_COMPILATION_RECIPE_VERSION:
            raise ObserverHabitError("unsupported habit compilation recipe version")
        for field_name in (
            "promotion_recipe_id",
            "grouping_recipe_id",
            "observation_schema_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        for field_name in (
            "allowed_guard_feature_keys",
            "required_guard_feature_keys",
            "forbidden_guard_feature_keys",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        allowed = set(self.allowed_guard_feature_keys)
        required = set(self.required_guard_feature_keys)
        forbidden = set(self.forbidden_guard_feature_keys)
        if not required <= allowed:
            raise ObserverHabitError("required guard keys must be allowed")
        if required & forbidden or allowed & forbidden:
            raise ObserverHabitError("guard feature key sets conflict")
        for field_name in (
            "maximum_guard_count",
            "maximum_counterexample_guard_count",
        ):
            _ensure_non_negative(getattr(self, field_name), field_name)
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_compilation_recipe_id != expected_id:
            raise ObserverHabitError("habit_compilation_recipe_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            allow_categorical_guards=self.allow_categorical_guards,
            allow_exact_guards=self.allow_exact_guards,
            allow_numeric_range_guards=self.allow_numeric_range_guards,
            allowed_guard_feature_keys=list(self.allowed_guard_feature_keys),
            forbidden_guard_feature_keys=list(self.forbidden_guard_feature_keys),
            grouping_recipe_id=self.grouping_recipe_id,
            maximum_counterexample_guard_count=self.maximum_counterexample_guard_count,
            maximum_guard_count=self.maximum_guard_count,
            observation_schema_id=self.observation_schema_id,
            promotion_recipe_id=self.promotion_recipe_id,
            require_counterexample_guards=self.require_counterexample_guards,
            required_guard_feature_keys=list(self.required_guard_feature_keys),
        )
        if include_id:
            payload["habit_compilation_recipe_id"] = self.habit_compilation_recipe_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitCompilationRecipeDTO":
        for key in (
            "allowed_guard_feature_keys",
            "required_guard_feature_keys",
            "forbidden_guard_feature_keys",
        ):
            values[key] = tuple(sorted(set(cast(Sequence[str], values.get(key, ())))))
        payload = _payload_with_version(
            OBSERVER_HABIT_COMPILATION_RECIPE_VERSION,
            **values,
        )
        return cls(habit_compilation_recipe_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitGuardDTO:
    habit_guard_id: str
    feature_key: str
    operator: str
    expected_type: str
    expected_value: object | None
    minimum_value: int | float | None
    maximum_value: int | float | None
    guard_role: str
    source_evidence_ids: tuple[str, ...]
    version: str = OBSERVER_HABIT_GUARD_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_GUARD_VERSION:
            raise ObserverHabitError("unsupported habit guard version")
        _require_non_empty(self.feature_key, "feature_key")
        if self.operator not in GUARD_OPERATORS:
            raise ObserverHabitError("unsupported guard operator")
        if self.expected_type not in GUARD_VALUE_TYPES:
            raise ObserverHabitError("unsupported guard expected_type")
        if self.guard_role not in GUARD_ROLES:
            raise ObserverHabitError("unsupported guard role")
        _ensure_sorted_unique(self.source_evidence_ids, "source_evidence_ids")
        if self.operator in {"equals", "not_equals"}:
            if _type_name(self.expected_value) != self.expected_type:
                raise ObserverHabitError("guard expected value type mismatch")
            if self.minimum_value is not None or self.maximum_value is not None:
                raise ObserverHabitError("equality guards cannot carry range bounds")
        elif self.operator == "in_closed_range":
            if self.expected_type not in {"int", "float", "number"}:
                raise ObserverHabitError("range guard requires numeric expected_type")
            if not _is_finite_number(self.minimum_value) or not _is_finite_number(
                self.maximum_value
            ):
                raise ObserverHabitError("range guard requires finite bounds")
            if float(cast(int | float, self.minimum_value)) > float(
                cast(int | float, self.maximum_value)
            ):
                raise ObserverHabitError("range guard bounds are invalid")
            if self.expected_value is not None:
                raise ObserverHabitError("range guard cannot carry expected_value")
        else:
            if (
                self.expected_value is not None
                or self.minimum_value is not None
                or self.maximum_value is not None
            ):
                raise ObserverHabitError("presence guards cannot carry values")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_guard_id != expected_id:
            raise ObserverHabitError("habit_guard_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            expected_type=self.expected_type,
            expected_value=self.expected_value,
            feature_key=self.feature_key,
            guard_role=self.guard_role,
            maximum_value=self.maximum_value,
            minimum_value=self.minimum_value,
            operator=self.operator,
            source_evidence_ids=list(self.source_evidence_ids),
        )
        if include_id:
            payload["habit_guard_id"] = self.habit_guard_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitGuardDTO":
        values["source_evidence_ids"] = tuple(
            sorted(set(cast(Sequence[str], values.get("source_evidence_ids", ()))))
        )
        payload = _payload_with_version(OBSERVER_HABIT_GUARD_VERSION, **values)
        return cls(habit_guard_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitCounterexampleDTO:
    counterexample_id: str
    habit_specification_id: str | None
    transition_key_id: str
    source_observation_artifact_id: str
    actual_action: str
    actual_target_state_class_id: str
    expected_target_state_class_id: str
    ledger_entry_id: str
    occurrence_id: str | None
    verification_status: str
    reason_codes: tuple[str, ...]
    candidate_guard_ids: tuple[str, ...]
    version: str = OBSERVER_HABIT_COUNTEREXAMPLE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_COUNTEREXAMPLE_VERSION:
            raise ObserverHabitError("unsupported counterexample version")
        for field_name in (
            "transition_key_id",
            "source_observation_artifact_id",
            "actual_action",
            "actual_target_state_class_id",
            "expected_target_state_class_id",
            "ledger_entry_id",
            "verification_status",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        _ensure_sorted_unique(self.candidate_guard_ids, "candidate_guard_ids")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.counterexample_id != expected_id:
            raise ObserverHabitError("counterexample_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            actual_action=self.actual_action,
            actual_target_state_class_id=self.actual_target_state_class_id,
            candidate_guard_ids=list(self.candidate_guard_ids),
            expected_target_state_class_id=self.expected_target_state_class_id,
            habit_specification_id=self.habit_specification_id,
            ledger_entry_id=self.ledger_entry_id,
            occurrence_id=self.occurrence_id,
            reason_codes=list(self.reason_codes),
            source_observation_artifact_id=self.source_observation_artifact_id,
            transition_key_id=self.transition_key_id,
            verification_status=self.verification_status,
        )
        if include_id:
            payload["counterexample_id"] = self.counterexample_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitCounterexampleDTO":
        for key in ("reason_codes", "candidate_guard_ids"):
            values[key] = tuple(sorted(set(cast(Sequence[str], values.get(key, ())))))
        payload = _payload_with_version(OBSERVER_HABIT_COUNTEREXAMPLE_VERSION, **values)
        return cls(counterexample_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitSpecificationDTO:
    habit_specification_id: str
    habit_compilation_recipe_id: str
    promotion_candidate_id: str
    promotion_analysis_id: str
    ledger_snapshot_id: str
    observation_graph_id: str
    grouping_recipe_id: str
    observation_schema_id: str
    transition_key_id: str
    source_state_class_id: str
    recommended_action: str
    expected_target_state_class_id: str
    positive_guards: tuple[ObserverHabitGuardDTO, ...]
    counterexample_guards: tuple[ObserverHabitGuardDTO, ...]
    supporting_occurrence_ids: tuple[str, ...]
    supporting_ledger_entry_ids: tuple[str, ...]
    status: str
    version: str = OBSERVER_HABIT_SPECIFICATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_SPECIFICATION_VERSION:
            raise ObserverHabitError("unsupported habit specification version")
        for field_name in (
            "habit_compilation_recipe_id",
            "promotion_candidate_id",
            "promotion_analysis_id",
            "ledger_snapshot_id",
            "observation_graph_id",
            "grouping_recipe_id",
            "observation_schema_id",
            "transition_key_id",
            "source_state_class_id",
            "recommended_action",
            "expected_target_state_class_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.status not in HABIT_SPECIFICATION_STATUSES:
            raise ObserverHabitError("unsupported habit status")
        _ensure_sorted_unique(
            self.supporting_occurrence_ids, "supporting_occurrence_ids"
        )
        _ensure_sorted_unique(
            self.supporting_ledger_entry_ids, "supporting_ledger_entry_ids"
        )
        guard_ids = tuple(
            item.habit_guard_id
            for item in self.positive_guards + self.counterexample_guards
        )
        _ensure_sorted_unique(tuple(sorted(guard_ids)), "habit_guard_ids")
        if any(item.guard_role != "positive" for item in self.positive_guards):
            raise ObserverHabitError("positive guard role mismatch")
        if any(
            item.guard_role != "counterexample" for item in self.counterexample_guards
        ):
            raise ObserverHabitError("counterexample guard role mismatch")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_specification_id != expected_id:
            raise ObserverHabitError("habit_specification_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            counterexample_guards=_canonical_tuple(self.counterexample_guards),
            expected_target_state_class_id=self.expected_target_state_class_id,
            grouping_recipe_id=self.grouping_recipe_id,
            habit_compilation_recipe_id=self.habit_compilation_recipe_id,
            ledger_snapshot_id=self.ledger_snapshot_id,
            observation_graph_id=self.observation_graph_id,
            observation_schema_id=self.observation_schema_id,
            positive_guards=_canonical_tuple(self.positive_guards),
            promotion_analysis_id=self.promotion_analysis_id,
            promotion_candidate_id=self.promotion_candidate_id,
            recommended_action=self.recommended_action,
            source_state_class_id=self.source_state_class_id,
            status=self.status,
            supporting_ledger_entry_ids=list(self.supporting_ledger_entry_ids),
            supporting_occurrence_ids=list(self.supporting_occurrence_ids),
            transition_key_id=self.transition_key_id,
        )
        if include_id:
            payload["habit_specification_id"] = self.habit_specification_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitSpecificationDTO":
        for key in ("supporting_occurrence_ids", "supporting_ledger_entry_ids"):
            values[key] = tuple(sorted(set(cast(Sequence[str], values[key]))))
        supporting_ledger_entry_ids = cast(
            tuple[str, ...], values["supporting_ledger_entry_ids"]
        )
        supporting_occurrence_ids = cast(
            tuple[str, ...], values["supporting_occurrence_ids"]
        )
        payload = _payload_with_version(
            OBSERVER_HABIT_SPECIFICATION_VERSION,
            counterexample_guards=_canonical_tuple(values["counterexample_guards"]),  # type: ignore[arg-type]
            expected_target_state_class_id=values["expected_target_state_class_id"],
            grouping_recipe_id=values["grouping_recipe_id"],
            habit_compilation_recipe_id=values["habit_compilation_recipe_id"],
            ledger_snapshot_id=values["ledger_snapshot_id"],
            observation_graph_id=values["observation_graph_id"],
            observation_schema_id=values["observation_schema_id"],
            positive_guards=_canonical_tuple(values["positive_guards"]),  # type: ignore[arg-type]
            promotion_analysis_id=values["promotion_analysis_id"],
            promotion_candidate_id=values["promotion_candidate_id"],
            recommended_action=values["recommended_action"],
            source_state_class_id=values["source_state_class_id"],
            status=values["status"],
            supporting_ledger_entry_ids=list(supporting_ledger_entry_ids),
            supporting_occurrence_ids=list(supporting_occurrence_ids),
            transition_key_id=values["transition_key_id"],
        )
        return cls(habit_specification_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitCompilationResultDTO:
    habit_compilation_result_id: str
    promotion_candidate_id: str
    compilation_recipe_id: str
    habit_specification: ObserverHabitSpecificationDTO | None
    counterexamples: tuple[ObserverHabitCounterexampleDTO, ...]
    disposition: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_COMPILATION_RESULT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_COMPILATION_RESULT_VERSION:
            raise ObserverHabitError("unsupported habit compilation result version")
        _require_non_empty(self.promotion_candidate_id, "promotion_candidate_id")
        _require_non_empty(self.compilation_recipe_id, "compilation_recipe_id")
        if self.disposition not in HABIT_COMPILATION_DISPOSITIONS:
            raise ObserverHabitError("unsupported compilation disposition")
        if (self.disposition == "compiled_for_shadow") != (
            self.habit_specification is not None
        ):
            raise ObserverHabitError("compiled disposition/specification mismatch")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_compilation_result_id != expected_id:
            raise ObserverHabitError("habit_compilation_result_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            compilation_recipe_id=self.compilation_recipe_id,
            counterexamples=_canonical_tuple(self.counterexamples),
            disposition=self.disposition,
            habit_specification=None
            if self.habit_specification is None
            else self.habit_specification.canonical_payload(),
            promotion_candidate_id=self.promotion_candidate_id,
            reason_codes=list(self.reason_codes),
        )
        if include_id:
            payload["habit_compilation_result_id"] = self.habit_compilation_result_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitCompilationResultDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        habit_specification = cast(
            ObserverHabitSpecificationDTO | None, values["habit_specification"]
        )
        reason_codes = cast(tuple[str, ...], values["reason_codes"])
        payload = _payload_with_version(
            OBSERVER_HABIT_COMPILATION_RESULT_VERSION,
            compilation_recipe_id=values["compilation_recipe_id"],
            counterexamples=_canonical_tuple(values["counterexamples"]),  # type: ignore[arg-type]
            disposition=values["disposition"],
            habit_specification=None
            if habit_specification is None
            else habit_specification.canonical_payload(),
            promotion_candidate_id=values["promotion_candidate_id"],
            reason_codes=list(reason_codes),
        )
        return cls(habit_compilation_result_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitGuardEvaluationDTO:
    guard_evaluation_id: str
    habit_guard_id: str
    observation_artifact_id: str
    status: str
    actual_type: str | None
    actual_value: object | None
    reason_code: str
    version: str = OBSERVER_HABIT_GUARD_EVALUATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_GUARD_EVALUATION_VERSION:
            raise ObserverHabitError("unsupported habit guard evaluation version")
        _require_non_empty(self.habit_guard_id, "habit_guard_id")
        _require_non_empty(self.observation_artifact_id, "observation_artifact_id")
        if self.status not in HABIT_GUARD_EVALUATION_STATUSES:
            raise ObserverHabitError("unsupported guard evaluation status")
        _require_non_empty(self.reason_code, "reason_code")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.guard_evaluation_id != expected_id:
            raise ObserverHabitError("guard_evaluation_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            actual_type=self.actual_type,
            actual_value=self.actual_value,
            habit_guard_id=self.habit_guard_id,
            observation_artifact_id=self.observation_artifact_id,
            reason_code=self.reason_code,
            status=self.status,
        )
        if include_id:
            payload["guard_evaluation_id"] = self.guard_evaluation_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitGuardEvaluationDTO":
        payload = _payload_with_version(
            OBSERVER_HABIT_GUARD_EVALUATION_VERSION, **values
        )
        return cls(guard_evaluation_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitEvaluationDTO:
    habit_evaluation_id: str
    habit_specification_id: str
    observation_artifact_id: str
    state_class_id: str | None
    guard_evaluations: tuple[ObserverHabitGuardEvaluationDTO, ...]
    decision: str
    recommended_action: str | None
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_EVALUATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_EVALUATION_VERSION:
            raise ObserverHabitError("unsupported habit evaluation version")
        _require_non_empty(self.habit_specification_id, "habit_specification_id")
        _require_non_empty(self.observation_artifact_id, "observation_artifact_id")
        if self.decision not in HABIT_EVALUATION_DECISIONS:
            raise ObserverHabitError("unsupported habit decision")
        if self.decision != "fire" and self.recommended_action is not None:
            raise ObserverHabitError("non-fire evaluation cannot recommend an action")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_evaluation_id != expected_id:
            raise ObserverHabitError("habit_evaluation_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            decision=self.decision,
            guard_evaluations=_canonical_tuple(self.guard_evaluations),
            habit_specification_id=self.habit_specification_id,
            observation_artifact_id=self.observation_artifact_id,
            reason_codes=list(self.reason_codes),
            recommended_action=self.recommended_action,
            state_class_id=self.state_class_id,
        )
        if include_id:
            payload["habit_evaluation_id"] = self.habit_evaluation_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitEvaluationDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        reason_codes = cast(tuple[str, ...], values["reason_codes"])
        payload = _payload_with_version(
            OBSERVER_HABIT_EVALUATION_VERSION,
            decision=values["decision"],
            guard_evaluations=_canonical_tuple(values["guard_evaluations"]),  # type: ignore[arg-type]
            habit_specification_id=values["habit_specification_id"],
            observation_artifact_id=values["observation_artifact_id"],
            reason_codes=list(reason_codes),
            recommended_action=values["recommended_action"],
            state_class_id=values["state_class_id"],
        )
        return cls(habit_evaluation_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitShadowOccurrenceDTO:
    shadow_occurrence_id: str
    habit_specification_id: str
    ledger_entry_id: str
    source_observation_artifact_id: str
    source_state_class_id: str
    habit_evaluation_id: str
    habit_decision: str
    habit_recommended_action: str | None
    authoritative_action: str
    actual_target_state_class_id: str
    expected_target_state_class_id: str
    outcome: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_SHADOW_OCCURRENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_SHADOW_OCCURRENCE_VERSION:
            raise ObserverHabitError("unsupported shadow occurrence version")
        for field_name in (
            "habit_specification_id",
            "ledger_entry_id",
            "source_observation_artifact_id",
            "source_state_class_id",
            "habit_evaluation_id",
            "habit_decision",
            "authoritative_action",
            "actual_target_state_class_id",
            "expected_target_state_class_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.outcome not in HABIT_SHADOW_OUTCOMES:
            raise ObserverHabitError("unsupported shadow outcome")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.shadow_occurrence_id != expected_id:
            raise ObserverHabitError("shadow_occurrence_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            actual_target_state_class_id=self.actual_target_state_class_id,
            authoritative_action=self.authoritative_action,
            expected_target_state_class_id=self.expected_target_state_class_id,
            habit_decision=self.habit_decision,
            habit_evaluation_id=self.habit_evaluation_id,
            habit_recommended_action=self.habit_recommended_action,
            habit_specification_id=self.habit_specification_id,
            ledger_entry_id=self.ledger_entry_id,
            outcome=self.outcome,
            reason_codes=list(self.reason_codes),
            source_observation_artifact_id=self.source_observation_artifact_id,
            source_state_class_id=self.source_state_class_id,
        )
        if include_id:
            payload["shadow_occurrence_id"] = self.shadow_occurrence_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitShadowOccurrenceDTO":
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        payload = _payload_with_version(
            OBSERVER_HABIT_SHADOW_OCCURRENCE_VERSION, **values
        )
        return cls(shadow_occurrence_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitShadowReplayDTO:
    habit_shadow_replay_id: str
    habit_specification_id: str
    ledger_snapshot_id: str
    shadow_occurrences: tuple[ObserverHabitShadowOccurrenceDTO, ...]
    applicable_count: int
    fire_count: int
    abstain_count: int
    correct_fire_count: int
    wrong_action_fire_count: int
    wrong_target_fire_count: int
    safe_abstention_count: int
    missed_opportunity_count: int
    invalid_count: int
    episode_ids: tuple[str, ...]
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_SHADOW_REPLAY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_SHADOW_REPLAY_VERSION:
            raise ObserverHabitError("unsupported shadow replay version")
        _require_non_empty(self.habit_specification_id, "habit_specification_id")
        _require_non_empty(self.ledger_snapshot_id, "ledger_snapshot_id")
        _ensure_sorted_unique(self.episode_ids, "episode_ids")
        _ensure_sorted_unique(self.failure_codes, "failure_codes")
        if self.status not in HABIT_SHADOW_REPLAY_STATUSES:
            raise ObserverHabitError("unsupported shadow replay status")
        outcomes = [item.outcome for item in self.shadow_occurrences]
        if self.correct_fire_count != outcomes.count("correct_fire"):
            raise ObserverHabitError("correct_fire_count mismatch")
        if self.wrong_action_fire_count != outcomes.count("wrong_action_fire"):
            raise ObserverHabitError("wrong_action_fire_count mismatch")
        if self.wrong_target_fire_count != outcomes.count("wrong_target_fire"):
            raise ObserverHabitError("wrong_target_fire_count mismatch")
        if self.safe_abstention_count != outcomes.count("safe_abstention"):
            raise ObserverHabitError("safe_abstention_count mismatch")
        if self.missed_opportunity_count != outcomes.count("missed_opportunity"):
            raise ObserverHabitError("missed_opportunity_count mismatch")
        if self.invalid_count != outcomes.count("invalid_evaluation"):
            raise ObserverHabitError("invalid_count mismatch")
        if self.fire_count != (
            self.correct_fire_count
            + self.wrong_action_fire_count
            + self.wrong_target_fire_count
        ):
            raise ObserverHabitError("fire_count mismatch")
        if self.abstain_count != (
            self.safe_abstention_count + self.missed_opportunity_count
        ):
            raise ObserverHabitError("abstain_count mismatch")
        if (
            self.applicable_count
            != self.fire_count + self.abstain_count + self.invalid_count
        ):
            raise ObserverHabitError("applicable_count mismatch")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_shadow_replay_id != expected_id:
            raise ObserverHabitError("habit_shadow_replay_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            abstain_count=self.abstain_count,
            applicable_count=self.applicable_count,
            correct_fire_count=self.correct_fire_count,
            episode_ids=list(self.episode_ids),
            failure_codes=list(self.failure_codes),
            fire_count=self.fire_count,
            habit_specification_id=self.habit_specification_id,
            invalid_count=self.invalid_count,
            ledger_snapshot_id=self.ledger_snapshot_id,
            missed_opportunity_count=self.missed_opportunity_count,
            safe_abstention_count=self.safe_abstention_count,
            shadow_occurrences=_canonical_tuple(self.shadow_occurrences),
            status=self.status,
            wrong_action_fire_count=self.wrong_action_fire_count,
            wrong_target_fire_count=self.wrong_target_fire_count,
        )
        if include_id:
            payload["habit_shadow_replay_id"] = self.habit_shadow_replay_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitShadowReplayDTO":
        occurrences = cast(
            tuple[ObserverHabitShadowOccurrenceDTO, ...], values["shadow_occurrences"]
        )
        outcomes = [item.outcome for item in occurrences]
        correct_fire_count = outcomes.count("correct_fire")
        wrong_action_fire_count = outcomes.count("wrong_action_fire")
        wrong_target_fire_count = outcomes.count("wrong_target_fire")
        safe_abstention_count = outcomes.count("safe_abstention")
        missed_opportunity_count = outcomes.count("missed_opportunity")
        invalid_count = outcomes.count("invalid_evaluation")
        fire_count = (
            correct_fire_count + wrong_action_fire_count + wrong_target_fire_count
        )
        abstain_count = safe_abstention_count + missed_opportunity_count
        applicable_count = fire_count + abstain_count + invalid_count
        values["correct_fire_count"] = correct_fire_count
        values["wrong_action_fire_count"] = wrong_action_fire_count
        values["wrong_target_fire_count"] = wrong_target_fire_count
        values["safe_abstention_count"] = safe_abstention_count
        values["missed_opportunity_count"] = missed_opportunity_count
        values["invalid_count"] = invalid_count
        values["fire_count"] = fire_count
        values["abstain_count"] = abstain_count
        values["applicable_count"] = applicable_count
        for key in ("episode_ids", "failure_codes"):
            values[key] = tuple(sorted(set(cast(Sequence[str], values.get(key, ())))))
        episode_ids = cast(tuple[str, ...], values["episode_ids"])
        failure_codes = cast(tuple[str, ...], values["failure_codes"])
        payload = _payload_with_version(
            OBSERVER_HABIT_SHADOW_REPLAY_VERSION,
            abstain_count=values["abstain_count"],
            applicable_count=values["applicable_count"],
            correct_fire_count=values["correct_fire_count"],
            episode_ids=list(episode_ids),
            failure_codes=list(failure_codes),
            fire_count=values["fire_count"],
            habit_specification_id=values["habit_specification_id"],
            invalid_count=values["invalid_count"],
            ledger_snapshot_id=values["ledger_snapshot_id"],
            missed_opportunity_count=values["missed_opportunity_count"],
            safe_abstention_count=values["safe_abstention_count"],
            shadow_occurrences=_canonical_tuple(occurrences),
            status=values["status"],
            wrong_action_fire_count=values["wrong_action_fire_count"],
            wrong_target_fire_count=values["wrong_target_fire_count"],
        )
        return cls(habit_shadow_replay_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitShadowEpisodeDTO:
    habit_shadow_episode_id: str
    habit_specification_id: str
    fixture_episode_result_id: str
    ledger_snapshot_id: str
    shadow_replay: ObserverHabitShadowReplayDTO
    authoritative_action_ids: tuple[str, ...]
    habit_fire_sequences: tuple[int, ...]
    habit_abstain_sequences: tuple[int, ...]
    status: str
    version: str = OBSERVER_HABIT_SHADOW_EPISODE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_SHADOW_EPISODE_VERSION:
            raise ObserverHabitError("unsupported shadow episode version")
        for field_name in (
            "habit_specification_id",
            "fixture_episode_result_id",
            "ledger_snapshot_id",
            "status",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_sorted_unique(self.authoritative_action_ids, "authoritative_action_ids")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_shadow_episode_id != expected_id:
            raise ObserverHabitError("habit_shadow_episode_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            authoritative_action_ids=list(self.authoritative_action_ids),
            fixture_episode_result_id=self.fixture_episode_result_id,
            habit_abstain_sequences=list(self.habit_abstain_sequences),
            habit_fire_sequences=list(self.habit_fire_sequences),
            habit_specification_id=self.habit_specification_id,
            ledger_snapshot_id=self.ledger_snapshot_id,
            shadow_replay=self.shadow_replay.canonical_payload(),
            status=self.status,
        )
        if include_id:
            payload["habit_shadow_episode_id"] = self.habit_shadow_episode_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitShadowEpisodeDTO":
        values["authoritative_action_ids"] = tuple(
            sorted(set(cast(Sequence[str], values["authoritative_action_ids"])))
        )
        authoritative_action_ids = cast(
            tuple[str, ...], values["authoritative_action_ids"]
        )
        habit_abstain_sequences = cast(
            tuple[int, ...], values["habit_abstain_sequences"]
        )
        habit_fire_sequences = cast(tuple[int, ...], values["habit_fire_sequences"])
        shadow_replay = cast(ObserverHabitShadowReplayDTO, values["shadow_replay"])
        payload = _payload_with_version(
            OBSERVER_HABIT_SHADOW_EPISODE_VERSION,
            authoritative_action_ids=list(authoritative_action_ids),
            fixture_episode_result_id=values["fixture_episode_result_id"],
            habit_abstain_sequences=list(habit_abstain_sequences),
            habit_fire_sequences=list(habit_fire_sequences),
            habit_specification_id=values["habit_specification_id"],
            ledger_snapshot_id=values["ledger_snapshot_id"],
            shadow_replay=shadow_replay.canonical_payload(),
            status=values["status"],
        )
        return cls(habit_shadow_episode_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitShadowAuditRecipeDTO:
    shadow_audit_recipe_id: str
    minimum_applicable_count: int
    minimum_episode_count: int
    minimum_correct_fire_count: int
    maximum_wrong_action_fire_count: int
    maximum_wrong_target_fire_count: int
    maximum_missed_opportunity_count: int
    maximum_invalid_count: int
    require_zero_false_fires: bool
    require_counterexample_coverage: bool
    version: str = OBSERVER_HABIT_SHADOW_AUDIT_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_SHADOW_AUDIT_RECIPE_VERSION:
            raise ObserverHabitError("unsupported shadow audit recipe version")
        for field_name in (
            "minimum_applicable_count",
            "minimum_episode_count",
            "minimum_correct_fire_count",
            "maximum_wrong_action_fire_count",
            "maximum_wrong_target_fire_count",
            "maximum_missed_opportunity_count",
            "maximum_invalid_count",
        ):
            _ensure_non_negative(getattr(self, field_name), field_name)
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.shadow_audit_recipe_id != expected_id:
            raise ObserverHabitError("shadow_audit_recipe_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            maximum_invalid_count=self.maximum_invalid_count,
            maximum_missed_opportunity_count=self.maximum_missed_opportunity_count,
            maximum_wrong_action_fire_count=self.maximum_wrong_action_fire_count,
            maximum_wrong_target_fire_count=self.maximum_wrong_target_fire_count,
            minimum_applicable_count=self.minimum_applicable_count,
            minimum_correct_fire_count=self.minimum_correct_fire_count,
            minimum_episode_count=self.minimum_episode_count,
            require_counterexample_coverage=self.require_counterexample_coverage,
            require_zero_false_fires=self.require_zero_false_fires,
        )
        if include_id:
            payload["shadow_audit_recipe_id"] = self.shadow_audit_recipe_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitShadowAuditRecipeDTO":
        payload = _payload_with_version(
            OBSERVER_HABIT_SHADOW_AUDIT_RECIPE_VERSION, **values
        )
        return cls(shadow_audit_recipe_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitCounterexampleCoverageDTO:
    coverage_id: str
    habit_specification_id: str
    counterexample_ids: tuple[str, ...]
    guarded_counterexample_ids: tuple[str, ...]
    unguarded_counterexample_ids: tuple[str, ...]
    coverage_complete: bool
    version: str = OBSERVER_HABIT_COUNTEREXAMPLE_COVERAGE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_COUNTEREXAMPLE_COVERAGE_VERSION:
            raise ObserverHabitError("unsupported counterexample coverage version")
        _require_non_empty(self.habit_specification_id, "habit_specification_id")
        for field_name in (
            "counterexample_ids",
            "guarded_counterexample_ids",
            "unguarded_counterexample_ids",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        if (
            tuple(
                sorted(
                    self.guarded_counterexample_ids + self.unguarded_counterexample_ids
                )
            )
            != self.counterexample_ids
        ):
            raise ObserverHabitError("counterexample coverage partition mismatch")
        if self.coverage_complete != (not self.unguarded_counterexample_ids):
            raise ObserverHabitError("coverage_complete mismatch")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.coverage_id != expected_id:
            raise ObserverHabitError("coverage_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            counterexample_ids=list(self.counterexample_ids),
            coverage_complete=self.coverage_complete,
            guarded_counterexample_ids=list(self.guarded_counterexample_ids),
            habit_specification_id=self.habit_specification_id,
            unguarded_counterexample_ids=list(self.unguarded_counterexample_ids),
        )
        if include_id:
            payload["coverage_id"] = self.coverage_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitCounterexampleCoverageDTO":
        for key in (
            "counterexample_ids",
            "guarded_counterexample_ids",
            "unguarded_counterexample_ids",
        ):
            values[key] = tuple(sorted(set(cast(Sequence[str], values.get(key, ())))))
        values["coverage_complete"] = not values["unguarded_counterexample_ids"]
        payload = _payload_with_version(
            OBSERVER_HABIT_COUNTEREXAMPLE_COVERAGE_VERSION, **values
        )
        return cls(coverage_id=canonical_id(payload), **values)  # type: ignore[arg-type]


@dataclass(frozen=True)
class ObserverHabitShadowAuditDTO:
    habit_shadow_audit_id: str
    habit_specification_id: str
    shadow_audit_recipe_id: str
    historical_shadow_replay_id: str
    evaluated_shadow_replay_ids: tuple[str, ...]
    live_shadow_episode_ids: tuple[str, ...]
    counterexample_coverage: ObserverHabitCounterexampleCoverageDTO | None
    disposition: str
    eligible_for_admission_review: bool
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_SHADOW_AUDIT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_SHADOW_AUDIT_VERSION:
            raise ObserverHabitError("unsupported shadow audit version")
        for field_name in (
            "habit_specification_id",
            "shadow_audit_recipe_id",
            "historical_shadow_replay_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_sorted_unique(self.live_shadow_episode_ids, "live_shadow_episode_ids")
        _ensure_sorted_unique(
            self.evaluated_shadow_replay_ids, "evaluated_shadow_replay_ids"
        )
        if self.historical_shadow_replay_id not in self.evaluated_shadow_replay_ids:
            raise ObserverHabitError(
                "evaluated replay IDs must include historical replay"
            )
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        if self.disposition not in HABIT_SHADOW_AUDIT_DISPOSITIONS:
            raise ObserverHabitError("unsupported shadow audit disposition")
        if self.eligible_for_admission_review != (
            self.disposition == "eligible_for_admission_review"
        ):
            raise ObserverHabitError("audit eligibility/disposition mismatch")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_shadow_audit_id != expected_id:
            raise ObserverHabitError("habit_shadow_audit_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload_with_version(
            self.version,
            counterexample_coverage=None
            if self.counterexample_coverage is None
            else self.counterexample_coverage.canonical_payload(),
            disposition=self.disposition,
            eligible_for_admission_review=self.eligible_for_admission_review,
            habit_specification_id=self.habit_specification_id,
            historical_shadow_replay_id=self.historical_shadow_replay_id,
            evaluated_shadow_replay_ids=list(self.evaluated_shadow_replay_ids),
            live_shadow_episode_ids=list(self.live_shadow_episode_ids),
            reason_codes=list(self.reason_codes),
            shadow_audit_recipe_id=self.shadow_audit_recipe_id,
        )
        if include_id:
            payload["habit_shadow_audit_id"] = self.habit_shadow_audit_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitShadowAuditDTO":
        values["live_shadow_episode_ids"] = tuple(
            sorted(set(cast(Sequence[str], values.get("live_shadow_episode_ids", ()))))
        )
        values["evaluated_shadow_replay_ids"] = tuple(
            sorted(
                set(cast(Sequence[str], values.get("evaluated_shadow_replay_ids", ())))
            )
        )
        values["reason_codes"] = tuple(
            sorted(set(cast(Sequence[str], values.get("reason_codes", ()))))
        )
        values["eligible_for_admission_review"] = (
            values["disposition"] == "eligible_for_admission_review"
        )
        coverage = cast(
            ObserverHabitCounterexampleCoverageDTO | None,
            values["counterexample_coverage"],
        )
        live_shadow_episode_ids = cast(
            tuple[str, ...], values["live_shadow_episode_ids"]
        )
        evaluated_shadow_replay_ids = cast(
            tuple[str, ...], values["evaluated_shadow_replay_ids"]
        )
        reason_codes = cast(tuple[str, ...], values["reason_codes"])
        payload = _payload_with_version(
            OBSERVER_HABIT_SHADOW_AUDIT_VERSION,
            counterexample_coverage=None
            if coverage is None
            else coverage.canonical_payload(),
            disposition=values["disposition"],
            eligible_for_admission_review=values["eligible_for_admission_review"],
            evaluated_shadow_replay_ids=list(evaluated_shadow_replay_ids),
            habit_specification_id=values["habit_specification_id"],
            historical_shadow_replay_id=values["historical_shadow_replay_id"],
            live_shadow_episode_ids=list(live_shadow_episode_ids),
            reason_codes=list(reason_codes),
            shadow_audit_recipe_id=values["shadow_audit_recipe_id"],
        )
        return cls(habit_shadow_audit_id=canonical_id(payload), **values)  # type: ignore[arg-type]
