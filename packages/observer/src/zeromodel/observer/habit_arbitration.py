"""Deterministic non-controlling arbitration plans for Observer habits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import (
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)
from zeromodel.observer.grouping import ObserverStateGroupingRecipeDTO
from zeromodel.observer.habit import ObserverHabitError, ObserverHabitSpecificationDTO
from zeromodel.observer.habit_overlap import ObserverHabitOverlapAnalysisDTO
from zeromodel.observer.habit_service import evaluate_observer_habit

OBSERVER_HABIT_ARBITRATION_PLAN_RECIPE_VERSION: Final = (
    "observer-habit-arbitration-plan-recipe/1"
)
OBSERVER_HABIT_ARBITRATION_PLAN_VERSION: Final = "observer-habit-arbitration-plan/1"
OBSERVER_HABIT_ARBITRATION_PLAN_COMPILATION_VERSION: Final = (
    "observer-habit-arbitration-plan-compilation/1"
)
OBSERVER_HABIT_ARBITRATION_EVALUATION_VERSION: Final = (
    "observer-habit-arbitration-evaluation/1"
)

ARBITRATION_STRATEGIES: Final = frozenset(
    {"strict_unique_fire", "most_specific_guard", "declared_order"}
)
PLAN_COMPILATION_DISPOSITIONS: Final = frozenset(
    {
        "compiled",
        "blocked_by_action_conflict",
        "blocked_by_target_conflict",
        "blocked_by_inconclusive_analysis",
        "unsupported_strategy",
        "invalid_declared_order",
        "too_many_habits",
        "invalid_lineage",
    }
)
ARBITRATION_DECISIONS: Final = frozenset(
    {"selected_habit", "fallback_no_fire", "fallback_ambiguous", "fallback_invalid"}
)


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _payload(version: str, **values: object) -> dict[str, object]:
    return {"version": version, **values}


@dataclass(frozen=True)
class ObserverHabitArbitrationPlanRecipeDTO:
    habit_arbitration_plan_recipe_id: str
    allowed_strategies: tuple[str, ...]
    default_strategy: str
    require_complete_pair_analysis: bool
    allow_different_action_overlap: bool
    allow_target_conflict: bool
    allow_inconclusive_pairs: bool
    maximum_habit_count: int
    version: str = OBSERVER_HABIT_ARBITRATION_PLAN_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_PLAN_RECIPE_VERSION:
            raise ObserverHabitError("unsupported arbitration plan recipe version")
        if self.maximum_habit_count < 2:
            raise ObserverHabitError("maximum_habit_count must allow two habits")
        if self.default_strategy not in self.allowed_strategies:
            raise ObserverHabitError("default strategy must be allowed")
        if set(self.allowed_strategies) - ARBITRATION_STRATEGIES:
            raise ObserverHabitError("unsupported arbitration strategy")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_plan_recipe_id != expected:
            raise ObserverHabitError("habit_arbitration_plan_recipe_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            allow_different_action_overlap=self.allow_different_action_overlap,
            allow_inconclusive_pairs=self.allow_inconclusive_pairs,
            allow_target_conflict=self.allow_target_conflict,
            allowed_strategies=list(self.allowed_strategies),
            default_strategy=self.default_strategy,
            maximum_habit_count=self.maximum_habit_count,
            require_complete_pair_analysis=self.require_complete_pair_analysis,
        )
        if include_id:
            payload["habit_arbitration_plan_recipe_id"] = (
                self.habit_arbitration_plan_recipe_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationPlanRecipeDTO":
        defaults = {
            "allowed_strategies": (
                "declared_order",
                "most_specific_guard",
                "strict_unique_fire",
            ),
            "default_strategy": "strict_unique_fire",
            "require_complete_pair_analysis": True,
            "allow_different_action_overlap": False,
            "allow_target_conflict": False,
            "allow_inconclusive_pairs": False,
            "maximum_habit_count": 8,
        }
        defaults.update(values)
        defaults["allowed_strategies"] = _sorted_unique(
            cast(Sequence[str], defaults["allowed_strategies"])
        )
        payload = _payload(
            OBSERVER_HABIT_ARBITRATION_PLAN_RECIPE_VERSION,
            allow_different_action_overlap=defaults["allow_different_action_overlap"],
            allow_inconclusive_pairs=defaults["allow_inconclusive_pairs"],
            allow_target_conflict=defaults["allow_target_conflict"],
            allowed_strategies=list(
                cast(tuple[str, ...], defaults["allowed_strategies"])
            ),
            default_strategy=defaults["default_strategy"],
            maximum_habit_count=defaults["maximum_habit_count"],
            require_complete_pair_analysis=defaults["require_complete_pair_analysis"],
        )
        return cls(
            habit_arbitration_plan_recipe_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_PLAN_RECIPE_VERSION,
            **defaults,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitArbitrationPlanDTO:
    habit_arbitration_plan_id: str
    habit_specification_ids: tuple[str, ...]
    habit_overlap_analysis_id: str
    activation_scope_id: str
    arbitration_strategy: str
    ordered_habit_ids: tuple[str, ...]
    tie_policy: str
    invalid_evaluation_policy: str
    no_fire_policy: str
    conflict_policy: str
    status: str
    version: str = OBSERVER_HABIT_ARBITRATION_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_PLAN_VERSION:
            raise ObserverHabitError("unsupported arbitration plan version")
        if self.status != "shadow_candidate":
            raise ObserverHabitError("arbitration plan must be shadow_candidate")
        if self.arbitration_strategy not in ARBITRATION_STRATEGIES:
            raise ObserverHabitError("unsupported arbitration strategy")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_plan_id != expected:
            raise ObserverHabitError("habit_arbitration_plan_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            activation_scope_id=self.activation_scope_id,
            arbitration_strategy=self.arbitration_strategy,
            conflict_policy=self.conflict_policy,
            habit_overlap_analysis_id=self.habit_overlap_analysis_id,
            habit_specification_ids=list(self.habit_specification_ids),
            invalid_evaluation_policy=self.invalid_evaluation_policy,
            no_fire_policy=self.no_fire_policy,
            ordered_habit_ids=list(self.ordered_habit_ids),
            status=self.status,
            tie_policy=self.tie_policy,
        )
        if include_id:
            payload["habit_arbitration_plan_id"] = self.habit_arbitration_plan_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationPlanDTO":
        for key in ("habit_specification_ids", "ordered_habit_ids"):
            values[key] = tuple(cast(Sequence[str], values[key]))
        payload = _payload(
            OBSERVER_HABIT_ARBITRATION_PLAN_VERSION,
            activation_scope_id=values["activation_scope_id"],
            arbitration_strategy=values["arbitration_strategy"],
            conflict_policy=values["conflict_policy"],
            habit_overlap_analysis_id=values["habit_overlap_analysis_id"],
            habit_specification_ids=list(
                cast(tuple[str, ...], values["habit_specification_ids"])
            ),
            invalid_evaluation_policy=values["invalid_evaluation_policy"],
            no_fire_policy=values["no_fire_policy"],
            ordered_habit_ids=list(cast(tuple[str, ...], values["ordered_habit_ids"])),
            status=values["status"],
            tie_policy=values["tie_policy"],
        )
        return cls(
            habit_arbitration_plan_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_PLAN_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitArbitrationPlanCompilationDTO:
    habit_arbitration_plan_compilation_id: str
    habit_overlap_analysis_id: str
    habit_arbitration_plan_recipe_id: str
    requested_strategy: str
    disposition: str
    reason_codes: tuple[str, ...]
    arbitration_plan: ObserverHabitArbitrationPlanDTO | None
    version: str = OBSERVER_HABIT_ARBITRATION_PLAN_COMPILATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_PLAN_COMPILATION_VERSION:
            raise ObserverHabitError("unsupported plan compilation version")
        if self.disposition not in PLAN_COMPILATION_DISPOSITIONS:
            raise ObserverHabitError("unsupported compilation disposition")
        if (self.disposition == "compiled") != (self.arbitration_plan is not None):
            raise ObserverHabitError("compiled disposition/plan mismatch")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_plan_compilation_id != expected:
            raise ObserverHabitError("habit_arbitration_plan_compilation_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            arbitration_plan=None
            if self.arbitration_plan is None
            else self.arbitration_plan.canonical_payload(),
            disposition=self.disposition,
            habit_arbitration_plan_recipe_id=self.habit_arbitration_plan_recipe_id,
            habit_overlap_analysis_id=self.habit_overlap_analysis_id,
            reason_codes=list(self.reason_codes),
            requested_strategy=self.requested_strategy,
        )
        if include_id:
            payload["habit_arbitration_plan_compilation_id"] = (
                self.habit_arbitration_plan_compilation_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationPlanCompilationDTO":
        values["reason_codes"] = _sorted_unique(
            cast(Sequence[str], values.get("reason_codes", ()))
        )
        payload = _payload(
            OBSERVER_HABIT_ARBITRATION_PLAN_COMPILATION_VERSION,
            arbitration_plan=None
            if values["arbitration_plan"] is None
            else cast(
                ObserverHabitArbitrationPlanDTO, values["arbitration_plan"]
            ).canonical_payload(),
            disposition=values["disposition"],
            habit_arbitration_plan_recipe_id=values["habit_arbitration_plan_recipe_id"],
            habit_overlap_analysis_id=values["habit_overlap_analysis_id"],
            reason_codes=list(cast(tuple[str, ...], values["reason_codes"])),
            requested_strategy=values["requested_strategy"],
        )
        return cls(
            habit_arbitration_plan_compilation_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_PLAN_COMPILATION_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitArbitrationEvaluationDTO:
    habit_arbitration_evaluation_id: str
    habit_arbitration_plan_id: str
    observation_artifact_id: str
    authoritative_fallback_action: str
    fired_habit_ids: tuple[str, ...]
    invalid_habit_ids: tuple[str, ...]
    selected_habit_id: str | None
    selected_action: str
    decision: str
    reason_codes: tuple[str, ...]
    habit_evaluation_ids: tuple[str, ...]
    version: str = OBSERVER_HABIT_ARBITRATION_EVALUATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_EVALUATION_VERSION:
            raise ObserverHabitError("unsupported arbitration evaluation version")
        if self.decision not in ARBITRATION_DECISIONS:
            raise ObserverHabitError("unsupported arbitration decision")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_evaluation_id != expected:
            raise ObserverHabitError("habit_arbitration_evaluation_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            authoritative_fallback_action=self.authoritative_fallback_action,
            decision=self.decision,
            fired_habit_ids=list(self.fired_habit_ids),
            habit_arbitration_plan_id=self.habit_arbitration_plan_id,
            habit_evaluation_ids=list(self.habit_evaluation_ids),
            invalid_habit_ids=list(self.invalid_habit_ids),
            observation_artifact_id=self.observation_artifact_id,
            reason_codes=list(self.reason_codes),
            selected_action=self.selected_action,
            selected_habit_id=self.selected_habit_id,
        )
        if include_id:
            payload["habit_arbitration_evaluation_id"] = (
                self.habit_arbitration_evaluation_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationEvaluationDTO":
        for key in (
            "fired_habit_ids",
            "invalid_habit_ids",
            "reason_codes",
            "habit_evaluation_ids",
        ):
            values[key] = _sorted_unique(cast(Sequence[str], values.get(key, ())))
        payload = _payload(
            OBSERVER_HABIT_ARBITRATION_EVALUATION_VERSION,
            authoritative_fallback_action=values["authoritative_fallback_action"],
            decision=values["decision"],
            fired_habit_ids=list(cast(tuple[str, ...], values["fired_habit_ids"])),
            habit_arbitration_plan_id=values["habit_arbitration_plan_id"],
            habit_evaluation_ids=list(
                cast(tuple[str, ...], values["habit_evaluation_ids"])
            ),
            invalid_habit_ids=list(cast(tuple[str, ...], values["invalid_habit_ids"])),
            observation_artifact_id=values["observation_artifact_id"],
            reason_codes=list(cast(tuple[str, ...], values["reason_codes"])),
            selected_action=values["selected_action"],
            selected_habit_id=values["selected_habit_id"],
        )
        return cls(
            habit_arbitration_evaluation_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_EVALUATION_VERSION,
            **values,  # type: ignore[arg-type]
        )


def compile_observer_habit_arbitration_plan(
    *,
    overlap_analysis: ObserverHabitOverlapAnalysisDTO,
    habit_specifications: tuple[ObserverHabitSpecificationDTO, ...],
    plan_recipe: ObserverHabitArbitrationPlanRecipeDTO,
    requested_strategy: str,
    declared_order: tuple[str, ...] = (),
) -> ObserverHabitArbitrationPlanCompilationDTO:
    habit_ids = tuple(
        sorted(item.habit_specification_id for item in habit_specifications)
    )
    disposition = "compiled"
    reasons: set[str] = {"compiled"}
    if requested_strategy not in plan_recipe.allowed_strategies:
        disposition, reasons = "unsupported_strategy", {"unsupported_strategy"}
    elif tuple(overlap_analysis.habit_specification_ids) != habit_ids:
        disposition, reasons = "invalid_lineage", {"invalid_lineage"}
    elif len(habit_ids) > plan_recipe.maximum_habit_count:
        disposition, reasons = "too_many_habits", {"too_many_habits"}
    elif (
        plan_recipe.require_complete_pair_analysis
        and overlap_analysis.inconclusive_pair_count
    ):
        disposition, reasons = (
            "blocked_by_inconclusive_analysis",
            {"blocked_by_inconclusive_analysis"},
        )
    elif (
        not plan_recipe.allow_different_action_overlap
        and overlap_analysis.different_action_conflict_count
    ):
        disposition, reasons = (
            "blocked_by_action_conflict",
            {"blocked_by_action_conflict"},
        )
    elif (
        not plan_recipe.allow_target_conflict and overlap_analysis.target_conflict_count
    ):
        disposition, reasons = (
            "blocked_by_target_conflict",
            {"blocked_by_target_conflict"},
        )
    elif (
        not plan_recipe.allow_inconclusive_pairs
        and overlap_analysis.inconclusive_pair_count
    ):
        disposition, reasons = (
            "blocked_by_inconclusive_analysis",
            {"blocked_by_inconclusive_analysis"},
        )
    if requested_strategy == "declared_order":
        if tuple(sorted(declared_order)) != habit_ids or len(
            set(declared_order)
        ) != len(declared_order):
            disposition, reasons = "invalid_declared_order", {"invalid_declared_order"}
    order = habit_ids if requested_strategy != "declared_order" else declared_order
    plan = None
    if disposition == "compiled":
        plan = ObserverHabitArbitrationPlanDTO.create(
            habit_specification_ids=habit_ids,
            habit_overlap_analysis_id=overlap_analysis.habit_overlap_analysis_id,
            activation_scope_id=overlap_analysis.activation_scope_id,
            arbitration_strategy=requested_strategy,
            ordered_habit_ids=order,
            tie_policy="fallback",
            invalid_evaluation_policy="fallback",
            no_fire_policy="fallback",
            conflict_policy="fallback",
            status="shadow_candidate",
        )
    return ObserverHabitArbitrationPlanCompilationDTO.create(
        habit_overlap_analysis_id=overlap_analysis.habit_overlap_analysis_id,
        habit_arbitration_plan_recipe_id=plan_recipe.habit_arbitration_plan_recipe_id,
        requested_strategy=requested_strategy,
        disposition=disposition,
        reason_codes=tuple(reasons),
        arbitration_plan=plan,
    )


def evaluate_observer_habit_arbitration(
    *,
    arbitration_plan: ObserverHabitArbitrationPlanDTO,
    habit_specifications: tuple[ObserverHabitSpecificationDTO, ...],
    observation: ObserverObservationArtifactDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    authoritative_fallback_action: str,
) -> ObserverHabitArbitrationEvaluationDTO:
    habits = {
        item.habit_specification_id: item
        for item in habit_specifications
        if item.habit_specification_id in arbitration_plan.habit_specification_ids
    }
    evaluations = tuple(
        evaluate_observer_habit(
            habit_specification=habits[habit_id],
            observation=observation,
            grouping_recipe=grouping_recipe,
            observation_schema=observation_schema,
        )
        for habit_id in arbitration_plan.ordered_habit_ids
    )
    by_id = {item.habit_specification_id: item for item in evaluations}
    fired = tuple(
        item.habit_specification_id for item in evaluations if item.decision == "fire"
    )
    invalid = tuple(
        item.habit_specification_id
        for item in evaluations
        if item.decision == "invalid"
    )
    selected: str | None = None
    reasons: set[str] = set()
    if invalid:
        decision = "fallback_invalid"
        reasons.add("invalid_habit_evaluation")
    elif arbitration_plan.arbitration_strategy == "strict_unique_fire":
        if len(fired) == 1:
            selected = fired[0]
            decision = "selected_habit"
            reasons.add("strict_unique_fire")
        elif not fired:
            decision = "fallback_no_fire"
            reasons.add("no_habit_fired")
        else:
            decision = "fallback_ambiguous"
            reasons.add("ambiguous_multiple_fire")
    elif arbitration_plan.arbitration_strategy == "declared_order":
        selected = next(
            (item for item in arbitration_plan.ordered_habit_ids if item in fired), None
        )
        decision = "selected_habit" if selected is not None else "fallback_no_fire"
        reasons.add("declared_order" if selected is not None else "no_habit_fired")
    else:
        selected = _most_specific_winner(fired, habits)
        if selected is None:
            decision = "fallback_no_fire" if not fired else "fallback_ambiguous"
            reasons.add("most_specific_no_unique_winner")
        else:
            decision = "selected_habit"
            reasons.add("most_specific_guard")
    selected_action = (
        authoritative_fallback_action
        if selected is None
        else cast(str, by_id[selected].recommended_action)
    )
    return ObserverHabitArbitrationEvaluationDTO.create(
        habit_arbitration_plan_id=arbitration_plan.habit_arbitration_plan_id,
        observation_artifact_id=observation.observation_artifact_id,
        authoritative_fallback_action=authoritative_fallback_action,
        fired_habit_ids=fired,
        invalid_habit_ids=invalid,
        selected_habit_id=selected,
        selected_action=selected_action,
        decision=decision,
        reason_codes=tuple(reasons),
        habit_evaluation_ids=tuple(item.habit_evaluation_id for item in evaluations),
    )


def _most_specific_winner(
    fired: tuple[str, ...], habits: Mapping[str, ObserverHabitSpecificationDTO]
) -> str | None:
    if not fired:
        return None
    counts = {
        habit_id: len(habits[habit_id].positive_guards)
        + len(habits[habit_id].counterexample_guards)
        for habit_id in fired
    }
    maximum = max(counts.values())
    winners = tuple(habit_id for habit_id, count in counts.items() if count == maximum)
    return winners[0] if len(winners) == 1 else None
