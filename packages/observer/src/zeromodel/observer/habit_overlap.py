"""Multi-habit overlap analysis for non-controlling Observer arbitration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import (
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)
from zeromodel.observer.fixture_predictor import _observation_for_state
from zeromodel.observer.graph import ObserverObservationGraphDTO
from zeromodel.observer.grouping import ObserverStateGroupingRecipeDTO
from zeromodel.observer.habit import (
    ObserverHabitError,
    ObserverHabitEvaluationDTO,
    ObserverHabitSpecificationDTO,
)
from zeromodel.observer.habit_activation import ObserverHabitActivationScopeDTO
from zeromodel.observer.habit_admission import ObserverHabitAdmissionDecisionDTO
from zeromodel.observer.habit_service import evaluate_observer_habit
from zeromodel.observer.ledger import ObserverTransitionLedgerSnapshotDTO

OBSERVER_HABIT_OVERLAP_ANALYSIS_RECIPE_VERSION: Final = (
    "observer-habit-overlap-analysis-recipe/1"
)
OBSERVER_HABIT_PAIR_OVERLAP_VERSION: Final = "observer-habit-pair-overlap/1"
OBSERVER_HABIT_OVERLAP_OCCURRENCE_VERSION: Final = "observer-habit-overlap-occurrence/1"
OBSERVER_HABIT_OVERLAP_ANALYSIS_VERSION: Final = "observer-habit-overlap-analysis/1"

SOURCE_RELATIONS: Final = frozenset(
    {
        "same_source_state_class",
        "different_source_state_class",
        "unknown_source_relation",
    }
)
GUARD_RELATIONS: Final = frozenset(
    {
        "equivalent",
        "left_subsumes_right",
        "right_subsumes_left",
        "partially_overlapping",
        "disjoint",
        "unknown",
    }
)
ACTION_RELATIONS: Final = frozenset(
    {"same_action", "different_action", "unknown_action_relation"}
)
TARGET_RELATIONS: Final = frozenset(
    {"same_expected_target", "different_expected_target", "unknown_target_relation"}
)
PAIR_STATUSES: Final = frozenset({"completed", "inconclusive", "failed"})
ANALYSIS_STATUSES: Final = frozenset(
    {"completed", "completed_with_conflicts", "inconclusive", "failed"}
)
OCCURRENCE_CLASSIFICATIONS: Final = frozenset(
    {
        "same_action_same_target",
        "same_action_different_target",
        "different_action",
        "invalid_evaluation",
    }
)


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _payload(version: str, **values: object) -> dict[str, object]:
    return {"version": version, **values}


@dataclass(frozen=True)
class ObserverHabitOverlapAnalysisRecipeDTO:
    habit_overlap_analysis_recipe_id: str
    require_same_observation_schema: bool
    require_same_grouping_recipe: bool
    require_same_fixture_scope: bool
    evaluate_guard_equivalence: bool
    evaluate_guard_subsumption: bool
    require_evidenced_overlap_for_conflict: bool
    maximum_pair_count: int
    allowed_guard_operators: tuple[str, ...]
    version: str = OBSERVER_HABIT_OVERLAP_ANALYSIS_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_OVERLAP_ANALYSIS_RECIPE_VERSION:
            raise ObserverHabitError("unsupported overlap recipe version")
        if self.maximum_pair_count < 1:
            raise ObserverHabitError("maximum_pair_count must be positive")
        if not self.allowed_guard_operators:
            raise ObserverHabitError("allowed_guard_operators must be non-empty")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_overlap_analysis_recipe_id != expected:
            raise ObserverHabitError("habit_overlap_analysis_recipe_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            allowed_guard_operators=list(self.allowed_guard_operators),
            evaluate_guard_equivalence=self.evaluate_guard_equivalence,
            evaluate_guard_subsumption=self.evaluate_guard_subsumption,
            maximum_pair_count=self.maximum_pair_count,
            require_evidenced_overlap_for_conflict=(
                self.require_evidenced_overlap_for_conflict
            ),
            require_same_fixture_scope=self.require_same_fixture_scope,
            require_same_grouping_recipe=self.require_same_grouping_recipe,
            require_same_observation_schema=self.require_same_observation_schema,
        )
        if include_id:
            payload["habit_overlap_analysis_recipe_id"] = (
                self.habit_overlap_analysis_recipe_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitOverlapAnalysisRecipeDTO":
        defaults = {
            "require_same_observation_schema": True,
            "require_same_grouping_recipe": True,
            "require_same_fixture_scope": True,
            "evaluate_guard_equivalence": True,
            "evaluate_guard_subsumption": True,
            "require_evidenced_overlap_for_conflict": False,
            "maximum_pair_count": 100,
            "allowed_guard_operators": ("equals", "is_present", "not_equals"),
        }
        defaults.update(values)
        defaults["allowed_guard_operators"] = _sorted_unique(
            cast(Sequence[str], defaults["allowed_guard_operators"])
        )
        payload = _payload(
            OBSERVER_HABIT_OVERLAP_ANALYSIS_RECIPE_VERSION,
            allowed_guard_operators=list(
                cast(tuple[str, ...], defaults["allowed_guard_operators"])
            ),
            evaluate_guard_equivalence=defaults["evaluate_guard_equivalence"],
            evaluate_guard_subsumption=defaults["evaluate_guard_subsumption"],
            maximum_pair_count=defaults["maximum_pair_count"],
            require_evidenced_overlap_for_conflict=defaults[
                "require_evidenced_overlap_for_conflict"
            ],
            require_same_fixture_scope=defaults["require_same_fixture_scope"],
            require_same_grouping_recipe=defaults["require_same_grouping_recipe"],
            require_same_observation_schema=defaults["require_same_observation_schema"],
        )
        return cls(
            habit_overlap_analysis_recipe_id=canonical_id(payload),
            version=OBSERVER_HABIT_OVERLAP_ANALYSIS_RECIPE_VERSION,
            **defaults,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitOverlapOccurrenceDTO:
    habit_overlap_occurrence_id: str
    left_habit_specification_id: str
    right_habit_specification_id: str
    observation_artifact_id: str
    ledger_entry_id: str | None
    episode_id: str | None
    left_evaluation_id: str
    right_evaluation_id: str
    left_recommended_action: str
    right_recommended_action: str
    left_expected_target_state_class_id: str
    right_expected_target_state_class_id: str
    occurrence_classification: str
    version: str = OBSERVER_HABIT_OVERLAP_OCCURRENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_OVERLAP_OCCURRENCE_VERSION:
            raise ObserverHabitError("unsupported overlap occurrence version")
        if self.occurrence_classification not in OCCURRENCE_CLASSIFICATIONS:
            raise ObserverHabitError("unsupported overlap occurrence classification")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_overlap_occurrence_id != expected:
            raise ObserverHabitError("habit_overlap_occurrence_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            episode_id=self.episode_id,
            ledger_entry_id=self.ledger_entry_id,
            left_evaluation_id=self.left_evaluation_id,
            left_expected_target_state_class_id=(
                self.left_expected_target_state_class_id
            ),
            left_habit_specification_id=self.left_habit_specification_id,
            left_recommended_action=self.left_recommended_action,
            observation_artifact_id=self.observation_artifact_id,
            occurrence_classification=self.occurrence_classification,
            right_evaluation_id=self.right_evaluation_id,
            right_expected_target_state_class_id=(
                self.right_expected_target_state_class_id
            ),
            right_habit_specification_id=self.right_habit_specification_id,
            right_recommended_action=self.right_recommended_action,
        )
        if include_id:
            payload["habit_overlap_occurrence_id"] = self.habit_overlap_occurrence_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitOverlapOccurrenceDTO":
        payload = _payload(OBSERVER_HABIT_OVERLAP_OCCURRENCE_VERSION, **values)
        return cls(
            habit_overlap_occurrence_id=canonical_id(payload),
            version=OBSERVER_HABIT_OVERLAP_OCCURRENCE_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitPairOverlapDTO:
    habit_pair_overlap_id: str
    left_habit_specification_id: str
    right_habit_specification_id: str
    source_state_relation: str
    guard_relation: str
    action_relation: str
    target_relation: str
    theoretical_overlap: bool
    evidenced_overlap: bool
    overlap_observation_ids: tuple[str, ...]
    overlap_ledger_entry_ids: tuple[str, ...]
    conflict_classifications: tuple[str, ...]
    analysis_status: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_PAIR_OVERLAP_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_PAIR_OVERLAP_VERSION:
            raise ObserverHabitError("unsupported pair overlap version")
        if self.left_habit_specification_id >= self.right_habit_specification_id:
            raise ObserverHabitError("habit pair IDs must be canonical")
        if self.source_state_relation not in SOURCE_RELATIONS:
            raise ObserverHabitError("unsupported source relation")
        if self.guard_relation not in GUARD_RELATIONS:
            raise ObserverHabitError("unsupported guard relation")
        if self.action_relation not in ACTION_RELATIONS:
            raise ObserverHabitError("unsupported action relation")
        if self.target_relation not in TARGET_RELATIONS:
            raise ObserverHabitError("unsupported target relation")
        if self.analysis_status not in PAIR_STATUSES:
            raise ObserverHabitError("unsupported pair status")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_pair_overlap_id != expected:
            raise ObserverHabitError("habit_pair_overlap_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            action_relation=self.action_relation,
            analysis_status=self.analysis_status,
            conflict_classifications=list(self.conflict_classifications),
            evidenced_overlap=self.evidenced_overlap,
            guard_relation=self.guard_relation,
            left_habit_specification_id=self.left_habit_specification_id,
            overlap_ledger_entry_ids=list(self.overlap_ledger_entry_ids),
            overlap_observation_ids=list(self.overlap_observation_ids),
            reason_codes=list(self.reason_codes),
            right_habit_specification_id=self.right_habit_specification_id,
            source_state_relation=self.source_state_relation,
            target_relation=self.target_relation,
            theoretical_overlap=self.theoretical_overlap,
        )
        if include_id:
            payload["habit_pair_overlap_id"] = self.habit_pair_overlap_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitPairOverlapDTO":
        for key in (
            "overlap_observation_ids",
            "overlap_ledger_entry_ids",
            "conflict_classifications",
            "reason_codes",
        ):
            values[key] = _sorted_unique(cast(Sequence[str], values.get(key, ())))
        payload = _payload(
            OBSERVER_HABIT_PAIR_OVERLAP_VERSION,
            action_relation=values["action_relation"],
            analysis_status=values["analysis_status"],
            conflict_classifications=list(
                cast(tuple[str, ...], values["conflict_classifications"])
            ),
            evidenced_overlap=values["evidenced_overlap"],
            guard_relation=values["guard_relation"],
            left_habit_specification_id=values["left_habit_specification_id"],
            overlap_ledger_entry_ids=list(
                cast(tuple[str, ...], values["overlap_ledger_entry_ids"])
            ),
            overlap_observation_ids=list(
                cast(tuple[str, ...], values["overlap_observation_ids"])
            ),
            reason_codes=list(cast(tuple[str, ...], values["reason_codes"])),
            right_habit_specification_id=values["right_habit_specification_id"],
            source_state_relation=values["source_state_relation"],
            target_relation=values["target_relation"],
            theoretical_overlap=values["theoretical_overlap"],
        )
        return cls(
            habit_pair_overlap_id=canonical_id(payload),
            version=OBSERVER_HABIT_PAIR_OVERLAP_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitOverlapAnalysisDTO:
    habit_overlap_analysis_id: str
    habit_overlap_analysis_recipe_id: str
    habit_specification_ids: tuple[str, ...]
    admission_decision_ids: tuple[str, ...]
    activation_scope_id: str
    observation_schema_id: str
    grouping_recipe_id: str
    ledger_snapshot_id: str
    observation_graph_id: str
    pair_overlaps: tuple[ObserverHabitPairOverlapDTO, ...]
    overlap_occurrences: tuple[ObserverHabitOverlapOccurrenceDTO, ...]
    pair_count: int
    evidenced_overlap_pair_count: int
    different_action_conflict_count: int
    target_conflict_count: int
    inconclusive_pair_count: int
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_OVERLAP_ANALYSIS_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_OVERLAP_ANALYSIS_VERSION:
            raise ObserverHabitError("unsupported overlap analysis version")
        if self.status not in ANALYSIS_STATUSES:
            raise ObserverHabitError("unsupported overlap analysis status")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_overlap_analysis_id != expected:
            raise ObserverHabitError("habit_overlap_analysis_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            activation_scope_id=self.activation_scope_id,
            admission_decision_ids=list(self.admission_decision_ids),
            different_action_conflict_count=self.different_action_conflict_count,
            evidenced_overlap_pair_count=self.evidenced_overlap_pair_count,
            failure_codes=list(self.failure_codes),
            grouping_recipe_id=self.grouping_recipe_id,
            habit_overlap_analysis_recipe_id=self.habit_overlap_analysis_recipe_id,
            habit_specification_ids=list(self.habit_specification_ids),
            inconclusive_pair_count=self.inconclusive_pair_count,
            ledger_snapshot_id=self.ledger_snapshot_id,
            observation_graph_id=self.observation_graph_id,
            observation_schema_id=self.observation_schema_id,
            overlap_occurrences=[
                item.canonical_payload() for item in self.overlap_occurrences
            ],
            pair_count=self.pair_count,
            pair_overlaps=[item.canonical_payload() for item in self.pair_overlaps],
            status=self.status,
            target_conflict_count=self.target_conflict_count,
        )
        if include_id:
            payload["habit_overlap_analysis_id"] = self.habit_overlap_analysis_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitOverlapAnalysisDTO":
        pair_overlaps = cast(
            tuple[ObserverHabitPairOverlapDTO, ...], values["pair_overlaps"]
        )
        occurrences = cast(
            tuple[ObserverHabitOverlapOccurrenceDTO, ...], values["overlap_occurrences"]
        )
        values["habit_specification_ids"] = _sorted_unique(
            cast(Sequence[str], values["habit_specification_ids"])
        )
        values["admission_decision_ids"] = _sorted_unique(
            cast(Sequence[str], values["admission_decision_ids"])
        )
        values["failure_codes"] = _sorted_unique(
            cast(Sequence[str], values.get("failure_codes", ()))
        )
        values["pair_count"] = len(pair_overlaps)
        values["evidenced_overlap_pair_count"] = sum(
            1 for item in pair_overlaps if item.evidenced_overlap
        )
        values["different_action_conflict_count"] = sum(
            1
            for item in pair_overlaps
            if "different_action_conflict" in item.conflict_classifications
        )
        values["target_conflict_count"] = sum(
            1
            for item in pair_overlaps
            if "same_action_target_conflict" in item.conflict_classifications
        )
        values["inconclusive_pair_count"] = sum(
            1 for item in pair_overlaps if item.analysis_status == "inconclusive"
        )
        payload = _payload(
            OBSERVER_HABIT_OVERLAP_ANALYSIS_VERSION,
            activation_scope_id=values["activation_scope_id"],
            admission_decision_ids=list(
                cast(tuple[str, ...], values["admission_decision_ids"])
            ),
            different_action_conflict_count=values["different_action_conflict_count"],
            evidenced_overlap_pair_count=values["evidenced_overlap_pair_count"],
            failure_codes=list(cast(tuple[str, ...], values["failure_codes"])),
            grouping_recipe_id=values["grouping_recipe_id"],
            habit_overlap_analysis_recipe_id=values["habit_overlap_analysis_recipe_id"],
            habit_specification_ids=list(
                cast(tuple[str, ...], values["habit_specification_ids"])
            ),
            inconclusive_pair_count=values["inconclusive_pair_count"],
            ledger_snapshot_id=values["ledger_snapshot_id"],
            observation_graph_id=values["observation_graph_id"],
            observation_schema_id=values["observation_schema_id"],
            overlap_occurrences=[item.canonical_payload() for item in occurrences],
            pair_count=values["pair_count"],
            pair_overlaps=[item.canonical_payload() for item in pair_overlaps],
            status=values["status"],
            target_conflict_count=values["target_conflict_count"],
        )
        return cls(
            habit_overlap_analysis_id=canonical_id(payload),
            version=OBSERVER_HABIT_OVERLAP_ANALYSIS_VERSION,
            **values,  # type: ignore[arg-type]
        )


def analyze_observer_habit_overlap(
    *,
    habit_specifications: tuple[ObserverHabitSpecificationDTO, ...],
    admission_decisions: tuple[ObserverHabitAdmissionDecisionDTO, ...],
    activation_scope: ObserverHabitActivationScopeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    observation_graph: ObserverObservationGraphDTO,
    analysis_recipe: ObserverHabitOverlapAnalysisRecipeDTO,
) -> ObserverHabitOverlapAnalysisDTO:
    failures = _validate_inputs(
        habit_specifications=habit_specifications,
        admission_decisions=admission_decisions,
        activation_scope=activation_scope,
        observation_schema=observation_schema,
        grouping_recipe=grouping_recipe,
        ledger_snapshot=ledger_snapshot,
        observation_graph=observation_graph,
        analysis_recipe=analysis_recipe,
    )
    habits = tuple(
        sorted(habit_specifications, key=lambda item: item.habit_specification_id)
    )
    pair_specs = tuple(combinations(habits, 2))
    pair_overlaps: list[ObserverHabitPairOverlapDTO] = []
    occurrences: list[ObserverHabitOverlapOccurrenceDTO] = []
    if not failures:
        for left, right in pair_specs:
            pair_occurrences = _overlap_occurrences_for_pair(
                left=left,
                right=right,
                ledger_snapshot=ledger_snapshot,
                observation_schema=observation_schema,
                grouping_recipe=grouping_recipe,
            )
            occurrences.extend(pair_occurrences)
            pair_overlaps.append(
                _pair_overlap(
                    left=left,
                    right=right,
                    occurrences=tuple(pair_occurrences),
                    recipe=analysis_recipe,
                )
            )
    status = _analysis_status(tuple(pair_overlaps), tuple(failures))
    return ObserverHabitOverlapAnalysisDTO.create(
        habit_overlap_analysis_recipe_id=analysis_recipe.habit_overlap_analysis_recipe_id,
        habit_specification_ids=tuple(item.habit_specification_id for item in habits),
        admission_decision_ids=tuple(
            item.habit_admission_decision_id for item in admission_decisions
        ),
        activation_scope_id=activation_scope.habit_activation_scope_id,
        observation_schema_id=observation_schema.schema_id,
        grouping_recipe_id=grouping_recipe.grouping_recipe_id,
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        observation_graph_id=observation_graph.observation_graph_id,
        pair_overlaps=tuple(pair_overlaps),
        overlap_occurrences=tuple(
            sorted(occurrences, key=lambda item: item.habit_overlap_occurrence_id)
        ),
        status=status,
        failure_codes=tuple(failures),
    )


def _validate_inputs(
    *,
    habit_specifications: tuple[ObserverHabitSpecificationDTO, ...],
    admission_decisions: tuple[ObserverHabitAdmissionDecisionDTO, ...],
    activation_scope: ObserverHabitActivationScopeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    observation_graph: ObserverObservationGraphDTO,
    analysis_recipe: ObserverHabitOverlapAnalysisRecipeDTO,
) -> tuple[str, ...]:
    failures: set[str] = set()
    if len(habit_specifications) < 2:
        failures.add("fewer_than_two_habits")
    habit_ids = tuple(item.habit_specification_id for item in habit_specifications)
    if len(set(habit_ids)) != len(habit_ids):
        failures.add("duplicate_habit_ids")
    if (
        len(tuple(combinations(habit_specifications, 2)))
        > analysis_recipe.maximum_pair_count
    ):
        failures.add("pair_count_exceeds_limit")
    decisions = {item.habit_specification_id: item for item in admission_decisions}
    if set(decisions) != set(habit_ids):
        failures.add("admission_decision_mismatch")
    for habit in habit_specifications:
        decision = decisions.get(habit.habit_specification_id)
        if decision is None or decision.decision != "admit":
            failures.add("habit_not_admitted")
        if habit.status != "shadow_candidate":
            failures.add("habit_not_shadow_candidate")
        if (
            analysis_recipe.require_same_observation_schema
            and habit.observation_schema_id != observation_schema.schema_id
        ):
            failures.add("schema_lineage_mismatch")
        if (
            analysis_recipe.require_same_grouping_recipe
            and habit.grouping_recipe_id != grouping_recipe.grouping_recipe_id
        ):
            failures.add("grouping_lineage_mismatch")
        if (
            analysis_recipe.require_same_fixture_scope
            and ledger_snapshot.fixture_id != activation_scope.fixture_id
        ):
            failures.add("fixture_scope_mismatch")
        if habit.recommended_action not in activation_scope.allowed_action_names:
            failures.add("action_outside_activation_scope")
        if habit.ledger_snapshot_id != ledger_snapshot.ledger_snapshot_id:
            failures.add("ledger_lineage_mismatch")
        if habit.observation_graph_id != observation_graph.observation_graph_id:
            failures.add("graph_lineage_mismatch")
        if any(
            guard.operator not in analysis_recipe.allowed_guard_operators
            for guard in habit.positive_guards + habit.counterexample_guards
        ):
            failures.add("unsupported_guard_operator")
    return tuple(sorted(failures))


def _pair_overlap(
    *,
    left: ObserverHabitSpecificationDTO,
    right: ObserverHabitSpecificationDTO,
    occurrences: tuple[ObserverHabitOverlapOccurrenceDTO, ...],
    recipe: ObserverHabitOverlapAnalysisRecipeDTO,
) -> ObserverHabitPairOverlapDTO:
    source_relation = (
        "same_source_state_class"
        if left.source_state_class_id == right.source_state_class_id
        else "different_source_state_class"
    )
    guard_relation = _guard_relation(left, right, recipe)
    action_relation = (
        "same_action"
        if left.recommended_action == right.recommended_action
        else "different_action"
    )
    target_relation = (
        "same_expected_target"
        if left.expected_target_state_class_id == right.expected_target_state_class_id
        else "different_expected_target"
    )
    theoretical = (
        source_relation == "same_source_state_class"
        and guard_relation not in {"disjoint", "unknown"}
    )
    evidenced = any(
        item.occurrence_classification != "invalid_evaluation" for item in occurrences
    )
    classifications: set[str] = set()
    reasons: set[str] = set()
    if not theoretical:
        classifications.add(
            "source_class_disjoint"
            if source_relation == "different_source_state_class"
            else "guard_disjoint"
        )
    elif not evidenced:
        classifications.add("theoretical_overlap_unobserved")
        reasons.add("insufficient_overlap_evidence")
    else:
        classifications.add("evidenced_cofire")
    if (
        guard_relation == "equivalent"
        and action_relation == "same_action"
        and target_relation == "same_expected_target"
    ):
        classifications.add("duplicate_habit")
    if guard_relation == "left_subsumes_right":
        classifications.add("redundant_broader_habit")
    if guard_relation == "right_subsumes_left":
        classifications.add("redundant_narrower_habit")
    conflict_visible = theoretical and (
        evidenced or not recipe.require_evidenced_overlap_for_conflict
    )
    if conflict_visible and action_relation == "different_action":
        classifications.add("different_action_conflict")
    elif conflict_visible and target_relation == "different_expected_target":
        classifications.add("same_action_target_conflict")
        reasons.add("target_prediction_disagreement")
    elif conflict_visible and action_relation == "same_action":
        classifications.add("same_action_target_agreement")
    if guard_relation == "unknown":
        classifications.add("analysis_inconclusive")
        reasons.add("guard_relation_unknown")
    status = (
        "inconclusive" if "analysis_inconclusive" in classifications else "completed"
    )
    return ObserverHabitPairOverlapDTO.create(
        left_habit_specification_id=left.habit_specification_id,
        right_habit_specification_id=right.habit_specification_id,
        source_state_relation=source_relation,
        guard_relation=guard_relation,
        action_relation=action_relation,
        target_relation=target_relation,
        theoretical_overlap=theoretical,
        evidenced_overlap=evidenced,
        overlap_observation_ids=tuple(
            item.observation_artifact_id
            for item in occurrences
            if item.occurrence_classification != "invalid_evaluation"
        ),
        overlap_ledger_entry_ids=tuple(
            item.ledger_entry_id
            for item in occurrences
            if item.ledger_entry_id is not None
            and item.occurrence_classification != "invalid_evaluation"
        ),
        conflict_classifications=tuple(classifications or {"none"}),
        analysis_status=status,
        reason_codes=tuple(reasons),
    )


def _overlap_occurrences_for_pair(
    *,
    left: ObserverHabitSpecificationDTO,
    right: ObserverHabitSpecificationDTO,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    observation_schema: ObserverObservationSchemaDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
) -> tuple[ObserverHabitOverlapOccurrenceDTO, ...]:
    occurrences: list[ObserverHabitOverlapOccurrenceDTO] = []
    entries = tuple(getattr(ledger_snapshot, "entries", ()))
    for entry in entries:
        try:
            observation = _observation_for_state(
                state=entry.source_state,
                action_effect="initial"
                if entry.source_state.step_index == 0
                else "source",
                observation_schema=observation_schema,
            )
            left_eval = evaluate_observer_habit(
                habit_specification=left,
                observation=observation,
                grouping_recipe=grouping_recipe,
                observation_schema=observation_schema,
            )
            right_eval = evaluate_observer_habit(
                habit_specification=right,
                observation=observation,
                grouping_recipe=grouping_recipe,
                observation_schema=observation_schema,
            )
            if left_eval.decision == "invalid" or right_eval.decision == "invalid":
                occurrences.append(
                    _occurrence(
                        left,
                        right,
                        observation,
                        entry.ledger_entry_id,
                        entry.episode_id,
                        left_eval,
                        right_eval,
                        "invalid_evaluation",
                    )
                )
            elif left_eval.decision == "fire" and right_eval.decision == "fire":
                occurrences.append(
                    _occurrence(
                        left,
                        right,
                        observation,
                        entry.ledger_entry_id,
                        entry.episode_id,
                        left_eval,
                        right_eval,
                        _occurrence_class(left, right),
                    )
                )
        except Exception:
            observation = ObserverObservationArtifactDTO.create(
                observation_schema=observation_schema,
                visible_state_features={
                    "action_effect": "invalid",
                    "agent_x": 0,
                    "target_x": 0,
                },
                recent_history_features={},
                hidden_state_uncertainty={},
                provenance={"reconstruction": "failed"},
                sequence_index=0,
            )
            invalid = _invalid_eval(left, observation)
            occurrences.append(
                _occurrence(
                    left,
                    right,
                    observation,
                    getattr(entry, "ledger_entry_id", None),
                    getattr(entry, "episode_id", None),
                    invalid,
                    _invalid_eval(right, observation),
                    "invalid_evaluation",
                )
            )
    return tuple(occurrences)


def _invalid_eval(
    habit: ObserverHabitSpecificationDTO, observation: ObserverObservationArtifactDTO
) -> ObserverHabitEvaluationDTO:
    return ObserverHabitEvaluationDTO.create(
        habit_specification_id=habit.habit_specification_id,
        observation_artifact_id=observation.observation_artifact_id,
        state_class_id=None,
        guard_evaluations=(),
        decision="invalid",
        recommended_action=None,
        reason_codes=("observation_reconstruction_failed",),
    )


def _occurrence(
    left: ObserverHabitSpecificationDTO,
    right: ObserverHabitSpecificationDTO,
    observation: ObserverObservationArtifactDTO,
    ledger_entry_id: str | None,
    episode_id: str | None,
    left_eval: ObserverHabitEvaluationDTO,
    right_eval: ObserverHabitEvaluationDTO,
    classification: str,
) -> ObserverHabitOverlapOccurrenceDTO:
    return ObserverHabitOverlapOccurrenceDTO.create(
        left_habit_specification_id=left.habit_specification_id,
        right_habit_specification_id=right.habit_specification_id,
        observation_artifact_id=observation.observation_artifact_id,
        ledger_entry_id=ledger_entry_id,
        episode_id=episode_id,
        left_evaluation_id=left_eval.habit_evaluation_id,
        right_evaluation_id=right_eval.habit_evaluation_id,
        left_recommended_action=left.recommended_action,
        right_recommended_action=right.recommended_action,
        left_expected_target_state_class_id=left.expected_target_state_class_id,
        right_expected_target_state_class_id=right.expected_target_state_class_id,
        occurrence_classification=classification,
    )


def _occurrence_class(
    left: ObserverHabitSpecificationDTO, right: ObserverHabitSpecificationDTO
) -> str:
    if left.recommended_action != right.recommended_action:
        return "different_action"
    if left.expected_target_state_class_id != right.expected_target_state_class_id:
        return "same_action_different_target"
    return "same_action_same_target"


def _guard_relation(
    left: ObserverHabitSpecificationDTO,
    right: ObserverHabitSpecificationDTO,
    recipe: ObserverHabitOverlapAnalysisRecipeDTO,
) -> str:
    left_guards = _equality_guard_map(left)
    right_guards = _equality_guard_map(right)
    if left_guards is None or right_guards is None:
        return "unknown"
    for key in set(left_guards) & set(right_guards):
        if left_guards[key] != right_guards[key]:
            return "disjoint"
    if left_guards == right_guards and recipe.evaluate_guard_equivalence:
        return "equivalent"
    if recipe.evaluate_guard_subsumption:
        # left_subsumes_right means every observation accepted by right is accepted by left.
        if set(left_guards.items()).issubset(set(right_guards.items())):
            return "left_subsumes_right"
        if set(right_guards.items()).issubset(set(left_guards.items())):
            return "right_subsumes_left"
    return "partially_overlapping"


def _equality_guard_map(
    habit: ObserverHabitSpecificationDTO,
) -> dict[str, tuple[str, object | None]] | None:
    result: dict[str, tuple[str, object | None]] = {}
    for guard in habit.positive_guards + habit.counterexample_guards:
        value: tuple[str, object | None]
        if guard.operator == "is_present":
            value = ("is_present", True)
        elif guard.operator == "equals":
            value = (guard.expected_type, guard.expected_value)
        else:
            return None
        key = guard.feature_key
        if key in result and result[key] != value:
            return None
        result[key] = value
    return result


def _analysis_status(
    pairs: tuple[ObserverHabitPairOverlapDTO, ...], failures: tuple[str, ...]
) -> str:
    if failures:
        return "failed"
    if any(item.analysis_status == "inconclusive" for item in pairs):
        return "inconclusive"
    if any(
        {"different_action_conflict", "same_action_target_conflict"}
        & set(item.conflict_classifications)
        for item in pairs
    ):
        return "completed_with_conflicts"
    return "completed"
