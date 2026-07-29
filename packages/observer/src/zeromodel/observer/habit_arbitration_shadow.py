"""Shadow replay and audit for Observer habit arbitration plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import ObserverObservationSchemaDTO
from zeromodel.observer.comparison import ObserverComparisonRecipeDTO
from zeromodel.observer.fixture import (
    ObserverFixtureActionDTO,
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
)
from zeromodel.observer.fixture_predictor import _observation_for_state
from zeromodel.observer.fixture_runtime import (
    ObserverFixtureEpisodeResultDTO,
    ObserverFixtureRuleScheduleEntryDTO,
    run_observer_fixture_episode,
)
from zeromodel.observer.grouping import ObserverStateGroupingRecipeDTO
from zeromodel.observer.habit import ObserverHabitError, ObserverHabitSpecificationDTO
from zeromodel.observer.habit_arbitration import (
    ObserverHabitArbitrationEvaluationDTO,
    ObserverHabitArbitrationPlanDTO,
    evaluate_observer_habit_arbitration,
)
from zeromodel.observer.habit_overlap import ObserverHabitOverlapAnalysisDTO
from zeromodel.observer.ledger import (
    ObserverTransitionLedgerEntryDTO,
    ObserverTransitionLedgerSnapshotDTO,
    build_observer_transition_ledger_snapshot,
)

OBSERVER_HABIT_ARBITRATION_SHADOW_OCCURRENCE_VERSION: Final = (
    "observer-habit-arbitration-shadow-occurrence/1"
)
OBSERVER_HABIT_ARBITRATION_SHADOW_REPLAY_VERSION: Final = (
    "observer-habit-arbitration-shadow-replay/1"
)
OBSERVER_HABIT_ARBITRATION_SHADOW_EPISODE_VERSION: Final = (
    "observer-habit-arbitration-shadow-episode/1"
)
OBSERVER_HABIT_ARBITRATION_AUDIT_RECIPE_VERSION: Final = (
    "observer-habit-arbitration-audit-recipe/1"
)
OBSERVER_HABIT_ARBITRATION_AUDIT_VERSION: Final = "observer-habit-arbitration-audit/1"

SHADOW_OUTCOMES: Final = frozenset(
    {
        "correct_selection",
        "wrong_action",
        "wrong_target",
        "fallback",
        "ambiguous_fallback",
        "missed_opportunity",
        "invalid_evaluation",
    }
)
AUDIT_DISPOSITIONS: Final = frozenset(
    {
        "eligible_for_multi_habit_activation_review",
        "insufficient_evidence",
        "action_conflict_detected",
        "target_conflict_detected",
        "wrong_action_detected",
        "wrong_target_detected",
        "excessive_ambiguity",
        "excessive_missed_opportunities",
        "invalid_evidence",
        "inconclusive",
    }
)


def _sorted_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(values)))


def _payload(version: str, **values: object) -> dict[str, object]:
    return {"version": version, **values}


@dataclass(frozen=True)
class ObserverHabitArbitrationShadowOccurrenceDTO:
    habit_arbitration_shadow_occurrence_id: str
    habit_arbitration_plan_id: str
    ledger_entry_id: str
    observation_artifact_id: str
    arbitration_evaluation_id: str
    selected_habit_id: str | None
    selected_action: str
    authoritative_action: str
    actual_target_state_class_id: str
    expected_target_state_class_id: str | None
    outcome: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_ARBITRATION_SHADOW_OCCURRENCE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_SHADOW_OCCURRENCE_VERSION:
            raise ObserverHabitError(
                "unsupported arbitration shadow occurrence version"
            )
        if self.outcome not in SHADOW_OUTCOMES:
            raise ObserverHabitError("unsupported arbitration shadow outcome")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_shadow_occurrence_id != expected:
            raise ObserverHabitError("habit_arbitration_shadow_occurrence_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            actual_target_state_class_id=self.actual_target_state_class_id,
            arbitration_evaluation_id=self.arbitration_evaluation_id,
            authoritative_action=self.authoritative_action,
            expected_target_state_class_id=self.expected_target_state_class_id,
            habit_arbitration_plan_id=self.habit_arbitration_plan_id,
            ledger_entry_id=self.ledger_entry_id,
            observation_artifact_id=self.observation_artifact_id,
            outcome=self.outcome,
            reason_codes=list(self.reason_codes),
            selected_action=self.selected_action,
            selected_habit_id=self.selected_habit_id,
        )
        if include_id:
            payload["habit_arbitration_shadow_occurrence_id"] = (
                self.habit_arbitration_shadow_occurrence_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationShadowOccurrenceDTO":
        values["reason_codes"] = _sorted_unique(
            cast(Sequence[str], values.get("reason_codes", ()))
        )
        payload = _payload(
            OBSERVER_HABIT_ARBITRATION_SHADOW_OCCURRENCE_VERSION,
            actual_target_state_class_id=values["actual_target_state_class_id"],
            arbitration_evaluation_id=values["arbitration_evaluation_id"],
            authoritative_action=values["authoritative_action"],
            expected_target_state_class_id=values["expected_target_state_class_id"],
            habit_arbitration_plan_id=values["habit_arbitration_plan_id"],
            ledger_entry_id=values["ledger_entry_id"],
            observation_artifact_id=values["observation_artifact_id"],
            outcome=values["outcome"],
            reason_codes=list(cast(tuple[str, ...], values["reason_codes"])),
            selected_action=values["selected_action"],
            selected_habit_id=values["selected_habit_id"],
        )
        return cls(
            habit_arbitration_shadow_occurrence_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_SHADOW_OCCURRENCE_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitArbitrationShadowReplayDTO:
    habit_arbitration_shadow_replay_id: str
    habit_arbitration_plan_id: str
    ledger_snapshot_id: str
    shadow_occurrences: tuple[ObserverHabitArbitrationShadowOccurrenceDTO, ...]
    evaluated_entry_ids: tuple[str, ...]
    applicable_count: int
    habit_selection_count: int
    fallback_count: int
    correct_selection_count: int
    wrong_action_count: int
    wrong_target_count: int
    ambiguous_fallback_count: int
    missed_opportunity_count: int
    invalid_count: int
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_ARBITRATION_SHADOW_REPLAY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_SHADOW_REPLAY_VERSION:
            raise ObserverHabitError("unsupported arbitration shadow replay version")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_shadow_replay_id != expected:
            raise ObserverHabitError("habit_arbitration_shadow_replay_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            ambiguous_fallback_count=self.ambiguous_fallback_count,
            applicable_count=self.applicable_count,
            correct_selection_count=self.correct_selection_count,
            evaluated_entry_ids=list(self.evaluated_entry_ids),
            failure_codes=list(self.failure_codes),
            fallback_count=self.fallback_count,
            habit_arbitration_plan_id=self.habit_arbitration_plan_id,
            habit_selection_count=self.habit_selection_count,
            invalid_count=self.invalid_count,
            ledger_snapshot_id=self.ledger_snapshot_id,
            missed_opportunity_count=self.missed_opportunity_count,
            shadow_occurrences=[
                item.canonical_payload() for item in self.shadow_occurrences
            ],
            status=self.status,
            wrong_action_count=self.wrong_action_count,
            wrong_target_count=self.wrong_target_count,
        )
        if include_id:
            payload["habit_arbitration_shadow_replay_id"] = (
                self.habit_arbitration_shadow_replay_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationShadowReplayDTO":
        occurrences = cast(
            tuple[ObserverHabitArbitrationShadowOccurrenceDTO, ...],
            values["shadow_occurrences"],
        )
        values["evaluated_entry_ids"] = _sorted_unique(
            cast(Sequence[str], values.get("evaluated_entry_ids", ()))
        )
        values["failure_codes"] = _sorted_unique(
            cast(Sequence[str], values.get("failure_codes", ()))
        )
        values["applicable_count"] = len(occurrences)
        values["habit_selection_count"] = sum(
            1 for item in occurrences if item.selected_habit_id is not None
        )
        values["fallback_count"] = sum(
            1 for item in occurrences if item.selected_habit_id is None
        )
        values["correct_selection_count"] = sum(
            1 for item in occurrences if item.outcome == "correct_selection"
        )
        values["wrong_action_count"] = sum(
            1 for item in occurrences if item.outcome == "wrong_action"
        )
        values["wrong_target_count"] = sum(
            1 for item in occurrences if item.outcome == "wrong_target"
        )
        values["ambiguous_fallback_count"] = sum(
            1 for item in occurrences if item.outcome == "ambiguous_fallback"
        )
        values["missed_opportunity_count"] = sum(
            1 for item in occurrences if item.outcome == "missed_opportunity"
        )
        values["invalid_count"] = sum(
            1 for item in occurrences if item.outcome == "invalid_evaluation"
        )
        payload = _payload(
            OBSERVER_HABIT_ARBITRATION_SHADOW_REPLAY_VERSION,
            ambiguous_fallback_count=values["ambiguous_fallback_count"],
            applicable_count=values["applicable_count"],
            correct_selection_count=values["correct_selection_count"],
            evaluated_entry_ids=list(
                cast(tuple[str, ...], values["evaluated_entry_ids"])
            ),
            failure_codes=list(cast(tuple[str, ...], values["failure_codes"])),
            fallback_count=values["fallback_count"],
            habit_arbitration_plan_id=values["habit_arbitration_plan_id"],
            habit_selection_count=values["habit_selection_count"],
            invalid_count=values["invalid_count"],
            ledger_snapshot_id=values["ledger_snapshot_id"],
            missed_opportunity_count=values["missed_opportunity_count"],
            shadow_occurrences=[item.canonical_payload() for item in occurrences],
            status=values["status"],
            wrong_action_count=values["wrong_action_count"],
            wrong_target_count=values["wrong_target_count"],
        )
        return cls(
            habit_arbitration_shadow_replay_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_SHADOW_REPLAY_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitArbitrationShadowEpisodeDTO:
    habit_arbitration_shadow_episode_id: str
    habit_arbitration_plan_id: str
    fixture_episode_result_id: str
    ledger_snapshot_id: str
    shadow_replay: ObserverHabitArbitrationShadowReplayDTO
    authoritative_action_ids: tuple[str, ...]
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_ARBITRATION_SHADOW_EPISODE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_SHADOW_EPISODE_VERSION:
            raise ObserverHabitError("unsupported arbitration shadow episode version")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_shadow_episode_id != expected:
            raise ObserverHabitError("habit_arbitration_shadow_episode_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            authoritative_action_ids=list(self.authoritative_action_ids),
            failure_codes=list(self.failure_codes),
            fixture_episode_result_id=self.fixture_episode_result_id,
            habit_arbitration_plan_id=self.habit_arbitration_plan_id,
            ledger_snapshot_id=self.ledger_snapshot_id,
            shadow_replay=self.shadow_replay.canonical_payload(),
            status=self.status,
        )
        if include_id:
            payload["habit_arbitration_shadow_episode_id"] = (
                self.habit_arbitration_shadow_episode_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationShadowEpisodeDTO":
        values["authoritative_action_ids"] = tuple(
            cast(Sequence[str], values.get("authoritative_action_ids", ()))
        )
        values["failure_codes"] = _sorted_unique(
            cast(Sequence[str], values.get("failure_codes", ()))
        )
        payload = _payload(
            OBSERVER_HABIT_ARBITRATION_SHADOW_EPISODE_VERSION,
            authoritative_action_ids=list(
                cast(tuple[str, ...], values["authoritative_action_ids"])
            ),
            failure_codes=list(cast(tuple[str, ...], values["failure_codes"])),
            fixture_episode_result_id=values["fixture_episode_result_id"],
            habit_arbitration_plan_id=values["habit_arbitration_plan_id"],
            ledger_snapshot_id=values["ledger_snapshot_id"],
            shadow_replay=cast(
                ObserverHabitArbitrationShadowReplayDTO, values["shadow_replay"]
            ).canonical_payload(),
            status=values["status"],
        )
        return cls(
            habit_arbitration_shadow_episode_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_SHADOW_EPISODE_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitArbitrationAuditRecipeDTO:
    habit_arbitration_audit_recipe_id: str
    minimum_evaluated_replay_count: int
    minimum_episode_count: int
    minimum_applicable_count: int
    minimum_habit_selection_count: int
    maximum_wrong_action_count: int
    maximum_wrong_target_count: int
    maximum_invalid_count: int
    maximum_ambiguous_fallback_count: int
    maximum_missed_opportunity_count: int
    require_zero_different_action_conflicts: bool
    require_zero_target_conflicts: bool
    require_complete_pair_analysis: bool
    version: str = OBSERVER_HABIT_ARBITRATION_AUDIT_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_AUDIT_RECIPE_VERSION:
            raise ObserverHabitError("unsupported arbitration audit recipe version")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_audit_recipe_id != expected:
            raise ObserverHabitError("habit_arbitration_audit_recipe_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            maximum_ambiguous_fallback_count=self.maximum_ambiguous_fallback_count,
            maximum_invalid_count=self.maximum_invalid_count,
            maximum_missed_opportunity_count=self.maximum_missed_opportunity_count,
            maximum_wrong_action_count=self.maximum_wrong_action_count,
            maximum_wrong_target_count=self.maximum_wrong_target_count,
            minimum_applicable_count=self.minimum_applicable_count,
            minimum_episode_count=self.minimum_episode_count,
            minimum_evaluated_replay_count=self.minimum_evaluated_replay_count,
            minimum_habit_selection_count=self.minimum_habit_selection_count,
            require_complete_pair_analysis=self.require_complete_pair_analysis,
            require_zero_different_action_conflicts=(
                self.require_zero_different_action_conflicts
            ),
            require_zero_target_conflicts=self.require_zero_target_conflicts,
        )
        if include_id:
            payload["habit_arbitration_audit_recipe_id"] = (
                self.habit_arbitration_audit_recipe_id
            )
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationAuditRecipeDTO":
        defaults: dict[str, object] = {
            "minimum_evaluated_replay_count": 1,
            "minimum_episode_count": 0,
            "minimum_applicable_count": 1,
            "minimum_habit_selection_count": 1,
            "maximum_wrong_action_count": 0,
            "maximum_wrong_target_count": 0,
            "maximum_invalid_count": 0,
            "maximum_ambiguous_fallback_count": 0,
            "maximum_missed_opportunity_count": 0,
            "require_zero_different_action_conflicts": True,
            "require_zero_target_conflicts": True,
            "require_complete_pair_analysis": True,
        }
        defaults.update(values)
        payload = _payload(OBSERVER_HABIT_ARBITRATION_AUDIT_RECIPE_VERSION, **defaults)
        return cls(
            habit_arbitration_audit_recipe_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_AUDIT_RECIPE_VERSION,
            **defaults,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitArbitrationAuditDTO:
    habit_arbitration_audit_id: str
    habit_arbitration_plan_id: str
    habit_overlap_analysis_id: str
    habit_arbitration_audit_recipe_id: str
    evaluated_shadow_replay_ids: tuple[str, ...]
    evaluated_shadow_episode_ids: tuple[str, ...]
    applicable_count: int
    habit_selection_count: int
    fallback_count: int
    correct_selection_count: int
    wrong_action_count: int
    wrong_target_count: int
    ambiguous_fallback_count: int
    missed_opportunity_count: int
    invalid_count: int
    episode_count: int
    eligible_for_multi_habit_activation_review: bool
    disposition: str
    reason_codes: tuple[str, ...]
    version: str = OBSERVER_HABIT_ARBITRATION_AUDIT_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ARBITRATION_AUDIT_VERSION:
            raise ObserverHabitError("unsupported arbitration audit version")
        if self.disposition not in AUDIT_DISPOSITIONS:
            raise ObserverHabitError("unsupported arbitration audit disposition")
        expected = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_arbitration_audit_id != expected:
            raise ObserverHabitError("habit_arbitration_audit_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = _payload(
            self.version,
            ambiguous_fallback_count=self.ambiguous_fallback_count,
            applicable_count=self.applicable_count,
            correct_selection_count=self.correct_selection_count,
            disposition=self.disposition,
            eligible_for_multi_habit_activation_review=(
                self.eligible_for_multi_habit_activation_review
            ),
            episode_count=self.episode_count,
            evaluated_shadow_episode_ids=list(self.evaluated_shadow_episode_ids),
            evaluated_shadow_replay_ids=list(self.evaluated_shadow_replay_ids),
            fallback_count=self.fallback_count,
            habit_arbitration_audit_recipe_id=self.habit_arbitration_audit_recipe_id,
            habit_arbitration_plan_id=self.habit_arbitration_plan_id,
            habit_overlap_analysis_id=self.habit_overlap_analysis_id,
            habit_selection_count=self.habit_selection_count,
            invalid_count=self.invalid_count,
            missed_opportunity_count=self.missed_opportunity_count,
            reason_codes=list(self.reason_codes),
            wrong_action_count=self.wrong_action_count,
            wrong_target_count=self.wrong_target_count,
        )
        if include_id:
            payload["habit_arbitration_audit_id"] = self.habit_arbitration_audit_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitArbitrationAuditDTO":
        for key in (
            "evaluated_shadow_replay_ids",
            "evaluated_shadow_episode_ids",
            "reason_codes",
        ):
            values[key] = _sorted_unique(cast(Sequence[str], values.get(key, ())))
        payload = _payload(
            OBSERVER_HABIT_ARBITRATION_AUDIT_VERSION,
            ambiguous_fallback_count=values["ambiguous_fallback_count"],
            applicable_count=values["applicable_count"],
            correct_selection_count=values["correct_selection_count"],
            disposition=values["disposition"],
            eligible_for_multi_habit_activation_review=values[
                "eligible_for_multi_habit_activation_review"
            ],
            episode_count=values["episode_count"],
            evaluated_shadow_episode_ids=list(
                cast(tuple[str, ...], values["evaluated_shadow_episode_ids"])
            ),
            evaluated_shadow_replay_ids=list(
                cast(tuple[str, ...], values["evaluated_shadow_replay_ids"])
            ),
            fallback_count=values["fallback_count"],
            habit_arbitration_audit_recipe_id=values[
                "habit_arbitration_audit_recipe_id"
            ],
            habit_arbitration_plan_id=values["habit_arbitration_plan_id"],
            habit_overlap_analysis_id=values["habit_overlap_analysis_id"],
            habit_selection_count=values["habit_selection_count"],
            invalid_count=values["invalid_count"],
            missed_opportunity_count=values["missed_opportunity_count"],
            reason_codes=list(cast(tuple[str, ...], values["reason_codes"])),
            wrong_action_count=values["wrong_action_count"],
            wrong_target_count=values["wrong_target_count"],
        )
        return cls(
            habit_arbitration_audit_id=canonical_id(payload),
            version=OBSERVER_HABIT_ARBITRATION_AUDIT_VERSION,
            **values,  # type: ignore[arg-type]
        )


def evaluate_observer_habit_arbitration_over_ledger(
    *,
    arbitration_plan: ObserverHabitArbitrationPlanDTO,
    habit_specifications: tuple[ObserverHabitSpecificationDTO, ...],
    ledger_entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverHabitArbitrationShadowReplayDTO:
    occurrences: list[ObserverHabitArbitrationShadowOccurrenceDTO] = []
    for entry in ledger_entries:
        observation = _observation_for_state(
            state=entry.source_state,
            action_effect="initial" if entry.source_state.step_index == 0 else "source",
            observation_schema=observation_schema,
        )
        action = _action_name(entry.action_id)
        evaluation = evaluate_observer_habit_arbitration(
            arbitration_plan=arbitration_plan,
            habit_specifications=habit_specifications,
            observation=observation,
            grouping_recipe=grouping_recipe,
            observation_schema=observation_schema,
            authoritative_fallback_action=action,
        )
        occurrences.append(
            _shadow_occurrence(
                arbitration_plan,
                entry,
                observation.observation_artifact_id,
                evaluation,
                habit_specifications,
                action,
            )
        )
    return ObserverHabitArbitrationShadowReplayDTO.create(
        habit_arbitration_plan_id=arbitration_plan.habit_arbitration_plan_id,
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        shadow_occurrences=tuple(occurrences),
        evaluated_entry_ids=tuple(item.ledger_entry_id for item in ledger_entries),
        status="completed",
        failure_codes=(),
    )


def run_observer_fixture_arbitration_shadow_episode(
    *,
    arbitration_plan: ObserverHabitArbitrationPlanDTO,
    habit_specifications: tuple[ObserverHabitSpecificationDTO, ...],
    initial_state: ObserverFixtureStateDTO,
    authoritative_actions: tuple[ObserverFixtureActionDTO, ...],
    predictor_rule_set: ObserverFixtureRuleSetDTO,
    environment_rule_schedule: tuple[ObserverFixtureRuleScheduleEntryDTO, ...],
    environment_rule_sets: tuple[ObserverFixtureRuleSetDTO, ...],
    observation_schema: ObserverObservationSchemaDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    comparison_recipe: ObserverComparisonRecipeDTO,
) -> tuple[
    ObserverFixtureEpisodeResultDTO,
    tuple[ObserverTransitionLedgerEntryDTO, ...],
    ObserverHabitArbitrationShadowEpisodeDTO,
]:
    episode, entries = run_observer_fixture_episode(
        initial_state=initial_state,
        actions=authoritative_actions,
        predictor_rule_set=predictor_rule_set,
        environment_rule_schedule=environment_rule_schedule,
        environment_rule_sets=environment_rule_sets,
        observation_schema=observation_schema,
        comparison_recipe=comparison_recipe,
    )
    replay = evaluate_observer_habit_arbitration_over_ledger(
        arbitration_plan=arbitration_plan,
        habit_specifications=habit_specifications,
        ledger_entries=entries,
        ledger_snapshot=episode.ledger_snapshot,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
    )
    shadow_episode = ObserverHabitArbitrationShadowEpisodeDTO.create(
        habit_arbitration_plan_id=arbitration_plan.habit_arbitration_plan_id,
        fixture_episode_result_id=episode.episode_result_id,
        ledger_snapshot_id=episode.ledger_snapshot.ledger_snapshot_id,
        shadow_replay=replay,
        authoritative_action_ids=tuple(entry.action_id for entry in entries),
        status="completed",
        failure_codes=(),
    )
    return episode, entries, shadow_episode


def audit_observer_habit_arbitration_shadow(
    *,
    arbitration_plan: ObserverHabitArbitrationPlanDTO,
    overlap_analysis: ObserverHabitOverlapAnalysisDTO,
    historical_shadow_replay: ObserverHabitArbitrationShadowReplayDTO,
    fixture_shadow_episodes: tuple[ObserverHabitArbitrationShadowEpisodeDTO, ...],
    audit_recipe: ObserverHabitArbitrationAuditRecipeDTO,
) -> ObserverHabitArbitrationAuditDTO:
    failures: set[str] = set()
    replays = [historical_shadow_replay] + [
        episode.shadow_replay for episode in fixture_shadow_episodes
    ]
    replay_ids = [item.habit_arbitration_shadow_replay_id for item in replays]
    if len(set(replay_ids)) != len(replay_ids):
        failures.add("duplicate_replay")
    if any(
        item.habit_arbitration_plan_id != arbitration_plan.habit_arbitration_plan_id
        for item in replays
    ):
        failures.add("foreign_replay")
    if any(
        episode.habit_arbitration_plan_id != arbitration_plan.habit_arbitration_plan_id
        for episode in fixture_shadow_episodes
    ):
        failures.add("foreign_episode")
    if tuple(overlap_analysis.habit_specification_ids) != tuple(
        arbitration_plan.habit_specification_ids
    ):
        failures.add("overlap_plan_habit_mismatch")
    applicable = sum(item.applicable_count for item in replays)
    selections = sum(item.habit_selection_count for item in replays)
    fallback = sum(item.fallback_count for item in replays)
    correct = sum(item.correct_selection_count for item in replays)
    wrong_action = sum(item.wrong_action_count for item in replays)
    wrong_target = sum(item.wrong_target_count for item in replays)
    ambiguity = sum(item.ambiguous_fallback_count for item in replays)
    missed = sum(item.missed_opportunity_count for item in replays)
    invalid = sum(item.invalid_count for item in replays)
    disposition = "eligible_for_multi_habit_activation_review"
    reasons: set[str] = {"eligible_for_multi_habit_activation_review"}
    if failures:
        disposition, reasons = "invalid_evidence", failures
    elif (
        len(replays) < audit_recipe.minimum_evaluated_replay_count
        or len(fixture_shadow_episodes) < audit_recipe.minimum_episode_count
        or applicable < audit_recipe.minimum_applicable_count
        or selections < audit_recipe.minimum_habit_selection_count
    ):
        disposition, reasons = "insufficient_evidence", {"insufficient_evidence"}
    elif (
        audit_recipe.require_complete_pair_analysis
        and overlap_analysis.inconclusive_pair_count
    ):
        disposition, reasons = "inconclusive", {"inconclusive_pair_analysis"}
    elif (
        audit_recipe.require_zero_different_action_conflicts
        and overlap_analysis.different_action_conflict_count
    ):
        disposition, reasons = "action_conflict_detected", {"action_conflict_detected"}
    elif (
        audit_recipe.require_zero_target_conflicts
        and overlap_analysis.target_conflict_count
    ):
        disposition, reasons = "target_conflict_detected", {"target_conflict_detected"}
    elif wrong_action > audit_recipe.maximum_wrong_action_count:
        disposition, reasons = "wrong_action_detected", {"wrong_action_detected"}
    elif wrong_target > audit_recipe.maximum_wrong_target_count:
        disposition, reasons = "wrong_target_detected", {"wrong_target_detected"}
    elif invalid > audit_recipe.maximum_invalid_count:
        disposition, reasons = "invalid_evidence", {"invalid_evidence"}
    elif ambiguity > audit_recipe.maximum_ambiguous_fallback_count:
        disposition, reasons = "excessive_ambiguity", {"excessive_ambiguity"}
    elif missed > audit_recipe.maximum_missed_opportunity_count:
        disposition, reasons = (
            "excessive_missed_opportunities",
            {"excessive_missed_opportunities"},
        )
    return ObserverHabitArbitrationAuditDTO.create(
        habit_arbitration_plan_id=arbitration_plan.habit_arbitration_plan_id,
        habit_overlap_analysis_id=overlap_analysis.habit_overlap_analysis_id,
        habit_arbitration_audit_recipe_id=audit_recipe.habit_arbitration_audit_recipe_id,
        evaluated_shadow_replay_ids=tuple(replay_ids),
        evaluated_shadow_episode_ids=tuple(
            item.habit_arbitration_shadow_episode_id for item in fixture_shadow_episodes
        ),
        applicable_count=applicable,
        habit_selection_count=selections,
        fallback_count=fallback,
        correct_selection_count=correct,
        wrong_action_count=wrong_action,
        wrong_target_count=wrong_target,
        ambiguous_fallback_count=ambiguity,
        missed_opportunity_count=missed,
        invalid_count=invalid,
        episode_count=len(fixture_shadow_episodes),
        eligible_for_multi_habit_activation_review=(
            disposition == "eligible_for_multi_habit_activation_review"
        ),
        disposition=disposition,
        reason_codes=tuple(reasons),
    )


def _shadow_occurrence(
    plan: ObserverHabitArbitrationPlanDTO,
    entry: ObserverTransitionLedgerEntryDTO,
    observation_id: str,
    evaluation: ObserverHabitArbitrationEvaluationDTO,
    habits: tuple[ObserverHabitSpecificationDTO, ...],
    authoritative_action: str,
) -> ObserverHabitArbitrationShadowOccurrenceDTO:
    habit = next(
        (
            item
            for item in habits
            if item.habit_specification_id == evaluation.selected_habit_id
        ),
        None,
    )
    expected = None if habit is None else habit.expected_target_state_class_id
    actual = entry.transition_verification.transition_record.observed_state_after_id
    if evaluation.decision == "fallback_invalid":
        outcome = "invalid_evaluation"
    elif evaluation.decision == "fallback_ambiguous":
        outcome = "ambiguous_fallback"
    elif evaluation.selected_habit_id is None:
        outcome = "fallback"
    elif evaluation.selected_action != authoritative_action:
        outcome = "wrong_action"
    elif expected is not None and expected != actual:
        outcome = "wrong_target"
    else:
        outcome = "correct_selection"
    return ObserverHabitArbitrationShadowOccurrenceDTO.create(
        habit_arbitration_plan_id=plan.habit_arbitration_plan_id,
        ledger_entry_id=entry.ledger_entry_id,
        observation_artifact_id=observation_id,
        arbitration_evaluation_id=evaluation.habit_arbitration_evaluation_id,
        selected_habit_id=evaluation.selected_habit_id,
        selected_action=evaluation.selected_action,
        authoritative_action=authoritative_action,
        actual_target_state_class_id=actual,
        expected_target_state_class_id=expected,
        outcome=outcome,
        reason_codes=(outcome,),
    )


def _action_name(action_id: str) -> str:
    for action_name in ("move_left", "move_right", "wait"):
        if (
            ObserverFixtureActionDTO.create(action_name=action_name).fixture_action_id
            == action_id
        ):
            return action_name
    return action_id


def ledger_snapshot_with_entries(
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
) -> ObserverTransitionLedgerSnapshotDTO:
    return build_observer_transition_ledger_snapshot(entries=entries)
