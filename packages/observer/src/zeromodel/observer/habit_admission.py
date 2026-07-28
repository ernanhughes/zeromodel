"""Admission decisions for shadow-audited Observer habits."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence, cast

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.habit import (
    ObserverHabitError,
    ObserverHabitShadowAuditDTO,
    ObserverHabitShadowEpisodeDTO,
    ObserverHabitShadowReplayDTO,
    ObserverHabitSpecificationDTO,
)

OBSERVER_HABIT_ADMISSION_RECIPE_VERSION: Final = "observer-habit-admission-recipe/1"
OBSERVER_HABIT_ADMISSION_DECISION_VERSION: Final = "observer-habit-admission-decision/1"

HABIT_ADMISSION_DECISIONS: Final = frozenset({"admit", "reject", "inconclusive"})


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverHabitError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverHabitError(f"{field_name} must be unique and sorted")


def _ensure_non_negative(value: int, field_name: str) -> None:
    if value < 0:
        raise ObserverHabitError(f"{field_name} must be non-negative")


@dataclass(frozen=True)
class ObserverHabitAdmissionRecipeDTO:
    habit_admission_recipe_id: str
    required_shadow_audit_disposition: str
    require_admission_review_eligibility: bool
    require_zero_false_fires: bool
    require_zero_invalid_evaluations: bool
    require_complete_counterexample_coverage: bool
    minimum_evaluated_replay_count: int
    minimum_episode_count: int
    minimum_correct_fire_count: int
    maximum_missed_opportunity_count: int
    allowed_observation_schema_ids: tuple[str, ...]
    allowed_grouping_recipe_ids: tuple[str, ...]
    allowed_habit_compilation_recipe_ids: tuple[str, ...]
    version: str = OBSERVER_HABIT_ADMISSION_RECIPE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ADMISSION_RECIPE_VERSION:
            raise ObserverHabitError("unsupported habit admission recipe version")
        _require_non_empty(
            self.required_shadow_audit_disposition,
            "required_shadow_audit_disposition",
        )
        for field_name in (
            "minimum_evaluated_replay_count",
            "minimum_episode_count",
            "minimum_correct_fire_count",
            "maximum_missed_opportunity_count",
        ):
            _ensure_non_negative(getattr(self, field_name), field_name)
        for field_name in (
            "allowed_observation_schema_ids",
            "allowed_grouping_recipe_ids",
            "allowed_habit_compilation_recipe_ids",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_admission_recipe_id != expected_id:
            raise ObserverHabitError("habit_admission_recipe_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "allowed_grouping_recipe_ids": list(self.allowed_grouping_recipe_ids),
            "allowed_habit_compilation_recipe_ids": list(
                self.allowed_habit_compilation_recipe_ids
            ),
            "allowed_observation_schema_ids": list(self.allowed_observation_schema_ids),
            "maximum_missed_opportunity_count": self.maximum_missed_opportunity_count,
            "minimum_correct_fire_count": self.minimum_correct_fire_count,
            "minimum_episode_count": self.minimum_episode_count,
            "minimum_evaluated_replay_count": self.minimum_evaluated_replay_count,
            "require_admission_review_eligibility": (
                self.require_admission_review_eligibility
            ),
            "require_complete_counterexample_coverage": (
                self.require_complete_counterexample_coverage
            ),
            "require_zero_false_fires": self.require_zero_false_fires,
            "require_zero_invalid_evaluations": self.require_zero_invalid_evaluations,
            "required_shadow_audit_disposition": (
                self.required_shadow_audit_disposition
            ),
            "version": self.version,
        }
        if include_id:
            payload["habit_admission_recipe_id"] = self.habit_admission_recipe_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitAdmissionRecipeDTO":
        for key in (
            "allowed_observation_schema_ids",
            "allowed_grouping_recipe_ids",
            "allowed_habit_compilation_recipe_ids",
        ):
            values[key] = tuple(sorted(set(cast(Sequence[str], values.get(key, ())))))
        payload = {
            **values,
            "allowed_grouping_recipe_ids": list(
                cast(tuple[str, ...], values["allowed_grouping_recipe_ids"])
            ),
            "allowed_habit_compilation_recipe_ids": list(
                cast(tuple[str, ...], values["allowed_habit_compilation_recipe_ids"])
            ),
            "allowed_observation_schema_ids": list(
                cast(tuple[str, ...], values["allowed_observation_schema_ids"])
            ),
            "version": OBSERVER_HABIT_ADMISSION_RECIPE_VERSION,
        }
        return cls(
            habit_admission_recipe_id=canonical_id(payload),
            version=OBSERVER_HABIT_ADMISSION_RECIPE_VERSION,
            **values,  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class ObserverHabitAdmissionDecisionDTO:
    habit_admission_decision_id: str
    habit_specification_id: str
    habit_shadow_audit_id: str
    habit_admission_recipe_id: str
    decision: str
    reason_codes: tuple[str, ...]
    admitted_registry_status: str | None
    evidence_replay_ids: tuple[str, ...]
    version: str = OBSERVER_HABIT_ADMISSION_DECISION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_HABIT_ADMISSION_DECISION_VERSION:
            raise ObserverHabitError("unsupported habit admission decision version")
        for field_name in (
            "habit_specification_id",
            "habit_shadow_audit_id",
            "habit_admission_recipe_id",
            "decision",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        if self.decision not in HABIT_ADMISSION_DECISIONS:
            raise ObserverHabitError("unsupported habit admission decision")
        _ensure_sorted_unique(self.reason_codes, "reason_codes")
        _ensure_sorted_unique(self.evidence_replay_ids, "evidence_replay_ids")
        if self.decision == "admit":
            if self.admitted_registry_status != "admitted_inactive":
                raise ObserverHabitError("admitted decision requires inactive status")
        elif self.admitted_registry_status is not None:
            raise ObserverHabitError("non-admitted decision cannot carry status")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.habit_admission_decision_id != expected_id:
            raise ObserverHabitError("habit_admission_decision_id mismatch")

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "admitted_registry_status": self.admitted_registry_status,
            "decision": self.decision,
            "evidence_replay_ids": list(self.evidence_replay_ids),
            "habit_admission_recipe_id": self.habit_admission_recipe_id,
            "habit_shadow_audit_id": self.habit_shadow_audit_id,
            "habit_specification_id": self.habit_specification_id,
            "reason_codes": list(self.reason_codes),
            "version": self.version,
        }
        if include_id:
            payload["habit_admission_decision_id"] = self.habit_admission_decision_id
        return payload

    @classmethod
    def create(cls, **values: object) -> "ObserverHabitAdmissionDecisionDTO":
        for key in ("reason_codes", "evidence_replay_ids"):
            values[key] = tuple(sorted(set(cast(Sequence[str], values.get(key, ())))))
        payload = {
            **values,
            "evidence_replay_ids": list(
                cast(tuple[str, ...], values["evidence_replay_ids"])
            ),
            "reason_codes": list(cast(tuple[str, ...], values["reason_codes"])),
            "version": OBSERVER_HABIT_ADMISSION_DECISION_VERSION,
        }
        return cls(
            habit_admission_decision_id=canonical_id(payload),
            version=OBSERVER_HABIT_ADMISSION_DECISION_VERSION,
            **values,  # type: ignore[arg-type]
        )


def admit_observer_habit(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    shadow_audit: ObserverHabitShadowAuditDTO,
    historical_shadow_replay: ObserverHabitShadowReplayDTO,
    live_shadow_episodes: tuple[ObserverHabitShadowEpisodeDTO, ...],
    admission_recipe: ObserverHabitAdmissionRecipeDTO,
) -> ObserverHabitAdmissionDecisionDTO:
    """Decide whether one shadow-audited habit may enter the inactive registry."""

    try:
        replays = _validated_evidence(
            habit_specification=habit_specification,
            shadow_audit=shadow_audit,
            historical_shadow_replay=historical_shadow_replay,
            live_shadow_episodes=live_shadow_episodes,
        )
    except ObserverHabitError as exc:
        return _decision(
            habit_specification=habit_specification,
            shadow_audit=shadow_audit,
            admission_recipe=admission_recipe,
            decision="inconclusive",
            reasons=(str(exc).replace(" ", "_"),),
            evidence_replay_ids=shadow_audit.evaluated_shadow_replay_ids,
        )

    reasons: set[str] = set()
    hard_failures: set[str] = set()
    if shadow_audit.disposition != admission_recipe.required_shadow_audit_disposition:
        hard_failures.add("shadow_audit_disposition_mismatch")
    if (
        admission_recipe.require_admission_review_eligibility
        and not shadow_audit.eligible_for_admission_review
    ):
        hard_failures.add("shadow_audit_not_eligible")
    coverage = shadow_audit.counterexample_coverage
    if admission_recipe.require_complete_counterexample_coverage and (
        coverage is None or not coverage.coverage_complete
    ):
        hard_failures.add("counterexample_coverage_incomplete")
    if (
        admission_recipe.allowed_observation_schema_ids
        and habit_specification.observation_schema_id
        not in admission_recipe.allowed_observation_schema_ids
    ):
        hard_failures.add("observation_schema_not_allowed")
    if (
        admission_recipe.allowed_grouping_recipe_ids
        and habit_specification.grouping_recipe_id
        not in admission_recipe.allowed_grouping_recipe_ids
    ):
        hard_failures.add("grouping_recipe_not_allowed")
    if (
        admission_recipe.allowed_habit_compilation_recipe_ids
        and habit_specification.habit_compilation_recipe_id
        not in admission_recipe.allowed_habit_compilation_recipe_ids
    ):
        hard_failures.add("habit_compilation_recipe_not_allowed")

    episode_ids = {
        episode_id for replay in replays for episode_id in replay.episode_ids
    }
    correct_fire_count = sum(replay.correct_fire_count for replay in replays)
    false_fire_count = sum(
        replay.wrong_action_fire_count + replay.wrong_target_fire_count
        for replay in replays
    )
    invalid_count = sum(replay.invalid_count for replay in replays)
    missed_count = sum(replay.missed_opportunity_count for replay in replays)
    if len(replays) < admission_recipe.minimum_evaluated_replay_count:
        hard_failures.add("minimum_evaluated_replay_count_not_met")
    if len(episode_ids) < admission_recipe.minimum_episode_count:
        hard_failures.add("minimum_episode_count_not_met")
    if correct_fire_count < admission_recipe.minimum_correct_fire_count:
        hard_failures.add("minimum_correct_fire_count_not_met")
    if admission_recipe.require_zero_false_fires and false_fire_count:
        hard_failures.add("false_fire_detected")
    if admission_recipe.require_zero_invalid_evaluations and invalid_count:
        hard_failures.add("invalid_evaluation_detected")
    if missed_count > admission_recipe.maximum_missed_opportunity_count:
        hard_failures.add("missed_opportunity_limit_exceeded")

    if hard_failures:
        return _decision(
            habit_specification=habit_specification,
            shadow_audit=shadow_audit,
            admission_recipe=admission_recipe,
            decision="reject",
            reasons=tuple(sorted(hard_failures)),
            evidence_replay_ids=tuple(
                replay.habit_shadow_replay_id for replay in replays
            ),
        )
    reasons.add("admission_requirements_met")
    return _decision(
        habit_specification=habit_specification,
        shadow_audit=shadow_audit,
        admission_recipe=admission_recipe,
        decision="admit",
        reasons=tuple(sorted(reasons)),
        evidence_replay_ids=tuple(replay.habit_shadow_replay_id for replay in replays),
    )


def _decision(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    shadow_audit: ObserverHabitShadowAuditDTO,
    admission_recipe: ObserverHabitAdmissionRecipeDTO,
    decision: str,
    reasons: tuple[str, ...],
    evidence_replay_ids: tuple[str, ...],
) -> ObserverHabitAdmissionDecisionDTO:
    return ObserverHabitAdmissionDecisionDTO.create(
        habit_specification_id=habit_specification.habit_specification_id,
        habit_shadow_audit_id=shadow_audit.habit_shadow_audit_id,
        habit_admission_recipe_id=admission_recipe.habit_admission_recipe_id,
        decision=decision,
        reason_codes=reasons,
        admitted_registry_status=("admitted_inactive" if decision == "admit" else None),
        evidence_replay_ids=evidence_replay_ids,
    )


def _validated_evidence(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    shadow_audit: ObserverHabitShadowAuditDTO,
    historical_shadow_replay: ObserverHabitShadowReplayDTO,
    live_shadow_episodes: tuple[ObserverHabitShadowEpisodeDTO, ...],
) -> tuple[ObserverHabitShadowReplayDTO, ...]:
    habit_id = habit_specification.habit_specification_id
    if shadow_audit.habit_specification_id != habit_id:
        raise ObserverHabitError("audit habit mismatch")
    if historical_shadow_replay.habit_shadow_replay_id != (
        shadow_audit.historical_shadow_replay_id
    ):
        raise ObserverHabitError("historical replay identity mismatch")
    replays_by_id: dict[str, ObserverHabitShadowReplayDTO] = {}
    for replay in (historical_shadow_replay,):
        if replay.habit_specification_id != habit_id:
            raise ObserverHabitError("foreign habit replay")
        replays_by_id[replay.habit_shadow_replay_id] = replay
    for episode in live_shadow_episodes:
        if episode.habit_specification_id != habit_id:
            raise ObserverHabitError("foreign live episode")
        replay = episode.shadow_replay
        if replay.habit_specification_id != habit_id:
            raise ObserverHabitError("foreign habit replay")
        if replay.ledger_snapshot_id != episode.ledger_snapshot_id:
            raise ObserverHabitError("live episode replay snapshot mismatch")
        if replay.habit_shadow_replay_id in replays_by_id:
            raise ObserverHabitError("duplicate replay evidence")
        replays_by_id[replay.habit_shadow_replay_id] = replay
    supplied_ids = tuple(sorted(replays_by_id))
    if supplied_ids != shadow_audit.evaluated_shadow_replay_ids:
        raise ObserverHabitError("evaluated replay evidence mismatch")
    return tuple(replays_by_id[key] for key in supplied_ids)
