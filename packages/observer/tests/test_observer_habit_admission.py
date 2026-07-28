import pytest

from zeromodel.observer import (
    ObserverHabitAdmissionRecipeDTO,
    ObserverHabitError,
    ObserverHabitShadowAuditDTO,
    ObserverHabitShadowAuditRecipeDTO,
    admit_observer_habit,
    audit_observer_habit_shadow,
)
from zeromodel.observer.habit import ObserverHabitSpecificationDTO
from zeromodel.observer.habit_admission import ObserverHabitAdmissionDecisionDTO

from test_observer_habit import compile_first, shadow_episode, shadow_replay


def admission_recipe(habit: ObserverHabitSpecificationDTO, **overrides):
    values = {
        "required_shadow_audit_disposition": "eligible_for_admission_review",
        "require_admission_review_eligibility": True,
        "require_zero_false_fires": True,
        "require_zero_invalid_evaluations": True,
        "require_complete_counterexample_coverage": False,
        "minimum_evaluated_replay_count": 2,
        "minimum_episode_count": 2,
        "minimum_correct_fire_count": 2,
        "maximum_missed_opportunity_count": 0,
        "allowed_observation_schema_ids": (habit.observation_schema_id,),
        "allowed_grouping_recipe_ids": (habit.grouping_recipe_id,),
        "allowed_habit_compilation_recipe_ids": (habit.habit_compilation_recipe_id,),
    }
    values.update(overrides)
    return ObserverHabitAdmissionRecipeDTO.create(**values)


def audit_recipe():
    return ObserverHabitShadowAuditRecipeDTO.create(
        minimum_applicable_count=2,
        minimum_episode_count=2,
        minimum_correct_fire_count=2,
        maximum_wrong_action_fire_count=0,
        maximum_wrong_target_fire_count=0,
        maximum_missed_opportunity_count=0,
        maximum_invalid_count=0,
        require_zero_false_fires=True,
        require_counterexample_coverage=False,
    )


def admitted_evidence():
    _, _, _, _, _, _, _, result = compile_first(action_name="wait")
    habit = result.habit_specification
    assert habit is not None
    historical = shadow_replay(
        habit, outcomes=("correct_fire",), episode_id="historical"
    )
    live = shadow_replay(habit, outcomes=("correct_fire",), episode_id="live")
    episode = shadow_episode(habit, live, "live-episode")
    audit = audit_observer_habit_shadow(
        habit_specification=habit,
        shadow_audit_recipe=audit_recipe(),
        historical_shadow_replay=historical,
        live_shadow_episodes=(episode,),
    )
    return habit, historical, episode, audit


def test_admit_valid_habit() -> None:
    habit, historical, episode, audit = admitted_evidence()
    decision = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=historical,
        live_shadow_episodes=(episode,),
        admission_recipe=admission_recipe(habit),
    )
    assert decision.decision == "admit"
    assert decision.admitted_registry_status == "admitted_inactive"
    assert decision.evidence_replay_ids == audit.evaluated_shadow_replay_ids


def test_reject_failed_shadow_audit() -> None:
    habit, historical, _, _ = admitted_evidence()
    failed = shadow_replay(habit, outcomes=("wrong_target_fire",), episode_id="live")
    audit = audit_observer_habit_shadow(
        habit_specification=habit,
        shadow_audit_recipe=audit_recipe(),
        historical_shadow_replay=historical,
        live_shadow_episodes=(shadow_episode(habit, failed, "live-episode"),),
    )
    decision = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=historical,
        live_shadow_episodes=(shadow_episode(habit, failed, "live-episode-2"),),
        admission_recipe=admission_recipe(habit),
    )
    assert decision.decision in {"reject", "inconclusive"}
    assert decision.admitted_registry_status is None


def test_inconclusive_missing_replay_evidence() -> None:
    habit, historical, _, audit = admitted_evidence()
    decision = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=historical,
        live_shadow_episodes=(),
        admission_recipe=admission_recipe(habit),
    )
    assert decision.decision == "inconclusive"


def test_recompute_thresholds_does_not_trust_audit_label() -> None:
    habit, historical, episode, audit = admitted_evidence()
    tampered = ObserverHabitShadowAuditDTO.create(
        habit_specification_id=audit.habit_specification_id,
        shadow_audit_recipe_id=audit.shadow_audit_recipe_id,
        historical_shadow_replay_id=audit.historical_shadow_replay_id,
        evaluated_shadow_replay_ids=audit.evaluated_shadow_replay_ids,
        live_shadow_episode_ids=audit.live_shadow_episode_ids,
        counterexample_coverage=audit.counterexample_coverage,
        disposition="eligible_for_admission_review",
        reason_codes=("shadow_thresholds_met",),
    )
    strict = admission_recipe(habit, minimum_correct_fire_count=999)
    decision = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=tampered,
        historical_shadow_replay=historical,
        live_shadow_episodes=(episode,),
        admission_recipe=strict,
    )
    assert decision.decision == "reject"
    assert "minimum_correct_fire_count_not_met" in decision.reason_codes


def test_duplicate_replay_evidence_is_inconclusive() -> None:
    habit, historical, _, audit = admitted_evidence()
    duplicate = shadow_episode(habit, historical, "duplicate")
    decision = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=historical,
        live_shadow_episodes=(duplicate,),
        admission_recipe=admission_recipe(habit),
    )
    assert decision.decision == "inconclusive"


def test_foreign_habit_replay_is_inconclusive() -> None:
    habit, _, episode, audit = admitted_evidence()
    _, _, _, _, _, _, _, foreign_result = compile_first(("move_right",))
    foreign_habit = foreign_result.habit_specification
    assert foreign_habit is not None
    foreign = shadow_replay(foreign_habit, outcomes=("correct_fire",), episode_id="f")
    decision = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=foreign,
        live_shadow_episodes=(episode,),
        admission_recipe=admission_recipe(habit),
    )
    assert decision.decision == "inconclusive"


def test_admission_recipe_sensitivity() -> None:
    habit, historical, episode, audit = admitted_evidence()
    loose = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=historical,
        live_shadow_episodes=(episode,),
        admission_recipe=admission_recipe(habit),
    )
    strict = admit_observer_habit(
        habit_specification=habit,
        shadow_audit=audit,
        historical_shadow_replay=historical,
        live_shadow_episodes=(episode,),
        admission_recipe=admission_recipe(habit, minimum_correct_fire_count=999),
    )
    assert loose.decision == "admit"
    assert strict.decision == "reject"
    assert loose.habit_admission_decision_id != strict.habit_admission_decision_id


def test_public_constructor_rejects_bad_decision_status() -> None:
    habit, _, _, audit = admitted_evidence()
    recipe = admission_recipe(habit)
    with pytest.raises(ObserverHabitError):
        ObserverHabitAdmissionDecisionDTO.create(
            habit_specification_id=habit.habit_specification_id,
            habit_shadow_audit_id=audit.habit_shadow_audit_id,
            habit_admission_recipe_id=recipe.habit_admission_recipe_id,
            decision="reject",
            reason_codes=("x",),
            admitted_registry_status="admitted_inactive",
            evidence_replay_ids=audit.evaluated_shadow_replay_ids,
        )
