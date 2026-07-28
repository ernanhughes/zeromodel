"""Compile inactive Observer habits and evaluate their guards."""

from __future__ import annotations

import math
from typing import Mapping

from zeromodel.observer._observation_replay import (
    source_observation_for_entry,
    target_observation_for_entry,
)
from zeromodel.observer.artifacts import (
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)
from zeromodel.observer.graph import ObserverObservationGraphBuildDTO
from zeromodel.observer.grouping import (
    ObserverGroupedFeatureValueDTO,
    ObserverGroupingFeatureDTO,
    ObserverStateClassDTO,
    ObserverStateGroupingRecipeDTO,
    assign_observation_to_state_class,
)
from zeromodel.observer.habit import (
    ObserverHabitCompilationRecipeDTO,
    ObserverHabitCompilationResultDTO,
    ObserverHabitCounterexampleDTO,
    ObserverHabitEvaluationDTO,
    ObserverHabitGuardDTO,
    ObserverHabitGuardEvaluationDTO,
    ObserverHabitSpecificationDTO,
)
from zeromodel.observer.ledger import (
    ObserverTransitionLedgerEntryDTO,
    ObserverTransitionLedgerSnapshotDTO,
)
from zeromodel.observer.promotion import (
    ObserverPromotionAnalysisDTO,
    ObserverPromotionCandidateDTO,
)


def compile_observer_habit_specification(
    *,
    promotion_analysis: ObserverPromotionAnalysisDTO,
    promotion_candidate: ObserverPromotionCandidateDTO,
    graph_build: ObserverObservationGraphBuildDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    compilation_recipe: ObserverHabitCompilationRecipeDTO,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
) -> ObserverHabitCompilationResultDTO:
    """Compile one eligible promotion candidate into an inactive habit spec."""

    invalid_reasons = _initial_compilation_failures(
        promotion_analysis=promotion_analysis,
        promotion_candidate=promotion_candidate,
        graph_build=graph_build,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
        compilation_recipe=compilation_recipe,
        ledger_snapshot=ledger_snapshot,
    )
    if invalid_reasons:
        return _blocked_result(
            promotion_candidate=promotion_candidate,
            compilation_recipe=compilation_recipe,
            disposition="invalid_candidate"
            if "candidate_not_eligible" in invalid_reasons
            else "schema_mismatch"
            if "schema_mismatch" in invalid_reasons
            else "unsupported",
            reason_codes=invalid_reasons,
        )
    graph = graph_build.graph
    assert graph is not None
    edge = next(
        (
            item
            for item in graph.edges
            if item.transition_key.transition_key_id
            == promotion_candidate.transition_key_id
        ),
        None,
    )
    if edge is None:
        return _blocked_result(
            promotion_candidate=promotion_candidate,
            compilation_recipe=compilation_recipe,
            disposition="invalid_candidate",
            reason_codes=("unknown_transition_key",),
        )
    source_class = next(
        (
            item
            for item in graph_build.state_classes
            if item.state_class_id == edge.transition_key.source_state_class_id
        ),
        None,
    )
    if source_class is None:
        return _blocked_result(
            promotion_candidate=promotion_candidate,
            compilation_recipe=compilation_recipe,
            disposition="invalid_candidate",
            reason_codes=("missing_source_class",),
        )
    occurrences_by_id = {
        item.occurrence_id: item for item in promotion_analysis.occurrences
    }
    supporting_occurrences = tuple(
        occurrences_by_id[item]
        for item in promotion_candidate.supporting_occurrence_ids
        if item in occurrences_by_id
    )
    if len(supporting_occurrences) != len(
        promotion_candidate.supporting_occurrence_ids
    ):
        return _blocked_result(
            promotion_candidate=promotion_candidate,
            compilation_recipe=compilation_recipe,
            disposition="invalid_candidate",
            reason_codes=("missing_supporting_occurrence",),
        )
    positive = _positive_guards(
        source_class=source_class,
        grouping_recipe=grouping_recipe,
        compilation_recipe=compilation_recipe,
        evidence_ids=promotion_candidate.supporting_occurrence_ids,
    )
    if isinstance(positive, str):
        return _blocked_result(
            promotion_candidate=promotion_candidate,
            compilation_recipe=compilation_recipe,
            disposition=positive,
            reason_codes=(positive,),
        )
    if len(positive) > compilation_recipe.maximum_guard_count:
        return _blocked_result(
            promotion_candidate=promotion_candidate,
            compilation_recipe=compilation_recipe,
            disposition="guard_limit_exceeded",
            reason_codes=("maximum_guard_count_exceeded",),
        )
    missing_required = tuple(
        key
        for key in compilation_recipe.required_guard_feature_keys
        if key not in {item.feature_key for item in positive}
    )
    if missing_required:
        return _blocked_result(
            promotion_candidate=promotion_candidate,
            compilation_recipe=compilation_recipe,
            disposition="insufficient_guard_evidence",
            reason_codes=("missing_required_guard_feature",),
        )
    source_target = _source_target_observations(
        entries=entries, observation_schema=observation_schema
    )
    counterexamples = _counterexamples(
        promotion_analysis=promotion_analysis,
        habit_specification_id=None,
        transition_key_id=edge.transition_key.transition_key_id,
        source_state_class_id=edge.transition_key.source_state_class_id,
        expected_target_state_class_id=edge.transition_key.target_state_class_id,
    )
    counterexample_guards = _counterexample_guards(
        counterexamples=counterexamples,
        source_target=source_target,
        positive_guards=positive,
        compilation_recipe=compilation_recipe,
    )
    if isinstance(counterexample_guards, str):
        return ObserverHabitCompilationResultDTO.create(
            promotion_candidate_id=promotion_candidate.promotion_candidate_id,
            compilation_recipe_id=compilation_recipe.habit_compilation_recipe_id,
            habit_specification=None,
            counterexamples=counterexamples,
            disposition=counterexample_guards,
            reason_codes=(counterexample_guards,),
        )
    if (
        compilation_recipe.require_counterexample_guards
        and counterexamples
        and not counterexample_guards
    ):
        return ObserverHabitCompilationResultDTO.create(
            promotion_candidate_id=promotion_candidate.promotion_candidate_id,
            compilation_recipe_id=compilation_recipe.habit_compilation_recipe_id,
            habit_specification=None,
            counterexamples=counterexamples,
            disposition="counterexample_conflict",
            reason_codes=("counterexample_guard_missing",),
        )
    if (
        len(counterexample_guards)
        > compilation_recipe.maximum_counterexample_guard_count
    ):
        return ObserverHabitCompilationResultDTO.create(
            promotion_candidate_id=promotion_candidate.promotion_candidate_id,
            compilation_recipe_id=compilation_recipe.habit_compilation_recipe_id,
            habit_specification=None,
            counterexamples=counterexamples,
            disposition="guard_limit_exceeded",
            reason_codes=("maximum_counterexample_guard_count_exceeded",),
        )
    habit = ObserverHabitSpecificationDTO.create(
        habit_compilation_recipe_id=compilation_recipe.habit_compilation_recipe_id,
        promotion_candidate_id=promotion_candidate.promotion_candidate_id,
        promotion_analysis_id=promotion_analysis.promotion_analysis_id,
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        observation_graph_id=graph.observation_graph_id,
        grouping_recipe_id=grouping_recipe.grouping_recipe_id,
        observation_schema_id=observation_schema.schema_id,
        transition_key_id=edge.transition_key.transition_key_id,
        source_state_class_id=edge.transition_key.source_state_class_id,
        recommended_action=edge.transition_key.action,
        expected_target_state_class_id=edge.transition_key.target_state_class_id,
        positive_guards=positive,
        counterexample_guards=counterexample_guards,
        supporting_occurrence_ids=promotion_candidate.supporting_occurrence_ids,
        supporting_ledger_entry_ids=promotion_candidate.supporting_ledger_entry_ids,
        status="shadow_candidate",
    )
    linked_counterexamples = tuple(
        ObserverHabitCounterexampleDTO.create(
            habit_specification_id=habit.habit_specification_id,
            transition_key_id=item.transition_key_id,
            source_observation_artifact_id=item.source_observation_artifact_id,
            actual_action=item.actual_action,
            actual_target_state_class_id=item.actual_target_state_class_id,
            expected_target_state_class_id=item.expected_target_state_class_id,
            ledger_entry_id=item.ledger_entry_id,
            occurrence_id=item.occurrence_id,
            verification_status=item.verification_status,
            reason_codes=item.reason_codes,
            candidate_guard_ids=tuple(
                guard.habit_guard_id
                for guard in counterexample_guards
                if item.counterexample_id in guard.source_evidence_ids
            ),
        )
        for item in counterexamples
    )
    return ObserverHabitCompilationResultDTO.create(
        promotion_candidate_id=promotion_candidate.promotion_candidate_id,
        compilation_recipe_id=compilation_recipe.habit_compilation_recipe_id,
        habit_specification=habit,
        counterexamples=linked_counterexamples,
        disposition="compiled_for_shadow",
        reason_codes=("compiled_for_shadow",),
    )


def evaluate_observer_habit(
    *,
    habit_specification: ObserverHabitSpecificationDTO,
    observation: ObserverObservationArtifactDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverHabitEvaluationDTO:
    """Evaluate an inactive habit against one observation."""

    if observation_schema.schema_id != habit_specification.observation_schema_id:
        return ObserverHabitEvaluationDTO.create(
            habit_specification_id=habit_specification.habit_specification_id,
            observation_artifact_id=observation.observation_artifact_id,
            state_class_id=None,
            guard_evaluations=(),
            decision="invalid",
            recommended_action=None,
            reason_codes=("schema_mismatch",),
        )
    assignment = assign_observation_to_state_class(
        observation=observation,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
    )
    if assignment.status != "assigned" or assignment.state_class_id is None:
        return ObserverHabitEvaluationDTO.create(
            habit_specification_id=habit_specification.habit_specification_id,
            observation_artifact_id=observation.observation_artifact_id,
            state_class_id=None,
            guard_evaluations=(),
            decision="invalid",
            recommended_action=None,
            reason_codes=("state_assignment_rejected",),
        )
    if assignment.state_class_id != habit_specification.source_state_class_id:
        return ObserverHabitEvaluationDTO.create(
            habit_specification_id=habit_specification.habit_specification_id,
            observation_artifact_id=observation.observation_artifact_id,
            state_class_id=assignment.state_class_id,
            guard_evaluations=(),
            decision="abstain",
            recommended_action=None,
            reason_codes=("source_class_mismatch",),
        )
    projected = _project_observation(observation)
    evaluations = tuple(
        _evaluate_guard(guard=guard, projected=projected, observation=observation)
        for guard in habit_specification.positive_guards
        + habit_specification.counterexample_guards
    )
    if any(item.status == "invalid" for item in evaluations):
        decision = "invalid"
        action = None
        reasons = ("guard_invalid",)
    elif all(item.status == "matched" for item in evaluations):
        decision = "fire"
        action = habit_specification.recommended_action
        reasons = ("guards_matched",)
    else:
        decision = "abstain"
        action = None
        reasons = ("guard_not_matched",)
    return ObserverHabitEvaluationDTO.create(
        habit_specification_id=habit_specification.habit_specification_id,
        observation_artifact_id=observation.observation_artifact_id,
        state_class_id=assignment.state_class_id,
        guard_evaluations=evaluations,
        decision=decision,
        recommended_action=action,
        reason_codes=reasons,
    )


def _initial_compilation_failures(
    *,
    promotion_analysis: ObserverPromotionAnalysisDTO,
    promotion_candidate: ObserverPromotionCandidateDTO,
    graph_build: ObserverObservationGraphBuildDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    compilation_recipe: ObserverHabitCompilationRecipeDTO,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
) -> tuple[str, ...]:
    failures: set[str] = set()
    graph = graph_build.graph
    if promotion_analysis.status != "built":
        failures.add("promotion_analysis_not_built")
    if (
        promotion_candidate.disposition != "eligible"
        or not promotion_candidate.eligible_for_compilation
    ):
        failures.add("candidate_not_eligible")
    if promotion_candidate.promotion_candidate_id not in {
        item.promotion_candidate_id for item in promotion_analysis.promotion_candidates
    }:
        failures.add("candidate_not_in_analysis")
    if graph is None or graph_build.status != "built":
        failures.add("graph_not_built")
    else:
        if graph.ledger_snapshot_id != ledger_snapshot.ledger_snapshot_id:
            failures.add("ledger_mismatch")
        if promotion_analysis.ledger_snapshot_id != ledger_snapshot.ledger_snapshot_id:
            failures.add("ledger_mismatch")
        if promotion_candidate.ledger_snapshot_id != ledger_snapshot.ledger_snapshot_id:
            failures.add("ledger_mismatch")
        if graph.grouping_recipe_id != grouping_recipe.grouping_recipe_id:
            failures.add("grouping_mismatch")
        if graph.observation_schema_id != observation_schema.schema_id:
            failures.add("schema_mismatch")
        if grouping_recipe.observation_schema_id != observation_schema.schema_id:
            failures.add("schema_mismatch")
    if compilation_recipe.promotion_recipe_id != promotion_analysis.promotion_recipe_id:
        failures.add("promotion_recipe_mismatch")
    if compilation_recipe.grouping_recipe_id != grouping_recipe.grouping_recipe_id:
        failures.add("grouping_mismatch")
    if compilation_recipe.observation_schema_id != observation_schema.schema_id:
        failures.add("schema_mismatch")
    return tuple(sorted(failures))


def _blocked_result(
    *,
    promotion_candidate: ObserverPromotionCandidateDTO,
    compilation_recipe: ObserverHabitCompilationRecipeDTO,
    disposition: str,
    reason_codes: tuple[str, ...],
) -> ObserverHabitCompilationResultDTO:
    return ObserverHabitCompilationResultDTO.create(
        promotion_candidate_id=promotion_candidate.promotion_candidate_id,
        compilation_recipe_id=compilation_recipe.habit_compilation_recipe_id,
        habit_specification=None,
        counterexamples=(),
        disposition=disposition,
        reason_codes=reason_codes,
    )


def _positive_guards(
    *,
    source_class: ObserverStateClassDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    compilation_recipe: ObserverHabitCompilationRecipeDTO,
    evidence_ids: tuple[str, ...],
) -> tuple[ObserverHabitGuardDTO, ...] | str:
    features = {item.feature_key: item for item in grouping_recipe.feature_groupings}
    guards: list[ObserverHabitGuardDTO] = []
    allowed = set(compilation_recipe.allowed_guard_feature_keys)
    forbidden = set(compilation_recipe.forbidden_guard_feature_keys)
    for grouped in source_class.state_class_key:
        feature = features.get(grouped.feature_key)
        if feature is None or feature.mode == "ignored":
            continue
        if grouped.feature_key not in allowed or grouped.feature_key in forbidden:
            continue
        guard = _guard_from_grouped(
            grouped=grouped,
            feature=feature,
            role="positive",
            evidence_ids=evidence_ids,
            compilation_recipe=compilation_recipe,
        )
        if guard is None:
            return "unsupported_grouping_mode"
        guards.append(guard)
    return tuple(sorted(guards, key=lambda item: item.feature_key))


def _guard_from_grouped(
    *,
    grouped: ObserverGroupedFeatureValueDTO,
    feature: ObserverGroupingFeatureDTO,
    role: str,
    evidence_ids: tuple[str, ...],
    compilation_recipe: ObserverHabitCompilationRecipeDTO,
) -> ObserverHabitGuardDTO | None:
    if grouped.grouped_kind == "exact":
        if not compilation_recipe.allow_exact_guards:
            return None
        value = grouped.grouped_value
        if not isinstance(value, Mapping):
            return None
        return ObserverHabitGuardDTO.create(
            feature_key=grouped.feature_key,
            operator="equals",
            expected_type=str(value["type"]),
            expected_value=value["value"],
            minimum_value=None,
            maximum_value=None,
            guard_role=role,
            source_evidence_ids=evidence_ids,
        )
    if grouped.grouped_kind == "category":
        if not compilation_recipe.allow_categorical_guards:
            return None
        raw_values = tuple(
            item.raw_value
            for item in feature.category_mapping
            if item.mapped_value == grouped.grouped_value
        )
        if len(raw_values) == 1:
            value = raw_values[0]
        elif (
            feature.unmapped_category_policy == "preserve"
            and grouped.grouped_value
            not in {item.mapped_value for item in feature.category_mapping}
        ):
            value = grouped.grouped_value
        else:
            return ObserverHabitGuardDTO.create(
                feature_key=grouped.feature_key,
                operator="is_present",
                expected_type="none",
                expected_value=None,
                minimum_value=None,
                maximum_value=None,
                guard_role=role,
                source_evidence_ids=evidence_ids,
            )
        return ObserverHabitGuardDTO.create(
            feature_key=grouped.feature_key,
            operator="equals",
            expected_type="str",
            expected_value=value,
            minimum_value=None,
            maximum_value=None,
            guard_role=role,
            source_evidence_ids=evidence_ids,
        )
    if grouped.grouped_kind == "numeric_bucket":
        if not compilation_recipe.allow_numeric_range_guards:
            return None
        value = grouped.grouped_value
        if not isinstance(value, Mapping):
            return None
        bucket_index = int(value["bucket_index"])
        bucket_size = float(value["bucket_size"])
        minimum = bucket_index * bucket_size
        maximum = ((bucket_index + 1) * bucket_size) - 1
        if not float(minimum).is_integer() or not float(maximum).is_integer():
            return None
        return ObserverHabitGuardDTO.create(
            feature_key=grouped.feature_key,
            operator="in_closed_range",
            expected_type=grouped.source_type or "number",
            expected_value=None,
            minimum_value=int(minimum),
            maximum_value=int(maximum),
            guard_role=role,
            source_evidence_ids=evidence_ids,
        )
    if grouped.grouped_kind == "missing":
        return ObserverHabitGuardDTO.create(
            feature_key=grouped.feature_key,
            operator="is_missing",
            expected_type="none",
            expected_value=None,
            minimum_value=None,
            maximum_value=None,
            guard_role=role,
            source_evidence_ids=evidence_ids,
        )
    return None


def _counterexamples(
    *,
    promotion_analysis: ObserverPromotionAnalysisDTO,
    habit_specification_id: str | None,
    transition_key_id: str,
    source_state_class_id: str,
    expected_target_state_class_id: str,
) -> tuple[ObserverHabitCounterexampleDTO, ...]:
    items = []
    for occurrence in promotion_analysis.occurrences:
        if occurrence.source_state_class_id != source_state_class_id:
            continue
        reasons: set[str] = set()
        if occurrence.transition_key_id == transition_key_id:
            if occurrence.verification_status == "contradicted":
                reasons.add("contradicted_transition")
            if occurrence.verification_status == "inconclusive":
                reasons.add("inconclusive_transition")
        elif occurrence.action == next(
            item.action
            for item in promotion_analysis.occurrences
            if item.transition_key_id == transition_key_id
        ):
            reasons.add("different_target")
        if occurrence.target_state_class_id != expected_target_state_class_id:
            reasons.add("different_target")
        if occurrence.predictor_rule_set_id != occurrence.environment_rule_set_id:
            reasons.add("rule_regime_mismatch")
        if not reasons:
            continue
        items.append(
            ObserverHabitCounterexampleDTO.create(
                habit_specification_id=habit_specification_id,
                transition_key_id=transition_key_id,
                source_observation_artifact_id=occurrence.source_observation_artifact_id,
                actual_action=occurrence.action,
                actual_target_state_class_id=occurrence.target_state_class_id,
                expected_target_state_class_id=expected_target_state_class_id,
                ledger_entry_id=occurrence.ledger_entry_id,
                occurrence_id=occurrence.occurrence_id,
                verification_status=occurrence.verification_status,
                reason_codes=tuple(sorted(reasons)),
                candidate_guard_ids=(),
            )
        )
    return tuple(sorted(items, key=lambda item: item.counterexample_id))


def _counterexample_guards(
    *,
    counterexamples: tuple[ObserverHabitCounterexampleDTO, ...],
    source_target: Mapping[
        str,
        tuple[
            ObserverObservationArtifactDTO | None, ObserverObservationArtifactDTO | None
        ],
    ],
    positive_guards: tuple[ObserverHabitGuardDTO, ...],
    compilation_recipe: ObserverHabitCompilationRecipeDTO,
) -> tuple[ObserverHabitGuardDTO, ...] | str:
    guards = []
    positive_keys = {item.feature_key for item in positive_guards}
    allowed = set(compilation_recipe.allowed_guard_feature_keys)
    forbidden = set(compilation_recipe.forbidden_guard_feature_keys)
    for counterexample in counterexamples:
        source, _ = source_target.get(counterexample.ledger_entry_id, (None, None))
        if source is None:
            return "counterexample_conflict"
        projected = _project_observation(source)
        guard = None
        for key in sorted(projected):
            if key in positive_keys or key not in allowed or key in forbidden:
                continue
            value = projected[key]
            guard = ObserverHabitGuardDTO.create(
                feature_key=key,
                operator="not_equals",
                expected_type=_type_name(value),
                expected_value=value,
                minimum_value=None,
                maximum_value=None,
                guard_role="counterexample",
                source_evidence_ids=(counterexample.counterexample_id,),
            )
            break
        if guard is None:
            return "counterexample_conflict"
        guards.append(guard)
    return tuple(sorted(guards, key=lambda item: item.habit_guard_id))


def _source_target_observations(
    *,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    observation_schema: ObserverObservationSchemaDTO,
) -> dict[
    str,
    tuple[ObserverObservationArtifactDTO | None, ObserverObservationArtifactDTO | None],
]:
    previous_target = None
    previous_effect = None
    result = {}
    for entry in entries:
        source = source_observation_for_entry(
            entry=entry,
            observation_schema=observation_schema,
            previous_target_observation=previous_target,
            previous_target_action_effect=previous_effect,
        )
        target = target_observation_for_entry(
            entry=entry, observation_schema=observation_schema
        )
        result[entry.ledger_entry_id] = (source, target)
        previous_target = target
        previous_effect = None if target is None else entry.executed_step.action_effect
    return result


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


def _evaluate_guard(
    *,
    guard: ObserverHabitGuardDTO,
    projected: Mapping[str, object],
    observation: ObserverObservationArtifactDTO,
) -> ObserverHabitGuardEvaluationDTO:
    if guard.operator == "is_missing":
        matched = guard.feature_key not in projected
        return _guard_eval(
            guard, observation, matched, None, None, "membership_checked"
        )
    if guard.operator == "is_present":
        matched = guard.feature_key in projected
        return _guard_eval(
            guard, observation, matched, None, None, "membership_checked"
        )
    if guard.feature_key not in projected:
        return _guard_eval(guard, observation, False, None, None, "feature_missing")
    value = projected[guard.feature_key]
    actual_type = _type_name(value)
    if actual_type != guard.expected_type and guard.expected_type != "number":
        return _guard_eval(
            guard, observation, None, actual_type, value, "type_mismatch"
        )
    if guard.operator == "equals":
        return _guard_eval(
            guard,
            observation,
            actual_type == guard.expected_type and value == guard.expected_value,
            actual_type,
            value,
            "exact_checked",
        )
    if guard.operator == "not_equals":
        if actual_type != guard.expected_type:
            return _guard_eval(
                guard, observation, None, actual_type, value, "type_mismatch"
            )
        return _guard_eval(
            guard,
            observation,
            value != guard.expected_value,
            actual_type,
            value,
            "exact_checked",
        )
    if guard.operator == "in_closed_range":
        if isinstance(value, bool) or not isinstance(value, int | float):
            return _guard_eval(
                guard, observation, None, actual_type, value, "type_mismatch"
            )
        if not math.isfinite(float(value)):
            return _guard_eval(
                guard, observation, None, actual_type, value, "invalid_number"
            )
        minimum_value = guard.minimum_value
        maximum_value = guard.maximum_value
        if minimum_value is None or maximum_value is None:
            return _guard_eval(
                guard, observation, None, actual_type, value, "invalid_range"
            )
        matched = float(minimum_value) <= float(value) <= float(maximum_value)
        return _guard_eval(
            guard, observation, matched, actual_type, value, "range_checked"
        )
    return _guard_eval(
        guard, observation, None, actual_type, value, "unsupported_guard"
    )


def _guard_eval(
    guard: ObserverHabitGuardDTO,
    observation: ObserverObservationArtifactDTO,
    matched: bool | None,
    actual_type: str | None,
    actual_value: object | None,
    reason_code: str,
) -> ObserverHabitGuardEvaluationDTO:
    return ObserverHabitGuardEvaluationDTO.create(
        habit_guard_id=guard.habit_guard_id,
        observation_artifact_id=observation.observation_artifact_id,
        status="invalid"
        if matched is None
        else "matched"
        if matched
        else "not_matched",
        actual_type=actual_type,
        actual_value=actual_value,
        reason_code=reason_code,
    )
