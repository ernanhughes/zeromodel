"""Derive bounded promotion-candidate evidence from Observer graphs."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping

from zeromodel.observer.artifacts import (
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)
from zeromodel.observer._observation_replay import (
    source_observation_for_entry,
    target_observation_for_entry,
)
from zeromodel.observer.graph import (
    ObserverObservationGraphBuildDTO,
    ObserverObservationGraphDTO,
    ObserverObservationGraphEdgeDTO,
    ObserverStateTransitionKeyDTO,
)
from zeromodel.observer.grouping import (
    ObserverStateClassAssignmentDTO,
    ObserverStateGroupingRecipeDTO,
)
from zeromodel.observer.ledger import (
    ObserverTransitionLedgerEntryDTO,
    ObserverTransitionLedgerSnapshotDTO,
)
from zeromodel.observer.promotion import (
    ObserverEvidenceIndependenceDTO,
    ObserverNoveltyEvidenceDTO,
    ObserverPromotionAnalysisDTO,
    ObserverPromotionCandidateDTO,
    ObserverPromotionEvidenceRecipeDTO,
    ObserverRuleChangeSurvivalDTO,
    ObserverRuleChangeTestDTO,
    ObserverRuleRegimeDTO,
    ObserverTransitionOccurrenceDTO,
    ObserverTransitionRecurrenceDTO,
    ObserverTransitionStabilityDTO,
)


@dataclass(frozen=True)
class _ObservationEvent:
    observation_artifact_id: str
    state_class_id: str
    ledger_entry_id: str
    ledger_sequence: int
    role_order: int


def analyze_observer_promotion_candidates(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    graph_build: ObserverObservationGraphBuildDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    promotion_recipe: ObserverPromotionEvidenceRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    rule_change_tests: tuple[ObserverRuleChangeTestDTO, ...] = (),
) -> ObserverPromotionAnalysisDTO:
    """Build deterministic novelty, recurrence, stability, and candidate evidence."""

    graph = graph_build.graph
    failure_codes = _initial_failure_codes(
        ledger_snapshot=ledger_snapshot,
        graph_build=graph_build,
        graph=graph,
        grouping_recipe=grouping_recipe,
        promotion_recipe=promotion_recipe,
        observation_schema=observation_schema,
        rule_change_tests=rule_change_tests,
    )
    if failure_codes or graph is None:
        return _failed_analysis(
            ledger_snapshot=ledger_snapshot,
            graph_id=graph.observation_graph_id if graph is not None else "",
            grouping_recipe_id=grouping_recipe.grouping_recipe_id,
            promotion_recipe_id=promotion_recipe.promotion_recipe_id,
            failure_codes=failure_codes or ("missing_graph",),
        )

    entries_by_id = {entry.ledger_entry_id: entry for entry in entries}
    assignments_by_observation = {
        assignment.observation_artifact_id: assignment
        for assignment in graph_build.assignments
        if assignment.status == "assigned" and assignment.state_class_id is not None
    }
    source_target = _source_target_observations(
        entries=entries,
        observation_schema=observation_schema,
    )
    novelty = _build_novelty_evidence(
        ledger_snapshot=ledger_snapshot,
        graph=graph,
        source_target=source_target,
        assignments_by_observation=assignments_by_observation,
        entries_by_id=entries_by_id,
    )
    regimes: dict[str, ObserverRuleRegimeDTO] = {}
    occurrences: list[ObserverTransitionOccurrenceDTO] = []
    seen_ledger_ids: set[str] = set()
    failures: set[str] = set()
    for edge in graph.edges:
        edge_occurrences = []
        for ledger_entry_id in edge.supporting_ledger_entry_ids:
            if ledger_entry_id not in entries_by_id:
                failures.add("edge_support_missing")
                continue
            if ledger_entry_id in seen_ledger_ids:
                failures.add("duplicate_occurrence")
                continue
            entry = entries_by_id[ledger_entry_id]
            source, target = source_target.get(ledger_entry_id, (None, None))
            if source is None or target is None:
                failures.add("edge_occurrence_mismatch")
                continue
            source_assignment = assignments_by_observation.get(
                source.observation_artifact_id
            )
            target_assignment = assignments_by_observation.get(
                target.observation_artifact_id
            )
            if source_assignment is None or target_assignment is None:
                failures.add("edge_occurrence_mismatch")
                continue
            key = ObserverStateTransitionKeyDTO.create(
                grouping_recipe_id=grouping_recipe.grouping_recipe_id,
                source_state_class_id=source_assignment.state_class_id or "",
                action=edge.transition_key.action,
                target_state_class_id=target_assignment.state_class_id or "",
            )
            if key.transition_key_id != edge.transition_key.transition_key_id:
                failures.add("transition_key_mismatch")
                continue
            regime = ObserverRuleRegimeDTO.create(
                predictor_rule_set_id=entry.predictor_rule_set_id,
                environment_rule_set_id=entry.environment_rule_set_id,
            )
            regimes[regime.rule_regime_id] = regime
            occurrence = ObserverTransitionOccurrenceDTO.create(
                transition_key_id=edge.transition_key.transition_key_id,
                graph_edge_id=edge.graph_edge_id,
                ledger_entry_id=entry.ledger_entry_id,
                ledger_sequence=entry.ledger_sequence,
                episode_id=entry.episode_id,
                source_state_class_id=source_assignment.state_class_id,
                source_observation_artifact_id=source.observation_artifact_id,
                target_state_class_id=target_assignment.state_class_id,
                target_observation_artifact_id=target.observation_artifact_id,
                action=edge.transition_key.action,
                verification_status=entry.transition_verification.verification_status,
                comparison_result_id=(
                    entry.transition_verification.comparison_result.comparison_result_id
                ),
                predictor_rule_set_id=entry.predictor_rule_set_id,
                environment_rule_set_id=entry.environment_rule_set_id,
                rule_regime_id=regime.rule_regime_id,
            )
            seen_ledger_ids.add(ledger_entry_id)
            edge_occurrences.append(occurrence)
            occurrences.append(occurrence)
        if not _edge_counts_match(edge, edge_occurrences):
            failures.add("edge_count_mismatch")
    if failures:
        return _failed_analysis(
            ledger_snapshot=ledger_snapshot,
            graph_id=graph.observation_graph_id,
            grouping_recipe_id=grouping_recipe.grouping_recipe_id,
            promotion_recipe_id=promotion_recipe.promotion_recipe_id,
            failure_codes=tuple(sorted(failures)),
        )

    by_key: dict[str, list[ObserverTransitionOccurrenceDTO]] = defaultdict(list)
    for occurrence in occurrences:
        by_key[occurrence.transition_key_id].append(occurrence)
    recurrences = tuple(
        _build_recurrence(graph=graph, transition_key_id=key, occurrences=items)
        for key, items in sorted(by_key.items())
    )
    recurrence_by_key = {item.transition_key_id: item for item in recurrences}
    stabilities = tuple(
        _build_stability(
            recurrence=recurrence,
            occurrences=by_key[recurrence.transition_key_id],
            recipe=promotion_recipe,
        )
        for recurrence in recurrences
    )
    independence_results = tuple(
        _build_independence(
            recurrence=recurrence,
            occurrences=by_key[recurrence.transition_key_id],
            recipe=promotion_recipe,
        )
        for recurrence in recurrences
    )
    rule_change_test = _selected_rule_change_test(
        promotion_recipe=promotion_recipe,
        rule_change_tests=rule_change_tests,
    )
    rule_change_results = tuple(
        _build_rule_change(
            transition_key_id=key,
            occurrences=items,
            rule_change_test=rule_change_test,
        )
        for key, items in sorted(by_key.items())
    )
    stability_by_key = {item.transition_key_id: item for item in stabilities}
    independence_by_key = {
        item.transition_key_id: item for item in independence_results
    }
    rule_change_by_key = {item.transition_key_id: item for item in rule_change_results}
    edge_by_key = {edge.transition_key.transition_key_id: edge for edge in graph.edges}
    candidates = tuple(
        _build_candidate(
            ledger_snapshot=ledger_snapshot,
            graph=graph,
            recipe=promotion_recipe,
            edge=edge_by_key[key],
            recurrence=recurrence_by_key[key],
            stability=stability_by_key[key],
            independence=independence_by_key[key],
            rule_change=rule_change_by_key[key],
        )
        for key in sorted(by_key)
    )
    eligible_ids = tuple(
        sorted(
            item.promotion_candidate_id
            for item in candidates
            if item.eligible_for_compilation
        )
    )
    rejected_ids = tuple(
        sorted(
            item.promotion_candidate_id
            for item in candidates
            if not item.eligible_for_compilation
        )
    )
    return ObserverPromotionAnalysisDTO.create(
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        observation_graph_id=graph.observation_graph_id,
        grouping_recipe_id=grouping_recipe.grouping_recipe_id,
        promotion_recipe_id=promotion_recipe.promotion_recipe_id,
        novelty_evidence=tuple(
            sorted(novelty, key=lambda item: item.novelty_evidence_id)
        ),
        occurrences=tuple(sorted(occurrences, key=lambda item: item.occurrence_id)),
        recurrences=recurrences,
        stabilities=stabilities,
        independence_results=independence_results,
        rule_change_results=rule_change_results,
        promotion_candidates=candidates,
        eligible_candidate_ids=eligible_ids,
        rejected_candidate_ids=rejected_ids,
        status="built",
        failure_codes=(),
    )


def _initial_failure_codes(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    graph_build: ObserverObservationGraphBuildDTO,
    graph: ObserverObservationGraphDTO | None,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    promotion_recipe: ObserverPromotionEvidenceRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    rule_change_tests: tuple[ObserverRuleChangeTestDTO, ...],
) -> tuple[str, ...]:
    failures: set[str] = set()
    if graph_build.status != "built":
        failures.add("failed_graph_build")
    if graph is None:
        failures.add("missing_graph")
    else:
        if graph.ledger_snapshot_id != ledger_snapshot.ledger_snapshot_id:
            failures.add("graph_ledger_mismatch")
        if graph.grouping_recipe_id != grouping_recipe.grouping_recipe_id:
            failures.add("graph_grouping_mismatch")
        if graph.observation_schema_id != observation_schema.schema_id:
            failures.add("schema_mismatch")
        if grouping_recipe.observation_schema_id != observation_schema.schema_id:
            failures.add("schema_mismatch")
        if promotion_recipe.observation_graph_id != graph.observation_graph_id:
            failures.add("graph_grouping_mismatch")
        if promotion_recipe.grouping_recipe_id != grouping_recipe.grouping_recipe_id:
            failures.add("graph_grouping_mismatch")
    if promotion_recipe.rule_change_test_id is not None and not any(
        item.rule_change_test_id == promotion_recipe.rule_change_test_id
        for item in rule_change_tests
    ):
        failures.add("rule_change_test_missing")
    return tuple(sorted(failures))


def _failed_analysis(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    graph_id: str,
    grouping_recipe_id: str,
    promotion_recipe_id: str,
    failure_codes: tuple[str, ...],
) -> ObserverPromotionAnalysisDTO:
    return ObserverPromotionAnalysisDTO.create(
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        observation_graph_id=graph_id,
        grouping_recipe_id=grouping_recipe_id,
        promotion_recipe_id=promotion_recipe_id,
        novelty_evidence=(),
        occurrences=(),
        recurrences=(),
        stabilities=(),
        independence_results=(),
        rule_change_results=(),
        promotion_candidates=(),
        eligible_candidate_ids=(),
        rejected_candidate_ids=(),
        status="failed",
        failure_codes=tuple(sorted(set(failure_codes))),
    )


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
        previous_effect = (
            entry.executed_step.action_effect if target is not None else None
        )
    return result


def _build_novelty_evidence(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    graph: ObserverObservationGraphDTO,
    source_target: Mapping[
        str,
        tuple[
            ObserverObservationArtifactDTO | None,
            ObserverObservationArtifactDTO | None,
        ],
    ],
    assignments_by_observation: Mapping[str, ObserverStateClassAssignmentDTO],
    entries_by_id: Mapping[str, ObserverTransitionLedgerEntryDTO],
) -> list[ObserverNoveltyEvidenceDTO]:
    events: dict[str, _ObservationEvent] = {}
    for ledger_entry_id, pair in source_target.items():
        for role, observation in (("source", pair[0]), ("target", pair[1])):
            if observation is None:
                continue
            entry = entries_by_id[ledger_entry_id]
            assignment = assignments_by_observation.get(
                observation.observation_artifact_id
            )
            if assignment is None or assignment.state_class_id is None:
                continue
            current = _ObservationEvent(
                observation_artifact_id=observation.observation_artifact_id,
                state_class_id=assignment.state_class_id,
                ledger_entry_id=ledger_entry_id,
                ledger_sequence=entry.ledger_sequence,
                role_order=0 if role == "source" else 1,
            )
            existing = events.get(observation.observation_artifact_id)
            if existing is None or (current.ledger_sequence, current.role_order) < (
                existing.ledger_sequence,
                existing.role_order,
            ):
                events[observation.observation_artifact_id] = current
    by_order = sorted(
        events.values(),
        key=lambda item: (
            item.ledger_sequence,
            item.role_order,
            item.observation_artifact_id,
        ),
    )
    seen_by_class: dict[str, int] = defaultdict(int)
    novelty = []
    for event in by_order:
        prior = seen_by_class[event.state_class_id]
        novelty.append(
            ObserverNoveltyEvidenceDTO.create(
                ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
                observation_graph_id=graph.observation_graph_id,
                state_class_id=event.state_class_id,
                first_observation_artifact_id=event.observation_artifact_id,
                first_ledger_entry_id=event.ledger_entry_id,
                first_ledger_sequence=event.ledger_sequence,
                previously_observed=prior > 0,
                prior_observation_count=prior,
                novelty_status="recurrent" if prior > 0 else "novel",
            )
        )
        seen_by_class[event.state_class_id] += 1
    return novelty


def _edge_counts_match(
    edge: ObserverObservationGraphEdgeDTO,
    occurrences: Iterable[ObserverTransitionOccurrenceDTO],
) -> bool:
    items = tuple(occurrences)
    return (
        edge.traversal_count == len(items)
        and edge.confirmed_count
        == sum(1 for item in items if item.verification_status == "confirmed")
        and edge.contradicted_count
        == sum(1 for item in items if item.verification_status == "contradicted")
        and edge.inconclusive_count
        == sum(1 for item in items if item.verification_status == "inconclusive")
    )


def _build_recurrence(
    *,
    graph: ObserverObservationGraphDTO,
    transition_key_id: str,
    occurrences: list[ObserverTransitionOccurrenceDTO],
) -> ObserverTransitionRecurrenceDTO:
    ordered = sorted(occurrences, key=lambda item: item.ledger_sequence)
    return ObserverTransitionRecurrenceDTO.create(
        transition_key_id=transition_key_id,
        observation_graph_id=graph.observation_graph_id,
        occurrence_ids=tuple(sorted(item.occurrence_id for item in ordered)),
        supporting_ledger_entry_ids=tuple(
            sorted(item.ledger_entry_id for item in ordered)
        ),
        episode_ids=tuple(sorted({item.episode_id for item in ordered})),
        source_observation_artifact_ids=tuple(
            sorted({item.source_observation_artifact_id for item in ordered})
        ),
        target_observation_artifact_ids=tuple(
            sorted({item.target_observation_artifact_id for item in ordered})
        ),
        predictor_rule_set_ids=tuple(
            sorted({item.predictor_rule_set_id for item in ordered})
        ),
        environment_rule_set_ids=tuple(
            sorted({item.environment_rule_set_id for item in ordered})
        ),
        rule_regime_ids=tuple(sorted({item.rule_regime_id for item in ordered})),
        traversal_count=len(ordered),
        independent_episode_count=len({item.episode_id for item in ordered}),
        distinct_source_observation_count=len(
            {item.source_observation_artifact_id for item in ordered}
        ),
        distinct_target_observation_count=len(
            {item.target_observation_artifact_id for item in ordered}
        ),
        distinct_rule_regime_count=len({item.rule_regime_id for item in ordered}),
        first_ledger_sequence=min(item.ledger_sequence for item in ordered),
        last_ledger_sequence=max(item.ledger_sequence for item in ordered),
    )


def _build_stability(
    *,
    recurrence: ObserverTransitionRecurrenceDTO,
    occurrences: list[ObserverTransitionOccurrenceDTO],
    recipe: ObserverPromotionEvidenceRecipeDTO,
) -> ObserverTransitionStabilityDTO:
    confirmed = tuple(
        sorted(
            item.occurrence_id
            for item in occurrences
            if item.verification_status == "confirmed"
        )
    )
    contradicted = tuple(
        sorted(
            item.occurrence_id
            for item in occurrences
            if item.verification_status == "contradicted"
        )
    )
    inconclusive = tuple(
        sorted(
            item.occurrence_id
            for item in occurrences
            if item.verification_status == "inconclusive"
        )
    )
    evaluated = len(confirmed) + len(contradicted)
    reasons: set[str] = set()
    reasons.add(
        "minimum_traversals_met"
        if recurrence.traversal_count >= recipe.minimum_traversal_count
        else "minimum_traversals_not_met"
    )
    reasons.add(
        "minimum_confirmations_met"
        if len(confirmed) >= recipe.minimum_confirmed_count
        else "minimum_confirmations_not_met"
    )
    reasons.add(
        "contradiction_limit_met"
        if len(contradicted) <= recipe.maximum_contradicted_count
        else "contradiction_limit_exceeded"
    )
    reasons.add(
        "inconclusive_limit_met"
        if len(inconclusive) <= recipe.maximum_inconclusive_count
        else "inconclusive_limit_exceeded"
    )
    if evaluated == 0:
        reasons.add("no_evaluated_evidence")
        ratio_ok = False
    else:
        ratio_ok = (
            len(confirmed) * recipe.minimum_confirmation_ratio_denominator
            >= recipe.minimum_confirmation_ratio_numerator * evaluated
        )
        reasons.add(
            "confirmation_ratio_met" if ratio_ok else "confirmation_ratio_not_met"
        )
    insufficient = (
        recurrence.traversal_count < recipe.minimum_traversal_count
        or len(confirmed) < recipe.minimum_confirmed_count
        or evaluated == 0
    )
    unstable = len(contradicted) > recipe.maximum_contradicted_count or not ratio_ok
    if insufficient:
        status = "insufficient_evidence"
    elif unstable:
        status = "unstable"
    elif len(inconclusive) > recipe.maximum_inconclusive_count:
        status = "mixed"
    else:
        status = "stable"
    return ObserverTransitionStabilityDTO.create(
        transition_key_id=recurrence.transition_key_id,
        recurrence_id=recurrence.recurrence_id,
        confirmed_occurrence_ids=confirmed,
        contradicted_occurrence_ids=contradicted,
        inconclusive_occurrence_ids=inconclusive,
        confirmed_count=len(confirmed),
        contradicted_count=len(contradicted),
        inconclusive_count=len(inconclusive),
        evaluated_count=evaluated,
        confirmation_ratio_numerator=len(confirmed),
        confirmation_ratio_denominator=evaluated,
        status=status,
        reason_codes=tuple(sorted(reasons)),
    )


def _build_independence(
    *,
    recurrence: ObserverTransitionRecurrenceDTO,
    occurrences: list[ObserverTransitionOccurrenceDTO],
    recipe: ObserverPromotionEvidenceRecipeDTO,
) -> ObserverEvidenceIndependenceDTO:
    source_classes = tuple(sorted({item.source_state_class_id for item in occurrences}))
    reasons: set[str] = set()
    reasons.add(
        "enough_episodes"
        if recurrence.independent_episode_count
        >= recipe.minimum_independent_episode_count
        else "not_enough_episodes"
    )
    reasons.add(
        "enough_source_observations"
        if recurrence.distinct_source_observation_count
        >= recipe.minimum_distinct_source_state_count
        else "not_enough_source_observations"
    )
    reasons.add(
        "enough_rule_regimes"
        if recurrence.distinct_rule_regime_count
        >= recipe.minimum_distinct_rule_regime_count
        else "not_enough_rule_regimes"
    )
    status = (
        "sufficient"
        if all(not item.startswith("not_") for item in reasons)
        else "insufficient"
    )
    return ObserverEvidenceIndependenceDTO.create(
        transition_key_id=recurrence.transition_key_id,
        episode_ids=recurrence.episode_ids,
        source_observation_artifact_ids=recurrence.source_observation_artifact_ids,
        source_state_class_ids=source_classes,
        rule_regime_ids=recurrence.rule_regime_ids,
        independent_episode_count=recurrence.independent_episode_count,
        distinct_source_observation_count=recurrence.distinct_source_observation_count,
        distinct_rule_regime_count=recurrence.distinct_rule_regime_count,
        status=status,
        reason_codes=tuple(sorted(reasons)),
    )


def _selected_rule_change_test(
    *,
    promotion_recipe: ObserverPromotionEvidenceRecipeDTO,
    rule_change_tests: tuple[ObserverRuleChangeTestDTO, ...],
) -> ObserverRuleChangeTestDTO | None:
    if promotion_recipe.rule_change_test_id is None:
        return None
    for item in rule_change_tests:
        if item.rule_change_test_id == promotion_recipe.rule_change_test_id:
            return item
    return None


def _build_rule_change(
    *,
    transition_key_id: str,
    occurrences: list[ObserverTransitionOccurrenceDTO],
    rule_change_test: ObserverRuleChangeTestDTO | None,
) -> ObserverRuleChangeSurvivalDTO:
    if rule_change_test is None:
        return ObserverRuleChangeSurvivalDTO.create(
            transition_key_id=transition_key_id,
            pre_change_occurrence_ids=(),
            post_change_occurrence_ids=(),
            post_change_confirmed_occurrence_ids=(),
            post_change_contradicted_occurrence_ids=(),
            survived_rule_change=False,
            status="not_tested",
            reason_codes=("rule_change_not_observed",),
        )
    pre = tuple(
        sorted(
            item.occurrence_id
            for item in occurrences
            if item.ledger_sequence < rule_change_test.change_start_ledger_sequence
            and item.environment_rule_set_id
            == rule_change_test.baseline_environment_rule_set_id
            and (
                rule_change_test.predictor_rule_set_id is None
                or item.predictor_rule_set_id == rule_change_test.predictor_rule_set_id
            )
        )
    )
    post_items = [
        item
        for item in occurrences
        if item.ledger_sequence >= rule_change_test.change_start_ledger_sequence
        and item.environment_rule_set_id
        == rule_change_test.changed_environment_rule_set_id
        and (
            rule_change_test.predictor_rule_set_id is None
            or item.predictor_rule_set_id == rule_change_test.predictor_rule_set_id
        )
    ]
    post = tuple(sorted(item.occurrence_id for item in post_items))
    post_confirmed = tuple(
        sorted(
            item.occurrence_id
            for item in post_items
            if item.verification_status == "confirmed"
        )
    )
    post_contradicted = tuple(
        sorted(
            item.occurrence_id
            for item in post_items
            if item.verification_status == "contradicted"
        )
    )
    reasons: set[str] = set()
    if pre:
        reasons.add("pre_change_observed")
    if post:
        reasons.add("post_change_observed")
    else:
        reasons.add("rule_change_not_observed")
    if post_confirmed:
        reasons.add("post_change_confirmed")
    else:
        reasons.add("post_change_confirmation_missing")
    if post_contradicted:
        reasons.add("post_change_contradicted")
    if not post:
        status = "not_tested"
    elif post_confirmed and not post_contradicted:
        status = "survived"
    else:
        status = "failed"
    return ObserverRuleChangeSurvivalDTO.create(
        transition_key_id=transition_key_id,
        pre_change_occurrence_ids=pre,
        post_change_occurrence_ids=post,
        post_change_confirmed_occurrence_ids=post_confirmed,
        post_change_contradicted_occurrence_ids=post_contradicted,
        survived_rule_change=status == "survived",
        status=status,
        reason_codes=tuple(sorted(reasons)),
    )


def _build_candidate(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    graph: ObserverObservationGraphDTO,
    recipe: ObserverPromotionEvidenceRecipeDTO,
    edge: ObserverObservationGraphEdgeDTO,
    recurrence: ObserverTransitionRecurrenceDTO,
    stability: ObserverTransitionStabilityDTO,
    independence: ObserverEvidenceIndependenceDTO,
    rule_change: ObserverRuleChangeSurvivalDTO,
) -> ObserverPromotionCandidateDTO:
    reasons: set[str] = set()
    for reason in stability.reason_codes:
        if reason in {
            "minimum_traversals_met",
            "minimum_traversals_not_met",
            "minimum_confirmations_met",
            "minimum_confirmations_not_met",
            "contradiction_limit_met",
            "contradiction_limit_exceeded",
            "inconclusive_limit_met",
            "inconclusive_limit_exceeded",
            "confirmation_ratio_met",
            "confirmation_ratio_not_met",
        }:
            reasons.add(reason)
    reasons.add(
        "episode_independence_met"
        if recurrence.independent_episode_count
        >= recipe.minimum_independent_episode_count
        else "episode_independence_not_met"
    )
    reasons.add(
        "source_diversity_met"
        if recurrence.distinct_source_observation_count
        >= recipe.minimum_distinct_source_state_count
        else "source_diversity_not_met"
    )
    reasons.add(
        "rule_regime_requirement_met"
        if recurrence.distinct_rule_regime_count
        >= recipe.minimum_distinct_rule_regime_count
        else "rule_regime_requirement_not_met"
    )
    rule_change_ok = (
        not recipe.require_post_rule_change_confirmation
        or rule_change.status == "survived"
    )
    reasons.add(
        "rule_change_survival_met" if rule_change_ok else "rule_change_survival_not_met"
    )
    reasons.add(
        "stability_met" if stability.status == "stable" else "stability_not_met"
    )
    if stability.status == "insufficient_evidence":
        disposition = "insufficient_evidence"
    elif stability.contradicted_count > recipe.maximum_contradicted_count:
        disposition = "contradicted"
    elif stability.status == "unstable":
        disposition = "unstable"
    elif independence.status != "sufficient":
        disposition = "not_independent"
    elif not rule_change_ok:
        disposition = "not_rule_change_tested"
    elif stability.status == "stable":
        disposition = "eligible"
    else:
        disposition = "unsupported"
    return ObserverPromotionCandidateDTO.create(
        promotion_recipe_id=recipe.promotion_recipe_id,
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        observation_graph_id=graph.observation_graph_id,
        transition_key_id=edge.transition_key.transition_key_id,
        graph_edge_id=edge.graph_edge_id,
        recurrence_id=recurrence.recurrence_id,
        stability_id=stability.stability_id,
        independence_id=independence.independence_id,
        rule_change_survival_id=rule_change.rule_change_survival_id,
        disposition=disposition,
        eligible_for_compilation=disposition == "eligible",
        reason_codes=tuple(sorted(reasons)),
        supporting_occurrence_ids=recurrence.occurrence_ids,
        supporting_ledger_entry_ids=recurrence.supporting_ledger_entry_ids,
    )
