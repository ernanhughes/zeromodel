"""Build rebuildable Observer observation graphs from transition ledgers."""

from __future__ import annotations

from collections import defaultdict
from typing import Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.artifacts import (
    ObserverObservationArtifactDTO,
    ObserverObservationSchemaDTO,
)
from zeromodel.observer.comparison import ObserverComparisonRecipeDTO
from zeromodel.observer.fixture import (
    ObserverFixtureRuleSetDTO,
    ObserverFixtureStateDTO,
)
from zeromodel.observer.graph import (
    OBSERVER_GRAPH_REBUILD_VERIFICATION_VERSION,
    OBSERVER_OBSERVATION_GRAPH_BUILD_VERSION,
    OBSERVER_OBSERVATION_GRAPH_VERSION,
    ObserverGraphRebuildVerificationDTO,
    ObserverObservationGraphBuildDTO,
    ObserverObservationGraphDTO,
    ObserverObservationGraphEdgeDTO,
    ObserverObservationGraphNodeDTO,
    ObserverStateTransitionKeyDTO,
)
from zeromodel.observer.grouping import (
    ObserverStateClassAssignmentDTO,
    ObserverStateClassDTO,
    ObserverStateGroupingRecipeDTO,
    assign_observation_to_state_class,
)
from zeromodel.observer.ledger import (
    ObserverTransitionLedgerEntryDTO,
    ObserverTransitionLedgerSnapshotDTO,
    replay_observer_fixture_ledger,
    verify_observer_transition_ledger_integrity,
)


def build_observer_observation_graph(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    comparison_recipe: ObserverComparisonRecipeDTO,
    predictor_rule_sets: tuple[ObserverFixtureRuleSetDTO, ...],
    environment_rule_sets: tuple[ObserverFixtureRuleSetDTO, ...],
) -> ObserverObservationGraphBuildDTO:
    """Build a disposable graph projection from immutable ledger entries."""

    predictor_map = {item.fixture_rule_set_id: item for item in predictor_rule_sets}
    environment_map = {item.fixture_rule_set_id: item for item in environment_rule_sets}
    integrity = verify_observer_transition_ledger_integrity(
        ledger_snapshot=ledger_snapshot, entries=entries
    )
    semantic = replay_observer_fixture_ledger(
        ledger_snapshot=ledger_snapshot,
        entries=entries,
        observation_schema=observation_schema,
        comparison_recipe=comparison_recipe,
        predictor_rule_sets=predictor_map,
        environment_rule_sets=environment_map,
    )
    failure_codes: set[str] = set()
    if integrity.status != "verified":
        failure_codes.add("ledger_integrity_failed")
    if semantic.status != "verified":
        failure_codes.add("ledger_semantic_replay_failed")
    if grouping_recipe.observation_schema_id != observation_schema.schema_id:
        failure_codes.add("schema_mismatch")

    assignments_by_observation: dict[str, ObserverStateClassAssignmentDTO] = {}
    observation_support: dict[str, set[str]] = defaultdict(set)
    observation_sequences: dict[str, list[int]] = defaultdict(list)
    source_observations: dict[str, ObserverObservationArtifactDTO] = {}
    target_observations: dict[str, ObserverObservationArtifactDTO] = {}
    rejected_entries: set[str] = set()
    previous_target_observation: ObserverObservationArtifactDTO | None = None

    for entry in entries:
        source = _source_observation_for_entry(
            entry=entry,
            observation_schema=observation_schema,
            previous_target_observation=previous_target_observation,
        )
        target = _target_observation_for_entry(
            entry=entry, observation_schema=observation_schema
        )
        if source is None:
            failure_codes.add("source_observation_unavailable")
            rejected_entries.add(entry.ledger_entry_id)
            previous_target_observation = target
            continue
        if target is None:
            failure_codes.add("target_observation_unavailable")
            rejected_entries.add(entry.ledger_entry_id)
            previous_target_observation = target
            continue
        source_observations[entry.ledger_entry_id] = source
        target_observations[entry.ledger_entry_id] = target
        for observation in (source, target):
            assignment = assignments_by_observation.get(
                observation.observation_artifact_id
            )
            if assignment is None:
                assignment = assign_observation_to_state_class(
                    observation=observation,
                    grouping_recipe=grouping_recipe,
                    observation_schema=observation_schema,
                )
                assignments_by_observation[observation.observation_artifact_id] = (
                    assignment
                )
            observation_support[observation.observation_artifact_id].add(
                entry.ledger_entry_id
            )
            observation_sequences[observation.observation_artifact_id].append(
                entry.ledger_sequence
            )
            if assignment.status == "rejected":
                failure_codes.add("assignment_rejected")
                rejected_entries.add(entry.ledger_entry_id)
        previous_target_observation = target

    assignments = tuple(
        sorted(
            assignments_by_observation.values(),
            key=lambda item: item.observation_artifact_id,
        )
    )
    state_classes = _state_classes_from_assignments(
        assignments=assignments,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
    )
    nodes = _build_nodes(
        assignments=assignments,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
        observation_support=observation_support,
        observation_sequences=observation_sequences,
    )
    edges = _build_edges(
        entries=entries,
        source_observations=source_observations,
        target_observations=target_observations,
        assignments_by_observation=assignments_by_observation,
        grouping_recipe=grouping_recipe,
        rejected_entries=rejected_entries,
    )
    graph = _build_graph(
        ledger_snapshot=ledger_snapshot,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
        nodes=nodes,
        edges=edges,
        assignments=assignments,
        rejected_ledger_entry_ids=tuple(sorted(rejected_entries)),
    )
    status = "built" if not failure_codes else "failed"
    payload = {
        "assignments": [item.canonical_payload() for item in assignments],
        "failure_codes": sorted(failure_codes),
        "graph": graph.canonical_payload(),
        "grouping_recipe_id": grouping_recipe.grouping_recipe_id,
        "ledger_integrity_result_id": integrity.ledger_replay_result_id,
        "ledger_semantic_replay_result_id": semantic.ledger_replay_result_id,
        "ledger_snapshot_id": ledger_snapshot.ledger_snapshot_id,
        "state_classes": [item.canonical_payload() for item in state_classes],
        "status": status,
        "version": OBSERVER_OBSERVATION_GRAPH_BUILD_VERSION,
    }
    return ObserverObservationGraphBuildDTO(
        graph_build_id=canonical_id(payload),
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        ledger_integrity_result_id=integrity.ledger_replay_result_id,
        ledger_semantic_replay_result_id=semantic.ledger_replay_result_id,
        grouping_recipe_id=grouping_recipe.grouping_recipe_id,
        assignments=assignments,
        state_classes=state_classes,
        graph=graph,
        status=status,
        failure_codes=tuple(sorted(failure_codes)),
    )


def verify_observer_graph_rebuild(
    *,
    expected_graph: ObserverObservationGraphDTO,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    comparison_recipe: ObserverComparisonRecipeDTO,
    predictor_rule_sets: tuple[ObserverFixtureRuleSetDTO, ...],
    environment_rule_sets: tuple[ObserverFixtureRuleSetDTO, ...],
) -> ObserverGraphRebuildVerificationDTO:
    rebuilt = build_observer_observation_graph(
        ledger_snapshot=ledger_snapshot,
        entries=entries,
        grouping_recipe=grouping_recipe,
        observation_schema=observation_schema,
        comparison_recipe=comparison_recipe,
        predictor_rule_sets=predictor_rule_sets,
        environment_rule_sets=environment_rule_sets,
    ).graph
    failures: set[str] = set()
    if expected_graph.observation_graph_id != rebuilt.observation_graph_id:
        failures.add("graph_id_mismatch")
    mismatched_nodes = _symmetric_difference(expected_graph.node_ids, rebuilt.node_ids)
    mismatched_edges = _symmetric_difference(expected_graph.edge_ids, rebuilt.edge_ids)
    mismatched_assignments = _symmetric_difference(
        expected_graph.assignment_ids, rebuilt.assignment_ids
    )
    if mismatched_nodes:
        failures.add("node_mismatch")
    if mismatched_edges:
        failures.add("edge_mismatch")
    if mismatched_assignments:
        failures.add("assignment_mismatch")
    status = "verified" if not failures else "failed"
    payload = {
        "expected_graph_id": expected_graph.observation_graph_id,
        "failure_codes": sorted(failures),
        "mismatched_assignment_ids": list(mismatched_assignments),
        "mismatched_edge_ids": list(mismatched_edges),
        "mismatched_node_ids": list(mismatched_nodes),
        "rebuilt_graph_id": rebuilt.observation_graph_id,
        "status": status,
        "version": OBSERVER_GRAPH_REBUILD_VERIFICATION_VERSION,
    }
    return ObserverGraphRebuildVerificationDTO(
        graph_rebuild_verification_id=canonical_id(payload),
        expected_graph_id=expected_graph.observation_graph_id,
        rebuilt_graph_id=rebuilt.observation_graph_id,
        status=status,
        mismatched_node_ids=mismatched_nodes,
        mismatched_edge_ids=mismatched_edges,
        mismatched_assignment_ids=mismatched_assignments,
        failure_codes=tuple(sorted(failures)),
    )


def _source_observation_for_entry(
    *,
    entry: ObserverTransitionLedgerEntryDTO,
    observation_schema: ObserverObservationSchemaDTO,
    previous_target_observation: ObserverObservationArtifactDTO | None,
) -> ObserverObservationArtifactDTO | None:
    if previous_target_observation is not None:
        if previous_target_observation.sequence_index != entry.source_state.step_index:
            return None
        return previous_target_observation
    return _observation_for_fixture_state(
        state=entry.source_state,
        action_effect="initial",
        observation_schema=observation_schema,
    )


def _target_observation_for_entry(
    *,
    entry: ObserverTransitionLedgerEntryDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverObservationArtifactDTO | None:
    target = _observation_for_fixture_state(
        state=entry.executed_step.actual_state,
        action_effect=entry.executed_step.action_effect,
        observation_schema=observation_schema,
    )
    if target.observation_artifact_id != entry.executed_step.actual_observation_id:
        return None
    return target


def _observation_for_fixture_state(
    *,
    state: ObserverFixtureStateDTO,
    action_effect: str,
    observation_schema: ObserverObservationSchemaDTO,
) -> ObserverObservationArtifactDTO:
    history = {}
    if state.previous_action is not None:
        history["previous_action"] = state.previous_action
    return ObserverObservationArtifactDTO.create(
        observation_schema=observation_schema,
        visible_state_features={
            "action_effect": action_effect,
            "agent_x": state.agent_x,
            "target_x": state.target_x,
        },
        recent_history_features=history,
        hidden_state_uncertainty={"cooldown_remaining": state.cooldown_remaining},
        provenance={"fixture_id": state.fixture_id, "rule_set_id": state.rule_set_id},
        sequence_index=state.step_index,
    )


def _state_classes_from_assignments(
    *,
    assignments: tuple[ObserverStateClassAssignmentDTO, ...],
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
) -> tuple[ObserverStateClassDTO, ...]:
    classes: dict[str, ObserverStateClassDTO] = {}
    for assignment in assignments:
        if assignment.status != "assigned" or assignment.state_class_id is None:
            continue
        state_class = ObserverStateClassDTO.create(
            grouping_recipe_id=grouping_recipe.grouping_recipe_id,
            observation_schema_id=observation_schema.schema_id,
            state_class_key=assignment.state_class_key,
        )
        classes[state_class.state_class_id] = state_class
    return tuple(sorted(classes.values(), key=lambda item: item.state_class_id))


def _build_nodes(
    *,
    assignments: tuple[ObserverStateClassAssignmentDTO, ...],
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    observation_support: Mapping[str, set[str]],
    observation_sequences: Mapping[str, list[int]],
) -> tuple[ObserverObservationGraphNodeDTO, ...]:
    by_class: dict[str, list[ObserverStateClassAssignmentDTO]] = defaultdict(list)
    for assignment in assignments:
        if assignment.status == "assigned" and assignment.state_class_id is not None:
            by_class[assignment.state_class_id].append(assignment)
    nodes = []
    for state_class_id, members in by_class.items():
        observation_ids = tuple(item.observation_artifact_id for item in members)
        sequences = [
            sequence
            for observation_id in observation_ids
            for sequence in observation_sequences[observation_id]
        ]
        nodes.append(
            ObserverObservationGraphNodeDTO.create(
                state_class_id=state_class_id,
                grouping_recipe_id=grouping_recipe.grouping_recipe_id,
                observation_schema_id=observation_schema.schema_id,
                observation_artifact_ids=observation_ids,
                assignment_ids=tuple(item.assignment_id for item in members),
                supporting_ledger_entry_ids=tuple(
                    ledger_id
                    for observation_id in observation_ids
                    for ledger_id in observation_support[observation_id]
                ),
                first_ledger_sequence=min(sequences) if sequences else 0,
                last_ledger_sequence=max(sequences) if sequences else 0,
            )
        )
    return tuple(sorted(nodes, key=lambda item: item.state_class_id))


def _build_edges(
    *,
    entries: tuple[ObserverTransitionLedgerEntryDTO, ...],
    source_observations: Mapping[str, ObserverObservationArtifactDTO],
    target_observations: Mapping[str, ObserverObservationArtifactDTO],
    assignments_by_observation: Mapping[str, ObserverStateClassAssignmentDTO],
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    rejected_entries: set[str],
) -> tuple[ObserverObservationGraphEdgeDTO, ...]:
    grouped: dict[str, list[ObserverTransitionLedgerEntryDTO]] = defaultdict(list)
    keys: dict[str, ObserverStateTransitionKeyDTO] = {}
    for entry in entries:
        if entry.ledger_entry_id in rejected_entries:
            continue
        source = source_observations.get(entry.ledger_entry_id)
        target = target_observations.get(entry.ledger_entry_id)
        if source is None or target is None:
            continue
        source_assignment = assignments_by_observation[source.observation_artifact_id]
        target_assignment = assignments_by_observation[target.observation_artifact_id]
        if (
            source_assignment.state_class_id is None
            or target_assignment.state_class_id is None
        ):
            continue
        key = ObserverStateTransitionKeyDTO.create(
            grouping_recipe_id=grouping_recipe.grouping_recipe_id,
            source_state_class_id=source_assignment.state_class_id,
            action=entry.transition_verification.transition_record.action,
            target_state_class_id=target_assignment.state_class_id,
        )
        keys[key.transition_key_id] = key
        grouped[key.transition_key_id].append(entry)
    edges = []
    for key_id, edge_entries in grouped.items():
        counts = {
            "confirmed": 0,
            "contradicted": 0,
            "inconclusive": 0,
        }
        for entry in edge_entries:
            counts[entry.transition_verification.verification_status] += 1
        edges.append(
            ObserverObservationGraphEdgeDTO.create(
                transition_key=keys[key_id],
                supporting_ledger_entry_ids=tuple(
                    entry.ledger_entry_id for entry in edge_entries
                ),
                transition_verification_ids=tuple(
                    entry.transition_verification.verification_id
                    for entry in edge_entries
                ),
                comparison_result_ids=tuple(
                    entry.transition_verification.comparison_result.comparison_result_id
                    for entry in edge_entries
                ),
                confirmed_count=counts["confirmed"],
                contradicted_count=counts["contradicted"],
                inconclusive_count=counts["inconclusive"],
                predictor_rule_set_ids=tuple(
                    entry.predictor_rule_set_id for entry in edge_entries
                ),
                environment_rule_set_ids=tuple(
                    entry.environment_rule_set_id for entry in edge_entries
                ),
            )
        )
    return tuple(
        sorted(
            edges,
            key=lambda item: (
                item.transition_key.source_state_class_id,
                item.transition_key.action,
                item.transition_key.target_state_class_id,
            ),
        )
    )


def _build_graph(
    *,
    ledger_snapshot: ObserverTransitionLedgerSnapshotDTO,
    grouping_recipe: ObserverStateGroupingRecipeDTO,
    observation_schema: ObserverObservationSchemaDTO,
    nodes: tuple[ObserverObservationGraphNodeDTO, ...],
    edges: tuple[ObserverObservationGraphEdgeDTO, ...],
    assignments: tuple[ObserverStateClassAssignmentDTO, ...],
    rejected_ledger_entry_ids: tuple[str, ...],
) -> ObserverObservationGraphDTO:
    node_ids = tuple(node.graph_node_id for node in nodes)
    edge_ids = tuple(edge.graph_edge_id for edge in edges)
    assignment_ids = tuple(sorted(item.assignment_id for item in assignments))
    payload = {
        "assignment_ids": list(assignment_ids),
        "edge_ids": list(edge_ids),
        "edges": [item.canonical_payload() for item in edges],
        "grouping_recipe_id": grouping_recipe.grouping_recipe_id,
        "ledger_snapshot_id": ledger_snapshot.ledger_snapshot_id,
        "node_ids": list(node_ids),
        "nodes": [item.canonical_payload() for item in nodes],
        "observation_schema_id": observation_schema.schema_id,
        "rejected_ledger_entry_ids": list(rejected_ledger_entry_ids),
        "version": OBSERVER_OBSERVATION_GRAPH_VERSION,
    }
    return ObserverObservationGraphDTO(
        observation_graph_id=canonical_id(payload),
        ledger_snapshot_id=ledger_snapshot.ledger_snapshot_id,
        grouping_recipe_id=grouping_recipe.grouping_recipe_id,
        observation_schema_id=observation_schema.schema_id,
        node_ids=node_ids,
        edge_ids=edge_ids,
        nodes=nodes,
        edges=edges,
        assignment_ids=assignment_ids,
        rejected_ledger_entry_ids=rejected_ledger_entry_ids,
    )


def _symmetric_difference(
    left: tuple[str, ...], right: tuple[str, ...]
) -> tuple[str, ...]:
    return tuple(sorted(set(left).symmetric_difference(right)))
