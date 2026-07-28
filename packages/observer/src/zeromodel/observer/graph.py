"""Canonical Observer observation graph DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from zeromodel.observer._canonical import canonical_id
from zeromodel.observer.grouping import (
    ObserverObservationGraphError,
    ObserverStateClassAssignmentDTO,
    ObserverStateClassDTO,
)

OBSERVER_OBSERVATION_GRAPH_NODE_VERSION: Final = "observer-observation-graph-node/1"
OBSERVER_STATE_TRANSITION_KEY_VERSION: Final = "observer-state-transition-key/1"
OBSERVER_OBSERVATION_GRAPH_EDGE_VERSION: Final = "observer-observation-graph-edge/1"
OBSERVER_OBSERVATION_GRAPH_VERSION: Final = "observer-observation-graph/1"
OBSERVER_OBSERVATION_GRAPH_BUILD_VERSION: Final = "observer-observation-graph-build/1"
OBSERVER_GRAPH_REBUILD_VERIFICATION_VERSION: Final = (
    "observer-graph-rebuild-verification/1"
)

GRAPH_BUILD_STATUSES: Final = frozenset({"built", "failed", "inconclusive"})
GRAPH_BUILD_FAILURE_CODES: Final = frozenset(
    {
        "ledger_integrity_failed",
        "ledger_semantic_replay_failed",
        "schema_mismatch",
        "source_observation_unavailable",
        "target_observation_unavailable",
        "assignment_rejected",
        "graph_invariant_failed",
    }
)
GRAPH_REBUILD_STATUSES: Final = frozenset({"verified", "failed", "inconclusive"})
GRAPH_REBUILD_FAILURE_CODES: Final = frozenset(
    {"graph_id_mismatch", "node_mismatch", "edge_mismatch", "assignment_mismatch"}
)


def _require_non_empty(value: str, field_name: str) -> None:
    if not value:
        raise ObserverObservationGraphError(f"{field_name} must be non-empty")


def _ensure_sorted_unique(values: tuple[str, ...], field_name: str) -> None:
    if values != tuple(sorted(set(values))):
        raise ObserverObservationGraphError(f"{field_name} must be unique and sorted")


def _ensure_unique(values: tuple[str, ...], field_name: str) -> None:
    if len(values) != len(set(values)):
        raise ObserverObservationGraphError(f"{field_name} must be unique")


@dataclass(frozen=True)
class ObserverObservationGraphNodeDTO:
    graph_node_id: str
    state_class_id: str
    grouping_recipe_id: str
    observation_schema_id: str
    observation_artifact_ids: tuple[str, ...]
    assignment_ids: tuple[str, ...]
    supporting_ledger_entry_ids: tuple[str, ...]
    first_ledger_sequence: int
    last_ledger_sequence: int
    visit_count: int
    version: str = OBSERVER_OBSERVATION_GRAPH_NODE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_OBSERVATION_GRAPH_NODE_VERSION:
            raise ObserverObservationGraphError("unsupported graph node version")
        for field_name in (
            "state_class_id",
            "grouping_recipe_id",
            "observation_schema_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_sorted_unique(self.observation_artifact_ids, "observation_artifact_ids")
        _ensure_sorted_unique(self.assignment_ids, "assignment_ids")
        _ensure_sorted_unique(
            self.supporting_ledger_entry_ids, "supporting_ledger_entry_ids"
        )
        if self.visit_count != len(self.observation_artifact_ids):
            raise ObserverObservationGraphError(
                "visit_count must match observation membership"
            )
        if self.visit_count and self.first_ledger_sequence > self.last_ledger_sequence:
            raise ObserverObservationGraphError("node ledger sequence range is invalid")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.graph_node_id != expected_id:
            raise ObserverObservationGraphError(
                "graph_node_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "assignment_ids": list(self.assignment_ids),
            "first_ledger_sequence": self.first_ledger_sequence,
            "grouping_recipe_id": self.grouping_recipe_id,
            "last_ledger_sequence": self.last_ledger_sequence,
            "observation_artifact_ids": list(self.observation_artifact_ids),
            "observation_schema_id": self.observation_schema_id,
            "state_class_id": self.state_class_id,
            "supporting_ledger_entry_ids": list(self.supporting_ledger_entry_ids),
            "version": self.version,
            "visit_count": self.visit_count,
        }
        if include_id:
            payload["graph_node_id"] = self.graph_node_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        state_class_id: str,
        grouping_recipe_id: str,
        observation_schema_id: str,
        observation_artifact_ids: tuple[str, ...],
        assignment_ids: tuple[str, ...],
        supporting_ledger_entry_ids: tuple[str, ...],
        first_ledger_sequence: int,
        last_ledger_sequence: int,
    ) -> "ObserverObservationGraphNodeDTO":
        observation_artifact_ids = tuple(sorted(set(observation_artifact_ids)))
        assignment_ids = tuple(sorted(set(assignment_ids)))
        supporting_ledger_entry_ids = tuple(sorted(set(supporting_ledger_entry_ids)))
        payload = {
            "assignment_ids": list(assignment_ids),
            "first_ledger_sequence": first_ledger_sequence,
            "grouping_recipe_id": grouping_recipe_id,
            "last_ledger_sequence": last_ledger_sequence,
            "observation_artifact_ids": list(observation_artifact_ids),
            "observation_schema_id": observation_schema_id,
            "state_class_id": state_class_id,
            "supporting_ledger_entry_ids": list(supporting_ledger_entry_ids),
            "version": OBSERVER_OBSERVATION_GRAPH_NODE_VERSION,
            "visit_count": len(observation_artifact_ids),
        }
        return cls(
            graph_node_id=canonical_id(payload),
            state_class_id=state_class_id,
            grouping_recipe_id=grouping_recipe_id,
            observation_schema_id=observation_schema_id,
            observation_artifact_ids=observation_artifact_ids,
            assignment_ids=assignment_ids,
            supporting_ledger_entry_ids=supporting_ledger_entry_ids,
            first_ledger_sequence=first_ledger_sequence,
            last_ledger_sequence=last_ledger_sequence,
            visit_count=len(observation_artifact_ids),
        )


@dataclass(frozen=True)
class ObserverStateTransitionKeyDTO:
    transition_key_id: str
    grouping_recipe_id: str
    source_state_class_id: str
    action: str
    target_state_class_id: str
    version: str = OBSERVER_STATE_TRANSITION_KEY_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_STATE_TRANSITION_KEY_VERSION:
            raise ObserverObservationGraphError("unsupported transition key version")
        for field_name in (
            "grouping_recipe_id",
            "source_state_class_id",
            "action",
            "target_state_class_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.transition_key_id != expected_id:
            raise ObserverObservationGraphError(
                "transition_key_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload = {
            "action": self.action,
            "grouping_recipe_id": self.grouping_recipe_id,
            "source_state_class_id": self.source_state_class_id,
            "target_state_class_id": self.target_state_class_id,
            "version": self.version,
        }
        if include_id:
            payload["transition_key_id"] = self.transition_key_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        grouping_recipe_id: str,
        source_state_class_id: str,
        action: str,
        target_state_class_id: str,
    ) -> "ObserverStateTransitionKeyDTO":
        payload = {
            "action": action,
            "grouping_recipe_id": grouping_recipe_id,
            "source_state_class_id": source_state_class_id,
            "target_state_class_id": target_state_class_id,
            "version": OBSERVER_STATE_TRANSITION_KEY_VERSION,
        }
        return cls(
            transition_key_id=canonical_id(payload),
            grouping_recipe_id=grouping_recipe_id,
            source_state_class_id=source_state_class_id,
            action=action,
            target_state_class_id=target_state_class_id,
        )


@dataclass(frozen=True)
class ObserverObservationGraphEdgeDTO:
    graph_edge_id: str
    transition_key: ObserverStateTransitionKeyDTO
    supporting_ledger_entry_ids: tuple[str, ...]
    transition_verification_ids: tuple[str, ...]
    comparison_result_ids: tuple[str, ...]
    traversal_count: int
    confirmed_count: int
    contradicted_count: int
    inconclusive_count: int
    predictor_rule_set_ids: tuple[str, ...]
    environment_rule_set_ids: tuple[str, ...]
    version: str = OBSERVER_OBSERVATION_GRAPH_EDGE_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_OBSERVATION_GRAPH_EDGE_VERSION:
            raise ObserverObservationGraphError("unsupported graph edge version")
        for field_name in (
            "supporting_ledger_entry_ids",
            "transition_verification_ids",
            "comparison_result_ids",
            "predictor_rule_set_ids",
            "environment_rule_set_ids",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        if self.traversal_count != len(self.supporting_ledger_entry_ids):
            raise ObserverObservationGraphError("traversal_count must match support")
        if (
            self.confirmed_count + self.contradicted_count + self.inconclusive_count
            != self.traversal_count
        ):
            raise ObserverObservationGraphError("status counts must sum to traversal")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.graph_edge_id != expected_id:
            raise ObserverObservationGraphError(
                "graph_edge_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "comparison_result_ids": list(self.comparison_result_ids),
            "confirmed_count": self.confirmed_count,
            "contradicted_count": self.contradicted_count,
            "environment_rule_set_ids": list(self.environment_rule_set_ids),
            "inconclusive_count": self.inconclusive_count,
            "predictor_rule_set_ids": list(self.predictor_rule_set_ids),
            "supporting_ledger_entry_ids": list(self.supporting_ledger_entry_ids),
            "transition_key": self.transition_key.canonical_payload(),
            "transition_verification_ids": list(self.transition_verification_ids),
            "traversal_count": self.traversal_count,
            "version": self.version,
        }
        if include_id:
            payload["graph_edge_id"] = self.graph_edge_id
        return payload

    @classmethod
    def create(
        cls,
        *,
        transition_key: ObserverStateTransitionKeyDTO,
        supporting_ledger_entry_ids: tuple[str, ...],
        transition_verification_ids: tuple[str, ...],
        comparison_result_ids: tuple[str, ...],
        confirmed_count: int,
        contradicted_count: int,
        inconclusive_count: int,
        predictor_rule_set_ids: tuple[str, ...],
        environment_rule_set_ids: tuple[str, ...],
    ) -> "ObserverObservationGraphEdgeDTO":
        supporting_ledger_entry_ids = tuple(sorted(set(supporting_ledger_entry_ids)))
        transition_verification_ids = tuple(sorted(set(transition_verification_ids)))
        comparison_result_ids = tuple(sorted(set(comparison_result_ids)))
        predictor_rule_set_ids = tuple(sorted(set(predictor_rule_set_ids)))
        environment_rule_set_ids = tuple(sorted(set(environment_rule_set_ids)))
        payload = {
            "comparison_result_ids": list(comparison_result_ids),
            "confirmed_count": confirmed_count,
            "contradicted_count": contradicted_count,
            "environment_rule_set_ids": list(environment_rule_set_ids),
            "inconclusive_count": inconclusive_count,
            "predictor_rule_set_ids": list(predictor_rule_set_ids),
            "supporting_ledger_entry_ids": list(supporting_ledger_entry_ids),
            "transition_key": transition_key.canonical_payload(),
            "transition_verification_ids": list(transition_verification_ids),
            "traversal_count": len(supporting_ledger_entry_ids),
            "version": OBSERVER_OBSERVATION_GRAPH_EDGE_VERSION,
        }
        return cls(
            graph_edge_id=canonical_id(payload),
            transition_key=transition_key,
            supporting_ledger_entry_ids=supporting_ledger_entry_ids,
            transition_verification_ids=transition_verification_ids,
            comparison_result_ids=comparison_result_ids,
            traversal_count=len(supporting_ledger_entry_ids),
            confirmed_count=confirmed_count,
            contradicted_count=contradicted_count,
            inconclusive_count=inconclusive_count,
            predictor_rule_set_ids=predictor_rule_set_ids,
            environment_rule_set_ids=environment_rule_set_ids,
        )


@dataclass(frozen=True)
class ObserverObservationGraphDTO:
    observation_graph_id: str
    ledger_snapshot_id: str
    grouping_recipe_id: str
    observation_schema_id: str
    node_ids: tuple[str, ...]
    edge_ids: tuple[str, ...]
    nodes: tuple[ObserverObservationGraphNodeDTO, ...]
    edges: tuple[ObserverObservationGraphEdgeDTO, ...]
    assignment_ids: tuple[str, ...]
    rejected_ledger_entry_ids: tuple[str, ...]
    version: str = OBSERVER_OBSERVATION_GRAPH_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_OBSERVATION_GRAPH_VERSION:
            raise ObserverObservationGraphError("unsupported graph version")
        for field_name in (
            "ledger_snapshot_id",
            "grouping_recipe_id",
            "observation_schema_id",
        ):
            _require_non_empty(getattr(self, field_name), field_name)
        _ensure_unique(self.node_ids, "node_ids")
        _ensure_unique(self.edge_ids, "edge_ids")
        _ensure_sorted_unique(self.assignment_ids, "assignment_ids")
        _ensure_sorted_unique(
            self.rejected_ledger_entry_ids, "rejected_ledger_entry_ids"
        )
        if self.node_ids != tuple(node.graph_node_id for node in self.nodes):
            raise ObserverObservationGraphError("node_ids must match nodes")
        if self.edge_ids != tuple(edge.graph_edge_id for edge in self.edges):
            raise ObserverObservationGraphError("edge_ids must match edges")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.observation_graph_id != expected_id:
            raise ObserverObservationGraphError(
                "observation_graph_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "assignment_ids": list(self.assignment_ids),
            "edge_ids": list(self.edge_ids),
            "edges": [item.canonical_payload() for item in self.edges],
            "grouping_recipe_id": self.grouping_recipe_id,
            "ledger_snapshot_id": self.ledger_snapshot_id,
            "node_ids": list(self.node_ids),
            "nodes": [item.canonical_payload() for item in self.nodes],
            "observation_schema_id": self.observation_schema_id,
            "rejected_ledger_entry_ids": list(self.rejected_ledger_entry_ids),
            "version": self.version,
        }
        if include_id:
            payload["observation_graph_id"] = self.observation_graph_id
        return payload


@dataclass(frozen=True)
class ObserverObservationGraphBuildDTO:
    graph_build_id: str
    ledger_snapshot_id: str
    ledger_integrity_result_id: str
    ledger_semantic_replay_result_id: str
    grouping_recipe_id: str
    assignments: tuple[ObserverStateClassAssignmentDTO, ...]
    state_classes: tuple[ObserverStateClassDTO, ...]
    graph: ObserverObservationGraphDTO
    status: str
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_OBSERVATION_GRAPH_BUILD_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_OBSERVATION_GRAPH_BUILD_VERSION:
            raise ObserverObservationGraphError("unsupported graph build version")
        if self.status not in GRAPH_BUILD_STATUSES:
            raise ObserverObservationGraphError("unsupported graph build status")
        _ensure_sorted_unique(self.failure_codes, "failure_codes")
        if set(self.failure_codes) - GRAPH_BUILD_FAILURE_CODES:
            raise ObserverObservationGraphError("unsupported graph build failure code")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.graph_build_id != expected_id:
            raise ObserverObservationGraphError(
                "graph_build_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "assignments": [item.canonical_payload() for item in self.assignments],
            "failure_codes": list(self.failure_codes),
            "graph": self.graph.canonical_payload(),
            "grouping_recipe_id": self.grouping_recipe_id,
            "ledger_integrity_result_id": self.ledger_integrity_result_id,
            "ledger_semantic_replay_result_id": self.ledger_semantic_replay_result_id,
            "ledger_snapshot_id": self.ledger_snapshot_id,
            "state_classes": [item.canonical_payload() for item in self.state_classes],
            "status": self.status,
            "version": self.version,
        }
        if include_id:
            payload["graph_build_id"] = self.graph_build_id
        return payload


@dataclass(frozen=True)
class ObserverGraphRebuildVerificationDTO:
    graph_rebuild_verification_id: str
    expected_graph_id: str
    rebuilt_graph_id: str
    status: str
    mismatched_node_ids: tuple[str, ...]
    mismatched_edge_ids: tuple[str, ...]
    mismatched_assignment_ids: tuple[str, ...]
    failure_codes: tuple[str, ...]
    version: str = OBSERVER_GRAPH_REBUILD_VERIFICATION_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVER_GRAPH_REBUILD_VERIFICATION_VERSION:
            raise ObserverObservationGraphError("unsupported rebuild verification")
        if self.status not in GRAPH_REBUILD_STATUSES:
            raise ObserverObservationGraphError("unsupported rebuild status")
        for field_name in (
            "mismatched_node_ids",
            "mismatched_edge_ids",
            "mismatched_assignment_ids",
            "failure_codes",
        ):
            _ensure_sorted_unique(getattr(self, field_name), field_name)
        if set(self.failure_codes) - GRAPH_REBUILD_FAILURE_CODES:
            raise ObserverObservationGraphError("unsupported rebuild failure code")
        expected_id = canonical_id(self.canonical_payload(include_id=False))
        if self.graph_rebuild_verification_id != expected_id:
            raise ObserverObservationGraphError(
                "graph_rebuild_verification_id disagrees with canonical payload"
            )

    def canonical_payload(self, *, include_id: bool = True) -> Mapping[str, object]:
        payload: dict[str, object] = {
            "expected_graph_id": self.expected_graph_id,
            "failure_codes": list(self.failure_codes),
            "mismatched_assignment_ids": list(self.mismatched_assignment_ids),
            "mismatched_edge_ids": list(self.mismatched_edge_ids),
            "mismatched_node_ids": list(self.mismatched_node_ids),
            "rebuilt_graph_id": self.rebuilt_graph_id,
            "status": self.status,
            "version": self.version,
        }
        if include_id:
            payload["graph_rebuild_verification_id"] = (
                self.graph_rebuild_verification_id
            )
        return payload
