"""WarehouseTransitionDomain: implements the same domain-neutral protocol
(``domains/protocol.py``) as ``ArcadeTransitionDomain``, using no arcade
component names and no arcade code."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from visual_transition_benchmark.domains.protocol import (
    ComponentSchema,
    DomainTransition,
    TransitionContract,
)
from visual_transition_benchmark.domains.warehouse import contracts as wc
from visual_transition_benchmark.domains.warehouse import dataset as wds
from visual_transition_benchmark.domains.warehouse import faults as wf
from visual_transition_benchmark.domains.warehouse import model as wm

_RELATION_FAULT_WITH_ADJACENCY_VIOLATION = frozenset(
    {
        "crate_moves_without_robot_adjacency",
        "two_crates_move_during_single_push",
        "expected_crate_remains_while_another_moves",
    }
)


def _sign(value: int) -> int:
    return -1 if value < 0 else (1 if value > 0 else 0)


def _contracts_for_action(action: str) -> Tuple[TransitionContract, ...]:
    expected = wc._EXPECTED_CHANGE_COMPONENTS.get(action, ())
    contracts = []
    for name in ("robot", "crate", "door"):
        kind = "presence_change" if name in expected else "presence_stable"
        contracts.append(
            TransitionContract(
                f"{name}-{kind}", name, kind, f"{name} {kind} on {action}"
            )
        )
    contracts.append(
        TransitionContract(
            "wall-stable", "wall", "presence_stable", "wall never legitimately changes"
        )
    )
    contracts.append(
        TransitionContract(
            "robot-direction", "robot", "direction", "robot delta must match the action"
        )
    )
    contracts.append(
        TransitionContract(
            "robot-magnitude",
            "robot",
            "magnitude",
            "robot delta must equal exactly one cell",
        )
    )
    contracts.append(
        TransitionContract(
            "battery-value",
            "battery",
            "value",
            "battery decrements by exactly one on a successful move/push",
        )
    )
    contracts.append(
        TransitionContract(
            "door-value",
            "door",
            "value",
            "door must read as open after OPEN_DOOR, closed otherwise",
        )
    )
    contracts.append(
        TransitionContract(
            "crate-adjacency-relation",
            "crate",
            "relation",
            "a crate change must coincide with robot adjacency",
        )
    )
    return tuple(contracts)


class WarehouseTransitionDomain:
    name = "warehouse"

    def generate_episode(
        self, *, seed: int, episode_id: str
    ) -> Tuple[DomainTransition, ...]:
        records = wf.generate_episode(episode_id, seed)
        return tuple(self._to_domain_transition(record) for record in records)

    def render(self, state: wm.WarehouseState) -> np.ndarray:
        return wds.render(state)

    def component_schema(self) -> ComponentSchema:
        return ComponentSchema(
            domain_name=self.name,
            component_names=wds.COMPONENT_NAMES,
            canvas_shape=wc.CANVAS_SHAPE,
        )

    def contracts_for_action(self, action: str) -> Tuple[TransitionContract, ...]:
        return _contracts_for_action(action)

    def build_component_analyzer(self) -> wc.WarehouseComponentAnalyzer:
        return wc.WarehouseComponentAnalyzer()

    def build_value_analyzer(self) -> wc.WarehouseValueAnalyzer:
        return wc.WarehouseValueAnalyzer()

    def _to_domain_transition(
        self, record: wds.WarehouseTransitionRecord
    ) -> DomainTransition:
        true_robot_delta = (
            record.state_after["robot"][0] - record.state_before["robot"][0],
            record.state_after["robot"][1] - record.state_before["robot"][1],
        )
        true_battery_after = record.state_after["battery"]
        true_door_state = "open" if record.state_after["door_open"] else "closed"

        relation_expected_satisfied = (
            record.category not in _RELATION_FAULT_WITH_ADJACENCY_VIOLATION
        )
        true_new_cell = _true_new_crate_target(record)
        identity_expected_id = (
            _true_identity_at(record, true_new_cell)
            if true_new_cell is not None
            else None
        )

        value_ground_truth = {
            "direction_expected_sign": tuple(_sign(v) for v in true_robot_delta),
            "magnitude_expected_delta": true_robot_delta,
            "value_expected_level": true_battery_after,
            "door_expected_level": true_door_state,
            "relation_expected_satisfied": relation_expected_satisfied,
            "true_target_after": true_new_cell,
            "identity_expected_id": identity_expected_id,
        }
        return DomainTransition(
            transition_id=record.transition_id,
            domain_name=self.name,
            episode_id=record.episode_id,
            step_number=record.step_number,
            seed=record.seed,
            action=record.action,
            category=record.category,
            frame_before=record.frame_before,
            frame_after=record.frame_after,
            expected_changed_components=record.expected_changed_components,
            observed_changed_components=record.observed_changed_components,
            fault_type=record.fault_type,
            is_faulty=record.is_faulty,
            expected_contracts=self.contracts_for_action(record.action),
            value_ground_truth=value_ground_truth,
            notes=record.notes,
        )


def _true_new_crate_target(record: wds.WarehouseTransitionRecord):
    before_crates = set(tuple(c) for c in record.state_before["crates"])
    after_crates = set(tuple(c) for c in record.state_after["crates"])
    new_cells = after_crates - before_crates
    if len(new_cells) == 1:
        return next(iter(new_cells))
    return None


def _true_identity_at(record: wds.WarehouseTransitionRecord, cell) -> Optional[int]:
    for index, position in enumerate(record.state_after["crates"]):
        if tuple(position) == cell:
            return index
    return None
