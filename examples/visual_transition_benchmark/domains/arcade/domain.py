"""Arcade domain wrapper: a thin adapter, not a reimplementation.

Every transition here is produced by the exact, unmodified stage-1 dataset
generator (``dataset.generate_episode``) and scored by the exact, unmodified
stage-1/stage-2 analyzers (``zeromodel_adapter.ArcadeBandZeroModelAnalyzer``,
``value_adapter.ValueAwareZeroModelAnalyzer``). This module only translates
between their result shapes and the domain-neutral protocol
(``domains/protocol.py``) -- see
``tests/test_cross_domain_arcade_regression.py`` for the proof that this
translation changes nothing observable.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from visual_transition_benchmark import dataset as ds
from visual_transition_benchmark import value_contracts as vc
from visual_transition_benchmark import value_metrics as vm
from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.domains.protocol import (
    AnalysisMetadata,
    ComponentAnalysisResult,
    ComponentSchema,
    DomainTransition,
    TransitionContract,
    ValueAnalysisResult,
)
from visual_transition_benchmark.value_adapter import ValueAwareZeroModelAnalyzer

_RELATION_VIOLATING_CATEGORIES = frozenset({"alien_disappears_without_hit", "unrelated_alien_change"})
_RELATION_NAME = "alien_substitution_without_cooldown_blocked"


def _sign(value: int) -> int:
    return -1 if value < 0 else (1 if value > 0 else 0)


def _contracts_for_action(action: str) -> Tuple[TransitionContract, ...]:
    contracts = []
    if action in ("LEFT", "RIGHT"):
        contracts.append(
            TransitionContract("tank-must-change", "tank", "presence_change", "tank must change on LEFT/RIGHT")
        )
    else:
        contracts.append(
            TransitionContract("tank-must-stay", "tank", "presence_stable", "tank must stay stable on STAY/FIRE")
        )
    contracts.append(
        TransitionContract("background-must-stay", "background", "presence_stable", "background never legitimately changes")
    )
    if action == "FIRE":
        contracts.append(
            TransitionContract("cooldown-must-change", "cooldown", "presence_change", "cooldown must change on FIRE")
        )
    contracts.append(TransitionContract("tank-direction", "tank", "direction", "tank delta sign must match the action"))
    contracts.append(TransitionContract("tank-magnitude", "tank", "magnitude", "tank delta must equal exactly one cell"))
    contracts.append(
        TransitionContract(
            "cooldown-value",
            "cooldown",
            "value",
            "FIRE always ends blocked; anything else always ends ready",
        )
    )
    contracts.append(
        TransitionContract(
            "alien-cooldown-relation",
            "alien",
            "relation",
            "an alien substitution can only coincide with a blocked cooldown",
        )
    )
    return tuple(contracts)


class _ComponentAnalyzerAdapter:
    def __init__(self) -> None:
        self._inner = zm.ArcadeBandZeroModelAnalyzer()

    def analyze(self, frame_before, frame_after, action, metadata: AnalysisMetadata) -> ComponentAnalysisResult:
        inner_metadata = zm.TransitionMetadata(transition_id=metadata.transition_id, step_number=metadata.step_number)
        result = self._inner.analyze(frame_before, frame_after, action, inner_metadata)
        return ComponentAnalysisResult(
            predicted_region_mask=result.predicted_region_mask,
            predicted_fields=result.predicted_fields,
            predicted_components=result.predicted_components,
            expected_components=result.expected_components,
            unexpected_components=result.unexpected_components,
            missing_components=result.missing_components,
            evidence_scores=result.evidence_scores,
            diagnostics=result.diagnostics,
        )


class _ValueAnalyzerAdapter:
    def __init__(self) -> None:
        self._inner = ValueAwareZeroModelAnalyzer()

    def analyze(self, frame_before, frame_after, action, metadata: AnalysisMetadata) -> ValueAnalysisResult:
        inner_metadata = zm.TransitionMetadata(transition_id=metadata.transition_id, step_number=metadata.step_number)
        analysis = self._inner.analyze(frame_before, frame_after, action, inner_metadata)
        values = analysis.values
        decoded = {
            "direction_decoded_sign": None if values.tank.delta_x is None else _sign(values.tank.delta_x),
            "magnitude_decoded_delta": values.tank.delta_x,
            "value_decoded_level": values.cooldown.after_level,
            "relation_decoded_satisfied": _RELATION_NAME not in analysis.verdict.relation_violations,
            "identity_decoded_id": None,
            "decoded_target_after": values.alien.after_x,
        }
        return ValueAnalysisResult(
            decoded=decoded,
            value_flags=analysis.value_flags,
            diagnostics={"conformance_status": analysis.component_analysis.diagnostics["conformance_status"]},
        )


class ArcadeTransitionDomain:
    name = "arcade"

    def generate_episode(self, *, seed: int, episode_id: str) -> Tuple[DomainTransition, ...]:
        records = ds.generate_episode(episode_id, seed)
        return tuple(self._to_domain_transition(record) for record in records)

    def render(self, state: ds.ArcadeState) -> np.ndarray:
        return ds.render(state)

    def component_schema(self) -> ComponentSchema:
        return ComponentSchema(
            domain_name=self.name,
            component_names=ds.COMPONENT_NAMES,
            canvas_shape=(zm.FRAME_HEIGHT, ds.WIDTH_PX),
        )

    def contracts_for_action(self, action: str) -> Tuple[TransitionContract, ...]:
        return _contracts_for_action(action)

    def build_component_analyzer(self) -> _ComponentAnalyzerAdapter:
        return _ComponentAnalyzerAdapter()

    def build_value_analyzer(self) -> _ValueAnalyzerAdapter:
        return _ValueAnalyzerAdapter()

    def _to_domain_transition(self, record: ds.TransitionRecord) -> DomainTransition:
        value_ground_truth = {
            "direction_expected_sign": _sign(vm.true_tank_delta(record)),
            "magnitude_expected_delta": vm.true_tank_delta(record),
            "value_expected_level": vm.true_cooldown_level(record),
            "relation_expected_satisfied": record.category not in _RELATION_VIOLATING_CATEGORIES,
            "true_target_after": vm.true_target_after(record),
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
