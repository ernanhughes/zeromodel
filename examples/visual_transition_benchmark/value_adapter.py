"""System D: value-aware ZeroModel.

Layers value_contracts.py's decoded values and contract verdicts on top of
System C's existing component-level result (``zeromodel_adapter.TransitionAnalysis``,
unchanged, reused as-is). Both layers are kept side by side in
``ValueTransitionAnalysis`` on purpose -- the whole point of this stage is
that a correct component-level verdict can coexist with a wrong value, so the
two must never be collapsed into a single pass/fail bit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from visual_transition_benchmark import value_contracts as vc
from visual_transition_benchmark.zeromodel_adapter import (
    ArcadeBandZeroModelAnalyzer,
    TransitionAnalysis,
    TransitionMetadata,
)


@dataclass(frozen=True)
class ValueTransitionAnalysis:
    component_analysis: TransitionAnalysis
    values: vc.DecodedValues
    verdict: vc.ValueContractVerdict
    value_flags: Tuple[str, ...]


def _value_flags(verdict: vc.ValueContractVerdict) -> Tuple[str, ...]:
    flags = []
    if verdict.tank_direction_ok is False:
        flags.append("tank_direction_violation")
    if verdict.tank_magnitude_ok is False:
        flags.append("tank_magnitude_violation")
    if not verdict.cooldown_value_ok:
        flags.append("cooldown_value_violation")
    flags.extend(f"relation:{name}" for name in verdict.relation_violations)
    return tuple(flags)


class ValueAwareZeroModelAnalyzer:
    """System D. ``analyze()`` mirrors System C's signature exactly (same
    non-privileged inputs: frame_before, frame_after, action, metadata)."""

    def __init__(self) -> None:
        self._component_analyzer = ArcadeBandZeroModelAnalyzer()

    def analyze(
        self,
        frame_before: np.ndarray,
        frame_after: np.ndarray,
        action: str,
        metadata: TransitionMetadata,
    ) -> ValueTransitionAnalysis:
        component_analysis = self._component_analyzer.analyze(frame_before, frame_after, action, metadata)
        transition_evidence = vc.build_value_transition_evidence(frame_before, frame_after)
        values = vc.decode_values(transition_evidence)
        verdict = vc.evaluate_contracts(action, values)
        return ValueTransitionAnalysis(
            component_analysis=component_analysis,
            values=values,
            verdict=verdict,
            value_flags=_value_flags(verdict),
        )
