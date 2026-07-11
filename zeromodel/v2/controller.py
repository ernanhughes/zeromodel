"""Trend-aware VPM controller distilled from Stephanie.

The controller is a consumer of metric rows. It does not mutate VPM artifacts;
it turns metric history into explicit workflow signals.
"""
from __future__ import annotations

import statistics as stats
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Mapping, Optional


class Signal(Enum):
    EDIT = auto()
    RESAMPLE = auto()
    ESCALATE = auto()
    STOP = auto()
    SPINOFF = auto()
    HOLD = auto()


@dataclass(frozen=True)
class Thresholds:
    mins: Dict[str, float]
    stop_margin: float = 0.02
    edit_margin: float = 0.01


@dataclass(frozen=True)
class Policy:
    window: int = 5
    patience: int = 3
    edit_margin: float = 0.05
    escalate_after: int = 2
    cooldown_steps: int = 1
    max_regressions: int = 2
    max_steps: int = 50
    spinoff_dim: str = "novelty"
    stickiness_dim: str = "stickiness"
    spinoff_novelty_min: float = 0.75
    spinoff_stickiness_max: float = 0.45
    local_gap_dims: tuple[str, ...] = ("citation_support", "entity_consistency", "lint_clean", "type_safe")
    zscore_clip_dims: tuple[str, ...] = ("coverage", "coherence", "correctness", "tests_pass_rate")
    zscore_clip_sigma: float = 3.5


@dataclass
class VPMRow:
    unit: str
    kind: str
    dims: Dict[str, float]
    timestamp: float = field(default_factory=time.time)
    step_idx: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Decision:
    signal: Signal
    reason: str
    params: Dict[str, Any] = field(default_factory=dict)
    snapshot: Dict[str, Any] = field(default_factory=dict)


class VPMController:
    def __init__(
        self,
        thresholds_code: Thresholds,
        thresholds_text: Thresholds,
        policy: Policy = Policy(),
        *,
        bandit_choose: Optional[Callable[[List[str]], str]] = None,
        bandit_update: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.thresholds_code = thresholds_code
        self.thresholds_text = thresholds_text
        self.policy = policy
        self.bandit_choose = bandit_choose
        self.bandit_update = bandit_update
        self.history: Dict[str, List[VPMRow]] = {}
        self.resample_counts: Dict[str, int] = {}
        self.cooldown_until_step: Dict[str, int] = {}
        self.last_signal: Dict[str, Signal] = {}

    def add_vpm_row(self, metrics: Mapping[str, Any], unit: str, *, candidate_exemplars: Optional[List[str]] = None) -> Decision:
        kind = "code" if "tests_pass_rate" in metrics else "text"
        row = VPMRow(
            unit=unit,
            kind=kind,
            dims={k: float(v) for k, v in dict(metrics).items() if isinstance(v, (int, float))},
            step_idx=metrics.get("step_idx") if isinstance(metrics.get("step_idx"), int) else None,
            meta=dict(metrics),
        )
        return self.add(row, candidate_exemplars=candidate_exemplars)

    def add(self, row: VPMRow, *, candidate_exemplars: Optional[List[str]] = None) -> Decision:
        row = self._clipped(row)
        history = self.history.setdefault(row.unit, [])
        history.append(row)
        if len(history) > 100:
            self.history[row.unit] = history[-100:]
            history = self.history[row.unit]

        thresholds = self.thresholds_code if row.kind == "code" else self.thresholds_text
        window = history[-self.policy.window:]

        if row.meta.get("total_steps", row.step_idx or 0) >= self.policy.max_steps:
            return self._decision(row, Signal.STOP, "Max steps reached", {})
        if self._in_cooldown(row):
            return self._decision(row, Signal.HOLD, "Cooldown", {"until_step": self.cooldown_until_step[row.unit]})
        if self._stable_above(window, thresholds):
            return self._decision(row, Signal.STOP, "Stable above thresholds", {})
        if self._should_spinoff(row):
            return self._decision(row, Signal.SPINOFF, "High novelty with low stickiness", {
                "novelty": row.dims.get(self.policy.spinoff_dim),
                "stickiness": row.dims.get(self.policy.stickiness_dim),
            })
        if self._regressions(window) > self.policy.max_regressions:
            return self._resample(row, "Too many regressions", candidate_exemplars)

        gaps = self._gaps(row, thresholds)
        local_gaps = [gap for gap in gaps if gap in self.policy.local_gap_dims]
        if local_gaps:
            return self._decision(row, Signal.EDIT, "Local gaps", {"gaps": local_gaps})
        if self._stagnating(window, thresholds):
            return self._resample(row, "Stagnation on core dims", candidate_exemplars)
        if gaps and self.resample_counts.get(row.unit, 0) >= self.policy.escalate_after:
            self._set_cooldown(row)
            return self._decision(row, Signal.ESCALATE, "Global gaps after resamples", {"gaps": gaps})
        if gaps:
            return self._resample(row, "Below thresholds", candidate_exemplars)
        return self._decision(row, Signal.EDIT, "Default edit", {"gaps": gaps})

    def _decision(self, row: VPMRow, signal: Signal, reason: str, params: Dict[str, Any]) -> Decision:
        self.last_signal[row.unit] = signal
        exemplar = row.meta.get("exemplar_id")
        if exemplar and self.bandit_update and signal in (Signal.EDIT, Signal.STOP):
            self.bandit_update(str(exemplar), self._reward(row))
        return Decision(
            signal=signal,
            reason=reason,
            params=params,
            snapshot={"unit": row.unit, "kind": row.kind, "step_idx": row.step_idx, "dims": dict(row.dims)},
        )

    def _resample(self, row: VPMRow, why: str, candidates: Optional[List[str]]) -> Decision:
        self.resample_counts[row.unit] = self.resample_counts.get(row.unit, 0) + 1
        self._set_cooldown(row)
        params: Dict[str, Any] = {"why": why}
        if candidates and self.bandit_choose:
            params["exemplar_id"] = self.bandit_choose(candidates)
        return self._decision(row, Signal.RESAMPLE, why, params)

    def _set_cooldown(self, row: VPMRow) -> None:
        if row.step_idx is not None:
            self.cooldown_until_step[row.unit] = row.step_idx + self.policy.cooldown_steps

    def _in_cooldown(self, row: VPMRow) -> bool:
        return row.step_idx is not None and row.unit in self.cooldown_until_step and row.step_idx < self.cooldown_until_step[row.unit]

    def _stable_above(self, window: List[VPMRow], thresholds: Thresholds) -> bool:
        if len(window) < self.policy.patience:
            return False
        recent = window[-self.policy.patience:]
        for item in recent:
            for dim, minimum in thresholds.mins.items():
                if item.dims.get(dim, 0.0) < minimum + thresholds.stop_margin:
                    return False
        return True

    def _should_spinoff(self, row: VPMRow) -> bool:
        novelty = row.dims.get(self.policy.spinoff_dim)
        stickiness = row.dims.get(self.policy.stickiness_dim)
        if novelty is None or stickiness is None:
            return False
        return novelty >= self.policy.spinoff_novelty_min and stickiness <= self.policy.spinoff_stickiness_max

    def _gaps(self, row: VPMRow, thresholds: Thresholds) -> List[str]:
        return [
            dim for dim, minimum in thresholds.mins.items()
            if row.dims.get(dim, 1.0) < minimum - self.policy.edit_margin
        ]

    def _regressions(self, window: List[VPMRow]) -> int:
        if len(window) < 2:
            return 0
        regressions = 0
        dims = set(window[-1].dims)
        for prev, cur in zip(window, window[1:]):
            dips = sum(1 for dim in dims if cur.dims.get(dim, 0.0) < prev.dims.get(dim, 0.0) - 1e-6)
            regressions += 1 if dips >= max(1, len(dims) // 4) else 0
        return regressions

    def _stagnating(self, window: List[VPMRow], thresholds: Thresholds) -> bool:
        if len(window) < self.policy.patience + 1:
            return False
        recent = window[-(self.policy.patience + 1):]
        for dim in thresholds.mins:
            if dim not in recent[-1].dims:
                continue
            if recent[-1].dims.get(dim, 0.0) - recent[0].dims.get(dim, 0.0) > 0.005:
                return False
        return True

    def _clipped(self, row: VPMRow) -> VPMRow:
        history = self.history.get(row.unit, [])
        if len(history) < 4:
            return row
        dims = dict(row.dims)
        for dim in self.policy.zscore_clip_dims:
            if dim not in dims:
                continue
            series = [item.dims[dim] for item in history if dim in item.dims]
            if len(series) < 4:
                continue
            mean = stats.mean(series)
            std = stats.pstdev(series) or 1e-6
            z = abs((dims[dim] - mean) / std)
            if z > self.policy.zscore_clip_sigma:
                dims[dim] = mean + self.policy.zscore_clip_sigma * (1 if dims[dim] > mean else -1) * std
        row.dims = dims
        return row

    def _reward(self, row: VPMRow) -> float:
        values = list(row.dims.values())
        return float(sum(values) / len(values)) if values else 0.0


def default_controller() -> VPMController:
    return VPMController(
        thresholds_code=Thresholds({"tests_pass_rate": 1.0, "coverage": 0.70, "type_safe": 1.0, "lint_clean": 1.0}),
        thresholds_text=Thresholds({"coverage": 0.80, "correctness": 0.75, "coherence": 0.75, "citation_support": 0.65, "entity_consistency": 0.80}),
    )
