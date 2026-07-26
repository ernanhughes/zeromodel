"""Stage-2 metrics: did the component change to the *correct* value?

These are reported **separately** from stage 1's component-attribution
metrics on purpose (per the task): a transition can be perfectly
label-correct (System C says "tank changed", full stop) while being
value-wrong (it moved to the wrong column, or moved too far). Collapsing the
two into one score would hide exactly the failure mode this stage exists to
find -- see ``label_correct_but_value_wrong`` below, which counts that
overlap directly.

All "accuracy" metrics compare the **decoded** value (what a vision-only
system reads off the frames) against the **true** simulated value (from
``TransitionRecord.state_before``/``state_after``, i.e. real
``TinyArcadeShooter`` state -- privileged, used here only as scoring ground
truth, never fed into ``ValueAwareZeroModelAnalyzer``). "Detection"/"false
alarm" metrics instead measure ZeroModel's own non-privileged contract
verdicts (``value_flags``) against that same ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

from visual_transition_benchmark.dataset import TransitionRecord
from visual_transition_benchmark.value_adapter import ValueTransitionAnalysis
from visual_transition_benchmark.value_contracts import DecodedValues
from visual_transition_benchmark.zeromodel_adapter import TransitionAnalysis


def _sign(value: int) -> int:
    return -1 if value < 0 else (1 if value > 0 else 0)


def true_tank_delta(record: TransitionRecord) -> int:
    return record.state_after["tank_x"] - record.state_before["tank_x"]


def true_cooldown_level(record: TransitionRecord) -> str:
    return "blocked" if record.state_after["cooldown"] == 1 else "ready"


def true_target_after(record: TransitionRecord):
    return record.state_after["target_x"]


def tank_direction_correct(record: TransitionRecord, values: DecodedValues) -> bool:
    if values.tank.delta_x is None:
        return False
    return _sign(values.tank.delta_x) == _sign(true_tank_delta(record))


def tank_magnitude_correct(record: TransitionRecord, values: DecodedValues) -> bool:
    return values.tank.delta_x == true_tank_delta(record)


def cooldown_value_correct(record: TransitionRecord, values: DecodedValues) -> bool:
    return values.cooldown.after_level == true_cooldown_level(record)


def target_selection_correct(record: TransitionRecord, values: DecodedValues) -> bool:
    return values.alien.after_x == true_target_after(record)


def value_fault_present(record: TransitionRecord, values: DecodedValues) -> bool:
    """True if ANY decoded dimension diverges from the true simulated state."""

    return not (
        tank_magnitude_correct(record, values)
        and cooldown_value_correct(record, values)
        and target_selection_correct(record, values)
    )


@dataclass(frozen=True)
class ValueAccuracySummary:
    n: int
    movement_direction_accuracy: float
    state_delta_accuracy: float
    cooldown_value_accuracy: float
    target_selection_accuracy: float


def value_accuracy_summary(
    records: Sequence[TransitionRecord], values_list: Sequence[DecodedValues]
) -> ValueAccuracySummary:
    n = len(records)
    if n == 0:
        return ValueAccuracySummary(0, 0.0, 0.0, 0.0, 0.0)
    direction_hits = sum(tank_direction_correct(r, v) for r, v in zip(records, values_list))
    delta_hits = sum(tank_magnitude_correct(r, v) for r, v in zip(records, values_list))
    cooldown_hits = sum(cooldown_value_correct(r, v) for r, v in zip(records, values_list))
    target_hits = sum(target_selection_correct(r, v) for r, v in zip(records, values_list))
    return ValueAccuracySummary(
        n=n,
        movement_direction_accuracy=direction_hits / n,
        state_delta_accuracy=delta_hits / n,
        cooldown_value_accuracy=cooldown_hits / n,
        target_selection_accuracy=target_hits / n,
    )


@dataclass(frozen=True)
class ValueFaultLocalizationSummary:
    n_relevant: int
    detection_rate: float
    n_clean: int
    false_alarm_rate_on_correct: float


def value_fault_localization_summary(
    records: Sequence[TransitionRecord],
    values_list: Sequence[DecodedValues],
    flags_list: Sequence[Tuple[str, ...]],
) -> ValueFaultLocalizationSummary:
    relevant = [
        (r, v, f) for r, v, f in zip(records, values_list, flags_list) if value_fault_present(r, v)
    ]
    clean = [
        (r, v, f) for r, v, f in zip(records, values_list, flags_list) if not value_fault_present(r, v)
    ]
    hits = sum(1 for _, _, f in relevant if f)
    alarms = sum(1 for _, _, f in clean if f)
    return ValueFaultLocalizationSummary(
        n_relevant=len(relevant),
        detection_rate=(hits / len(relevant) if relevant else 0.0),
        n_clean=len(clean),
        false_alarm_rate_on_correct=(alarms / len(clean) if clean else 0.0),
    )


def relation_violation_rate_by_category(
    records: Sequence[TransitionRecord], flags_list: Sequence[Tuple[str, ...]]
) -> dict:
    by_category: dict = {}
    for record, flags in zip(records, flags_list):
        bucket = by_category.setdefault(record.category, [0, 0])
        bucket[0] += 1
        if any(flag.startswith("relation:") for flag in flags):
            bucket[1] += 1
    return {
        category: (count[1] / count[0] if count[0] else 0.0) for category, count in by_category.items()
    }


@dataclass(frozen=True)
class HiddenValueFaultSummary:
    n_faulty: int
    label_clean_but_value_wrong: int

    @property
    def rate(self) -> float:
        return self.label_clean_but_value_wrong / self.n_faulty if self.n_faulty else 0.0


def label_correct_but_value_wrong(
    records: Sequence[TransitionRecord],
    component_outputs: Sequence[TransitionAnalysis],
    values_list: Sequence[DecodedValues],
) -> HiddenValueFaultSummary:
    """How many faulty transitions look completely clean to System C (no
    missing/unexpected component finding) yet are demonstrably value-wrong?
    This is the number that justifies stage 2's existence."""

    faulty = 0
    hidden = 0
    for record, component, values in zip(records, component_outputs, values_list):
        if not record.is_faulty:
            continue
        faulty += 1
        label_clean = not component.missing_components and not component.unexpected_components
        if label_clean and value_fault_present(record, values):
            hidden += 1
    return HiddenValueFaultSummary(n_faulty=faulty, label_clean_but_value_wrong=hidden)
