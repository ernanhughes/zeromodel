"""Metric definitions for the visual-transition debugging benchmark (section 6).

Unit of measurement, declared once (section 6.1): all region precision/recall is
computed at **field** granularity -- the same 4x1-pixel P4A tiles System C uses
internally (``zeromodel_adapter.FIELD_SCHEMA``). This keeps System A (pixel diff)
and System C (ZeroModel) directly comparable: both output a set of field ids.
Ground truth for "did this field actually change" is the exact pixel comparison
between frame_before/frame_after (threshold 1, no privileged labels involved).

Component-level metrics (6.3-6.7) operate over the four declared component
names: tank / alien / cooldown / background.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Iterable, Mapping, Sequence, Tuple

import numpy as np

from visual_transition_benchmark import zeromodel_adapter as zm
from visual_transition_benchmark.baselines import SystemOutput
from visual_transition_benchmark.dataset import COMPONENT_NAMES, TransitionRecord

COMPONENT_UNIVERSE = frozenset(COMPONENT_NAMES)


def ground_truth_changed_fields(frame_before: np.ndarray, frame_after: np.ndarray) -> Tuple[str, ...]:
    changed = []
    for field in zm.FIELD_SCHEMA.fields:
        before_region = frame_before[field.y0 : field.y1, field.x0 : field.x1]
        after_region = frame_after[field.y0 : field.y1, field.x0 : field.x1]
        if np.any(before_region != after_region):
            changed.append(field.field_id)
    return tuple(sorted(changed))


def field_precision_recall(
    predicted_fields: Iterable[str], truth_fields: Iterable[str]
) -> Tuple[float, float]:
    """Zero-division convention (matches scikit-learn's zero_division=0 default):

    - predicted empty, truth empty  -> precision=1.0 (trivially correct)
    - predicted empty, truth nonempty -> precision=0.0, recall=0.0
    - predicted nonempty, truth empty -> recall=1.0 (vacuously nothing to find)
    """

    predicted = set(predicted_fields)
    truth = set(truth_fields)
    tp = len(predicted & truth)
    precision = tp / len(predicted) if predicted else (1.0 if not truth else 0.0)
    recall = tp / len(truth) if truth else 1.0
    return precision, recall


def unexpected_ground_truth(record: TransitionRecord) -> frozenset:
    return frozenset(record.observed_changed_components) - frozenset(record.expected_changed_components)


def missing_ground_truth(record: TransitionRecord) -> frozenset:
    return frozenset(record.expected_changed_components) - frozenset(record.observed_changed_components)


def fault_ground_truth(record: TransitionRecord) -> frozenset:
    return unexpected_ground_truth(record) | missing_ground_truth(record)


def flagged_relevant(output: SystemOutput) -> frozenset:
    """Every component a system surfaces as worth a human's attention."""

    return (
        frozenset(output.predicted_components)
        | frozenset(output.missing_components)
        | frozenset(output.unexpected_components)
    )


def false_implicated_components(record: TransitionRecord, output: SystemOutput) -> frozenset:
    return flagged_relevant(output) - frozenset(record.observed_changed_components)


def _prf(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def component_multilabel_metrics(
    predicted_sets: Sequence[Iterable[str]], true_sets: Sequence[Iterable[str]]
) -> Mapping[str, float]:
    if len(predicted_sets) != len(true_sets):
        raise ValueError("predicted_sets and true_sets must be the same length")
    tp = fp = fn = 0
    exact = 0
    for predicted_raw, true_raw in zip(predicted_sets, true_sets):
        predicted = set(predicted_raw)
        truth = set(true_raw)
        tp += len(predicted & truth)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
        if predicted == truth:
            exact += 1
    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_set_accuracy": exact / len(predicted_sets) if predicted_sets else 0.0,
        "n": len(predicted_sets),
    }


def per_transition_component_f1(predicted: Iterable[str], truth: Iterable[str]) -> float:
    predicted_set = set(predicted)
    truth_set = set(truth)
    tp = len(predicted_set & truth_set)
    fp = len(predicted_set - truth_set)
    fn = len(truth_set - predicted_set)
    _, _, f1 = _prf(tp, fp, fn)
    return f1


@dataclass(frozen=True)
class UnexpectedMissingSummary:
    n_relevant: int
    precision: float
    recall: float
    detection_rate: float
    fault_localization_rate: float


def unexpected_change_summary(
    records: Sequence[TransitionRecord], outputs: Sequence[SystemOutput]
) -> UnexpectedMissingSummary:
    tp = fp = fn = 0
    hits = 0
    localized = 0
    relevant = 0
    for record, output in zip(records, outputs):
        truth = unexpected_ground_truth(record)
        if not truth:
            continue
        relevant += 1
        predicted = frozenset(output.unexpected_components)
        tp += len(predicted & truth)
        fp += len(predicted - truth)
        fn += len(truth - predicted)
        if predicted & truth:
            hits += 1
        flagged = flagged_relevant(output)
        if flagged & fault_ground_truth(record) and flagged != COMPONENT_UNIVERSE:
            localized += 1
    precision, recall, _ = _prf(tp, fp, fn)
    return UnexpectedMissingSummary(
        n_relevant=relevant,
        precision=precision,
        recall=recall,
        detection_rate=(hits / relevant if relevant else 0.0),
        fault_localization_rate=(localized / relevant if relevant else 0.0),
    )


@dataclass(frozen=True)
class MissingChangeSummary:
    n_relevant: int
    detection_rate: float
    false_alarm_rate_on_correct: float


def missing_change_summary(
    records: Sequence[TransitionRecord], outputs: Sequence[SystemOutput]
) -> MissingChangeSummary:
    relevant = 0
    hits = 0
    for record, output in zip(records, outputs):
        truth = missing_ground_truth(record)
        if not truth:
            continue
        relevant += 1
        if truth & frozenset(output.missing_components):
            hits += 1

    ordinary = [(r, o) for r, o in zip(records, outputs) if not r.is_faulty]
    false_alarms = sum(
        1
        for r, o in ordinary
        if frozenset(o.missing_components) or frozenset(o.unexpected_components)
    )
    return MissingChangeSummary(
        n_relevant=relevant,
        detection_rate=(hits / relevant if relevant else 0.0),
        false_alarm_rate_on_correct=(false_alarms / len(ordinary) if ordinary else 0.0),
    )


@dataclass(frozen=True)
class FalseImplicatedSummary:
    mean_count: float
    median_count: float
    pct_zero: float


def false_implicated_summary(
    records: Sequence[TransitionRecord], outputs: Sequence[SystemOutput]
) -> FalseImplicatedSummary:
    counts = [
        len(false_implicated_components(record, output)) for record, output in zip(records, outputs)
    ]
    if not counts:
        return FalseImplicatedSummary(mean_count=0.0, median_count=0.0, pct_zero=0.0)
    return FalseImplicatedSummary(
        mean_count=sum(counts) / len(counts),
        median_count=float(median(counts)),
        pct_zero=sum(1 for c in counts if c == 0) / len(counts),
    )


@dataclass(frozen=True)
class ImprovementSummary:
    n: int
    better: int
    equal: int
    worse: int

    @property
    def pct_better(self) -> float:
        return self.better / self.n if self.n else 0.0

    @property
    def pct_equal(self) -> float:
        return self.equal / self.n if self.n else 0.0

    @property
    def pct_worse(self) -> float:
        return self.worse / self.n if self.n else 0.0


def compare_to_pixel_diff(
    records: Sequence[TransitionRecord],
    zeromodel_outputs: Sequence[SystemOutput],
    pixel_outputs: Sequence[SystemOutput],
) -> Tuple[ImprovementSummary, Tuple[str, ...]]:
    """Per-transition ZeroModel-vs-pixel-diff verdicts (section 6.7)."""

    better = equal = worse = 0
    verdicts = []
    for record, zm_out, pd_out in zip(records, zeromodel_outputs, pixel_outputs):
        truth = record.observed_changed_components
        zm_f1 = per_transition_component_f1(zm_out.predicted_components, truth)
        pd_f1 = per_transition_component_f1(pd_out.predicted_components, truth)
        missing_gt = missing_ground_truth(record)
        zm_flags_absence = bool(missing_gt) and bool(missing_gt & frozenset(zm_out.missing_components))
        if zm_flags_absence:
            verdict = "better"
        elif zm_f1 > pd_f1 + 1e-9:
            verdict = "better"
        elif zm_f1 < pd_f1 - 1e-9:
            verdict = "worse"
        else:
            verdict = "equal"
        verdicts.append(verdict)
        if verdict == "better":
            better += 1
        elif verdict == "worse":
            worse += 1
        else:
            equal += 1
    return (
        ImprovementSummary(n=len(records), better=better, equal=equal, worse=worse),
        tuple(verdicts),
    )
