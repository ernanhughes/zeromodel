"""Domain-neutral metrics for the cross-domain replication experiment.

Presence-level metrics are not reimplemented: ``metrics.py``'s existing
functions (``component_multilabel_metrics``, ``unexpected_change_summary``,
``missing_change_summary``, ``false_implicated_components``,
``field_precision_recall``) already operate only on component-name strings
and generic attributes (``observed_changed_components``,
``expected_changed_components``, ``is_faulty``, ``predicted_components``,
``missing_components``, ``unexpected_components``) -- exactly the attributes
``DomainTransition`` and ``ComponentAnalysisResult`` expose. They are reused
here unchanged, imported directly by ``cross_domain_report.py``.

This module adds only what stage 1's metrics could not already express:
value-level capability classes (direction, magnitude, value, relation,
identity), scored generically off the shared vocabulary declared in
``domains/protocol.py`` -- no arcade or warehouse component name appears
anywhere below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

from visual_transition_benchmark.domains.protocol import (
    DomainTransition,
    ValueAnalysisResult,
)


def direction_correct(transition: DomainTransition, decoded) -> Optional[bool]:
    expected = transition.value_ground_truth.get("direction_expected_sign")
    if expected is None:
        return None
    return decoded.get("direction_decoded_sign") == expected


def magnitude_correct(transition: DomainTransition, decoded) -> Optional[bool]:
    expected = transition.value_ground_truth.get("magnitude_expected_delta")
    if expected is None:
        return None
    return decoded.get("magnitude_decoded_delta") == expected


def value_level_correct(transition: DomainTransition, decoded) -> Optional[bool]:
    """Generically checks every declared ``*_expected_level`` /
    ``*_decoded_level`` pair (arcade has one -- cooldown; warehouse has two --
    battery and door) without knowing their names."""

    keys = [
        key[: -len("_expected_level")]
        for key in transition.value_ground_truth
        if key.endswith("_expected_level")
    ]
    if not keys:
        return None
    return all(
        decoded.get(f"{key}_decoded_level")
        == transition.value_ground_truth.get(f"{key}_expected_level")
        for key in keys
    )


def relation_correct(transition: DomainTransition, decoded) -> Optional[bool]:
    expected = transition.value_ground_truth.get("relation_expected_satisfied")
    if expected is None:
        return None
    return decoded.get("relation_decoded_satisfied") == expected


def identity_correct(transition: DomainTransition, decoded) -> Optional[bool]:
    expected = transition.value_ground_truth.get("identity_expected_id")
    if expected is None:
        return None
    return decoded.get("identity_decoded_id") == expected


_CHECKS = {
    "direction": direction_correct,
    "magnitude": magnitude_correct,
    "value": value_level_correct,
    "relation": relation_correct,
    "identity": identity_correct,
}


@dataclass(frozen=True)
class CapabilityRate:
    n_applicable: int
    n_correct: int

    @property
    def rate(self) -> Optional[float]:
        return (self.n_correct / self.n_applicable) if self.n_applicable else None


def capability_rate(
    capability: str,
    transitions: Sequence[DomainTransition],
    analyses: Sequence[ValueAnalysisResult],
) -> CapabilityRate:
    check = _CHECKS[capability]
    applicable = 0
    correct = 0
    for transition, analysis in zip(transitions, analyses):
        result = check(transition, analysis.decoded)
        if result is None:
            continue
        applicable += 1
        if result:
            correct += 1
    return CapabilityRate(n_applicable=applicable, n_correct=correct)


def value_fault_present(transition: DomainTransition, decoded) -> Optional[bool]:
    """True if any applicable value-level check disagrees with ground truth.
    None only when *no* value-level check applies to this transition at all."""

    results = [check(transition, decoded) for check in _CHECKS.values()]
    applicable = [r for r in results if r is not None]
    if not applicable:
        return None
    return not all(applicable)


@dataclass(frozen=True)
class HiddenValueFaultRate:
    n_faulty: int
    label_clean_but_value_wrong: int

    @property
    def rate(self) -> float:
        return (
            self.label_clean_but_value_wrong / self.n_faulty if self.n_faulty else 0.0
        )


def label_correct_but_value_wrong(
    transitions: Sequence[DomainTransition],
    component_outputs: Sequence,
    value_analyses: Sequence[ValueAnalysisResult],
) -> HiddenValueFaultRate:
    faulty = 0
    hidden = 0
    for transition, component, value in zip(
        transitions, component_outputs, value_analyses
    ):
        if not transition.is_faulty:
            continue
        faulty += 1
        label_clean = (
            not component.missing_components and not component.unexpected_components
        )
        fault_present = value_fault_present(transition, value.decoded)
        if label_clean and fault_present:
            hidden += 1
    return HiddenValueFaultRate(n_faulty=faulty, label_clean_but_value_wrong=hidden)


@dataclass(frozen=True)
class ValueDetectionRate:
    n_relevant: int
    detection_rate: float
    n_clean: int
    false_alarm_rate_on_correct: float


def value_fault_detection(
    transitions: Sequence[DomainTransition],
    value_analyses: Sequence[ValueAnalysisResult],
) -> ValueDetectionRate:
    relevant = []
    clean = []
    for transition, value in zip(transitions, value_analyses):
        present = value_fault_present(transition, value.decoded)
        if present is None:
            continue
        (relevant if present else clean).append(value.value_flags)

    hits = sum(1 for flags in relevant if flags)
    alarms = sum(1 for flags in clean if flags)
    return ValueDetectionRate(
        n_relevant=len(relevant),
        detection_rate=(hits / len(relevant) if relevant else 0.0),
        n_clean=len(clean),
        false_alarm_rate_on_correct=(alarms / len(clean) if clean else 0.0),
    )
