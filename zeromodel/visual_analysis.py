"""Exploratory analysis for governed visual-address benchmark traces.

The helpers in this module deliberately separate ranking from calibration. They
may be used to describe existing evaluation traces, but thresholds selected on
those traces are not validated deployment thresholds. A promoted operating
point requires an independent rejection-calibration split and an untouched
final evaluation split.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMValidationError


EXPLORATORY_TEST_REUSE_WARNING = (
    "Exploratory curve over evaluation traces. Do not select and report a "
    "deployment threshold on the same observations; use independent rejection "
    "calibration and untouched final evaluation families."
)


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


@dataclass(frozen=True)
class VisualTracePoint:
    observation_id: str
    family_id: str
    expected_disposition: str
    expected_row_id: Optional[str]
    expected_action_id: Optional[str]
    top1_row_id: Optional[str]
    top1_action_id: Optional[str]
    score: Optional[float]

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "VisualTracePoint":
        decision = value.get("decision") or {}
        top1_row = value.get("top1_row_id")
        if top1_row is None:
            top1_row = decision.get("nearest_row_id")
        top1_action = value.get("top1_action_id")
        if top1_action is None:
            top1_action = value.get("predicted_action_id")
        score = decision.get("nearest_score")
        return cls(
            observation_id=str(value["observation_id"]),
            family_id=str(value["family_id"]),
            expected_disposition=str(value["expected_disposition"]),
            expected_row_id=(
                None
                if value.get("expected_row_id") is None
                else str(value.get("expected_row_id"))
            ),
            expected_action_id=(
                None
                if value.get("expected_action_id") is None
                else str(value.get("expected_action_id"))
            ),
            top1_row_id=None if top1_row is None else str(top1_row),
            top1_action_id=None if top1_action is None else str(top1_action),
            score=None if score is None else float(score),
        )

    @property
    def expected_accept(self) -> bool:
        return self.expected_disposition == "accept"

    @property
    def expected_reject(self) -> bool:
        return self.expected_disposition == "reject"

    @property
    def row_correct(self) -> bool:
        return bool(
            self.expected_accept
            and self.top1_row_id is not None
            and self.top1_row_id == self.expected_row_id
        )

    @property
    def action_correct(self) -> bool:
        return bool(
            self.expected_accept
            and self.top1_action_id is not None
            and self.top1_action_id == self.expected_action_id
        )


def trace_points(values: Iterable[Mapping[str, Any]]) -> Tuple[VisualTracePoint, ...]:
    points = tuple(VisualTracePoint.from_dict(value) for value in values)
    identifiers = [point.observation_id for point in points]
    if len(identifiers) != len(set(identifiers)):
        raise VPMValidationError("visual analysis trace ids must be unique")
    return points


def score_thresholds(
    points: Sequence[VisualTracePoint],
    *,
    maximum_points: int = 101,
) -> Tuple[float, ...]:
    if maximum_points < 2:
        raise VPMValidationError("maximum_points must be at least two")
    scores = sorted({float(point.score) for point in points if point.score is not None})
    if not scores:
        return ()
    if len(scores) <= maximum_points:
        return tuple(scores)
    indices = np.linspace(0, len(scores) - 1, num=maximum_points)
    return tuple(scores[int(round(index))] for index in indices)


def operating_curve(
    points: Sequence[VisualTracePoint],
    thresholds: Optional[Sequence[float]] = None,
) -> Tuple[Mapping[str, Any], ...]:
    items = tuple(points)
    benign = tuple(point for point in items if point.expected_accept)
    rejected = tuple(point for point in items if point.expected_reject)
    selected_thresholds = (
        tuple(float(value) for value in thresholds)
        if thresholds is not None
        else score_thresholds(items)
    )
    rows = []
    for threshold in selected_thresholds:
        accepted_benign = tuple(
            point for point in benign
            if point.score is not None and point.score >= threshold
        )
        false_accepts = tuple(
            point for point in rejected
            if point.score is not None and point.score >= threshold
        )
        correct_rows = sum(point.row_correct for point in accepted_benign)
        correct_actions = sum(point.action_correct for point in accepted_benign)
        rows.append(
            {
                "threshold": float(threshold),
                "benign_count": len(benign),
                "rejection_count": len(rejected),
                "accepted_benign_count": len(accepted_benign),
                "false_accept_count": len(false_accepts),
                "coverage": _rate(len(accepted_benign), len(benign)),
                "accepted_row_precision": _rate(correct_rows, len(accepted_benign)),
                "accepted_action_precision": _rate(
                    correct_actions, len(accepted_benign)
                ),
                "row_recall": _rate(correct_rows, len(benign)),
                "action_recall": _rate(correct_actions, len(benign)),
                "false_acceptance_rate": _rate(len(false_accepts), len(rejected)),
                "false_rejection_rate": _rate(
                    len(benign) - len(accepted_benign), len(benign)
                ),
            }
        )
    return tuple(rows)


def family_summary(points: Sequence[VisualTracePoint]) -> Mapping[str, Mapping[str, Any]]:
    result: Dict[str, Mapping[str, Any]] = {}
    for family_id in sorted({point.family_id for point in points}):
        family = tuple(point for point in points if point.family_id == family_id)
        benign = tuple(point for point in family if point.expected_accept)
        rejected = tuple(point for point in family if point.expected_reject)
        result[family_id] = {
            "observation_count": len(family),
            "benign_count": len(benign),
            "rejection_count": len(rejected),
            "top1_row_accuracy": _rate(
                sum(point.row_correct for point in benign), len(benign)
            ),
            "top1_action_accuracy": _rate(
                sum(point.action_correct for point in benign), len(benign)
            ),
            "score_min": (
                min(point.score for point in family if point.score is not None)
                if any(point.score is not None for point in family)
                else None
            ),
            "score_max": (
                max(point.score for point in family if point.score is not None)
                if any(point.score is not None for point in family)
                else None
            ),
        }
    return result


def paired_top1_outcomes(
    left: Sequence[VisualTracePoint],
    right: Sequence[VisualTracePoint],
) -> Mapping[str, Any]:
    left_by_id = {point.observation_id: point for point in left if point.expected_accept}
    right_by_id = {point.observation_id: point for point in right if point.expected_accept}
    if set(left_by_id) != set(right_by_id):
        raise VPMValidationError("paired visual traces must cover identical benign ids")

    row_counts = {"both_correct": 0, "left_only": 0, "right_only": 0, "neither": 0}
    action_counts = dict(row_counts)
    for observation_id in sorted(left_by_id):
        left_point = left_by_id[observation_id]
        right_point = right_by_id[observation_id]
        for name, left_correct, right_correct in (
            ("row", left_point.row_correct, right_point.row_correct),
            ("action", left_point.action_correct, right_point.action_correct),
        ):
            target = row_counts if name == "row" else action_counts
            key = (
                "both_correct" if left_correct and right_correct
                else "left_only" if left_correct
                else "right_only" if right_correct
                else "neither"
            )
            target[key] += 1
    return {
        "observation_count": len(left_by_id),
        "row": row_counts,
        "action": action_counts,
        "note": (
            "Off-diagonal counts support paired analysis. Use state/family-"
            "clustered uncertainty for research conclusions."
        ),
    }


def analyze_trace_sets(
    traces_by_system: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Mapping[str, Any]:
    parsed = {
        str(system_id): trace_points(values)
        for system_id, values in traces_by_system.items()
    }
    systems = {
        system_id: {
            "operating_curve": list(operating_curve(points)),
            "family_summary": family_summary(points),
        }
        for system_id, points in parsed.items()
    }
    paired = None
    if "B" in parsed and "D" in parsed:
        paired = paired_top1_outcomes(parsed["B"], parsed["D"])
    return {
        "status": "exploratory",
        "warning": EXPLORATORY_TEST_REUSE_WARNING,
        "systems": systems,
        "paired_B_D": paired,
    }
