from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from zeromodel.critic.errors import CriticEvaluationError


def _arrays(
    labels: Sequence[float], scores: Sequence[float]
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(labels, dtype=np.float64)
    s = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or s.ndim != 1 or y.size != s.size or y.size == 0:
        raise CriticEvaluationError(
            "labels and scores must be non-empty aligned vectors"
        )
    if not np.isin(y, [0.0, 1.0]).all() or not np.isfinite(s).all():
        raise CriticEvaluationError("labels must be binary and scores finite")
    return y, s


def accuracy(
    labels: Sequence[float], scores: Sequence[float], *, threshold: float = 0.5
) -> float:
    y, s = _arrays(labels, scores)
    return float(np.mean((s >= threshold) == (y == 1.0)))


def brier_score(labels: Sequence[float], scores: Sequence[float]) -> float:
    y, s = _arrays(labels, scores)
    return float(np.mean((s - y) ** 2))


def auroc(labels: Sequence[float], scores: Sequence[float]) -> float:
    y, s = _arrays(labels, scores)
    positives = np.sum(y == 1.0)
    negatives = np.sum(y == 0.0)
    if positives == 0 or negatives == 0:
        raise CriticEvaluationError("AUROC requires both classes")
    order = np.argsort(s, kind="mergesort")
    sorted_scores = s[order]
    ranks = np.empty_like(s, dtype=np.float64)
    start = 0
    while start < s.size:
        end = start + 1
        while end < s.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end
    rank_sum_positive = float(np.sum(ranks[y == 1.0]))
    return float(
        (rank_sum_positive - positives * (positives + 1.0) / 2.0)
        / (positives * negatives)
    )


def expected_calibration_error(
    labels: Sequence[float], scores: Sequence[float], *, bin_count: int = 10
) -> float:
    y, s = _arrays(labels, scores)
    if bin_count <= 0:
        raise CriticEvaluationError("bin_count must be positive")
    edges = np.linspace(0.0, 1.0, bin_count + 1)
    total = 0.0
    for idx in range(bin_count):
        left = edges[idx]
        right = edges[idx + 1]
        mask = (s >= left) & (s <= right if idx == bin_count - 1 else s < right)
        if np.any(mask):
            total += float(np.mean(mask)) * abs(
                float(np.mean(s[mask])) - float(np.mean(y[mask]))
            )
    return float(total)


def evaluate_binary_critic(
    labels: Sequence[float], scores: Sequence[float], *, bin_count: int = 10
) -> dict[str, float | int | str]:
    y, s = _arrays(labels, scores)
    return {
        "sample_count": int(y.size),
        "positive_rate": float(np.mean(y)),
        "accuracy": accuracy(y, s),
        "auroc": auroc(y, s),
        "brier": brier_score(y, s),
        "ece": expected_calibration_error(y, s, bin_count=bin_count),
        "ece_bin_count": int(bin_count),
        "ece_binning_method": "equal_width_closed_last",
    }


def grouped_selection_metrics(
    group_ids: Sequence[str], labels: Sequence[float], scores: Sequence[float]
) -> dict[str, float | int]:
    y, s = _arrays(labels, scores)
    groups: dict[str, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        groups.setdefault(str(group_id), []).append(index)
    selected_labels = []
    preferred_first = []
    ties = 0
    margins = []
    for indexes in groups.values():
        best_score = max(float(s[index]) for index in indexes)
        winners = [index for index in indexes if float(s[index]) == best_score]
        winner = sorted(winners, key=lambda index: str(index))[0]
        selected_labels.append(float(y[winner]))
        if len(winners) > 1:
            ties += 1
        positives = [index for index in indexes if y[index] == 1.0]
        negatives = [index for index in indexes if y[index] == 0.0]
        if positives and negatives:
            best_positive = max(float(s[index]) for index in positives)
            best_negative = max(float(s[index]) for index in negatives)
            preferred_first.append(float(best_positive > best_negative))
            margins.append(best_positive - best_negative)
    return {
        "group_count": len(groups),
        "selected_positive_rate": float(np.mean(selected_labels)),
        "preferred_ranked_first_rate": float(np.mean(preferred_first))
        if preferred_first
        else 0.0,
        "tie_rate": float(ties / len(groups)) if groups else 0.0,
        "mean_preferred_minus_rejected_margin": float(np.mean(margins))
        if margins
        else 0.0,
    }


def budget_selection_metrics(
    labels: Sequence[float], scores: Sequence[float], budgets: Sequence[float]
) -> list[dict[str, float | int]]:
    y, s = _arrays(labels, scores)
    order = sorted(range(y.size), key=lambda index: (-float(s[index]), str(index)))
    overall = float(np.mean(y))
    rows = []
    for budget in budgets:
        count = (
            int(np.ceil(float(budget) * y.size))
            if 0.0 < float(budget) <= 1.0
            else int(budget)
        )
        count = max(0, min(count, y.size))
        chosen = order[:count]
        positive_rate = float(np.mean(y[chosen])) if chosen else 0.0
        rows.append(
            {
                "budget": float(budget),
                "number_selected": int(count),
                "positive_rate_selected": positive_rate,
                "overall_positive_rate": overall,
                "lift_vs_random_expectation": 0.0
                if overall == 0.0
                else positive_rate / overall,
            }
        )
    return rows


@dataclass(frozen=True, slots=True)
class CriticEvaluationResultDTO:
    metrics: Mapping[str, Any] = field(default_factory=dict)
