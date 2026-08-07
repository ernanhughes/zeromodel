from __future__ import annotations

import pytest

from zeromodel.critic.counterexamples import find_counterexamples
from zeromodel.critic.evaluation import (
    auroc,
    budget_selection_metrics,
    brier_score,
    expected_calibration_error,
    grouped_selection_metrics,
)
from zeromodel.critic.promotion import CriticPromotionPolicyDTO, evaluate_promotion


def test_known_evaluation_metrics() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.4, 0.35, 0.8]
    assert auroc(labels, scores) == pytest.approx(0.75)
    assert brier_score(labels, scores) == pytest.approx(
        ((0.1**2) + (0.4**2) + (0.65**2) + (0.2**2)) / 4
    )
    assert expected_calibration_error(labels, scores, bin_count=2) >= 0.0


def test_grouped_selection_uses_selected_label() -> None:
    metrics = grouped_selection_metrics(
        ["a", "a", "b", "b"],
        [1, 0, 0, 1],
        [0.4, 0.9, 0.8, 0.7],
    )
    assert metrics["selected_positive_rate"] == pytest.approx(0.0)
    assert metrics["preferred_ranked_first_rate"] == pytest.approx(0.0)


def test_budget_promotion_and_counterexamples() -> None:
    rows = budget_selection_metrics([1, 0, 1, 0], [0.9, 0.8, 0.2, 0.1], [0.5])
    assert rows[0]["number_selected"] == 2
    assert rows[0]["positive_rate_selected"] == pytest.approx(0.5)
    rejected = evaluate_promotion(
        current_metrics={"auroc": 0.7, "ece": 0.05},
        candidate_metrics={"auroc": 0.69, "ece": 0.2},
        policy=CriticPromotionPolicyDTO(min_candidate_auroc=0.7),
    )
    assert not rejected.recommended
    accepted = evaluate_promotion(
        current_metrics={"auroc": 0.7, "ece": 0.05},
        candidate_metrics={"auroc": 0.8, "ece": 0.04},
        policy=CriticPromotionPolicyDTO(min_candidate_auroc=0.7, min_auroc_gain=0.05),
    )
    assert accepted.recommended
    disagreements = find_counterexamples(
        ["x", "y"],
        [1, 0],
        [0.4, 0.2],
        [0.8, 0.7],
        threshold=0.5,
    )
    assert [item.artifact_id for item in disagreements] == ["x", "y"]
    assert disagreements[0].correct_critic == "candidate"
