from __future__ import annotations

import pytest

from zeromodel.analysis.learning import (
    LearningObservation,
    build_learning_vpm,
)
from zeromodel.core.artifact import VPMValidationError


def test_learning_vpm_requires_train_heldout_and_regression_evidence() -> None:
    assessment = build_learning_vpm(
        [
            LearningObservation(unit_id="claim-a", before=0.40, after=0.70, split="train"),
            LearningObservation(unit_id="claim-b", before=0.50, after=0.62, split="heldout"),
            LearningObservation(unit_id="claim-c", before=0.80, after=0.80, split="regression"),
        ]
    )

    assert assessment.learned is True
    assert assessment.train_delta == pytest.approx(0.30)
    assert assessment.heldout_delta == pytest.approx(0.12)
    assert assessment.regression_degradation == pytest.approx(0.0)
    assert assessment.artifact.source.metric_ids == (
        "learning_score",
        "delta_positive",
        "feedback_alignment",
        "generalization",
        "regression_safety",
        "after_score",
    )
    assert assessment.artifact.provenance["kind"] == "learning_trace"
    assert assessment.artifact.provenance["learned"] is True


def test_tracking_without_heldout_evidence_is_not_learning() -> None:
    assessment = build_learning_vpm(
        [
            {"unit_id": "same-example", "before": 0.2, "after": 0.9, "split": "train"},
            {"unit_id": "safety-check", "before": 0.8, "after": 0.8, "split": "regression"},
        ]
    )

    assert assessment.train_delta > 0
    assert assessment.heldout_delta == 0.0
    assert assessment.learned is False


def test_regression_prevents_learning_claim() -> None:
    assessment = build_learning_vpm(
        [
            LearningObservation(unit_id="claim-a", before=0.40, after=0.75, split="train"),
            LearningObservation(unit_id="claim-b", before=0.50, after=0.64, split="heldout"),
            LearningObservation(unit_id="claim-c", before=0.90, after=0.70, split="regression"),
        ]
    )

    assert assessment.regression_degradation == pytest.approx(0.20)
    assert assessment.learned is False


def test_learning_vpm_cell_maps_to_learning_observation() -> None:
    assessment = build_learning_vpm(
        [
            LearningObservation(unit_id="train-unit", before=0.3, after=0.6, split="train"),
            LearningObservation(unit_id="heldout-unit", before=0.4, after=0.55, split="heldout"),
            LearningObservation(unit_id="regression-unit", before=0.9, after=0.9, split="regression"),
        ]
    )

    first_cell = assessment.artifact.cell(0, 0)

    assert first_cell.metric_id == "learning_score"
    assert first_cell.row_id in {"train:train-unit", "heldout:heldout-unit", "regression:regression-unit"}
    assert 0.0 <= first_cell.normalized_value <= 1.0


def test_learning_observation_rejects_invalid_scores_and_splits() -> None:
    with pytest.raises(VPMValidationError, match="Unsupported learning split"):
        LearningObservation(unit_id="x", before=0.1, after=0.2, split="future")

    with pytest.raises(VPMValidationError, match="scores must be in"):
        LearningObservation(unit_id="x", before=0.1, after=1.2)


def test_learning_vpm_rejects_duplicate_split_unit_rows() -> None:
    with pytest.raises(VPMValidationError, match="unique split:unit_id"):
        build_learning_vpm(
            [
                LearningObservation(unit_id="same", before=0.1, after=0.2, split="train"),
                LearningObservation(unit_id="same", before=0.2, after=0.3, split="train"),
            ]
        )
