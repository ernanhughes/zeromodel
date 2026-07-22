from __future__ import annotations

import pytest

from zeromodel.analysis.training import (
    TrainingCheckpoint,
    build_training_progress_vpm,
)
from zeromodel.core.artifact import VPMValidationError


def checkpoints():
    return [
        TrainingCheckpoint(
            step=1000,
            metrics={
                "train_loss": 1.00,
                "heldout_score": 0.50,
                "regression_safety": 0.99,
                "stability": 0.96,
                "tokens_per_second": 1000,
            },
        ),
        TrainingCheckpoint(
            step=2000,
            metrics={
                "train_loss": 0.80,
                "heldout_score": 0.56,
                "regression_safety": 0.98,
                "stability": 0.94,
                "tokens_per_second": 1100,
            },
        ),
        TrainingCheckpoint(
            step=3000,
            metrics={
                "train_loss": 0.65,
                "heldout_score": 0.64,
                "regression_safety": 0.97,
                "stability": 0.92,
                "tokens_per_second": 1250,
            },
        ),
    ]


def test_training_progress_vpm_selects_best_checkpoint() -> None:
    assessment = build_training_progress_vpm(
        checkpoints(),
        stability_metric="stability",
        efficiency_metric="tokens_per_second",
    )

    assert assessment.learned is True
    assert assessment.best_checkpoint_id == "step_3000"
    assert assessment.best_step == 3000
    assert assessment.warnings == ()
    assert assessment.artifact.source.metric_ids == (
        "progress_score",
        "train_progress",
        "heldout_progress",
        "regression_safety",
        "stability",
        "efficiency",
    )
    assert assessment.artifact.provenance["kind"] == "training_progress"
    assert assessment.artifact.provenance["learned"] is True


def test_training_progress_warns_when_train_improves_without_heldout_transfer() -> None:
    assessment = build_training_progress_vpm(
        [
            {
                "step": 0,
                "metrics": {
                    "train_loss": 1.0,
                    "heldout_score": 0.50,
                    "regression_safety": 0.99,
                },
            },
            {
                "step": 1,
                "metrics": {
                    "train_loss": 0.5,
                    "heldout_score": 0.50,
                    "regression_safety": 0.99,
                },
            },
        ]
    )

    assert assessment.learned is False
    assert "train_progress_without_heldout_transfer" in assessment.warnings


def test_training_progress_regression_blocks_learning() -> None:
    assessment = build_training_progress_vpm(
        [
            {
                "step": 0,
                "metrics": {
                    "train_loss": 1.0,
                    "heldout_score": 0.50,
                    "regression_safety": 0.99,
                },
            },
            {
                "step": 1,
                "metrics": {
                    "train_loss": 0.7,
                    "heldout_score": 0.60,
                    "regression_safety": 0.70,
                },
            },
        ]
    )

    assert assessment.learned is False
    assert "regression_safety_below_threshold" in assessment.warnings


def test_training_progress_cell_maps_to_checkpoint() -> None:
    assessment = build_training_progress_vpm(
        checkpoints(), stability_metric="stability"
    )
    cell = assessment.artifact.cell(0, 0)

    assert cell.metric_id == "progress_score"
    assert cell.row_id in {"step_1000", "step_2000", "step_3000"}
    assert 0.0 <= cell.normalized_value <= 1.0


def test_training_progress_rejects_invalid_checkpoints() -> None:
    with pytest.raises(VPMValidationError, match="at least two checkpoints"):
        build_training_progress_vpm(
            [{"step": 0, "metrics": {"train_loss": 1.0, "heldout_score": 0.5}}]
        )

    with pytest.raises(VPMValidationError, match="ordered"):
        build_training_progress_vpm(
            [
                {"step": 2, "metrics": {"train_loss": 1.0, "heldout_score": 0.5}},
                {"step": 1, "metrics": {"train_loss": 0.9, "heldout_score": 0.6}},
            ]
        )

    with pytest.raises(VPMValidationError, match="finite"):
        TrainingCheckpoint(step=1, metrics={"train_loss": float("nan")})
