"""Training progress artifacts for model training telemetry.

This module turns checkpoint telemetry into deterministic VPM artifacts. It is not a
trainer and it does not inspect model internals. It summarizes observable training
evidence: train improvement, held-out improvement, regression safety, stability,
efficiency, and checkpoint selection signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import LayoutRecipe, ScoreTable, VPMArtifact, VPMValidationError, build_vpm

TRAINING_METRICS: Tuple[str, ...] = (
    "progress_score",
    "train_progress",
    "heldout_progress",
    "regression_safety",
    "stability",
    "efficiency",
)

_DIRECTION_VALUES = ("increase", "decrease")


@dataclass(frozen=True)
class TrainingCheckpoint:
    """One observed checkpoint from a training run.

    ``metrics`` is intentionally a plain mapping so callers can adapt TensorBoard,
    W&B, Trackio, JSONL logs, or custom training loops without depending on any
    tracker-specific SDK.
    """

    step: int
    metrics: Mapping[str, float]
    checkpoint_id: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        step = int(self.step)
        if step < 0:
            raise VPMValidationError("TrainingCheckpoint step must be non-negative")
        if not self.metrics:
            raise VPMValidationError("TrainingCheckpoint requires at least one metric")
        normalized_metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            metric_id = str(key)
            number = float(value)
            if not np.isfinite(number):
                raise VPMValidationError("TrainingCheckpoint metrics must be finite")
            normalized_metrics[metric_id] = number
        checkpoint_id = self.checkpoint_id or "step_%s" % step
        object.__setattr__(self, "step", step)
        object.__setattr__(self, "checkpoint_id", str(checkpoint_id))
        object.__setattr__(self, "metrics", normalized_metrics)

    def metric(self, name: str) -> float:
        try:
            return float(self.metrics[name])
        except KeyError as exc:
            raise VPMValidationError("Missing checkpoint metric: %s" % name) from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "checkpoint_id": self.checkpoint_id,
            "metrics": dict(self.metrics),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class TrainingProgressAssessment:
    """Summary and VPM artifact for a training run."""

    artifact: VPMArtifact
    checkpoints: Tuple[TrainingCheckpoint, ...]
    best_checkpoint_id: str
    best_step: int
    learned: bool
    warnings: Tuple[str, ...]
    thresholds: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact.artifact_id,
            "best_checkpoint_id": self.best_checkpoint_id,
            "best_step": self.best_step,
            "learned": self.learned,
            "warnings": list(self.warnings),
            "thresholds": dict(self.thresholds),
            "checkpoints": [checkpoint.to_dict() for checkpoint in self.checkpoints],
        }


def training_progress_recipe() -> LayoutRecipe:
    """Default recipe that places strongest training progress first."""
    return LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "training-progress-first",
            "row_order": {
                "kind": "lexicographic",
                "keys": [
                    {"metric_id": "progress_score", "direction": "desc"},
                    {"metric_id": "heldout_progress", "direction": "desc"},
                    {"metric_id": "regression_safety", "direction": "desc"},
                ],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )


def _validate_direction(direction: str) -> str:
    direction = str(direction)
    if direction not in _DIRECTION_VALUES:
        raise VPMValidationError("direction must be one of %r" % (_DIRECTION_VALUES,))
    return direction


def _relative_progress(baseline: float, current: float, direction: str) -> float:
    """Return non-negative relative progress from baseline toward current."""
    denom = max(abs(float(baseline)), 1e-12)
    if direction == "decrease":
        delta = float(baseline) - float(current)
    else:
        delta = float(current) - float(baseline)
    return max(0.0, float(delta / denom))


def _bounded01(value: float, *, label: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise VPMValidationError("%s must be finite" % label)
    return float(np.clip(number, 0.0, 1.0))


def _metric_progress(
    checkpoints: Sequence[TrainingCheckpoint],
    metric: str,
    direction: str,
) -> list[float]:
    baseline = checkpoints[0].metric(metric)
    return [_relative_progress(baseline, checkpoint.metric(metric), direction) for checkpoint in checkpoints]


def _direct_metric(checkpoints: Sequence[TrainingCheckpoint], metric: Optional[str], default: float) -> list[float]:
    if metric is None:
        return [float(default) for _ in checkpoints]
    return [_bounded01(checkpoint.metric(metric), label=metric) for checkpoint in checkpoints]


def _efficiency_progress(
    checkpoints: Sequence[TrainingCheckpoint],
    metric: Optional[str],
    direction: str,
    default: float,
) -> list[float]:
    if metric is None:
        return [float(default) for _ in checkpoints]
    return [min(1.0, value) for value in _metric_progress(checkpoints, metric, direction)]


def _weighted_score(
    train_progress: float,
    heldout_progress: float,
    regression_safety: float,
    stability: float,
    efficiency: float,
    weights: Mapping[str, float],
) -> float:
    total = 0.0
    total_weight = 0.0
    values = {
        "train_progress": train_progress,
        "heldout_progress": heldout_progress,
        "regression_safety": regression_safety,
        "stability": stability,
        "efficiency": efficiency,
    }
    for key, value in values.items():
        weight = float(weights.get(key, 0.0))
        if weight < 0:
            raise VPMValidationError("training score weights must be non-negative")
        total += weight * float(value)
        total_weight += weight
    if total_weight <= 0:
        raise VPMValidationError("training score weights must contain positive mass")
    return float(total / total_weight)


def build_training_progress_vpm(
    checkpoints: Sequence[TrainingCheckpoint | Mapping[str, Any]],
    *,
    train_metric: str = "train_loss",
    train_direction: str = "decrease",
    heldout_metric: str = "heldout_score",
    heldout_direction: str = "increase",
    regression_metric: Optional[str] = "regression_safety",
    stability_metric: Optional[str] = None,
    efficiency_metric: Optional[str] = None,
    efficiency_direction: str = "increase",
    min_train_progress: float = 0.05,
    min_heldout_progress: float = 0.02,
    min_regression_safety: float = 0.95,
    min_stability: float = 0.80,
    score_weights: Optional[Mapping[str, float]] = None,
    recipe: Optional[LayoutRecipe] = None,
    provenance: Optional[Mapping[str, Any]] = None,
) -> TrainingProgressAssessment:
    """Build a checkpoint-level training progress VPM.

    ``learned=True`` means the best checkpoint shows enough train progress, enough
    held-out progress, acceptable regression safety, and acceptable stability. It
    does not prove a model's internal mechanism; it proves that the observed
    checkpoint telemetry meets the declared evidence thresholds.
    """
    normalized_checkpoints = tuple(
        item if isinstance(item, TrainingCheckpoint) else TrainingCheckpoint(**dict(item))
        for item in checkpoints
    )
    if len(normalized_checkpoints) < 2:
        raise VPMValidationError("build_training_progress_vpm requires at least two checkpoints")

    steps = [checkpoint.step for checkpoint in normalized_checkpoints]
    if steps != sorted(steps):
        raise VPMValidationError("Training checkpoints must be ordered by non-decreasing step")
    checkpoint_ids = [checkpoint.checkpoint_id for checkpoint in normalized_checkpoints]
    if len(set(checkpoint_ids)) != len(checkpoint_ids):
        raise VPMValidationError("Training checkpoints require unique checkpoint_id values")

    train_direction = _validate_direction(train_direction)
    heldout_direction = _validate_direction(heldout_direction)
    efficiency_direction = _validate_direction(efficiency_direction)

    train_progress = _metric_progress(normalized_checkpoints, train_metric, train_direction)
    heldout_progress = _metric_progress(normalized_checkpoints, heldout_metric, heldout_direction)
    regression_safety = _direct_metric(normalized_checkpoints, regression_metric, 1.0)
    stability = _direct_metric(normalized_checkpoints, stability_metric, 1.0)
    efficiency = _efficiency_progress(normalized_checkpoints, efficiency_metric, efficiency_direction, 1.0)

    weights = {
        "train_progress": 0.25,
        "heldout_progress": 0.35,
        "regression_safety": 0.20,
        "stability": 0.15,
        "efficiency": 0.05,
    }
    if score_weights:
        weights.update({str(key): float(value) for key, value in score_weights.items()})

    rows: list[Tuple[float, ...]] = []
    progress_scores: list[float] = []
    for values in zip(train_progress, heldout_progress, regression_safety, stability, efficiency):
        score = _weighted_score(*values, weights=weights)
        progress_scores.append(score)
        rows.append((score, *values))

    best_index = max(
        range(len(normalized_checkpoints)),
        key=lambda index: (progress_scores[index], normalized_checkpoints[index].step),
    )
    best = normalized_checkpoints[best_index]

    warnings: list[str] = []
    if train_progress[best_index] >= min_train_progress and heldout_progress[best_index] < min_heldout_progress:
        warnings.append("train_progress_without_heldout_transfer")
    if regression_safety[best_index] < min_regression_safety:
        warnings.append("regression_safety_below_threshold")
    if stability[best_index] < min_stability:
        warnings.append("stability_below_threshold")
    if progress_scores[-1] < progress_scores[best_index]:
        warnings.append("latest_checkpoint_is_not_best")

    learned = bool(
        train_progress[best_index] >= float(min_train_progress)
        and heldout_progress[best_index] >= float(min_heldout_progress)
        and regression_safety[best_index] >= float(min_regression_safety)
        and stability[best_index] >= float(min_stability)
    )

    table = ScoreTable(
        values=rows,
        row_ids=tuple(checkpoint.checkpoint_id for checkpoint in normalized_checkpoints),
        metric_ids=TRAINING_METRICS,
        metadata={
            "kind": "training_progress",
            "train_metric": train_metric,
            "heldout_metric": heldout_metric,
            "regression_metric": regression_metric,
            "stability_metric": stability_metric,
            "efficiency_metric": efficiency_metric,
            "checkpoints": [checkpoint.to_dict() for checkpoint in normalized_checkpoints],
        },
    )

    artifact = build_vpm(
        table,
        recipe or training_progress_recipe(),
        provenance={
            "kind": "training_progress",
            "parents": [],
            "best_checkpoint_id": best.checkpoint_id,
            "best_step": best.step,
            "learned": learned,
            "warnings": warnings,
            **dict(provenance or {}),
        },
    )

    return TrainingProgressAssessment(
        artifact=artifact,
        checkpoints=normalized_checkpoints,
        best_checkpoint_id=best.checkpoint_id,
        best_step=best.step,
        learned=learned,
        warnings=tuple(warnings),
        thresholds={
            "min_train_progress": float(min_train_progress),
            "min_heldout_progress": float(min_heldout_progress),
            "min_regression_safety": float(min_regression_safety),
            "min_stability": float(min_stability),
        },
    )
