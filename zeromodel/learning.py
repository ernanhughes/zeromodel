"""Learning trace artifacts for showing durable behavior change.

This module distinguishes tracking from learning. A score moving over time is only
tracking. Learning requires a before/after change tied to feedback plus evidence
that the change transfers to held-out work without unacceptable regression.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import LayoutRecipe, ScoreTable, VPMArtifact, VPMValidationError, build_vpm

LEARNING_METRICS: Tuple[str, ...] = (
    "learning_score",
    "delta_positive",
    "feedback_alignment",
    "generalization",
    "regression_safety",
    "after_score",
)

VALID_SPLITS = ("train", "heldout", "regression")


@dataclass(frozen=True)
class LearningObservation:
    """One before/after measurement for a unit affected by feedback.

    ``split`` controls how the observation contributes to the learning claim:

    - ``train``: did the corrected/experienced unit improve?
    - ``heldout``: did related future or unseen work improve?
    - ``regression``: did unrelated or previously good work stay intact?
    """

    unit_id: str
    before: float
    after: float
    split: str = "train"
    feedback_alignment: float = 1.0
    weight: float = 1.0
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_id = str(self.unit_id)
        split = str(self.split)
        before = float(self.before)
        after = float(self.after)
        feedback_alignment = float(self.feedback_alignment)
        weight = float(self.weight)
        if not unit_id:
            raise VPMValidationError("LearningObservation requires a non-empty unit_id")
        if split not in VALID_SPLITS:
            raise VPMValidationError("Unsupported learning split: %r" % split)
        values = [before, after, feedback_alignment, weight]
        if not np.isfinite(values).all():
            raise VPMValidationError("LearningObservation numeric values must be finite")
        if not (0.0 <= before <= 1.0 and 0.0 <= after <= 1.0):
            raise VPMValidationError("LearningObservation before/after scores must be in [0, 1]")
        if not (0.0 <= feedback_alignment <= 1.0):
            raise VPMValidationError("feedback_alignment must be in [0, 1]")
        if weight <= 0:
            raise VPMValidationError("weight must be positive")
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "before", before)
        object.__setattr__(self, "after", after)
        object.__setattr__(self, "feedback_alignment", feedback_alignment)
        object.__setattr__(self, "weight", weight)

    @property
    def delta(self) -> float:
        return float(self.after - self.before)

    @property
    def improvement(self) -> float:
        return max(self.delta, 0.0)

    @property
    def degradation(self) -> float:
        return max(-self.delta, 0.0)

    @property
    def regression_safety(self) -> float:
        return max(0.0, 1.0 - self.degradation)

    @property
    def generalization(self) -> float:
        return self.improvement if self.split == "heldout" else 0.0

    @property
    def learning_score(self) -> float:
        if self.split == "regression":
            return self.regression_safety
        return self.improvement * self.feedback_alignment

    def metric_row(self) -> Tuple[float, ...]:
        return (
            self.learning_score,
            self.improvement,
            self.feedback_alignment,
            self.generalization,
            self.regression_safety,
            self.after,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "before": self.before,
            "after": self.after,
            "delta": self.delta,
            "split": self.split,
            "feedback_alignment": self.feedback_alignment,
            "weight": self.weight,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class LearningAssessment:
    """Summary of whether a trace shows learning rather than mere tracking."""

    artifact: VPMArtifact
    observations: Tuple[LearningObservation, ...]
    train_delta: float
    heldout_delta: float
    regression_degradation: float
    learned: bool
    thresholds: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact.artifact_id,
            "learned": self.learned,
            "train_delta": self.train_delta,
            "heldout_delta": self.heldout_delta,
            "regression_degradation": self.regression_degradation,
            "thresholds": dict(self.thresholds),
            "observations": [observation.to_dict() for observation in self.observations],
        }


def learning_recipe() -> LayoutRecipe:
    """Default recipe that places strongest learning evidence first."""
    return LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "learning-evidence-first",
            "row_order": {
                "kind": "lexicographic",
                "keys": [
                    {"metric_id": "learning_score", "direction": "desc"},
                    {"metric_id": "generalization", "direction": "desc"},
                    {"metric_id": "regression_safety", "direction": "desc"},
                ],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values:
        return 0.0
    total_weight = float(sum(weights))
    if total_weight <= 0:
        return 0.0
    return float(sum(value * weight for value, weight in zip(values, weights)) / total_weight)


def _mean_delta(observations: Sequence[LearningObservation], split: str) -> float:
    subset = [observation for observation in observations if observation.split == split]
    return _weighted_mean([observation.delta for observation in subset], [observation.weight for observation in subset])


def _mean_degradation(observations: Sequence[LearningObservation], split: str) -> float:
    subset = [observation for observation in observations if observation.split == split]
    return _weighted_mean([observation.degradation for observation in subset], [observation.weight for observation in subset])


def build_learning_vpm(
    observations: Sequence[LearningObservation | Mapping[str, Any]],
    *,
    recipe: Optional[LayoutRecipe] = None,
    min_train_delta: float = 0.05,
    min_heldout_delta: float = 0.02,
    max_regression_degradation: float = 0.01,
    provenance: Optional[Mapping[str, Any]] = None,
) -> LearningAssessment:
    """Build a learning-evidence VPM and decide whether learning is demonstrated.

    The returned assessment marks ``learned=True`` only when:

    1. training/corrected units improve enough;
    2. held-out or future units also improve enough;
    3. regression observations do not degrade beyond the allowed threshold.
    """
    normalized_observations = tuple(
        item if isinstance(item, LearningObservation) else LearningObservation(**dict(item))
        for item in observations
    )
    if not normalized_observations:
        raise VPMValidationError("build_learning_vpm requires at least one observation")

    row_ids = tuple("%s:%s" % (observation.split, observation.unit_id) for observation in normalized_observations)
    if len(set(row_ids)) != len(row_ids):
        raise VPMValidationError("Learning observations require unique split:unit_id row identifiers")

    table = ScoreTable(
        values=[observation.metric_row() for observation in normalized_observations],
        row_ids=row_ids,
        metric_ids=LEARNING_METRICS,
        metadata={
            "kind": "learning_trace",
            "observations": [observation.to_dict() for observation in normalized_observations],
        },
    )

    train_delta = _mean_delta(normalized_observations, "train")
    heldout_delta = _mean_delta(normalized_observations, "heldout")
    regression_degradation = _mean_degradation(normalized_observations, "regression")
    learned = bool(
        train_delta >= float(min_train_delta)
        and heldout_delta >= float(min_heldout_delta)
        and regression_degradation <= float(max_regression_degradation)
    )

    artifact = build_vpm(
        table,
        recipe or learning_recipe(),
        provenance={
            "kind": "learning_trace",
            "parents": [],
            "train_delta": train_delta,
            "heldout_delta": heldout_delta,
            "regression_degradation": regression_degradation,
            "learned": learned,
            **dict(provenance or {}),
        },
    )

    return LearningAssessment(
        artifact=artifact,
        observations=normalized_observations,
        train_delta=train_delta,
        heldout_delta=heldout_delta,
        regression_degradation=regression_degradation,
        learned=learned,
        thresholds={
            "min_train_delta": float(min_train_delta),
            "min_heldout_delta": float(min_heldout_delta),
            "max_regression_degradation": float(max_regression_degradation),
        },
    )
