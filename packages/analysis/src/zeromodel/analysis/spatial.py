"""Spatial optimization for dense ZeroModel view profiles.

The core artifact layer makes one dense source table inspectable. ``ViewProfile``
lets a caller choose a policy lens over that table. This module adds the first
spatial-calculus step: learn a deterministic metric-weight view profile that
concentrates high-signal mass into the top-left inspection region.

This module deliberately avoids claiming task accuracy, semantic correctness, or
universal optimality. It optimizes one explicit geometric objective over scored
matrices: top-left mass after column and row ordering.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import ScoreTable, VPMArtifact, VPMValidationError
from zeromodel.core.views import ViewProfile, build_view


def _normalize_per_metric(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise VPMValidationError("spatial optimization expects 2D matrices")
    if not np.isfinite(matrix).all():
        raise VPMValidationError("spatial optimization values must be finite")
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    ranges = maxs - mins
    out = np.zeros_like(matrix, dtype=np.float64)
    active = ranges > 0.0
    out[:, active] = (matrix[:, active] - mins[active]) / ranges[active]
    constant = ~active
    if np.any(constant):
        out[:, constant] = matrix[:, constant]
    return np.clip(out, 0.0, 1.0)


def _as_table(source: ScoreTable | VPMArtifact) -> ScoreTable:
    if isinstance(source, VPMArtifact):
        return source.source
    if isinstance(source, ScoreTable):
        return source
    raise VPMValidationError("spatial optimizer requires a ScoreTable or VPMArtifact")


def _as_tables(source: ScoreTable | VPMArtifact | Sequence[ScoreTable | VPMArtifact]) -> tuple[ScoreTable, ...]:
    if isinstance(source, (ScoreTable, VPMArtifact)):
        return (_as_table(source),)
    tables = tuple(_as_table(item) for item in source)
    if not tables:
        raise VPMValidationError("spatial optimizer requires at least one source table")
    first_metrics = tables[0].metric_ids
    for table in tables[1:]:
        if table.metric_ids != first_metrics:
            raise VPMValidationError("all spatial optimizer tables must have identical metric_ids")
    return tables


def _simplex(weights: np.ndarray) -> np.ndarray:
    values = np.asarray(weights, dtype=np.float64)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    total = float(values.sum())
    if total <= 0.0:
        return np.full(values.shape, 1.0 / values.size, dtype=np.float64)
    return values / total


@dataclass(frozen=True)
class SpatialOptimizationResult:
    """Result of fitting a spatial top-left concentration profile."""

    profile: ViewProfile
    metric_ids: Tuple[str, ...]
    metric_weights: Mapping[str, float]
    canonical_metric_ids: Tuple[str, ...]
    baseline_mass: float
    optimized_mass: float
    improvement: float
    iterations: int
    objective: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile.to_dict(),
            "metric_ids": list(self.metric_ids),
            "metric_weights": dict(self.metric_weights),
            "canonical_metric_ids": list(self.canonical_metric_ids),
            "baseline_mass": float(self.baseline_mass),
            "optimized_mass": float(self.optimized_mass),
            "improvement": float(self.improvement),
            "iterations": int(self.iterations),
            "objective": dict(self.objective),
        }


class SpatialOptimizer:
    """Learn a deterministic view profile for top-left mass concentration.

    ``Kc`` controls how many prominent columns participate in row scoring.
    ``Kr`` controls how many prominent rows contribute to the top-left objective.
    ``alpha`` controls spatial decay from the top-left corner.

    The optimizer is intentionally small and NumPy-only. It uses deterministic
    coordinate ascent over non-negative metric weights and emits a ``ViewProfile``
    rather than a separate layout format.
    """

    def __init__(
        self,
        *,
        Kc: int = 16,
        Kr: int = 32,
        alpha: float = 0.95,
        l2: float = 1e-3,
        max_iters: int | None = None,
        max_evals: int | None = None,
        min_step: float = 1e-4,
    ) -> None:
        if Kc <= 0:
            raise VPMValidationError("Kc must be positive")
        if Kr <= 0:
            raise VPMValidationError("Kr must be positive")
        if not (0.0 < alpha < 1.0):
            raise VPMValidationError("alpha must be in (0, 1)")
        if l2 < 0.0:
            raise VPMValidationError("l2 must be non-negative")
        if max_evals is not None and max_iters is not None and int(max_evals) != int(max_iters):
            raise VPMValidationError("use max_evals or max_iters, not conflicting values")
        eval_budget = int(max_evals if max_evals is not None else (80 if max_iters is None else max_iters))
        if eval_budget <= 0:
            raise VPMValidationError("max_evals must be positive")
        if min_step <= 0.0:
            raise VPMValidationError("min_step must be positive")
        self.Kc = int(Kc)
        self.Kr = int(Kr)
        self.alpha = float(alpha)
        self.l2 = float(l2)
        self.max_evals = eval_budget
        # Backwards-compatible alias for callers that still inspect max_iters.
        self.max_iters = eval_budget
        self.min_step = float(min_step)

    def top_left_mass(self, ordered_values: np.ndarray) -> float:
        """Return spatially decayed mass in the top-left ``Kr`` x ``Kc`` block."""
        values = np.asarray(ordered_values, dtype=np.float64)
        if values.ndim != 2:
            raise VPMValidationError("top_left_mass requires a 2D matrix")
        rows = min(self.Kr, values.shape[0])
        cols = min(self.Kc, values.shape[1])
        if rows == 0 or cols == 0:
            return 0.0
        i, j = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        decay = self.alpha ** (i + j)
        return float(np.sum(values[:rows, :cols] * decay))

    def ordered_values(self, values: np.ndarray, weights: Sequence[float]) -> tuple[np.ndarray, tuple[int, ...], tuple[int, ...]]:
        """Order columns by weight and rows by weighted intensity."""
        matrix = _normalize_per_metric(np.asarray(values, dtype=np.float64))
        w = _simplex(np.asarray(weights, dtype=np.float64))
        if matrix.shape[1] != w.size:
            raise VPMValidationError("weight count must match matrix column count")

        column_order = np.argsort(-w, kind="stable")
        k = min(self.Kc, matrix.shape[1])
        weighted_columns = column_order[:k]
        row_scores = matrix[:, weighted_columns] @ w[weighted_columns]
        row_order = np.argsort(-row_scores, kind="stable")
        ordered = matrix[np.ix_(row_order, column_order)]
        return ordered, tuple(int(i) for i in row_order), tuple(int(i) for i in column_order)

    def score_weights(self, tables: Sequence[ScoreTable], weights: Sequence[float]) -> float:
        """Evaluate a metric-weight vector on the explicit top-left objective."""
        w = _simplex(np.asarray(weights, dtype=np.float64))
        total = 0.0
        for table in tables:
            ordered, _, _ = self.ordered_values(table.values, w)
            total += self.top_left_mass(ordered)
        mean_mass = total / max(1, len(tables))
        return float(mean_mass - self.l2 * np.sum(w**2))

    def _initial_candidates(self, tables: Sequence[ScoreTable]) -> list[np.ndarray]:
        metric_count = len(tables[0].metric_ids)
        candidates: list[np.ndarray] = []
        candidates.append(np.full(metric_count, 1.0 / metric_count, dtype=np.float64))

        mean_signal = np.zeros(metric_count, dtype=np.float64)
        variance_signal = np.zeros(metric_count, dtype=np.float64)
        for table in tables:
            matrix = _normalize_per_metric(table.values)
            mean_signal += matrix.mean(axis=0)
            variance_signal += matrix.var(axis=0)
        candidates.append(_simplex(mean_signal + 1e-9))
        candidates.append(_simplex(mean_signal + variance_signal + 1e-9))

        for index in range(metric_count):
            one_hot = np.zeros(metric_count, dtype=np.float64)
            one_hot[index] = 1.0
            candidates.append(one_hot)
        return candidates

    def learn_weights(self, source: ScoreTable | VPMArtifact | Sequence[ScoreTable | VPMArtifact]) -> tuple[np.ndarray, int, float, float]:
        """Learn non-negative metric weights for the source table or table series."""
        tables = _as_tables(source)
        metric_count = len(tables[0].metric_ids)
        if metric_count == 0:
            raise VPMValidationError("spatial optimizer requires at least one metric")

        candidates = self._initial_candidates(tables)
        best = max(candidates, key=lambda weights: self.score_weights(tables, weights))
        best = _simplex(best)
        baseline = self.score_weights(tables, np.full(metric_count, 1.0 / metric_count))
        best_score = self.score_weights(tables, best)

        step = 0.50
        evaluations = 0
        while evaluations < self.max_evals and step >= self.min_step:
            improved = False
            for metric_index in range(metric_count):
                candidate = best.copy()
                candidate[metric_index] += step
                candidate = _simplex(candidate)
                score = self.score_weights(tables, candidate)
                evaluations += 1
                if score > best_score + 1e-12:
                    best = candidate
                    best_score = score
                    improved = True
                if evaluations >= self.max_evals:
                    break
            if not improved:
                step *= 0.5
        return best, evaluations, float(baseline), float(best_score)

    def fit(
        self,
        source: ScoreTable | VPMArtifact | Sequence[ScoreTable | VPMArtifact],
        *,
        name: str = "spatial-optimized",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> SpatialOptimizationResult:
        """Fit and return a ``ViewProfile`` optimized for top-left mass."""
        tables = _as_tables(source)
        metric_ids = tuple(str(metric_id) for metric_id in tables[0].metric_ids)
        weights, evaluations, baseline, optimized = self.learn_weights(tables)
        weight_map = {metric_id: float(weights[index]) for index, metric_id in enumerate(metric_ids)}
        canonical_metric_ids = tuple(
            metric_id
            for metric_id, _ in sorted(
                weight_map.items(),
                key=lambda item: (-float(item[1]), item[0]),
            )
        )
        profile = ViewProfile(
            name=name,
            metric_weights=weight_map,
            metadata={
                "kind": "spatial_optimization",
                "objective": "top_left_mass",
                "Kc": self.Kc,
                "Kr": self.Kr,
                "alpha": self.alpha,
                "l2": self.l2,
                **dict(metadata or {}),
            },
        )
        return SpatialOptimizationResult(
            profile=profile,
            metric_ids=metric_ids,
            metric_weights=weight_map,
            canonical_metric_ids=canonical_metric_ids,
            baseline_mass=baseline,
            optimized_mass=optimized,
            improvement=float(optimized - baseline),
            iterations=evaluations,
            objective={
                "name": "top_left_mass",
                "Kc": self.Kc,
                "Kr": self.Kr,
                "alpha": self.alpha,
                "l2": self.l2,
                "max_evals": self.max_evals,
                "max_iters_alias": self.max_iters,
                "min_step": self.min_step,
            },
        )


def optimize_view_profile(
    source: ScoreTable | VPMArtifact | Sequence[ScoreTable | VPMArtifact],
    *,
    name: str = "spatial-optimized",
    optimizer: Optional[SpatialOptimizer] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> SpatialOptimizationResult:
    """Convenience wrapper for fitting an optimized ``ViewProfile``."""
    return (optimizer or SpatialOptimizer()).fit(source, name=name, metadata=metadata)


def build_optimized_view(
    source: ScoreTable | VPMArtifact | Sequence[ScoreTable | VPMArtifact],
    *,
    name: str = "spatial-optimized",
    optimizer: Optional[SpatialOptimizer] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> VPMArtifact:
    """Fit an optimized profile and build a VPM view for the first source table."""
    result = optimize_view_profile(source, name=name, optimizer=optimizer, metadata=metadata)
    table = _as_tables(source)[0]
    return build_view(
        table,
        result.profile,
        provenance={
            "kind": "spatial_optimized_view",
            "spatial_optimization": result.to_dict(),
        },
    )
