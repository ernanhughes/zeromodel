"""Temporal decision manifolds for dense ZeroModel panels.

A single VPM is a spatial view over one scored table. A decision manifold is the
smallest temporal extension: a sequence of dense panels, each converted into an
optimized view, then summarized by how its spatial decision landscape changes
over time.

This module deliberately avoids claims about semantic correctness or decision
accuracy. It exposes deterministic geometry over scored panels: optimized mass,
metric-weight movement, row/column order movement, metric affinity, and
inflection points worth inspecting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import ScoreTable, VPMArtifact, VPMValidationError
from zeromodel.analysis.spatial import SpatialOptimizationResult, SpatialOptimizer
from zeromodel.core.views import build_view


SourcePanel = ScoreTable | VPMArtifact


def _as_table(source: SourcePanel) -> ScoreTable:
    if isinstance(source, VPMArtifact):
        return source.source
    if isinstance(source, ScoreTable):
        return source
    raise VPMValidationError(
        "decision manifold panels must be ScoreTable or VPMArtifact"
    )


def _as_tables(source: Sequence[SourcePanel]) -> tuple[ScoreTable, ...]:
    tables = tuple(_as_table(item) for item in source)
    if not tables:
        raise VPMValidationError("decision manifold requires at least one panel")
    metric_ids = tables[0].metric_ids
    row_ids = tables[0].row_ids
    shape = tables[0].shape
    for table in tables[1:]:
        if table.shape != shape:
            raise VPMValidationError(
                "all decision manifold panels must have identical shape"
            )
        if table.metric_ids != metric_ids:
            raise VPMValidationError(
                "all decision manifold panels must have identical metric_ids"
            )
        if table.row_ids != row_ids:
            raise VPMValidationError(
                "all decision manifold panels must have identical row_ids"
            )
    return tables


def _weight_vector(result: SpatialOptimizationResult) -> np.ndarray:
    return np.array(
        [float(result.metric_weights[metric_id]) for metric_id in result.metric_ids],
        dtype=np.float64,
    )


def _permutation_delta(left: Sequence[int], right: Sequence[int]) -> float:
    """Return normalized average absolute position change for two permutations."""
    if len(left) != len(right):
        raise VPMValidationError("permutation delta requires equal-length permutations")
    if not left:
        return 0.0
    left_pos = {int(value): index for index, value in enumerate(left)}
    right_pos = {int(value): index for index, value in enumerate(right)}
    if set(left_pos) != set(right_pos):
        raise VPMValidationError("permutation delta requires the same elements")
    denom = max(1, len(left) - 1)
    total = sum(abs(left_pos[value] - right_pos[value]) / denom for value in left_pos)
    return float(total / len(left))


@dataclass(frozen=True)
class ManifoldFrame:
    """One optimized panel in a temporal decision manifold."""

    frame_index: int
    source_digest: str
    artifact: VPMArtifact
    optimization: SpatialOptimizationResult
    top_left_mass: float
    row_order: Tuple[int, ...]
    column_order: Tuple[int, ...]
    metric_weights: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "source_digest": self.source_digest,
            "artifact_id": self.artifact.artifact_id,
            "optimization": self.optimization.to_dict(),
            "top_left_mass": float(self.top_left_mass),
            "row_order": list(self.row_order),
            "column_order": list(self.column_order),
            "metric_weights": dict(self.metric_weights),
        }


@dataclass(frozen=True)
class ManifoldTransition:
    """Change summary between adjacent manifold frames."""

    from_index: int
    to_index: int
    mass_delta: float
    weight_delta: float
    row_order_delta: float
    column_order_delta: float
    curvature: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "from_index": int(self.from_index),
            "to_index": int(self.to_index),
            "mass_delta": float(self.mass_delta),
            "weight_delta": float(self.weight_delta),
            "row_order_delta": float(self.row_order_delta),
            "column_order_delta": float(self.column_order_delta),
            "curvature": float(self.curvature),
        }


@dataclass(frozen=True)
class ManifoldSummary:
    """Resolved temporal manifold over optimized dense views."""

    frames: Tuple[ManifoldFrame, ...]
    transitions: Tuple[ManifoldTransition, ...]
    inflection_indices: Tuple[int, ...]
    metric_ids: Tuple[str, ...]
    metric_graph: np.ndarray
    objective: Mapping[str, Any] = field(default_factory=dict)

    @property
    def mass_series(self) -> Tuple[float, ...]:
        return tuple(float(frame.top_left_mass) for frame in self.frames)

    @property
    def curvature_series(self) -> Tuple[float, ...]:
        values = [0.0] * len(self.frames)
        for transition in self.transitions:
            values[transition.to_index] = float(transition.curvature)
        return tuple(values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames": [frame.to_dict() for frame in self.frames],
            "transitions": [transition.to_dict() for transition in self.transitions],
            "inflection_indices": list(self.inflection_indices),
            "metric_ids": list(self.metric_ids),
            "metric_graph": self.metric_graph.tolist(),
            "mass_series": list(self.mass_series),
            "curvature_series": list(self.curvature_series),
            "objective": dict(self.objective),
        }


class DecisionManifold:
    """Build a temporal manifold from a sequence of dense scored panels.

    Each panel is fitted with ``SpatialOptimizer`` and converted into a normal
    VPM artifact. The manifold then compares adjacent frames to surface where
    the optimized view, top-left mass, or spatial order changes most.
    """

    def __init__(
        self,
        optimizer: Optional[SpatialOptimizer] = None,
        *,
        inflection_top_k: int = 3,
        mass_weight: float = 1.0,
        metric_weight: float = 1.0,
        row_order_weight: float = 0.5,
        column_order_weight: float = 0.5,
    ) -> None:
        if inflection_top_k <= 0:
            raise VPMValidationError("inflection_top_k must be positive")
        for name, value in {
            "mass_weight": mass_weight,
            "metric_weight": metric_weight,
            "row_order_weight": row_order_weight,
            "column_order_weight": column_order_weight,
        }.items():
            if float(value) < 0.0:
                raise VPMValidationError("%s must be non-negative" % name)
        self.optimizer = optimizer or SpatialOptimizer()
        self.inflection_top_k = int(inflection_top_k)
        self.mass_weight = float(mass_weight)
        self.metric_weight = float(metric_weight)
        self.row_order_weight = float(row_order_weight)
        self.column_order_weight = float(column_order_weight)

    def build(
        self,
        panels: Sequence[SourcePanel],
        *,
        name: str = "decision-manifold",
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> ManifoldSummary:
        tables = _as_tables(panels)
        frames: list[ManifoldFrame] = []

        for index, table in enumerate(tables):
            optimization = self.optimizer.fit(
                table,
                name="%s-frame-%04d" % (name, index),
                metadata={
                    "manifold": name,
                    "frame_index": index,
                    **dict(metadata or {}),
                },
            )
            artifact = build_view(
                table,
                optimization.profile,
                provenance={
                    "kind": "decision_manifold_frame",
                    "manifold": name,
                    "frame_index": index,
                    "spatial_optimization": optimization.to_dict(),
                },
            )
            frames.append(
                ManifoldFrame(
                    frame_index=index,
                    source_digest="sha256:%s" % table.digest,
                    artifact=artifact,
                    optimization=optimization,
                    top_left_mass=self.optimizer.top_left_mass(
                        artifact.normalized_values
                    ),
                    row_order=tuple(artifact.row_order),
                    column_order=tuple(artifact.column_order),
                    metric_weights=dict(optimization.metric_weights),
                )
            )

        transitions = self._transitions(frames)
        inflections = find_inflection_points(transitions, top_k=self.inflection_top_k)
        metric_graph = self.metric_graph(frames)
        return ManifoldSummary(
            frames=tuple(frames),
            transitions=tuple(transitions),
            inflection_indices=inflections,
            metric_ids=tuple(str(metric_id) for metric_id in tables[0].metric_ids),
            metric_graph=metric_graph,
            objective={
                "name": "temporal_spatial_change",
                "optimizer": self.optimizer.fit(tables[0]).objective,
                "inflection_top_k": self.inflection_top_k,
                "mass_weight": self.mass_weight,
                "metric_weight": self.metric_weight,
                "row_order_weight": self.row_order_weight,
                "column_order_weight": self.column_order_weight,
            },
        )

    def _transitions(self, frames: Sequence[ManifoldFrame]) -> list[ManifoldTransition]:
        transitions: list[ManifoldTransition] = []
        for left, right in zip(frames, frames[1:]):
            left_weights = _weight_vector(left.optimization)
            right_weights = _weight_vector(right.optimization)
            mass_delta = abs(float(right.top_left_mass) - float(left.top_left_mass))
            weight_delta = float(np.sum(np.abs(right_weights - left_weights)))
            row_order_delta = _permutation_delta(left.row_order, right.row_order)
            column_order_delta = _permutation_delta(
                left.column_order, right.column_order
            )
            curvature = (
                self.mass_weight * mass_delta
                + self.metric_weight * weight_delta
                + self.row_order_weight * row_order_delta
                + self.column_order_weight * column_order_delta
            )
            transitions.append(
                ManifoldTransition(
                    from_index=left.frame_index,
                    to_index=right.frame_index,
                    mass_delta=mass_delta,
                    weight_delta=weight_delta,
                    row_order_delta=row_order_delta,
                    column_order_delta=column_order_delta,
                    curvature=float(curvature),
                )
            )
        return transitions

    def metric_graph(self, frames: Sequence[ManifoldFrame]) -> np.ndarray:
        """Return a simple metric affinity graph from frame weight co-activation."""
        if not frames:
            raise VPMValidationError("metric_graph requires at least one frame")
        vectors = np.stack(
            [_weight_vector(frame.optimization) for frame in frames], axis=0
        )
        graph = np.zeros((vectors.shape[1], vectors.shape[1]), dtype=np.float64)
        for vector in vectors:
            graph += np.outer(vector, vector)
        graph /= float(len(frames))
        return graph


def find_inflection_points(
    transitions: Sequence[ManifoldTransition],
    *,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> Tuple[int, ...]:
    """Return frame indices where adjacent-frame curvature is largest.

    ``top_k`` selects the largest curvature transitions. ``threshold`` selects
    all transitions at or above a fixed curvature. If neither is supplied, a
    mean-plus-standard-deviation threshold is used.
    """
    items = tuple(transitions)
    if not items:
        return tuple()
    if top_k is not None and top_k <= 0:
        raise VPMValidationError("top_k must be positive when provided")

    if threshold is None and top_k is None:
        curvatures = np.array(
            [transition.curvature for transition in items], dtype=np.float64
        )
        threshold = float(curvatures.mean() + curvatures.std())

    if threshold is not None:
        selected = [
            transition
            for transition in items
            if transition.curvature >= float(threshold)
        ]
    else:
        selected = list(items)

    if top_k is not None:
        selected = sorted(
            selected,
            key=lambda transition: (-transition.curvature, transition.to_index),
        )[: int(top_k)]

    return tuple(sorted(int(transition.to_index) for transition in selected))


def build_decision_manifold(
    panels: Sequence[SourcePanel],
    *,
    optimizer: Optional[SpatialOptimizer] = None,
    name: str = "decision-manifold",
    inflection_top_k: int = 3,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ManifoldSummary:
    """Convenience wrapper for building a temporal decision manifold."""
    manifold = DecisionManifold(optimizer=optimizer, inflection_top_k=inflection_top_k)
    return manifold.build(panels, name=name, metadata=metadata)
