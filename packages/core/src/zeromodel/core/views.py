"""Multi-view profiles for dense ZeroModel artifacts.

A ZeroModel source table can contain many possible signals at once. A
``ViewProfile`` is a policy lens over that dense table: it turns selected metric
weights up or down, emits a deterministic layout recipe, and builds a VPM view
without regenerating or mutating the underlying evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

from zeromodel.core.artifact import (
    LayoutRecipe,
    ScoreTable,
    VPMArtifact,
    VPMValidationError,
    build_vpm,
)


@dataclass(frozen=True)
class ViewProfile:
    """A policy lens over a dense score table.

    Positive metric weights mean "higher is more salient". Negative metric
    weights mean "lower is more salient". Building a view does not alter the
    source table; it only changes row and column ordering for inspection.
    """

    name: str
    metric_weights: Mapping[str, float]
    include_unweighted_columns: bool = True
    normalization_clip: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        if not name:
            raise VPMValidationError("ViewProfile name must be non-empty")

        weights: dict[str, float] = {}
        for metric_id, weight in dict(self.metric_weights).items():
            metric = str(metric_id).strip()
            if not metric:
                raise VPMValidationError("ViewProfile metric ids must be non-empty")
            number = float(weight)
            if not np.isfinite(number):
                raise VPMValidationError(
                    "ViewProfile weight for %s must be finite" % metric
                )
            if abs(number) <= 1e-12:
                continue
            weights[metric] = number
        if not weights:
            raise VPMValidationError(
                "ViewProfile requires at least one non-zero metric weight"
            )

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "metric_weights", weights)
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_metric(
        cls, metric_id: str, *, name: Optional[str] = None, weight: float = 1.0
    ) -> "ViewProfile":
        """Create a one-metric view profile."""
        return cls(
            name=name or str(metric_id), metric_weights={str(metric_id): float(weight)}
        )

    def weighted_keys(self) -> tuple[dict[str, Any], ...]:
        """Return LayoutRecipe weighted-score keys for this profile."""
        keys: list[dict[str, Any]] = []
        for metric_id, weight in sorted(
            self.metric_weights.items(),
            key=lambda item: (-abs(float(item[1])), item[0]),
        ):
            keys.append(
                {
                    "metric_id": metric_id,
                    "direction": "desc" if weight >= 0 else "asc",
                    "weight": abs(float(weight)),
                }
            )
        return tuple(keys)

    def _metric_sets(
        self, table: ScoreTable
    ) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        metrics = tuple(str(metric_id) for metric_id in table.metric_ids)
        missing = sorted(set(self.metric_weights) - set(metrics))
        if missing:
            raise VPMValidationError(
                "ViewProfile references unknown metrics: %s" % ", ".join(missing)
            )
        weighted = tuple(
            metric_id
            for metric_id, _ in sorted(
                self.metric_weights.items(),
                key=lambda item: (-abs(float(item[1])), item[0]),
            )
        )
        unweighted = tuple(metric for metric in metrics if metric not in set(weighted))
        return metrics, weighted, unweighted

    def column_metric_ids(self, table: ScoreTable) -> tuple[str, ...]:
        """Return the full metric order for this profile.

        VPM artifacts preserve a full column permutation for source mapping. When
        ``include_unweighted_columns`` is false, unweighted columns are still
        retained after the weighted columns, but the recipe records them as
        hidden so renderers and consumers can crop or de-emphasize them honestly.
        """
        _, weighted, unweighted = self._metric_sets(table)
        return weighted + unweighted

    def visible_metric_ids(self, table: ScoreTable) -> tuple[str, ...]:
        """Return metrics intended to be visible under this profile."""
        metrics, weighted, _ = self._metric_sets(table)
        return metrics if self.include_unweighted_columns else weighted

    def hidden_metric_ids(self, table: ScoreTable) -> tuple[str, ...]:
        """Return retained-but-hidden metrics under this profile."""
        _, _, unweighted = self._metric_sets(table)
        return tuple() if self.include_unweighted_columns else unweighted

    def to_recipe(self, table: ScoreTable) -> LayoutRecipe:
        """Build a deterministic layout recipe for ``table``."""
        metric_ids = self.column_metric_ids(table)
        view_profile = {
            **self.to_dict(),
            "visible_metric_ids": list(self.visible_metric_ids(table)),
            "hidden_metric_ids": list(self.hidden_metric_ids(table)),
        }
        return LayoutRecipe.from_dict(
            {
                "version": "vpm-layout/0",
                "name": "view:%s" % self.name,
                "row_order": {
                    "kind": "weighted_score",
                    "keys": list(self.weighted_keys()),
                    "tie_break": "row_id",
                },
                "column_order": {
                    "kind": "explicit",
                    "metric_ids": list(metric_ids),
                },
                "normalization": {
                    "kind": "per_metric_minmax",
                    "clip": bool(self.normalization_clip),
                },
                "view_profile": view_profile,
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_weights": dict(self.metric_weights),
            "include_unweighted_columns": bool(self.include_unweighted_columns),
            "normalization_clip": bool(self.normalization_clip),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class ViewSet:
    """A named collection of related view profiles."""

    profiles: Tuple[ViewProfile, ...]

    def __init__(self, profiles: Sequence[ViewProfile]) -> None:
        normalized = tuple(profiles)
        if not normalized:
            raise VPMValidationError("ViewSet requires at least one profile")
        names = [profile.name for profile in normalized]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise VPMValidationError(
                "Duplicate ViewProfile names: %s" % ", ".join(duplicates)
            )
        object.__setattr__(self, "profiles", normalized)

    def build(self, source: ScoreTable | VPMArtifact) -> dict[str, VPMArtifact]:
        """Build one VPM artifact per profile."""
        return {profile.name: build_view(source, profile) for profile in self.profiles}


def build_view(
    source: ScoreTable | VPMArtifact,
    profile: ViewProfile,
    *,
    provenance: Optional[Mapping[str, Any]] = None,
) -> VPMArtifact:
    """Build a VPM view from a score table or existing artifact.

    Passing an existing artifact reuses its source table and records the parent
    artifact id in provenance. The source evidence remains unchanged; only the
    view policy changes.
    """
    if isinstance(source, VPMArtifact):
        table = source.source
        parents = [source.artifact_id]
    elif isinstance(source, ScoreTable):
        table = source
        parents = []
    else:
        raise VPMValidationError(
            "build_view requires a ScoreTable or VPMArtifact source"
        )

    recipe = profile.to_recipe(table)
    merged_provenance = {
        "kind": "view_profile",
        "view_profile": profile.to_dict(),
        "visible_metric_ids": list(profile.visible_metric_ids(table)),
        "hidden_metric_ids": list(profile.hidden_metric_ids(table)),
        "source_digest": "sha256:%s" % table.digest,
        "parents": parents,
    }
    if provenance:
        merged_provenance.update(dict(provenance))
    return build_vpm(table, recipe, provenance=merged_provenance)


def build_views(
    source: ScoreTable | VPMArtifact,
    profiles: Sequence[ViewProfile],
) -> dict[str, VPMArtifact]:
    """Build many named views from one dense source."""
    return ViewSet(profiles).build(source)
