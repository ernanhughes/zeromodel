"""State-addressed policy lookup over immutable VPM artifacts.

This module is a tiny consumer layer: it does not change artifact identity, source
mapping, normalization, or rendering.  It treats source rows as discretized state
signs and metric columns as candidate actions, then returns the best action for a
runtime state row.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .artifact import VPMArtifact, VPMCell, VPMValidationError


@dataclass(frozen=True)
class PolicyLookupDecision:
    """A single state-addressed decision recovered from a VPM artifact."""

    artifact_id: str
    row_id: str
    action: str
    value: float
    source_row_index: int
    source_metric_index: int
    view_row: int
    view_column: int
    candidates: Mapping[str, float]

    @property
    def metric_id(self) -> str:
        """Alias for the action metric that produced the selected sign."""
        return self.action

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "row_id": self.row_id,
            "action": self.action,
            "metric_id": self.metric_id,
            "value": float(self.value),
            "source_row_index": int(self.source_row_index),
            "source_metric_index": int(self.source_metric_index),
            "view_row": int(self.view_row),
            "view_column": int(self.view_column),
            "candidates": {str(k): float(v) for k, v in self.candidates.items()},
        }


class VPMPolicyLookup:
    """Read an action sign from a state-addressed VPM artifact.

    Rows are discretized states. Metrics are actions. The consumer receives a
    row id for the current state, reads the candidate action values in that row,
    and returns the deterministic argmax plus the exact cell/source coordinates
    that produced the decision.

    The default comparison uses raw source values because action values in a
    policy table are usually comparable within a row.  Use
    ``value_source="normalized"`` only when the policy intentionally wants to
    compare rendered view intensities instead.
    """

    def __init__(
        self,
        artifact: VPMArtifact,
        *,
        action_metric_ids: Optional[Sequence[str]] = None,
        value_source: str = "raw",
        tie_break: str = "metric_order",
    ) -> None:
        if value_source not in {"raw", "normalized"}:
            raise VPMValidationError("value_source must be 'raw' or 'normalized'")
        if tie_break not in {"metric_order", "metric_id"}:
            raise VPMValidationError("tie_break must be 'metric_order' or 'metric_id'")

        actions = tuple(str(metric_id) for metric_id in (action_metric_ids or artifact.source.metric_ids))
        if not actions:
            raise VPMValidationError("VPMPolicyLookup requires at least one action metric")
        missing = [metric_id for metric_id in actions if metric_id not in artifact.source.metric_ids]
        if missing:
            raise VPMValidationError("Unknown action metric ids: %s" % ", ".join(sorted(missing)))

        self.artifact = artifact
        self.action_metric_ids = actions
        self.value_source = value_source
        self.tie_break = tie_break
        self._row_view_index = {
            artifact.source.row_ids[source_row_index]: view_row
            for view_row, source_row_index in enumerate(artifact.row_order)
        }
        self._metric_view_index = {
            artifact.source.metric_ids[source_metric_index]: view_column
            for view_column, source_metric_index in enumerate(artifact.column_order)
        }

    def read(self, row_id: str) -> PolicyLookupDecision:
        """Return the selected action for ``row_id``.

        The returned decision contains the selected action and the exact VPM cell
        coordinates needed to audit or replay the move.
        """
        key = str(row_id)
        if key not in self._row_view_index:
            raise VPMValidationError("Unknown policy row_id: %s" % key)

        view_row = self._row_view_index[key]
        cells: list[tuple[int, str, VPMCell, float]] = []
        candidates: Dict[str, float] = {}
        for rank, metric_id in enumerate(self.action_metric_ids):
            view_column = self._metric_view_index[metric_id]
            cell = self.artifact.cell(view_row, view_column)
            value = cell.raw_value if self.value_source == "raw" else cell.normalized_value
            if not np.isfinite(value):
                raise VPMValidationError("Policy value must be finite for %s/%s" % (key, metric_id))
            candidates[metric_id] = float(value)
            cells.append((rank, metric_id, cell, float(value)))

        if self.tie_break == "metric_id":
            _, action, cell, value = max(cells, key=lambda item: (item[3], tuple(-ord(ch) for ch in item[1])))
        else:
            _, action, cell, value = max(cells, key=lambda item: (item[3], -item[0]))

        return PolicyLookupDecision(
            artifact_id=self.artifact.artifact_id,
            row_id=key,
            action=action,
            value=value,
            source_row_index=cell.source_row_index,
            source_metric_index=cell.source_metric_index,
            view_row=cell.view_row,
            view_column=cell.view_column,
            candidates=candidates,
        )

    def trace(self, row_ids: Sequence[str]) -> tuple[PolicyLookupDecision, ...]:
        """Read many state rows and return a deterministic decision trace."""
        return tuple(self.read(row_id) for row_id in row_ids)


# Blog-facing vocabulary: the code API remains VPMPolicyLookup, while the 1.0
# explanation can talk about the reader as a sign reader.
SignReader = VPMPolicyLookup
