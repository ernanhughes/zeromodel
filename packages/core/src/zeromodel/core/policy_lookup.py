"""State-addressed policy lookup over immutable VPM artifacts.

This module is a tiny consumer layer: it does not change artifact identity, source
mapping, normalization, or rendering. It treats source rows as discretized state
signs and metric columns as candidate actions, then returns the best action for a
runtime state row.

The reader compiles its immutable artifact and fixed lookup configuration into a
compact in-memory execution plan at construction time. Runtime reads therefore do
not allocate one ``VPMCell`` per candidate or recompute the winning action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from zeromodel.core.artifact import VPMArtifact, VPMValidationError


POLICY_PLAN_VERSION = "zeromodel-policy-plan/v1"


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
    evidence: Mapping[str, float] = field(default_factory=dict)

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
            "evidence": {str(k): float(v) for k, v in self.evidence.items()},
        }


class VPMPolicyLookup:
    """Read action signs from a state-addressed VPM artifact.

    Rows are discretized states. Metrics are actions. The consumer receives a
    row id for the current state and returns the deterministic argmax plus the
    exact cell/source coordinates that produced the decision.

    Optional evidence metrics are returned alongside the decision but never
    participate in action selection. Tables produced by ``with_q_diagnostics``
    declare their action and evidence metrics in metadata, so the reader can use
    the safe separation automatically when those arguments are omitted.

    The default comparison uses raw source values because action values in a
    policy table are usually comparable within a row. Use
    ``value_source="normalized"`` only when the policy intentionally wants to
    compare rendered view intensities instead.

    Construction compiles row/metric mappings, candidate matrices, evidence
    matrices and the winning action for every immutable source row. ``choose``
    is the minimal hot path; ``read`` adds the full candidate/evidence trace.
    """

    def __init__(
        self,
        artifact: VPMArtifact,
        *,
        action_metric_ids: Optional[Sequence[str]] = None,
        evidence_metric_ids: Optional[Sequence[str]] = None,
        value_source: str = "raw",
        evidence_value_source: str = "raw",
        tie_break: str = "metric_order",
    ) -> None:
        if value_source not in {"raw", "normalized"}:
            raise VPMValidationError("value_source must be 'raw' or 'normalized'")
        if evidence_value_source not in {"raw", "normalized"}:
            raise VPMValidationError(
                "evidence_value_source must be 'raw' or 'normalized'"
            )
        if tie_break not in {"metric_order", "metric_id"}:
            raise VPMValidationError("tie_break must be 'metric_order' or 'metric_id'")

        diagnostic_metadata = artifact.source.metadata.get("policy_diagnostics")
        declared_actions: Sequence[str] | None = None
        declared_evidence: list[str] = []
        if isinstance(diagnostic_metadata, Mapping):
            metadata_actions = diagnostic_metadata.get("action_metric_ids")
            if isinstance(metadata_actions, (tuple, list)):
                declared_actions = tuple(str(value) for value in metadata_actions)
            for key in ("criticality", "decision_margin"):
                item = diagnostic_metadata.get(key)
                if isinstance(item, Mapping) and item.get("metric_id") is not None:
                    declared_evidence.append(str(item["metric_id"]))

        action_source = (
            action_metric_ids
            if action_metric_ids is not None
            else declared_actions or artifact.source.metric_ids
        )
        evidence_source = (
            evidence_metric_ids
            if evidence_metric_ids is not None
            else declared_evidence
        )

        actions = tuple(str(metric_id) for metric_id in action_source)
        if not actions:
            raise VPMValidationError(
                "VPMPolicyLookup requires at least one action metric"
            )
        if len(set(actions)) != len(actions):
            raise VPMValidationError("action_metric_ids must be unique")

        evidence = tuple(str(metric_id) for metric_id in evidence_source)
        if len(set(evidence)) != len(evidence):
            raise VPMValidationError("evidence_metric_ids must be unique")
        overlap = sorted(set(actions).intersection(evidence))
        if overlap:
            raise VPMValidationError(
                "Evidence metrics cannot also be action metrics: %s"
                % ", ".join(overlap)
            )

        metric_source_index = {
            metric_id: index
            for index, metric_id in enumerate(artifact.source.metric_ids)
        }
        missing_actions = [
            metric_id for metric_id in actions if metric_id not in metric_source_index
        ]
        if missing_actions:
            raise VPMValidationError(
                "Unknown action metric ids: %s" % ", ".join(sorted(missing_actions))
            )
        missing_evidence = [
            metric_id for metric_id in evidence if metric_id not in metric_source_index
        ]
        if missing_evidence:
            raise VPMValidationError(
                "Unknown evidence metric ids: %s" % ", ".join(sorted(missing_evidence))
            )

        self.artifact = artifact
        self.action_metric_ids = actions
        self.evidence_metric_ids = evidence
        self.value_source = value_source
        self.evidence_value_source = evidence_value_source
        self.tie_break = tie_break

        row_count = len(artifact.source.row_ids)
        metric_count = len(artifact.source.metric_ids)

        self._row_source_index = {
            row_id: source_row
            for source_row, row_id in enumerate(artifact.source.row_ids)
        }
        self._row_view_index = {
            artifact.source.row_ids[source_row]: view_row
            for view_row, source_row in enumerate(artifact.row_order)
        }
        self._metric_view_index = {
            artifact.source.metric_ids[source_metric]: view_column
            for view_column, source_metric in enumerate(artifact.column_order)
        }

        source_to_view_row = np.empty(row_count, dtype=np.intp)
        for view_row, source_row in enumerate(artifact.row_order):
            source_to_view_row[source_row] = view_row
        source_to_view_row.flags.writeable = False
        self._source_to_view_row = source_to_view_row

        source_to_view_column = np.empty(metric_count, dtype=np.intp)
        for view_column, source_metric in enumerate(artifact.column_order):
            source_to_view_column[source_metric] = view_column
        source_to_view_column.flags.writeable = False
        self._source_to_view_column = source_to_view_column

        action_source_columns = np.asarray(
            [metric_source_index[metric_id] for metric_id in actions],
            dtype=np.intp,
        )
        action_source_columns.flags.writeable = False
        self._action_source_columns = action_source_columns

        action_view_columns = np.asarray(
            source_to_view_column[action_source_columns],
            dtype=np.intp,
        )
        action_view_columns.flags.writeable = False
        self._action_view_columns = action_view_columns

        evidence_source_columns = np.asarray(
            [metric_source_index[metric_id] for metric_id in evidence],
            dtype=np.intp,
        )
        evidence_source_columns.flags.writeable = False
        self._evidence_source_columns = evidence_source_columns

        evidence_view_columns = np.asarray(
            source_to_view_column[evidence_source_columns],
            dtype=np.intp,
        )
        evidence_view_columns.flags.writeable = False
        self._evidence_view_columns = evidence_view_columns

        self._action_values = self._compile_value_matrix(
            value_source=value_source,
            source_columns=action_source_columns,
            view_columns=action_view_columns,
        )
        self._evidence_values = self._compile_value_matrix(
            value_source=evidence_value_source,
            source_columns=evidence_source_columns,
            view_columns=evidence_view_columns,
        )
        self._winner_indices = self._compile_winner_indices()

    def _compile_value_matrix(
        self,
        *,
        value_source: str,
        source_columns: np.ndarray,
        view_columns: np.ndarray,
    ) -> np.ndarray:
        row_count = len(self.artifact.source.row_ids)
        if source_columns.size == 0:
            values = np.empty((row_count, 0), dtype=np.float64)
        elif value_source == "raw":
            values = np.ascontiguousarray(
                self.artifact.source.values[:, source_columns],
                dtype=np.float64,
            )
        else:
            values = np.ascontiguousarray(
                self.artifact.normalized_values[
                    self._source_to_view_row[:, None],
                    view_columns[None, :],
                ],
                dtype=np.float64,
            )
        if not np.isfinite(values).all():
            raise VPMValidationError("Compiled policy values must be finite")
        values.flags.writeable = False
        return values

    def _compile_winner_indices(self) -> np.ndarray:
        if self.tie_break == "metric_order":
            winners = np.argmax(self._action_values, axis=1).astype(
                np.intp,
                copy=False,
            )
        else:
            winners = np.asarray(
                [
                    max(
                        range(len(self.action_metric_ids)),
                        key=lambda index: (
                            float(row_values[index]),
                            tuple(-ord(ch) for ch in self.action_metric_ids[index]),
                        ),
                    )
                    for row_values in self._action_values
                ],
                dtype=np.intp,
            )
        winners.flags.writeable = False
        return winners

    def _source_row_for(self, row_id: str) -> tuple[str, int]:
        key = str(row_id)
        try:
            return key, self._row_source_index[key]
        except KeyError as exc:
            raise VPMValidationError("Unknown policy row_id: %s" % key) from exc

    def choose(self, row_id: str) -> str:
        """Return only the selected action for the fastest runtime path."""
        _, source_row = self._source_row_for(row_id)
        winner = int(self._winner_indices[source_row])
        return self.action_metric_ids[winner]

    def choose_many(self, row_ids: Sequence[str]) -> tuple[str, ...]:
        """Return selected actions for many row ids without trace allocation."""
        source_rows = [self._source_row_for(row_id)[1] for row_id in row_ids]
        return tuple(
            self.action_metric_ids[int(self._winner_indices[source_row])]
            for source_row in source_rows
        )

    def read(self, row_id: str) -> PolicyLookupDecision:
        """Return the selected action and complete candidate/evidence trace."""
        key, source_row = self._source_row_for(row_id)
        winner = int(self._winner_indices[source_row])
        action = self.action_metric_ids[winner]
        value = float(self._action_values[source_row, winner])
        source_metric_index = int(self._action_source_columns[winner])
        view_row = int(self._source_to_view_row[source_row])
        view_column = int(self._action_view_columns[winner])

        candidates: Dict[str, float] = {
            metric_id: float(candidate_value)
            for metric_id, candidate_value in zip(
                self.action_metric_ids,
                self._action_values[source_row],
            )
        }
        evidence: Dict[str, float] = {
            metric_id: float(evidence_value)
            for metric_id, evidence_value in zip(
                self.evidence_metric_ids,
                self._evidence_values[source_row],
            )
        }

        return PolicyLookupDecision(
            artifact_id=self.artifact.artifact_id,
            row_id=key,
            action=action,
            value=value,
            source_row_index=source_row,
            source_metric_index=source_metric_index,
            view_row=view_row,
            view_column=view_column,
            candidates=candidates,
            evidence=evidence,
        )

    def trace(
        self,
        row_ids: Sequence[str],
    ) -> tuple[PolicyLookupDecision, ...]:
        """Read many state rows and return a deterministic decision trace."""
        return tuple(self.read(row_id) for row_id in row_ids)

    def to_compiled_plan(self) -> dict[str, Any]:
        """Return a portable consumer plan derived from this artifact and reader.

        The plan is not a second policy artifact and does not replace the VPM
        artifact identity. It records the exact artifact and lookup semantics used
        to produce a compact runtime consumer in another language.
        """
        return {
            "format": POLICY_PLAN_VERSION,
            "artifact_id": self.artifact.artifact_id,
            "consumer": "VPMPolicyLookup",
            "value_source": self.value_source,
            "evidence_value_source": self.evidence_value_source,
            "tie_break": self.tie_break,
            "row_ids": list(self.artifact.source.row_ids),
            "action_metric_ids": list(self.action_metric_ids),
            "evidence_metric_ids": list(self.evidence_metric_ids),
            "action_values": self._action_values.tolist(),
            "evidence_values": self._evidence_values.tolist(),
            "winner_indices": [int(value) for value in self._winner_indices],
            "source_to_view_rows": [int(value) for value in self._source_to_view_row],
            "action_source_metric_indices": [
                int(value) for value in self._action_source_columns
            ],
            "action_view_columns": [int(value) for value in self._action_view_columns],
            "evidence_source_metric_indices": [
                int(value) for value in self._evidence_source_columns
            ],
            "evidence_view_columns": [
                int(value) for value in self._evidence_view_columns
            ],
        }


# Blog-facing vocabulary: the code API remains VPMPolicyLookup, while the
# explanation can talk about the reader as a sign reader.
SignReader = VPMPolicyLookup
