"""Derived evidence metrics for Q-bearing policy tables."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from zeromodel.core.artifact import ScoreTable, VPMValidationError

CRITICALITY_METRIC_ID = "criticality"
DECISION_MARGIN_METRIC_ID = "decision_margin"


def with_q_diagnostics(
    table: ScoreTable,
    *,
    action_metric_ids: Sequence[str],
    criticality_metric_id: str = CRITICALITY_METRIC_ID,
    decision_margin_metric_id: str = DECISION_MARGIN_METRIC_ID,
) -> ScoreTable:
    """Return ``table`` with criticality and decision-margin evidence columns.

    ``criticality`` is the best-minus-worst action value used by VIPER-style
    critical-state weighting when the source columns carry Q-values or an
    equivalent consequence-bearing teacher signal.

    ``decision_margin`` is the best-minus-second-best action value. It measures
    how decisively the winning action defeats its nearest alternative.

    The returned object is a new immutable :class:`ScoreTable`; the source table
    is not mutated. Runtime consumers should continue to pass only the original
    action metric ids to ``VPMPolicyLookup.action_metric_ids``.
    """

    actions = tuple(str(metric_id) for metric_id in action_metric_ids)
    if len(actions) < 2:
        raise VPMValidationError("with_q_diagnostics requires at least two action metrics")
    if len(set(actions)) != len(actions):
        raise VPMValidationError("action_metric_ids must be unique")

    missing = [metric_id for metric_id in actions if metric_id not in table.metric_ids]
    if missing:
        raise VPMValidationError(
            "Unknown action metric ids: %s" % ", ".join(sorted(missing))
        )

    derived_ids = (str(criticality_metric_id), str(decision_margin_metric_id))
    if len(set(derived_ids)) != len(derived_ids):
        raise VPMValidationError("diagnostic metric ids must be distinct")
    conflicts = [metric_id for metric_id in derived_ids if metric_id in table.metric_ids]
    if conflicts:
        raise VPMValidationError(
            "Diagnostic metric ids already exist: %s" % ", ".join(sorted(conflicts))
        )

    action_indices = [table.metric_index(metric_id) for metric_id in actions]
    action_values = np.asarray(table.values[:, action_indices], dtype=np.float64)

    best = np.max(action_values, axis=1)
    worst = np.min(action_values, axis=1)
    sorted_values = np.sort(action_values, axis=1)
    second_best = sorted_values[:, -2]

    criticality = best - worst
    decision_margin = best - second_best
    values = np.column_stack((table.values, criticality, decision_margin))

    metadata = dict(table.to_identity_payload()["metadata"])
    metadata["policy_diagnostics"] = {
        "version": "q-policy-diagnostics/1",
        "action_metric_ids": list(actions),
        "criticality": {
            "metric_id": derived_ids[0],
            "definition": "best_minus_worst",
            "requires": "q_values_or_equivalent_consequence_signal",
        },
        "decision_margin": {
            "metric_id": derived_ids[1],
            "definition": "best_minus_second_best",
        },
    }

    return ScoreTable(
        values=values,
        row_ids=table.row_ids,
        metric_ids=table.metric_ids + derived_ids,
        metadata=metadata,
    )
