"""Tiny edge-style consumers for VPM artifacts.

These consumers demonstrate the blog claim that a deployed decision can inspect
an already-built artifact without running a model at decision time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from zeromodel.analysis.compose import as_field


@dataclass(frozen=True)
class TopLeftGateResult:
    accepted: bool
    score: float
    threshold: float
    rows: int
    columns: int


class TopLeftGate:
    """Evaluate the mean intensity in a top-left region."""

    def __init__(
        self,
        threshold: float = 0.75,
        rows: int | None = None,
        columns: int | None = None,
        fraction: float = 0.25,
    ):
        self.threshold = float(threshold)
        self.rows = rows
        self.columns = columns
        self.fraction = float(fraction)

    def evaluate(self, field: Any) -> TopLeftGateResult:
        arr = as_field(field)
        if self.rows is None:
            rows = max(1, int(round(np.sqrt(self.fraction) * arr.shape[0])))
        else:
            rows = int(self.rows)
        if self.columns is None:
            columns = max(1, int(round(np.sqrt(self.fraction) * arr.shape[1])))
        else:
            columns = int(self.columns)
        rows = max(1, min(rows, arr.shape[0]))
        columns = max(1, min(columns, arr.shape[1]))
        score = float(arr[:rows, :columns].mean())
        return TopLeftGateResult(
            accepted=bool(score >= self.threshold),
            score=score,
            threshold=self.threshold,
            rows=rows,
            columns=columns,
        )
