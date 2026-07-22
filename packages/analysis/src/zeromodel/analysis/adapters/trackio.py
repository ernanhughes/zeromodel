"""Trackio training telemetry export adapter.

Supports JSON/JSONL/CSV metric exports shaped as flat rows or rows with nested
``metrics`` dictionaries. No Trackio runtime dependency is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from zeromodel.analysis.training import TrainingCheckpoint

from zeromodel.analysis.adapters.common import checkpoints_from_export

TRACKIO_DEFAULT_ALIASES = {
    "train/loss": "train_loss",
    "train_loss": "train_loss",
    "eval/loss": "eval_loss",
    "validation/loss": "eval_loss",
    "eval/accuracy": "heldout_score",
    "validation/accuracy": "heldout_score",
    "eval/reward": "heldout_score",
    "regression_safety": "regression_safety",
    "safety/regression": "regression_safety",
    "tokens/sec": "tokens_per_second",
    "tokens_per_second": "tokens_per_second",
}


def checkpoints_from_trackio_export(
    path: str | Path,
    *,
    step_keys: Sequence[str] = ("step", "global_step", "_step"),
    metric_aliases: Mapping[str, str] | None = None,
) -> list[TrainingCheckpoint]:
    """Load checkpoints from a Trackio metrics export."""
    aliases = dict(TRACKIO_DEFAULT_ALIASES)
    aliases.update(dict(metric_aliases or {}))
    return checkpoints_from_export(path, step_keys=step_keys, metric_aliases=aliases)
