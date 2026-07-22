"""Weights & Biases history export adapter.

Supports exported history JSONL/JSON/CSV files. No W&B SDK is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from zeromodel.analysis.training import TrainingCheckpoint

from zeromodel.analysis.adapters.common import checkpoints_from_export

WANDB_DEFAULT_ALIASES = {
    "train/loss": "train_loss",
    "train_loss": "train_loss",
    "eval/loss": "eval_loss",
    "val/loss": "eval_loss",
    "validation/loss": "eval_loss",
    "eval/accuracy": "heldout_score",
    "val/accuracy": "heldout_score",
    "validation/accuracy": "heldout_score",
    "eval/reward": "heldout_score",
    "safety/regression": "regression_safety",
    "regression/safety": "regression_safety",
    "system/tokens_per_second": "tokens_per_second",
    "tokens/sec": "tokens_per_second",
}


def checkpoints_from_wandb_export(
    path: str | Path,
    *,
    step_keys: Sequence[str] = ("_step", "step", "global_step"),
    metric_aliases: Mapping[str, str] | None = None,
) -> list[TrainingCheckpoint]:
    """Load checkpoints from a W&B history export.

    The adapter accepts flat history rows and nested ``metrics`` rows. Metric names
    with slashes are normalized to underscores after aliasing.
    """
    aliases = dict(WANDB_DEFAULT_ALIASES)
    aliases.update(dict(metric_aliases or {}))
    return checkpoints_from_export(path, step_keys=step_keys, metric_aliases=aliases)
