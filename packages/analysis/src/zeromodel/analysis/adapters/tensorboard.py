"""TensorBoard scalar export adapter.

This adapter parses exported scalar CSV/JSON/JSONL files. It deliberately does not
parse TensorBoard event protobufs directly, keeping ZeroModel free of TensorBoard
runtime dependencies.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from zeromodel.analysis.training import TrainingCheckpoint

from zeromodel.analysis.adapters.common import checkpoints_from_export

TENSORBOARD_DEFAULT_ALIASES = {
    "train/loss": "train_loss",
    "loss/train": "train_loss",
    "eval/loss": "eval_loss",
    "validation/loss": "eval_loss",
    "eval/accuracy": "heldout_score",
    "validation/accuracy": "heldout_score",
    "eval/reward": "heldout_score",
    "tokens/sec": "tokens_per_second",
    "tokens_per_second": "tokens_per_second",
}


def checkpoints_from_tensorboard_scalars(
    path: str | Path,
    *,
    step_keys: Sequence[str] = ("step", "global_step"),
    tag_key: str = "tag",
    value_key: str = "value",
    metric_aliases: Mapping[str, str] | None = None,
) -> list[TrainingCheckpoint]:
    """Load checkpoints from TensorBoard scalar exports.

    TensorBoard's scalar dashboard can export rows shaped like
    ``wall_time,step,tag,value``. Those rows are grouped into one checkpoint per
    step.
    """
    aliases = dict(TENSORBOARD_DEFAULT_ALIASES)
    aliases.update(dict(metric_aliases or {}))
    return checkpoints_from_export(
        path,
        step_keys=step_keys,
        tag_key=tag_key,
        value_key=value_key,
        metric_aliases=aliases,
    )
