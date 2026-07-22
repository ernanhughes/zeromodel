"""Generic JSON/JSONL/CSV training telemetry adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from zeromodel.analysis.training import TrainingCheckpoint

from zeromodel.analysis.adapters.common import checkpoints_from_export


def checkpoints_from_jsonl(
    path: str | Path,
    *,
    step_keys: Sequence[str] = ("step", "global_step", "_step"),
    metric_aliases: Mapping[str, str] | None = None,
) -> list[TrainingCheckpoint]:
    """Load checkpoint telemetry from a JSONL export.

    Each line may contain a nested ``metrics`` object or flat numeric metric
    columns. The return value can be passed directly to
    ``build_training_progress_vpm``.
    """
    return checkpoints_from_export(
        path, step_keys=step_keys, metric_aliases=metric_aliases
    )


def checkpoints_from_json(
    path: str | Path,
    *,
    step_keys: Sequence[str] = ("step", "global_step", "_step"),
    metric_aliases: Mapping[str, str] | None = None,
) -> list[TrainingCheckpoint]:
    """Load checkpoint telemetry from a JSON export."""
    return checkpoints_from_export(
        path, step_keys=step_keys, metric_aliases=metric_aliases
    )


def checkpoints_from_csv(
    path: str | Path,
    *,
    step_keys: Sequence[str] = ("step", "global_step", "_step"),
    metric_aliases: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> list[TrainingCheckpoint]:
    """Load checkpoint telemetry from a CSV export."""
    return checkpoints_from_export(
        path, step_keys=step_keys, metric_aliases=metric_aliases, **kwargs
    )
