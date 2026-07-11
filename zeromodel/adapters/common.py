"""Common helpers for converting tracker exports into training checkpoints.

The adapters in this package intentionally parse exported files rather than live
tracker APIs. That keeps ZeroModel dependency-light and makes the resulting
training progress artifacts reproducible from checked-in fixtures.
"""
from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from zeromodel.artifact import VPMValidationError
from zeromodel.training import TrainingCheckpoint

_DEFAULT_IGNORE_KEYS = {
    "step",
    "global_step",
    "_step",
    "checkpoint_id",
    "metadata",
    "wall_time",
    "timestamp",
    "time",
    "epoch",
    "tag",
    "name",
    "value",
}


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _load_json(path: Path) -> list[Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for key in ("records", "history", "checkpoints", "rows", "data"):
            if isinstance(payload.get(key), list):
                records = payload[key]
                break
        else:
            records = [payload]
    else:
        raise VPMValidationError("JSON tracker export must be an object or list of objects")
    if not all(isinstance(item, Mapping) for item in records):
        raise VPMValidationError("Tracker export records must be objects")
    return [dict(item) for item in records]


def _load_jsonl(path: Path) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        item = json.loads(text)
        if not isinstance(item, Mapping):
            raise VPMValidationError("JSONL tracker record on line %s must be an object" % line_number)
        records.append(dict(item))
    return records


def _load_csv(path: Path) -> list[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def load_tracker_records(path: str | Path) -> list[Mapping[str, Any]]:
    """Load JSON, JSONL, or CSV tracker records from ``path``."""
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".json":
        return _load_json(source)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(source)
    if suffix == ".csv":
        return _load_csv(source)
    raise VPMValidationError("Unsupported tracker export format: %s" % suffix)


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> str | None:
    for key in keys:
        if key in record:
            return key
    return None


def _metric_name(name: str, aliases: Mapping[str, str]) -> str:
    return str(aliases.get(name, name)).replace("/", "_").replace(" ", "_")


def _metadata_for_record(record: Mapping[str, Any], metadata_keys: Iterable[str]) -> dict[str, Any]:
    return {key: record[key] for key in metadata_keys if key in record}


def records_to_checkpoints(
    records: Sequence[Mapping[str, Any]],
    *,
    step_keys: Sequence[str] = ("step", "global_step", "_step"),
    metrics_key: str = "metrics",
    tag_key: str | None = None,
    value_key: str | None = None,
    checkpoint_id_key: str = "checkpoint_id",
    metric_aliases: Mapping[str, str] | None = None,
    metadata_keys: Sequence[str] = ("wall_time", "timestamp", "time", "epoch"),
    ignore_keys: Iterable[str] = _DEFAULT_IGNORE_KEYS,
) -> list[TrainingCheckpoint]:
    """Convert tracker records into ordered ``TrainingCheckpoint`` objects.

    Two export shapes are supported:

    1. one row per checkpoint, with either a nested ``metrics`` object or flat
       numeric columns;
    2. one row per scalar, with ``tag`` and ``value`` columns that are grouped by
       step, as in TensorBoard scalar CSV exports.
    """
    if not records:
        raise VPMValidationError("Tracker export contained no records")

    aliases = {str(key): str(value) for key, value in dict(metric_aliases or {}).items()}
    ignored = {str(key) for key in ignore_keys}
    by_step: "OrderedDict[int, dict[str, Any]]" = OrderedDict()

    for record in records:
        step_key = _first_present(record, step_keys)
        if step_key is None:
            raise VPMValidationError("Tracker record is missing a step key; tried %r" % (tuple(step_keys),))
        step_number = _as_number(record[step_key])
        if step_number is None:
            raise VPMValidationError("Tracker step value must be numeric")
        step = int(step_number)
        if step not in by_step:
            by_step[step] = {"metrics": {}, "metadata": {}}
        bucket = by_step[step]

        if tag_key and value_key and tag_key in record and value_key in record:
            value = _as_number(record[value_key])
            if value is not None:
                bucket["metrics"][_metric_name(str(record[tag_key]), aliases)] = value
        elif isinstance(record.get(metrics_key), Mapping):
            for key, value in record[metrics_key].items():
                number = _as_number(value)
                if number is not None:
                    bucket["metrics"][_metric_name(str(key), aliases)] = number
        else:
            for key, value in record.items():
                if key in ignored:
                    continue
                number = _as_number(value)
                if number is not None:
                    bucket["metrics"][_metric_name(str(key), aliases)] = number

        bucket["metadata"].update(_metadata_for_record(record, metadata_keys))
        if checkpoint_id_key in record:
            bucket["checkpoint_id"] = str(record[checkpoint_id_key])

    checkpoints: list[TrainingCheckpoint] = []
    for step, payload in by_step.items():
        metrics = payload["metrics"]
        if not metrics:
            continue
        checkpoints.append(
            TrainingCheckpoint(
                step=step,
                checkpoint_id=payload.get("checkpoint_id") or "step_%s" % step,
                metrics=metrics,
                metadata=payload.get("metadata", {}),
            )
        )
    if not checkpoints:
        raise VPMValidationError("Tracker export contained no numeric metrics")
    return sorted(checkpoints, key=lambda checkpoint: checkpoint.step)


def checkpoints_from_export(
    path: str | Path,
    **kwargs: Any,
) -> list[TrainingCheckpoint]:
    """Load records from a tracker export and convert them to checkpoints."""
    return records_to_checkpoints(load_tracker_records(path), **kwargs)
