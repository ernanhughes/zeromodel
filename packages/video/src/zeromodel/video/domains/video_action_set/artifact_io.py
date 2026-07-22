"""Video action-set artifact io."""

from __future__ import annotations

import json
from pathlib import Path
from typing import (
    Any,
    Mapping,
)
from zeromodel.video.domains.video_action_set.canonical_json import (
    canonical_json_bytes,
    canonical_json_value,
    canonical_sha256,
)


def _json_ready(value: Any) -> Any:
    return canonical_json_value(value)


def _json_bytes(value: Any) -> bytes:
    return canonical_json_bytes(value)


def _sha256(value: Any) -> str:
    return canonical_sha256(value)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(_json_ready(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(_json_ready(row[key]), sort_keys=True)
                    if isinstance(row.get(key), (dict, list, tuple))
                    else row.get(key, "")
                    for key in fieldnames
                }
            )


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
