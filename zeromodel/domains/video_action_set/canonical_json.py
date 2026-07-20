from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import TypeAlias

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


def canonical_json_value(value: object) -> JsonValue:
    if isinstance(value, Mapping):
        return {str(key): canonical_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical_json_value(item) for item in value]
    if value is None or isinstance(value, bool | int | float | str):
        return value
    raise TypeError(f"value is not JSON serializable: {type(value).__name__}")


def canonical_json_text(value: object) -> str:
    return json.dumps(
        canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def canonical_json_bytes(value: object) -> bytes:
    return canonical_json_text(value).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


__all__ = [
    "JsonValue",
    "canonical_json_bytes",
    "canonical_json_text",
    "canonical_json_value",
    "canonical_sha256",
]
