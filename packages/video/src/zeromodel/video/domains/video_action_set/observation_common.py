from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.dto import CanonicalJsonDTO


def is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and value.startswith("sha256:")
        and len(value) == 71
        and all(item in "0123456789abcdef" for item in value[7:])
    )


def sha256(value: object, message: str) -> str:
    if not is_sha256(value):
        raise VPMValidationError(message)
    return str(value)


def optional_sha256(value: object, message: str) -> str | None:
    if value is None:
        return None
    return sha256(value, message)


def require_keys(
    payload: Mapping[str, object],
    keys: tuple[str, ...],
    message: str,
) -> None:
    if set(payload) != set(keys):
        raise VPMValidationError(message)


def mapping(value: object, message: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise VPMValidationError(message)
    return cast(Mapping[str, object], value)


def sequence(value: object, message: str) -> Sequence[object]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise VPMValidationError(message)
    return cast(Sequence[object], value)


def string(payload: Mapping[str, object], key: str, message: str) -> str:
    value = payload[key]
    if not isinstance(value, str) or not value:
        raise VPMValidationError(message)
    return value


def optional_string(
    payload: Mapping[str, object],
    key: str,
    message: str,
) -> str | None:
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise VPMValidationError(message)
    return value


def integer(payload: Mapping[str, object], key: str, message: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise VPMValidationError(message)
    return value


def boolean(payload: Mapping[str, object], key: str, message: str) -> bool:
    value = payload[key]
    if not isinstance(value, bool):
        raise VPMValidationError(message)
    return value


def string_tuple(value: object, message: str) -> tuple[str | None, ...]:
    items = sequence(value, message)
    result: list[str | None] = []
    for item in items:
        result.append(None if item is None else sha256(item, message))
    return tuple(result)


def json_mapping(dto: CanonicalJsonDTO, message: str) -> Mapping[str, object]:
    return mapping(dto.to_value(), message)


__all__ = [
    "boolean",
    "integer",
    "is_sha256",
    "json_mapping",
    "mapping",
    "optional_sha256",
    "optional_string",
    "require_keys",
    "sequence",
    "sha256",
    "string",
    "string_tuple",
]
