from __future__ import annotations

from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

from zeromodel.artifacts import ArtifactRef, canonical_json_bytes, sha256_digest
from zeromodel.core.artifact import VPMValidationError


def freeze_json(value: Any) -> Any:
    if isinstance(value, np.generic):
        raise VPMValidationError("metadata must use plain JSON scalar types")
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): freeze_json(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    return value


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def require_nonempty(value: object, field: str) -> str:
    text = str(value)
    if not text:
        raise VPMValidationError(f"{field} must be non-empty")
    return text


def require_unique(values: Sequence[object], field: str) -> None:
    normalized = [str(value) for value in values]
    if len(set(normalized)) != len(normalized):
        raise VPMValidationError(f"{field} must not contain duplicates")


def require_finite(value: float, field: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise VPMValidationError(f"{field} must be finite")
    return number


def ref_payload(ref: ArtifactRef) -> dict[str, str]:
    return {"artifact_kind": ref.artifact_kind, "artifact_id": ref.artifact_id}


def refs_payload(refs: Sequence[ArtifactRef]) -> list[dict[str, str]]:
    return [ref_payload(ref) for ref in refs]


def ref_from_dict(data: Mapping[str, Any]) -> ArtifactRef:
    return ArtifactRef(
        artifact_kind=str(data["artifact_kind"]),
        artifact_id=str(data["artifact_id"]),
    )


def refs_from_dict(items: Sequence[Mapping[str, Any]]) -> tuple[ArtifactRef, ...]:
    return tuple(ref_from_dict(item) for item in items)


def digest_payload(kind: str, payload: Mapping[str, Any]) -> str:
    return sha256_digest(canonical_json_bytes({"kind": kind, "payload": payload}))
