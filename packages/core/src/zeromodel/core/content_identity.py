from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
from struct import pack
from typing import Any, Mapping

import numpy as np

from zeromodel.core.artifact import VPMValidationError

PROTOTYPE_UNIVERSE_IDENTITY_VERSION = "zeromodel-video-prototype-universe/v1"


@dataclass(frozen=True)
class PrototypeUniverseIdentity:
    version: str
    policy_artifact_id: str
    source_scope: str
    row_ids: tuple[str, ...]
    digest: str


@dataclass(frozen=True)
class UnresolvedArtifactIdentity:
    label: str
    reason: str

    def __post_init__(self) -> None:
        if self.label.startswith("sha256:"):
            raise VPMValidationError(
                "unresolved identities must not masquerade as sha256 digests"
            )


def _normalize_scalar(value: Any) -> Any:
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise VPMValidationError("canonical JSON rejects non-finite floats")
        return value
    if isinstance(value, np.generic):
        return _normalize_scalar(value.item())
    raise VPMValidationError(f"unsupported canonical JSON scalar: {type(value)!r}")


def _normalize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        items = sorted((str(key), _normalize_json(item)) for key, item in value.items())
        return OrderedDict(items)
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return _normalize_scalar(value)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        normalized = _normalize_json(value)
        return json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("value is not canonically JSON serializable") from exc


def sha256_digest(value: Any) -> str:
    payload = (
        value
        if isinstance(value, (bytes, bytearray, memoryview))
        else canonical_json_bytes(value)
    )
    return "sha256:" + hashlib.sha256(bytes(payload)).hexdigest()


def canonical_float64_bytes(value: float) -> bytes:
    number = float(value)
    if not np.isfinite(number):
        raise VPMValidationError("raw score must be finite")
    return pack(">d", np.float64(number).item())


def array_content_digest(array: np.ndarray) -> str:
    pixels = np.asarray(array)
    if pixels.ndim < 1:
        raise VPMValidationError("array_content_digest requires an array")
    canonical = np.ascontiguousarray(pixels)
    payload = {
        "dtype": canonical.dtype.str,
        "shape": list(canonical.shape),
        "content_hex": canonical.tobytes(order="C").hex(),
    }
    return sha256_digest(payload)


def prototype_universe_identity(
    *,
    prototypes: Mapping[str, tuple[str, str, str, Any]],
    policy_artifact_id: str,
    source_scope: str,
) -> PrototypeUniverseIdentity:
    rows = []
    for observation_id, (row_id, action_id, _digest, observation) in sorted(
        prototypes.items()
    ):
        rows.append(
            {
                "observation_id": str(observation_id),
                "row_id": str(row_id),
                "action_id": str(action_id),
                "array_dtype": observation.pixels.dtype.str,
                "array_shape": list(observation.pixels.shape),
                "pixel_digest": array_content_digest(observation.pixels),
            }
        )
    digest = sha256_digest(
        {
            "version": PROTOTYPE_UNIVERSE_IDENTITY_VERSION,
            "policy_artifact_id": policy_artifact_id,
            "source_scope": source_scope,
            "rows": rows,
        }
    )
    return PrototypeUniverseIdentity(
        version=PROTOTYPE_UNIVERSE_IDENTITY_VERSION,
        policy_artifact_id=policy_artifact_id,
        source_scope=source_scope,
        row_ids=tuple(row["row_id"] for row in rows),
        digest=digest,
    )


__all__ = [
    "PROTOTYPE_UNIVERSE_IDENTITY_VERSION",
    "PrototypeUniverseIdentity",
    "UnresolvedArtifactIdentity",
    "array_content_digest",
    "canonical_float64_bytes",
    "canonical_json_bytes",
    "prototype_universe_identity",
    "sha256_digest",
]
