from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any, Callable

import numpy as np

from ...artifact import VPMValidationError
from ...matrix_blob import MatrixBlob
from .canonical_json import canonical_json_bytes
from .contracts import FRAME_SHAPE

_record_loader: Callable[[Mapping[str, object]], "MaterializedObservationDTO"] | None
_record_loader = None


@dataclass(frozen=True, slots=True)
class MaterializedObservationDTO:
    observation: Any
    matrix_blob: MatrixBlob | None

    def __post_init__(self) -> None:
        if self.observation.split == "final":
            raise VPMValidationError(
                "final split observation materialization is prohibited"
            )
        if self.matrix_blob is None:
            if self.observation.matrix_blob_id is not None:
                raise VPMValidationError("observation matrix blob mismatch")
            return
        if self.observation.matrix_blob_id != self.matrix_blob.blob_id:
            raise VPMValidationError("observation matrix blob mismatch")
        validate_observation_matrix_blob(self.observation, self.matrix_blob)

    @classmethod
    def from_record(
        cls,
        record: Mapping[str, object],
    ) -> MaterializedObservationDTO:
        if _record_loader is None:
            raise VPMValidationError("observation record loader is not registered")
        return _record_loader(record)

    def to_record(self, *, include_pixels: bool = True) -> dict[str, object]:
        return self.observation.to_record(
            matrix_blob=self.matrix_blob,
            include_pixels=include_pixels,
        )


def validate_observation_matrix_blob(
    observation: Any,
    blob: MatrixBlob,
) -> None:
    if blob.dtype != "uint8" or tuple(blob.shape) != FRAME_SHAPE:
        raise VPMValidationError(
            "observation matrix blob does not match declared pixels"
        )
    if _pixel_digest_from_bytes(blob.data) != observation.observation_pixel_digest:
        raise VPMValidationError(
            "observation matrix blob does not match declared pixels"
        )
    metadata = dict(blob.metadata)
    if metadata != {
        "kind": "video_action_set_frame_pixels",
        "pixel_digest": observation.observation_pixel_digest,
    }:
        raise VPMValidationError(
            "observation matrix blob does not match declared pixels"
        )
    descriptor = observation.provider_observation_descriptor
    if descriptor is None:
        raise VPMValidationError("provider observation descriptor mismatch")
    if descriptor.shape != tuple(
        blob.shape
    ) or descriptor.raw_digest != _provider_raw_digest(blob):
        raise VPMValidationError("provider observation descriptor mismatch")


def blob_from_record(
    record: Mapping[str, object],
    pixel_digest: str | None,
) -> MatrixBlob | None:
    pixels = record.get("pixels")
    if pixels is None:
        return None
    if pixel_digest is None or _pixel_digest_from_array(pixels) != pixel_digest:
        raise VPMValidationError("observation pixel digest mismatch")
    return MatrixBlob.from_array(
        np.ascontiguousarray(pixels, dtype=np.uint8),
        dtype="uint8",
        metadata={
            "kind": "video_action_set_frame_pixels",
            "pixel_digest": pixel_digest,
        },
    )


def register_materialized_observation_loader(
    loader: Callable[[Mapping[str, object]], MaterializedObservationDTO],
) -> None:
    global _record_loader
    _record_loader = loader


def _pixel_digest_from_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _pixel_digest_from_array(pixels: object) -> str:
    return _pixel_digest_from_bytes(
        np.ascontiguousarray(pixels, dtype=np.uint8).tobytes(order="C")
    )


def _provider_raw_digest(blob: MatrixBlob) -> str:
    payload = (
        b"zeromodel.image-observation.raw.v1\0"
        + canonical_json_bytes(list(blob.shape))
        + blob.data
    )
    return "sha256:" + hashlib.sha256(payload).hexdigest()


__all__ = [
    "MaterializedObservationDTO",
    "blob_from_record",
    "register_materialized_observation_loader",
    "validate_observation_matrix_blob",
]
