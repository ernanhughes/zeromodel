from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
from typing import Any, Callable

import numpy as np

from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.video.domains.video_action_set.canonical_json import canonical_json_bytes
from zeromodel.video.domains.video_action_set.contracts import FRAME_SHAPE
from zeromodel.video.domains.video_action_set.pixel_digest import (
    array_digest,
    pixel_digest_from_bytes,
)

_record_loader: Callable[..., "MaterializedObservationDTO"] | None
_record_loader = None


@dataclass(frozen=True, slots=True)
class MaterializedObservationDTO:
    observation: Any
    matrix_blob: MatrixBlob | None
    final_access_id: str | None = None

    def __post_init__(self) -> None:
        if self.observation.split == "final" and self.final_access_id is None:
            raise VPMValidationError(
                "final split observation materialization is prohibited"
            )
        if self.final_access_id is not None:
            if self.observation.split != "final":
                raise VPMValidationError("final access id is only valid for final")
            if self.observation.final_access_id != self.final_access_id:
                raise VPMValidationError("final access id mismatch")
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

    @classmethod
    def from_authorized_final_record(
        cls,
        record: Mapping[str, object],
        *,
        final_access_id: str,
    ) -> MaterializedObservationDTO:
        if _record_loader is None:
            raise VPMValidationError("observation record loader is not registered")
        return _record_loader(record, final_access_id=final_access_id)

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
    if pixel_digest_from_bytes(blob.data) != observation.observation_pixel_digest:
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
    normalized_pixels = np.ascontiguousarray(pixels, dtype=np.uint8)
    if pixel_digest is None or array_digest(normalized_pixels) != pixel_digest:
        raise VPMValidationError("observation pixel digest mismatch")
    return MatrixBlob.from_array(
        normalized_pixels,
        dtype="uint8",
        metadata={
            "kind": "video_action_set_frame_pixels",
            "pixel_digest": pixel_digest,
        },
    )


def register_materialized_observation_loader(
    loader: Callable[..., MaterializedObservationDTO],
) -> None:
    global _record_loader
    _record_loader = loader


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
