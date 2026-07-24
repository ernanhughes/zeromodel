"""Deterministic addressable VPM field schemas for Stage P4A.

P4A establishes exact spatial fields only. It does not estimate relevance, assign
importance, create Evidence VPMs, or claim causal meaning.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Iterable, Mapping

import numpy as np

from .representation import SourceVPMDTO

FIELD_SCHEMA_VERSION: Final = "perception-field-schema/1"
FIELD_SAMPLE_VERSION: Final = "perception-field-sample/1"
_ALLOWED_CHANNEL_MODES: Final = {"joint", "separate"}
_ALLOWED_MASK_MODES: Final = {"keep", "remove"}


class PerceptionFieldError(ValueError):
    """Raised when a VPM field contract is invalid or incompatible."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class VPMFieldAddressDTO:
    """One half-open rectangular field in a declared channel range."""

    field_id: str
    field_schema_id: str
    channel_start: int
    channel_end: int
    x0: int
    y0: int
    x1: int
    y1: int
    level: int = 0

    def __post_init__(self) -> None:
        if not self.field_id or not self.field_schema_id:
            raise PerceptionFieldError("field identities must be non-empty")
        if self.channel_start < 0 or self.channel_end <= self.channel_start:
            raise PerceptionFieldError("field channel range must be positive")
        if self.x0 < 0 or self.y0 < 0 or self.x1 <= self.x0 or self.y1 <= self.y0:
            raise PerceptionFieldError("field spatial bounds must be positive")
        if self.level < 0:
            raise PerceptionFieldError("field level must be non-negative")

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def channels(self) -> int:
        return self.channel_end - self.channel_start


@dataclass(frozen=True)
class VPMFieldSchemaDTO:
    """Reusable deterministic partition contract for one source shape/spec."""

    field_schema_id: str
    source_encoder_spec_id: str
    width: int
    height: int
    channels: int
    tile_width: int
    tile_height: int
    channel_mode: str
    fields: tuple[VPMFieldAddressDTO, ...]
    version: str = FIELD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.field_schema_id or not self.source_encoder_spec_id:
            raise PerceptionFieldError("schema identities must be non-empty")
        if min(self.width, self.height, self.channels, self.tile_width, self.tile_height) <= 0:
            raise PerceptionFieldError("schema dimensions must be positive")
        if self.channel_mode not in _ALLOWED_CHANNEL_MODES:
            raise PerceptionFieldError(
                f"channel_mode must be one of {sorted(_ALLOWED_CHANNEL_MODES)}"
            )
        if not self.fields:
            raise PerceptionFieldError("field schema requires at least one field")
        ids = tuple(field.field_id for field in self.fields)
        if ids != tuple(sorted(ids)) or len(ids) != len(set(ids)):
            raise PerceptionFieldError("fields must be unique and sorted by field_id")
        if any(field.field_schema_id != self.field_schema_id for field in self.fields):
            raise PerceptionFieldError("every field must reference its owning schema")
        self._validate_exact_partition()

    def _validate_exact_partition(self) -> None:
        coverage = np.zeros((self.height, self.width, self.channels), dtype=np.uint8)
        for field in self.fields:
            if field.x1 > self.width or field.y1 > self.height:
                raise PerceptionFieldError("field exceeds schema spatial bounds")
            if field.channel_end > self.channels:
                raise PerceptionFieldError("field exceeds schema channel bounds")
            coverage[
                field.y0 : field.y1,
                field.x0 : field.x1,
                field.channel_start : field.channel_end,
            ] += 1
        if not np.all(coverage == 1):
            raise PerceptionFieldError("fields must cover every source value exactly once")

    def field(self, field_id: str) -> VPMFieldAddressDTO:
        for item in self.fields:
            if item.field_id == field_id:
                return item
        raise KeyError(field_id)


@dataclass(frozen=True)
class ExtractedFieldDTO:
    """Exact immutable values extracted from one Source VPM field."""

    sample_id: str
    source_vpm_id: str
    field_schema_id: str
    field_id: str
    shape: tuple[int, int, int]
    dtype: str
    values_digest: str
    values: bytes
    version: str = FIELD_SAMPLE_VERSION

    def __post_init__(self) -> None:
        if not all((self.sample_id, self.source_vpm_id, self.field_schema_id, self.field_id)):
            raise PerceptionFieldError("sample identities must be non-empty")
        if self.dtype != "uint8":
            raise PerceptionFieldError("P4A samples must use uint8")
        expected = self.shape[0] * self.shape[1] * self.shape[2]
        if expected <= 0 or len(self.values) != expected:
            raise PerceptionFieldError("sample byte length does not match shape")

    def to_array(self) -> np.ndarray:
        return np.frombuffer(self.values, dtype=np.uint8).reshape(self.shape).copy()


def _schema_payload(
    source: SourceVPMDTO,
    *,
    tile_width: int,
    tile_height: int,
    channel_mode: str,
) -> Mapping[str, object]:
    return {
        "channel_mode": channel_mode,
        "channels": source.channels,
        "height": source.height,
        "source_encoder_spec_id": source.encoder_spec_id,
        "tile_height": tile_height,
        "tile_width": tile_width,
        "version": FIELD_SCHEMA_VERSION,
        "width": source.width,
    }


def build_grid_field_schema(
    source: SourceVPMDTO,
    *,
    tile_width: int,
    tile_height: int,
    channel_mode: str = "joint",
) -> VPMFieldSchemaDTO:
    """Partition a source shape into deterministic edge-aware rectangular fields."""

    if tile_width <= 0 or tile_height <= 0:
        raise PerceptionFieldError("tile dimensions must be positive")
    if channel_mode not in _ALLOWED_CHANNEL_MODES:
        raise PerceptionFieldError(
            f"channel_mode must be one of {sorted(_ALLOWED_CHANNEL_MODES)}"
        )
    payload = _schema_payload(
        source,
        tile_width=tile_width,
        tile_height=tile_height,
        channel_mode=channel_mode,
    )
    schema_id = _digest(_canonical_json(payload))
    channels = (
        ((0, source.channels),)
        if channel_mode == "joint"
        else tuple((channel, channel + 1) for channel in range(source.channels))
    )
    fields: list[VPMFieldAddressDTO] = []
    for y0 in range(0, source.height, tile_height):
        for x0 in range(0, source.width, tile_width):
            y1 = min(y0 + tile_height, source.height)
            x1 = min(x0 + tile_width, source.width)
            for channel_start, channel_end in channels:
                address_payload: Mapping[str, object] = {
                    "channel_end": channel_end,
                    "channel_start": channel_start,
                    "field_schema_id": schema_id,
                    "level": 0,
                    "x0": x0,
                    "x1": x1,
                    "y0": y0,
                    "y1": y1,
                }
                fields.append(
                    VPMFieldAddressDTO(
                        field_id=_digest(_canonical_json(address_payload)),
                        field_schema_id=schema_id,
                        channel_start=channel_start,
                        channel_end=channel_end,
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                    )
                )
    return VPMFieldSchemaDTO(
        field_schema_id=schema_id,
        source_encoder_spec_id=source.encoder_spec_id,
        width=source.width,
        height=source.height,
        channels=source.channels,
        tile_width=tile_width,
        tile_height=tile_height,
        channel_mode=channel_mode,
        fields=tuple(sorted(fields, key=lambda field: field.field_id)),
    )


def _canonical_source_array(source: SourceVPMDTO) -> np.ndarray:
    array = source.to_array()
    if source.channels == 1:
        return array.reshape(source.height, source.width, 1)
    return array.reshape(source.height, source.width, source.channels)


def validate_source_for_schema(source: SourceVPMDTO, schema: VPMFieldSchemaDTO) -> None:
    if source.encoder_spec_id != schema.source_encoder_spec_id:
        raise PerceptionFieldError("source encoder spec does not match field schema")
    if (source.width, source.height, source.channels) != (
        schema.width,
        schema.height,
        schema.channels,
    ):
        raise PerceptionFieldError("source shape does not match field schema")


def extract_source_fields(
    source: SourceVPMDTO, schema: VPMFieldSchemaDTO
) -> tuple[ExtractedFieldDTO, ...]:
    """Extract every schema field without pooling, quantization, or information loss."""

    validate_source_for_schema(source, schema)
    array = _canonical_source_array(source)
    samples: list[ExtractedFieldDTO] = []
    for field in schema.fields:
        values_array = np.ascontiguousarray(
            array[
                field.y0 : field.y1,
                field.x0 : field.x1,
                field.channel_start : field.channel_end,
            ],
            dtype=np.uint8,
        )
        values = values_array.tobytes()
        values_digest = _digest(values)
        sample_payload: Mapping[str, object] = {
            "field_id": field.field_id,
            "field_schema_id": schema.field_schema_id,
            "shape": list(values_array.shape),
            "source_vpm_id": source.source_vpm_id,
            "values_digest": values_digest,
            "version": FIELD_SAMPLE_VERSION,
        }
        samples.append(
            ExtractedFieldDTO(
                sample_id=_digest(_canonical_json(sample_payload)),
                source_vpm_id=source.source_vpm_id,
                field_schema_id=schema.field_schema_id,
                field_id=field.field_id,
                shape=tuple(int(value) for value in values_array.shape),
                dtype="uint8",
                values_digest=values_digest,
                values=values,
            )
        )
    return tuple(samples)


def reconstruct_source_array(
    samples: Iterable[ExtractedFieldDTO], schema: VPMFieldSchemaDTO
) -> np.ndarray:
    """Reconstruct the exact normalized source values represented by all fields."""

    by_id = {sample.field_id: sample for sample in samples}
    expected_ids = {field.field_id for field in schema.fields}
    if set(by_id) != expected_ids:
        raise PerceptionFieldError("samples must contain exactly one value for every field")
    output = np.zeros((schema.height, schema.width, schema.channels), dtype=np.uint8)
    for field in schema.fields:
        sample = by_id[field.field_id]
        if sample.field_schema_id != schema.field_schema_id:
            raise PerceptionFieldError("sample field schema identity mismatch")
        if _digest(sample.values) != sample.values_digest:
            raise PerceptionFieldError("sample value digest mismatch")
        expected_shape = (field.height, field.width, field.channels)
        if sample.shape != expected_shape:
            raise PerceptionFieldError("sample shape does not match field address")
        output[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ] = sample.to_array()
    if schema.channels == 1:
        return output[:, :, 0]
    return output


def mask_source_fields(
    source: SourceVPMDTO,
    schema: VPMFieldSchemaDTO,
    field_ids: Iterable[str],
    *,
    mode: str,
    neutral_value: int = 0,
) -> np.ndarray:
    """Apply a deterministic keep/remove mask without claiming intervention validity."""

    validate_source_for_schema(source, schema)
    if mode not in _ALLOWED_MASK_MODES:
        raise PerceptionFieldError(f"mode must be one of {sorted(_ALLOWED_MASK_MODES)}")
    if not 0 <= neutral_value <= 255:
        raise PerceptionFieldError("neutral_value must be in [0, 255]")
    selected = set(field_ids)
    known = {field.field_id for field in schema.fields}
    unknown = selected - known
    if unknown:
        raise PerceptionFieldError(f"unknown field identities: {sorted(unknown)}")
    source_array = _canonical_source_array(source)
    output = (
        np.full_like(source_array, neutral_value)
        if mode == "keep"
        else source_array.copy()
    )
    for field in schema.fields:
        is_selected = field.field_id in selected
        region = np.s_[
            field.y0 : field.y1,
            field.x0 : field.x1,
            field.channel_start : field.channel_end,
        ]
        if mode == "keep" and is_selected:
            output[region] = source_array[region]
        elif mode == "remove" and is_selected:
            output[region] = neutral_value
    if schema.channels == 1:
        return output[:, :, 0]
    return output
