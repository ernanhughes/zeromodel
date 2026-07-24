"""Deterministic source-image and target-action VPM representations.

Stage P1 is intentionally narrow. It normalizes bounded images into canonical PNG
bytes and encodes bounded discrete actions into canonical one-hot PNG fields.
No learning, similarity, evidence weighting, or inference behavior belongs here.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass
from io import BytesIO
from typing import Final, Mapping, Sequence

import numpy as np
from PIL import Image

SOURCE_ENCODER_VERSION: Final = "perception-source-vpm/1"
ACTION_SCHEMA_VERSION: Final = "perception-action-schema/1"
TARGET_ENCODER_VERSION: Final = "perception-target-vpm/1"

_ALLOWED_COLOR_SPACES: Final = {"L": 1, "RGB": 3, "RGBA": 4}


class PerceptionRepresentationError(ValueError):
    """Raised when an image or action cannot be represented canonically."""


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


def _png_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=9)
    return buffer.getvalue()


@dataclass(frozen=True)
class SourceImageEncoderSpecDTO:
    """Explicit bounded normalization contract for source images."""

    color_space: str = "RGB"
    max_width: int = 4096
    max_height: int = 4096
    max_pixels: int = 16_777_216
    max_input_bytes: int = 64 * 1024 * 1024
    version: str = SOURCE_ENCODER_VERSION

    def __post_init__(self) -> None:
        if self.color_space not in _ALLOWED_COLOR_SPACES:
            raise PerceptionRepresentationError(
                f"unsupported color_space {self.color_space!r}; "
                f"expected one of {sorted(_ALLOWED_COLOR_SPACES)}"
            )
        for name, value in (
            ("max_width", self.max_width),
            ("max_height", self.max_height),
            ("max_pixels", self.max_pixels),
            ("max_input_bytes", self.max_input_bytes),
        ):
            if value <= 0:
                raise PerceptionRepresentationError(f"{name} must be positive")
        if not self.version:
            raise PerceptionRepresentationError("version must be non-empty")

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "color_space": self.color_space,
            "max_height": self.max_height,
            "max_input_bytes": self.max_input_bytes,
            "max_pixels": self.max_pixels,
            "max_width": self.max_width,
            "version": self.version,
        }

    @property
    def encoder_spec_id(self) -> str:
        return _digest(_canonical_json(self.canonical_payload()))


@dataclass(frozen=True)
class SourceVPMDTO:
    """Canonical PNG representation of one bounded source image."""

    source_vpm_id: str
    encoder_spec_id: str
    width: int
    height: int
    channels: int
    color_space: str
    dtype: str
    pixel_digest: str
    png_digest: str
    png_bytes: bytes

    def to_array(self) -> np.ndarray:
        """Decode the canonical PNG back into an immutable-shape uint8 array."""

        with Image.open(BytesIO(self.png_bytes)) as image:
            decoded = np.asarray(image.convert(self.color_space), dtype=np.uint8)
        if self.color_space == "L":
            decoded = decoded.reshape(self.height, self.width)
        return decoded


@dataclass(frozen=True)
class DiscreteActionSchemaDTO:
    """Stable ordered vocabulary for canonical discrete action fields."""

    labels: tuple[str, ...]
    version: str = ACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.labels:
            raise PerceptionRepresentationError("action schema requires at least one label")
        if any(not isinstance(label, str) or not label for label in self.labels):
            raise PerceptionRepresentationError("action labels must be non-empty strings")
        if len(set(self.labels)) != len(self.labels):
            raise PerceptionRepresentationError("action labels must be unique")
        if tuple(sorted(self.labels)) != self.labels:
            raise PerceptionRepresentationError(
                "action labels must be supplied in canonical sorted order"
            )
        if not self.version:
            raise PerceptionRepresentationError("version must be non-empty")

    @classmethod
    def from_labels(cls, labels: Sequence[str]) -> "DiscreteActionSchemaDTO":
        return cls(labels=tuple(sorted(labels)))

    def canonical_payload(self) -> Mapping[str, object]:
        return {"labels": list(self.labels), "version": self.version}

    @property
    def action_schema_id(self) -> str:
        return _digest(_canonical_json(self.canonical_payload()))

    def index_of(self, label: str) -> int:
        try:
            return self.labels.index(label)
        except ValueError as exc:
            raise PerceptionRepresentationError(
                f"action {label!r} is not present in schema"
            ) from exc


@dataclass(frozen=True)
class TargetVPMDTO:
    """Canonical one-row grayscale PNG representing one discrete action."""

    target_vpm_id: str
    action_schema_id: str
    action_label: str
    encoder_version: str
    width: int
    height: int
    channels: int
    png_digest: str
    png_bytes: bytes

    def scores(self) -> tuple[int, ...]:
        with Image.open(BytesIO(self.png_bytes)) as image:
            values = np.asarray(image.convert("L"), dtype=np.uint8)
        return tuple(int(value) for value in values.reshape(-1))


def _validate_dimensions(width: int, height: int, spec: SourceImageEncoderSpecDTO) -> None:
    if width <= 0 or height <= 0:
        raise PerceptionRepresentationError("image dimensions must be positive")
    if width > spec.max_width or height > spec.max_height:
        raise PerceptionRepresentationError(
            f"image dimensions {width}x{height} exceed "
            f"maximum {spec.max_width}x{spec.max_height}"
        )
    if width * height > spec.max_pixels:
        raise PerceptionRepresentationError(
            f"image pixel count {width * height} exceeds maximum {spec.max_pixels}"
        )


def _normalize_pil_image(
    image: Image.Image, spec: SourceImageEncoderSpecDTO
) -> Image.Image:
    _validate_dimensions(image.width, image.height, spec)
    # Pillow does not apply EXIF orientation implicitly. This preserves encoded
    # pixel coordinates instead of silently transposing the observation.
    normalized = image.convert(spec.color_space)
    normalized.load()
    return normalized


def encode_source_image_bytes(
    payload: bytes, spec: SourceImageEncoderSpecDTO | None = None
) -> SourceVPMDTO:
    """Decode bounded image bytes and emit a canonical source VPM PNG."""

    resolved = spec or SourceImageEncoderSpecDTO()
    if len(payload) > resolved.max_input_bytes:
        raise PerceptionRepresentationError(
            f"input payload size {len(payload)} exceeds maximum "
            f"{resolved.max_input_bytes}"
        )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(payload)) as image:
                normalized = _normalize_pil_image(image, resolved)
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise PerceptionRepresentationError("image exceeds Pillow safety limits") from exc
    except PerceptionRepresentationError:
        raise
    except Exception as exc:
        raise PerceptionRepresentationError("input is not a supported image") from exc
    return _source_vpm_from_normalized(normalized, resolved)


def encode_source_array(
    values: np.ndarray, spec: SourceImageEncoderSpecDTO | None = None
) -> SourceVPMDTO:
    """Normalize a bounded uint8 NumPy image into a canonical source VPM PNG."""

    resolved = spec or SourceImageEncoderSpecDTO()
    array = np.asarray(values)
    if array.dtype != np.uint8:
        raise PerceptionRepresentationError("source arrays must use dtype uint8")
    expected_channels = _ALLOWED_COLOR_SPACES[resolved.color_space]
    if resolved.color_space == "L":
        if array.ndim != 2:
            raise PerceptionRepresentationError("L source arrays must have shape (H, W)")
        height, width = array.shape
    else:
        if array.ndim != 3 or array.shape[2] != expected_channels:
            raise PerceptionRepresentationError(
                f"{resolved.color_space} source arrays must have shape "
                f"(H, W, {expected_channels})"
            )
        height, width, _ = array.shape
    _validate_dimensions(width, height, resolved)
    normalized = Image.fromarray(np.ascontiguousarray(array), mode=resolved.color_space)
    return _source_vpm_from_normalized(normalized, resolved)


def _source_vpm_from_normalized(
    normalized: Image.Image, spec: SourceImageEncoderSpecDTO
) -> SourceVPMDTO:
    array = np.asarray(normalized, dtype=np.uint8)
    canonical_pixels = np.ascontiguousarray(array).tobytes(order="C")
    png = _png_bytes(normalized)
    channels = _ALLOWED_COLOR_SPACES[spec.color_space]
    pixel_digest = _digest(
        _canonical_json(
            {
                "color_space": spec.color_space,
                "dtype": "uint8",
                "height": normalized.height,
                "width": normalized.width,
            }
        ),
        canonical_pixels,
    )
    png_digest = _digest(png)
    source_vpm_id = _digest(
        spec.encoder_spec_id.encode("ascii"),
        pixel_digest.encode("ascii"),
    )
    return SourceVPMDTO(
        source_vpm_id=source_vpm_id,
        encoder_spec_id=spec.encoder_spec_id,
        width=normalized.width,
        height=normalized.height,
        channels=channels,
        color_space=spec.color_space,
        dtype="uint8",
        pixel_digest=pixel_digest,
        png_digest=png_digest,
        png_bytes=png,
    )


def encode_discrete_action(
    action_label: str, schema: DiscreteActionSchemaDTO
) -> TargetVPMDTO:
    """Encode one schema-owned action as a canonical one-hot grayscale PNG."""

    index = schema.index_of(action_label)
    values = np.zeros((1, len(schema.labels)), dtype=np.uint8)
    values[0, index] = 255
    png = _png_bytes(Image.fromarray(values, mode="L"))
    png_digest = _digest(png)
    target_vpm_id = _digest(
        schema.action_schema_id.encode("ascii"),
        TARGET_ENCODER_VERSION.encode("ascii"),
        action_label.encode("utf-8"),
        png_digest.encode("ascii"),
    )
    return TargetVPMDTO(
        target_vpm_id=target_vpm_id,
        action_schema_id=schema.action_schema_id,
        action_label=action_label,
        encoder_version=TARGET_ENCODER_VERSION,
        width=len(schema.labels),
        height=1,
        channels=1,
        png_digest=png_digest,
        png_bytes=png,
    )


def decode_discrete_action(
    target: TargetVPMDTO, schema: DiscreteActionSchemaDTO
) -> str:
    """Decode and validate a canonical one-hot target VPM."""

    if target.action_schema_id != schema.action_schema_id:
        raise PerceptionRepresentationError("target action schema identity mismatch")
    if target.encoder_version != TARGET_ENCODER_VERSION:
        raise PerceptionRepresentationError("unsupported target encoder version")
    if target.width != len(schema.labels) or target.height != 1 or target.channels != 1:
        raise PerceptionRepresentationError("target VPM shape does not match schema")
    if _digest(target.png_bytes) != target.png_digest:
        raise PerceptionRepresentationError("target PNG digest mismatch")
    scores = target.scores()
    if len(scores) != len(schema.labels):
        raise PerceptionRepresentationError("target PNG field count does not match schema")
    active = [index for index, value in enumerate(scores) if value == 255]
    if len(active) != 1 or any(value not in (0, 255) for value in scores):
        raise PerceptionRepresentationError("target VPM is not canonical one-hot data")
    label = schema.labels[active[0]]
    if label != target.action_label:
        raise PerceptionRepresentationError("target action metadata disagrees with PNG")
    return label
