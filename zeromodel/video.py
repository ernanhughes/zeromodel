"""Deterministic video transport and identity contracts.

The canonical benchmark path uses owned lossless frame arrays rather than codec
bytes. Optional file decoders may produce these contracts, but codec output is
not itself the canonical clip identity unless a caller explicitly pins it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

import numpy as np

from .artifact import VPMValidationError


VIDEO_FRAME_VERSION = "zeromodel-video-frame/v1"
VIDEO_CLIP_MANIFEST_VERSION = "zeromodel-video-clip-manifest/v1"
VIDEO_FRAME_SOURCE_VERSION = "zeromodel-video-frame-source/v1"


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, np.generic):
        raise VPMValidationError("video JSON must use plain scalar types")
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            _thaw(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("video values must be JSON-serializable") from exc


def _sha256(prefix: bytes, descriptor: Any, payload: bytes = b"") -> str:
    digest = hashlib.sha256(prefix + _json_bytes(descriptor) + payload).hexdigest()
    return "sha256:" + digest


def _validate_pixels(pixels: np.ndarray) -> np.ndarray:
    array = np.asarray(pixels)
    if array.dtype != np.uint8:
        raise VPMValidationError("video frames must use uint8 samples")
    if not (array.ndim == 2 or (array.ndim == 3 and array.shape[2] in {3, 4})):
        raise VPMValidationError("video frames must be HxW or HxWx3/4")
    if array.size == 0:
        raise VPMValidationError("video frames cannot be empty")
    owned = np.array(array, dtype=np.uint8, order="C", copy=True)
    owned.flags.writeable = False
    return owned


@dataclass(frozen=True)
class VideoFrame:
    """One owned frame with stable source, order, time, and payload identity."""

    clip_id: str
    frame_index: int
    timestamp_seconds: float
    pixels: np.ndarray
    source_digest: str
    frame_id: str = ""
    decoding_order: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_FRAME_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_FRAME_VERSION:
            raise VPMValidationError("unsupported video frame version")
        if not str(self.clip_id):
            raise VPMValidationError("clip_id cannot be empty")
        if not str(self.source_digest):
            raise VPMValidationError("source_digest cannot be empty")
        if int(self.frame_index) < 0:
            raise VPMValidationError("frame_index must be non-negative")
        timestamp = float(self.timestamp_seconds)
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise VPMValidationError("timestamp_seconds must be finite and non-negative")
        decoding_order = self.frame_index if self.decoding_order is None else int(self.decoding_order)
        if decoding_order < 0:
            raise VPMValidationError("decoding_order must be non-negative")
        owned = _validate_pixels(self.pixels)
        metadata = _freeze(self.metadata)
        _json_bytes(metadata)
        object.__setattr__(self, "pixels", owned)
        object.__setattr__(self, "frame_index", int(self.frame_index))
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "decoding_order", decoding_order)
        object.__setattr__(self, "clip_id", str(self.clip_id))
        object.__setattr__(self, "source_digest", str(self.source_digest))
        object.__setattr__(self, "metadata", metadata)
        computed_id = self.frame_digest
        object.__setattr__(self, "frame_id", str(self.frame_id) or computed_id)

    @property
    def channels(self) -> int:
        return 1 if self.pixels.ndim == 2 else int(self.pixels.shape[2])

    @property
    def frame_digest(self) -> str:
        descriptor = {
            "version": self.version,
            "clip_id": str(self.clip_id),
            "frame_index": int(self.frame_index),
            "decoding_order": int(self.frame_index if self.decoding_order is None else self.decoding_order),
            "timestamp_seconds": float(self.timestamp_seconds),
            "source_digest": str(self.source_digest),
            "shape": list(np.asarray(self.pixels).shape),
            "metadata": _thaw(self.metadata),
        }
        return _sha256(
            b"zeromodel.video-frame.v1\0",
            descriptor,
            np.asarray(self.pixels, dtype=np.uint8).tobytes(order="C"),
        )

    def to_descriptor(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "clip_id": self.clip_id,
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "decoding_order": self.decoding_order,
            "timestamp_seconds": self.timestamp_seconds,
            "source_digest": self.source_digest,
            "frame_digest": self.frame_digest,
            "shape": list(self.pixels.shape),
            "channels": self.channels,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True)
class VideoClipManifest:
    """Ordered identity record for one lossless decoded clip."""

    clip_id: str
    source_kind: str
    source_digest: str
    frame_count: int
    width: int
    height: int
    channels: int
    nominal_fps: Optional[float]
    frame_ids: Tuple[str, ...]
    frame_digests: Tuple[str, ...]
    timestamps_seconds: Tuple[float, ...]
    payload_digest: str
    decode_warnings: Tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    version: str = VIDEO_CLIP_MANIFEST_VERSION

    def __post_init__(self) -> None:
        if self.version != VIDEO_CLIP_MANIFEST_VERSION:
            raise VPMValidationError("unsupported video clip manifest version")
        for name in ("clip_id", "source_kind", "source_digest", "payload_digest"):
            if not str(getattr(self, name)):
                raise VPMValidationError("%s cannot be empty" % name)
        if int(self.frame_count) <= 0:
            raise VPMValidationError("frame_count must be positive")
        if int(self.width) <= 0 or int(self.height) <= 0 or int(self.channels) not in {1, 3, 4}:
            raise VPMValidationError("invalid clip dimensions or channels")
        fps = None if self.nominal_fps is None else float(self.nominal_fps)
        if fps is not None and (not math.isfinite(fps) or fps <= 0.0):
            raise VPMValidationError("nominal_fps must be finite and positive")
        frame_ids = tuple(str(value) for value in self.frame_ids)
        frame_digests = tuple(str(value) for value in self.frame_digests)
        timestamps = tuple(float(value) for value in self.timestamps_seconds)
        warnings = tuple(str(value) for value in self.decode_warnings)
        expected = int(self.frame_count)
        if not (len(frame_ids) == len(frame_digests) == len(timestamps) == expected):
            raise VPMValidationError("manifest frame vectors must match frame_count")
        if len(set(frame_ids)) != len(frame_ids):
            raise VPMValidationError("frame_ids must be unique")
        if any(not math.isfinite(value) or value < 0.0 for value in timestamps):
            raise VPMValidationError("manifest timestamps must be finite and non-negative")
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise VPMValidationError("manifest timestamps must be monotonic")
        metadata = _freeze(self.metadata)
        _json_bytes(metadata)
        object.__setattr__(self, "frame_count", expected)
        object.__setattr__(self, "width", int(self.width))
        object.__setattr__(self, "height", int(self.height))
        object.__setattr__(self, "channels", int(self.channels))
        object.__setattr__(self, "nominal_fps", fps)
        object.__setattr__(self, "frame_ids", frame_ids)
        object.__setattr__(self, "frame_digests", frame_digests)
        object.__setattr__(self, "timestamps_seconds", timestamps)
        object.__setattr__(self, "decode_warnings", warnings)
        object.__setattr__(self, "metadata", metadata)

    @property
    def manifest_id(self) -> str:
        return _sha256(
            b"zeromodel.video-clip-manifest.v1\0",
            self.to_dict(include_manifest_id=False),
        )

    def to_dict(self, *, include_manifest_id: bool = True) -> Dict[str, Any]:
        result = {
            "version": self.version,
            "clip_id": self.clip_id,
            "source_kind": self.source_kind,
            "source_digest": self.source_digest,
            "frame_count": self.frame_count,
            "width": self.width,
            "height": self.height,
            "channels": self.channels,
            "nominal_fps": self.nominal_fps,
            "frame_ids": list(self.frame_ids),
            "frame_digests": list(self.frame_digests),
            "timestamps_seconds": list(self.timestamps_seconds),
            "payload_digest": self.payload_digest,
            "decode_warnings": list(self.decode_warnings),
            "metadata": _thaw(self.metadata),
        }
        if include_manifest_id:
            result["manifest_id"] = self.manifest_id
        return result


@runtime_checkable
class VideoFrameSource(Protocol):
    def manifest(self) -> VideoClipManifest:
        ...

    def frames(self) -> Iterable[VideoFrame]:
        ...


class InMemoryVideoFrameSource:
    """Canonical dependency-light source backed by owned lossless frames."""

    def __init__(
        self,
        frames: Sequence[VideoFrame],
        *,
        source_kind: str = "lossless_frame_stream",
        nominal_fps: Optional[float] = None,
        decode_warnings: Sequence[str] = (),
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        owned = tuple(frames)
        if not owned:
            raise VPMValidationError("video source requires at least one frame")
        first = owned[0]
        expected_shape = first.pixels.shape
        for expected_index, frame in enumerate(owned):
            if not isinstance(frame, VideoFrame):
                raise VPMValidationError("video source requires VideoFrame values")
            if frame.clip_id != first.clip_id:
                raise VPMValidationError("all frames must share one clip_id")
            if frame.source_digest != first.source_digest:
                raise VPMValidationError("all frames must share one source_digest")
            if frame.frame_index != expected_index:
                raise VPMValidationError("frame indices must be contiguous and ordered")
            if frame.decoding_order != expected_index:
                raise VPMValidationError("decoding order must be contiguous and ordered")
            if frame.pixels.shape != expected_shape:
                raise VPMValidationError("frame shape changes are not allowed")
            if expected_index and frame.timestamp_seconds < owned[expected_index - 1].timestamp_seconds:
                raise VPMValidationError("frame timestamps must be monotonic")
        payload_digest = _sha256(
            b"zeromodel.video-clip-payload.v1\0",
            {
                "clip_id": first.clip_id,
                "source_digest": first.source_digest,
                "frame_digests": [frame.frame_digest for frame in owned],
            },
        )
        channels = 1 if len(expected_shape) == 2 else int(expected_shape[2])
        self._frames = owned
        self._manifest = VideoClipManifest(
            clip_id=first.clip_id,
            source_kind=str(source_kind),
            source_digest=first.source_digest,
            frame_count=len(owned),
            width=int(expected_shape[1]),
            height=int(expected_shape[0]),
            channels=channels,
            nominal_fps=nominal_fps,
            frame_ids=tuple(frame.frame_id for frame in owned),
            frame_digests=tuple(frame.frame_digest for frame in owned),
            timestamps_seconds=tuple(frame.timestamp_seconds for frame in owned),
            payload_digest=payload_digest,
            decode_warnings=tuple(decode_warnings),
            metadata=metadata or {},
        )

    @classmethod
    def from_arrays(
        cls,
        frames: Sequence[np.ndarray],
        *,
        clip_id: str,
        nominal_fps: float,
        source_id: str = "",
        timestamps_seconds: Optional[Sequence[float]] = None,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> "InMemoryVideoFrameSource":
        arrays = tuple(_validate_pixels(frame) for frame in frames)
        if not arrays:
            raise VPMValidationError("video source requires at least one frame")
        if not math.isfinite(float(nominal_fps)) or float(nominal_fps) <= 0.0:
            raise VPMValidationError("nominal_fps must be finite and positive")
        if timestamps_seconds is None:
            timestamps = tuple(index / float(nominal_fps) for index in range(len(arrays)))
        else:
            timestamps = tuple(float(value) for value in timestamps_seconds)
            if len(timestamps) != len(arrays):
                raise VPMValidationError("timestamps must match frame count")
        source_digest = _sha256(
            b"zeromodel.video-source.v1\0",
            {
                "clip_id": str(clip_id),
                "source_id": str(source_id),
                "nominal_fps": float(nominal_fps),
                "shapes": [list(array.shape) for array in arrays],
            },
            b"".join(array.tobytes(order="C") for array in arrays),
        )
        video_frames = tuple(
            VideoFrame(
                clip_id=str(clip_id),
                frame_index=index,
                decoding_order=index,
                timestamp_seconds=timestamps[index],
                pixels=array,
                source_digest=source_digest,
                metadata={"source_id": str(source_id)},
            )
            for index, array in enumerate(arrays)
        )
        return cls(
            video_frames,
            source_kind="lossless_frame_stream",
            nominal_fps=float(nominal_fps),
            metadata=metadata,
        )

    def manifest(self) -> VideoClipManifest:
        return self._manifest

    def frames(self) -> Iterator[VideoFrame]:
        return iter(self._frames)
