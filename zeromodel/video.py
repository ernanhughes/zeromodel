"""Deterministic, dependency-light video transport contracts."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

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
            _thaw(value), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise VPMValidationError("video values must be JSON-serializable") from exc


def _digest(prefix: bytes, descriptor: Any, payload: bytes = b"") -> str:
    return "sha256:" + hashlib.sha256(
        prefix + _json_bytes(descriptor) + payload
    ).hexdigest()


def _owned_pixels(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
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
        if not str(self.clip_id) or not str(self.source_digest):
            raise VPMValidationError("clip_id and source_digest cannot be empty")
        index = int(self.frame_index)
        order = index if self.decoding_order is None else int(self.decoding_order)
        timestamp = float(self.timestamp_seconds)
        if index < 0 or order < 0:
            raise VPMValidationError("frame index and decoding order must be non-negative")
        if not math.isfinite(timestamp) or timestamp < 0.0:
            raise VPMValidationError("timestamp_seconds must be finite and non-negative")
        metadata = _freeze(self.metadata)
        _json_bytes(metadata)
        object.__setattr__(self, "pixels", _owned_pixels(self.pixels))
        object.__setattr__(self, "clip_id", str(self.clip_id))
        object.__setattr__(self, "source_digest", str(self.source_digest))
        object.__setattr__(self, "frame_index", index)
        object.__setattr__(self, "decoding_order", order)
        object.__setattr__(self, "timestamp_seconds", timestamp)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "frame_id", str(self.frame_id) or self.frame_digest)

    @property
    def channels(self) -> int:
        return 1 if self.pixels.ndim == 2 else int(self.pixels.shape[2])

    @property
    def pixel_digest(self) -> str:
        return _digest(
            b"zeromodel.video-frame-pixels.v1\0",
            {"shape": list(self.pixels.shape)},
            self.pixels.tobytes(order="C"),
        )

    @property
    def frame_digest(self) -> str:
        descriptor = {
            "version": self.version,
            "clip_id": self.clip_id,
            "frame_index": self.frame_index,
            "decoding_order": self.decoding_order,
            "timestamp_seconds": self.timestamp_seconds,
            "source_digest": self.source_digest,
            "shape": list(self.pixels.shape),
            "metadata": _thaw(self.metadata),
        }
        return _digest(
            b"zeromodel.video-frame.v1\0", descriptor,
            self.pixels.tobytes(order="C"),
        )

    def to_descriptor(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "clip_id": self.clip_id,
            "frame_id": self.frame_id,
            "frame_index": self.frame_index,
            "decoding_order": self.decoding_order,
            "timestamp_seconds": self.timestamp_seconds,
            "source_digest": self.source_digest,
            "frame_digest": self.frame_digest,
            "pixel_digest": self.pixel_digest,
            "shape": list(self.pixels.shape),
            "channels": self.channels,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True)
class VideoClipManifest:
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
        if any(not str(getattr(self, name)) for name in (
            "clip_id", "source_kind", "source_digest", "payload_digest"
        )):
            raise VPMValidationError("manifest identities cannot be empty")
        count, width, height, channels = (
            int(self.frame_count), int(self.width), int(self.height), int(self.channels)
        )
        if count <= 0 or width <= 0 or height <= 0 or channels not in {1, 3, 4}:
            raise VPMValidationError("invalid clip dimensions or frame count")
        fps = None if self.nominal_fps is None else float(self.nominal_fps)
        if fps is not None and (not math.isfinite(fps) or fps <= 0.0):
            raise VPMValidationError("nominal_fps must be finite and positive")
        ids = tuple(str(item) for item in self.frame_ids)
        digests = tuple(str(item) for item in self.frame_digests)
        timestamps = tuple(float(item) for item in self.timestamps_seconds)
        if not (len(ids) == len(digests) == len(timestamps) == count):
            raise VPMValidationError("manifest frame vectors must match frame_count")
        if len(set(ids)) != len(ids):
            raise VPMValidationError("frame_ids must be unique")
        if any(not math.isfinite(item) or item < 0.0 for item in timestamps):
            raise VPMValidationError("manifest timestamps must be finite and non-negative")
        if any(right < left for left, right in zip(timestamps, timestamps[1:])):
            raise VPMValidationError("manifest timestamps must be monotonic")
        metadata = _freeze(self.metadata)
        _json_bytes(metadata)
        for name, value in (
            ("frame_count", count), ("width", width), ("height", height),
            ("channels", channels), ("nominal_fps", fps), ("frame_ids", ids),
            ("frame_digests", digests), ("timestamps_seconds", timestamps),
            ("decode_warnings", tuple(str(item) for item in self.decode_warnings)),
            ("metadata", metadata),
        ):
            object.__setattr__(self, name, value)

    @property
    def manifest_id(self) -> str:
        return _digest(
            b"zeromodel.video-clip-manifest.v1\0",
            self.to_dict(include_manifest_id=False),
        )

    def to_dict(self, *, include_manifest_id: bool = True) -> dict[str, Any]:
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
    def manifest(self) -> VideoClipManifest: ...
    def frames(self) -> Iterable[VideoFrame]: ...


class InMemoryVideoFrameSource:
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
        first, shape = owned[0], owned[0].pixels.shape
        for expected, frame in enumerate(owned):
            if not isinstance(frame, VideoFrame):
                raise VPMValidationError("video source requires VideoFrame values")
            if frame.clip_id != first.clip_id or frame.source_digest != first.source_digest:
                raise VPMValidationError("all frames must share clip and source identity")
            if frame.frame_index != expected or frame.decoding_order != expected:
                raise VPMValidationError("frame indices and decoding order must be contiguous and ordered")
            if frame.pixels.shape != shape:
                raise VPMValidationError("frame shape changes are not allowed")
            if expected and frame.timestamp_seconds < owned[expected - 1].timestamp_seconds:
                raise VPMValidationError("frame timestamps must be monotonic")
        frame_digests = tuple(frame.frame_digest for frame in owned)
        payload_digest = _digest(
            b"zeromodel.video-clip-payload.v1\0",
            {"clip_id": first.clip_id, "source_digest": first.source_digest,
             "frame_digests": list(frame_digests)},
        )
        self._frames = owned
        self._manifest = VideoClipManifest(
            clip_id=first.clip_id,
            source_kind=str(source_kind),
            source_digest=first.source_digest,
            frame_count=len(owned),
            width=int(shape[1]), height=int(shape[0]),
            channels=1 if len(shape) == 2 else int(shape[2]),
            nominal_fps=nominal_fps,
            frame_ids=tuple(frame.frame_id for frame in owned),
            frame_digests=frame_digests,
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
        arrays = tuple(_owned_pixels(frame) for frame in frames)
        fps = float(nominal_fps)
        if not arrays:
            raise VPMValidationError("video source requires at least one frame")
        if not math.isfinite(fps) or fps <= 0.0:
            raise VPMValidationError("nominal_fps must be finite and positive")
        timestamps = (
            tuple(index / fps for index in range(len(arrays)))
            if timestamps_seconds is None
            else tuple(float(item) for item in timestamps_seconds)
        )
        if len(timestamps) != len(arrays):
            raise VPMValidationError("timestamps must match frame count")
        source_digest = _digest(
            b"zeromodel.video-source.v1\0",
            {"clip_id": str(clip_id), "source_id": str(source_id),
             "nominal_fps": fps, "shapes": [list(item.shape) for item in arrays]},
            b"".join(item.tobytes(order="C") for item in arrays),
        )
        values = tuple(
            VideoFrame(
                clip_id=str(clip_id), frame_index=index, decoding_order=index,
                timestamp_seconds=timestamps[index], pixels=array,
                source_digest=source_digest, metadata={"source_id": str(source_id)},
            )
            for index, array in enumerate(arrays)
        )
        return cls(values, nominal_fps=fps, metadata=metadata)

    def manifest(self) -> VideoClipManifest:
        return self._manifest

    def frames(self) -> Iterator[VideoFrame]:
        return iter(self._frames)
