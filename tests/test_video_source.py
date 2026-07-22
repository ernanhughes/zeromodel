from __future__ import annotations

import json

import numpy as np
import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.video import InMemoryVideoFrameSource, VideoFrame


def test_lossless_frame_source_owns_memory_and_has_stable_identity() -> None:
    caller = np.zeros((2, 3), dtype=np.uint8)
    source = InMemoryVideoFrameSource.from_arrays(
        [caller, np.ones((2, 3), dtype=np.uint8)],
        clip_id="clip-1",
        nominal_fps=5.0,
        source_id="fixture",
    )
    manifest = source.manifest()
    before = manifest.manifest_id
    caller[0, 0] = 255

    frames = tuple(source.frames())
    assert frames[0].pixels.flags.writeable is False
    assert frames[0].pixels[0, 0] == 0
    assert source.manifest().manifest_id == before
    assert manifest.frame_count == 2
    assert manifest.timestamps_seconds == (0.0, 0.2)
    assert frames[0].pixel_digest != frames[1].pixel_digest
    assert json.loads(json.dumps(manifest.to_dict())) == manifest.to_dict()


def test_duplicate_timestamps_are_explicit_and_permitted() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [np.zeros((1, 1), dtype=np.uint8)] * 2,
        clip_id="duplicate-time",
        nominal_fps=10.0,
        timestamps_seconds=(0.0, 0.0),
    )
    frames = tuple(source.frames())
    assert source.manifest().timestamps_seconds == (0.0, 0.0)
    assert frames[0].pixel_digest == frames[1].pixel_digest
    assert frames[0].frame_digest != frames[1].frame_digest


def test_frame_shape_changes_are_rejected() -> None:
    first = VideoFrame(
        clip_id="clip",
        frame_index=0,
        timestamp_seconds=0.0,
        pixels=np.zeros((2, 2), dtype=np.uint8),
        source_digest="sha256:source",
    )
    second = VideoFrame(
        clip_id="clip",
        frame_index=1,
        timestamp_seconds=0.1,
        pixels=np.zeros((3, 2), dtype=np.uint8),
        source_digest="sha256:source",
    )
    with pytest.raises(VPMValidationError, match="shape changes"):
        InMemoryVideoFrameSource((first, second), nominal_fps=10.0)


def test_frame_indices_and_decoding_order_must_be_contiguous() -> None:
    first = VideoFrame(
        clip_id="clip",
        frame_index=0,
        timestamp_seconds=0.0,
        pixels=np.zeros((2, 2), dtype=np.uint8),
        source_digest="sha256:source",
    )
    skipped = VideoFrame(
        clip_id="clip",
        frame_index=2,
        decoding_order=2,
        timestamp_seconds=0.2,
        pixels=np.zeros((2, 2), dtype=np.uint8),
        source_digest="sha256:source",
    )
    with pytest.raises(VPMValidationError, match="contiguous and ordered"):
        InMemoryVideoFrameSource((first, skipped), nominal_fps=10.0)


def test_negative_and_non_finite_timestamps_are_rejected() -> None:
    with pytest.raises(VPMValidationError, match="non-negative"):
        VideoFrame(
            clip_id="clip",
            frame_index=0,
            timestamp_seconds=-0.1,
            pixels=np.zeros((2, 2), dtype=np.uint8),
            source_digest="sha256:source",
        )
    with pytest.raises(VPMValidationError, match="finite and non-negative"):
        VideoFrame(
            clip_id="clip",
            frame_index=0,
            timestamp_seconds=float("nan"),
            pixels=np.zeros((2, 2), dtype=np.uint8),
            source_digest="sha256:source",
        )


def test_duplicate_frame_ids_are_rejected() -> None:
    first = VideoFrame(
        clip_id="clip",
        frame_index=0,
        timestamp_seconds=0.0,
        pixels=np.zeros((2, 2), dtype=np.uint8),
        source_digest="sha256:source",
        frame_id="frame-0",
    )
    second = VideoFrame(
        clip_id="clip",
        frame_index=1,
        timestamp_seconds=0.1,
        pixels=np.ones((2, 2), dtype=np.uint8),
        source_digest="sha256:source",
        frame_id="frame-0",
    )
    with pytest.raises(VPMValidationError, match="frame IDs must be unique"):
        InMemoryVideoFrameSource((first, second), nominal_fps=10.0)


def test_owned_frame_pixels_cannot_be_mutated_by_callers() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [np.zeros((2, 2), dtype=np.uint8)],
        clip_id="clip",
        nominal_fps=10.0,
    )
    frame = next(source.frames())
    with pytest.raises(ValueError):
        frame.pixels[0, 0] = 255
