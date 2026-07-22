from __future__ import annotations

import numpy as np
import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video import InMemoryVideoFrameSource, VideoFrame


def test_frame_identity_and_memory_isolation() -> None:
    pixels = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    frame = VideoFrame(
        clip_id="clip-a",
        frame_index=0,
        timestamp_seconds=0.0,
        pixels=pixels,
        source_digest="source-a",
        metadata={"expected_action": "left"},
    )

    before = frame.pixel_digest
    pixels[0, 0] = 99

    assert frame.pixels[0, 0] == 1
    assert frame.pixels.flags.writeable is False
    assert frame.pixel_digest == before
    assert frame.frame_id == frame.frame_digest
    assert frame.to_descriptor()["metadata"] == {"expected_action": "left"}


def test_clip_manifest_ordering_round_trip_and_rejections() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [
            np.zeros((2, 2), dtype=np.uint8),
            np.ones((2, 2), dtype=np.uint8),
        ],
        clip_id="clip-a",
        nominal_fps=2.0,
        source_id="fixture",
    )
    manifest = source.manifest()

    assert manifest.frame_count == 2
    assert manifest.timestamps_seconds == (0.0, 0.5)
    assert list(frame.frame_index for frame in source.frames()) == [0, 1]
    assert manifest.to_dict()["manifest_id"] == manifest.manifest_id

    duplicate = tuple(source.frames())
    with pytest.raises(
        VPMValidationError,
        match="frame indices and decoding order|frame IDs",
    ):
        InMemoryVideoFrameSource((duplicate[0], duplicate[0]))

    with pytest.raises(VPMValidationError, match="requires at least one frame"):
        InMemoryVideoFrameSource.from_arrays(
            [],
            clip_id="empty",
            nominal_fps=1.0,
        )
