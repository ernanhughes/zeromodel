from __future__ import annotations

from functools import lru_cache
import json

import numpy as np
import pytest

from examples.arcade_shooter_policy import ACTIONS, ShooterConfig, compile_policy_artifact
from examples.arcade_visual_sign_reader import (
    compile_visual_index_artifact,
    make_visual_reader,
    render_state_frame,
)
from examples.arcade_visual_video_baseline import (
    arcade_transition_spec,
    build_canonical_arcade_clip,
    run_exact_video_baseline,
)
from zeromodel import VPMPolicyLookup
from zeromodel.artifact import VPMValidationError
from zeromodel.video import InMemoryVideoFrameSource, VideoFrame
from zeromodel.video_policy import VideoPolicyReader
from zeromodel.visual_policy import DeterministicVisualAddressProvider


# The reader is immutable; sharing it avoids recompiling the 112-row visual index.
@lru_cache(maxsize=None)
def _reader(*, maximum_identical_frame_run: int | None = None) -> VideoPolicyReader:
    config = ShooterConfig()
    policy = compile_policy_artifact(config)
    visual_build = compile_visual_index_artifact(config, policy_artifact=policy)
    provider = DeterministicVisualAddressProvider(
        make_visual_reader(policy, visual_build),
        source_scope="fixture:test-video",
    )
    return VideoPolicyReader(
        provider,
        VPMPolicyLookup(policy, action_metric_ids=ACTIONS),
        arcade_transition_spec(config),
        maximum_identical_frame_run=maximum_identical_frame_run,
    )


def test_exact_canonical_video_reproduces_symbolic_rows_actions_and_trace() -> None:
    result = run_exact_video_baseline()
    assert result["accepted_frames"] == result["frame_count"]
    assert result["exact_row_sequence_match"]
    assert result["action_sequence_match"]
    assert json.loads(json.dumps(result["trace"])) == result["trace"]


def test_impossible_transition_rejects_but_retains_raw_prediction() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [
            render_state_frame(0, 0, 0),
            render_state_frame(6, 6, 0),
        ],
        clip_id="impossible-jump",
        nominal_fps=10.0,
    )
    trace = _reader().read(source)

    assert trace.decisions[0].accepted
    assert not trace.decisions[1].accepted
    assert trace.decisions[1].reason == "transition_impossible"
    assert trace.decisions[1].raw_row_id == "tank=6|target=6|cooldown=0"
    assert trace.decisions[1].raw_action_id == "FIRE"
    assert trace.decisions[1].policy is None


def test_rejected_frame_is_not_silently_carried_forward_and_recovery_is_explicit() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [
            render_state_frame(3, 0, 0),
            np.zeros((16, 28), dtype=np.uint8),
            render_state_frame(2, 0, 0),
        ],
        clip_id="occlusion-gap",
        nominal_fps=10.0,
    )
    trace = _reader().read(source)

    assert trace.decisions[0].accepted
    assert not trace.decisions[1].accepted
    assert trace.decisions[1].accepted_row_id is None
    assert trace.decisions[1].temporal.current_frame_independently_supported is False
    assert trace.decisions[2].accepted
    assert trace.decisions[2].temporal.transition.status == "possible_with_gap"
    assert trace.decisions[2].temporal.previous_accepted_row_id == "tank=3|target=0|cooldown=0"


def test_stale_repeated_frame_is_rejected_after_declared_horizon() -> None:
    frame = render_state_frame(3, None, 0)
    source = InMemoryVideoFrameSource.from_arrays(
        [frame, frame, frame],
        clip_id="stale-repeat",
        nominal_fps=10.0,
    )
    trace = _reader(maximum_identical_frame_run=2).read(source)

    assert trace.decisions[0].accepted
    assert trace.decisions[1].accepted
    assert not trace.decisions[2].accepted
    assert "stale_repeated_frame" in trace.decisions[2].rejection_reasons


def test_reader_detects_reordered_frames_against_manifest() -> None:
    source, _, _ = build_canonical_arcade_clip()

    class ReorderedSource:
        def manifest(self):
            return source.manifest()

        def frames(self):
            frames = list(source.frames())
            frames[0], frames[1] = frames[1], frames[0]
            return frames

    with pytest.raises(VPMValidationError, match="frame order"):
        _reader().read(ReorderedSource())


def test_trace_identity_is_deterministic_and_json_safe() -> None:
    source, _, _ = build_canonical_arcade_clip()
    trace_a = _reader().read(source)
    trace_b = _reader().read(source)

    assert trace_a.trace_id == trace_b.trace_id
    assert json.loads(json.dumps(trace_a.to_dict())) == trace_a.to_dict()


def test_reader_rejects_tampered_manifest_frame_id() -> None:
    source, _, _ = build_canonical_arcade_clip()

    class TamperedManifestSource:
        def manifest(self):
            manifest = source.manifest()
            return type(manifest)(
                clip_id=manifest.clip_id,
                source_kind=manifest.source_kind,
                source_digest=manifest.source_digest,
                frame_count=manifest.frame_count,
                width=manifest.width,
                height=manifest.height,
                channels=manifest.channels,
                nominal_fps=manifest.nominal_fps,
                frame_ids=("tampered-frame-id",) + manifest.frame_ids[1:],
                frame_digests=manifest.frame_digests,
                timestamps_seconds=manifest.timestamps_seconds,
                payload_digest=manifest.payload_digest,
                decode_warnings=manifest.decode_warnings,
                metadata=manifest.metadata,
            )

        def frames(self):
            return source.frames()

    with pytest.raises(VPMValidationError, match="frame_id does not match manifest"):
        _reader().read(TamperedManifestSource())


def test_reader_rejects_tampered_manifest_frame_digest() -> None:
    source, _, _ = build_canonical_arcade_clip()

    class TamperedManifestSource:
        def manifest(self):
            manifest = source.manifest()
            return type(manifest)(
                clip_id=manifest.clip_id,
                source_kind=manifest.source_kind,
                source_digest=manifest.source_digest,
                frame_count=manifest.frame_count,
                width=manifest.width,
                height=manifest.height,
                channels=manifest.channels,
                nominal_fps=manifest.nominal_fps,
                frame_ids=manifest.frame_ids,
                frame_digests=("sha256:tampered",) + manifest.frame_digests[1:],
                timestamps_seconds=manifest.timestamps_seconds,
                payload_digest=manifest.payload_digest,
                decode_warnings=manifest.decode_warnings,
                metadata=manifest.metadata,
            )

        def frames(self):
            return source.frames()

    with pytest.raises(VPMValidationError, match="frame digest does not match manifest"):
        _reader().read(TamperedManifestSource())


def test_reader_rejects_manifest_and_source_frame_count_mismatch() -> None:
    source, _, _ = build_canonical_arcade_clip()

    class MissingFrameSource:
        def manifest(self):
            return source.manifest()

        def frames(self):
            frames = list(source.frames())
            return frames[:-1]

    with pytest.raises(VPMValidationError, match="frame count does not match manifest"):
        _reader().read(MissingFrameSource())


def test_reader_rejects_source_digest_mismatch() -> None:
    source, _, _ = build_canonical_arcade_clip()
    frames = list(source.frames())
    first = frames[0]
    frames[0] = VideoFrame(
        clip_id=first.clip_id,
        frame_index=first.frame_index,
        decoding_order=first.decoding_order,
        timestamp_seconds=first.timestamp_seconds,
        pixels=first.pixels,
        source_digest="sha256:tampered-source",
        frame_id=first.frame_id,
        metadata=first.metadata,
    )

    class TamperedSourceDigestSource:
        def manifest(self):
            return source.manifest()

        def frames(self):
            return frames

    with pytest.raises(VPMValidationError, match="source digest does not match manifest"):
        _reader().read(TamperedSourceDigestSource())


def test_reader_never_accepts_without_independent_current_frame_evidence() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [
            render_state_frame(3, 0, 0),
            np.zeros((16, 28), dtype=np.uint8),
            np.zeros((16, 28), dtype=np.uint8),
        ],
        clip_id="unsupported-gap",
        nominal_fps=10.0,
    )
    trace = _reader().read(source)

    assert trace.decisions[1].accepted is False
    assert trace.decisions[1].temporal.current_frame_independently_supported is False
    assert trace.decisions[2].accepted is False
    assert trace.decisions[2].temporal.current_frame_independently_supported is False
