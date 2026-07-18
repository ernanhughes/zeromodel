from __future__ import annotations

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
from zeromodel.video import InMemoryVideoFrameSource
from zeromodel.video_policy import VideoPolicyReader
from zeromodel.visual_policy import DeterministicVisualAddressProvider


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
