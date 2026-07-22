"""Generic VideoPolicyReader contracts, split out of research.

These 10 tests were originally part of
research/video_action_set/tests/test_video_policy.py, built on top of the
full arcade shooter fixture (examples/arcade_shooter_policy.py +
examples/arcade_visual_sign_reader.py + examples/arcade_visual_video_baseline.py).
Every one of them asserts a property that holds for ANY policy/provider, not
anything specific to the arcade closed-world action space - arcade frames
were only ever a convenient concrete fixture. They are moved here with a
small synthetic fixture (matching the FakeProvider pattern already used by
test_temporal_policy.py in this same directory) so packages/video/tests no
longer needs to import examples/ or research/ to exercise its own
VideoPolicyReader contract.

The one genuinely arcade-specific test from that file - the full
closed-world symbolic-trace reproduction proof - was intentionally left in
research/video_action_set/tests/test_video_policy.py, since it depends on
exhaustive properties of the arcade action space itself and would not
generalize to an arbitrary policy. See
docs/reviews/post-split-stage-a2-test-ownership-changes.csv for the full
before/after mapping.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from zeromodel.core import LayoutRecipe, ScoreTable, build_vpm
from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.policy_lookup import VPMPolicyLookup
from zeromodel.core.policy_transitions import PolicyTransitionSpec
from zeromodel.observation import (
    ImageObservation,
    VisualAddressContract,
    VisualAddressDecision,
)
from zeromodel.video import InMemoryVideoFrameSource, VideoFrame, VideoPolicyReader


def _policy():
    table = ScoreTable(
        values=[[1.0, 0.0], [0.0, 1.0]],
        row_ids=["left", "right"],
        metric_ids=["A", "B"],
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "video-policy-reader-contracts",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )
    return build_vpm(table, recipe)


class FakeProvider:
    """Deterministically replays a fixed accept/reject/row sequence."""

    def __init__(self, policy_id: str, rows: tuple[str | None, ...]) -> None:
        self.rows = rows
        self.index = 0
        self._contract = VisualAddressContract(
            provider_kind="fake",
            provider_version="fake/v1",
            score_semantics="distance",
            observation_spec_digest="obs",
            representation_spec_digest="repr",
            address_artifact_id="addr",
            calibration_artifact_id="cal",
            policy_artifact_id=policy_id,
        )

    def contract(self) -> VisualAddressContract:
        return self._contract

    def read(self, observation: ImageObservation) -> VisualAddressDecision:
        row = self.rows[self.index]
        self.index += 1
        accepted = row is not None
        return VisualAddressDecision(
            accepted=accepted,
            reason="accepted" if accepted else "provider_rejected",
            observation_digest=observation.raw_digest,
            representation_digest=observation.raw_digest,
            provider_kind=self._contract.provider_kind,
            provider_version=self._contract.provider_version,
            score_semantics=self._contract.score_semantics,
            address_artifact_id=self._contract.address_artifact_id,
            calibration_artifact_id=self._contract.calibration_artifact_id,
            policy_artifact_id=self._contract.policy_artifact_id,
            nearest_row_id=row,
            nearest_score=0.0 if accepted else None,
            second_row_id=None,
            second_score=None,
            ambiguity_measure=None,
            matched_row_id=row,
            exact_match=accepted,
            accepted_by=("fake",) if accepted else (),
        )


def _reader(
    rows: tuple[str | None, ...], *, maximum_identical_frame_run: int | None = None
) -> VideoPolicyReader:
    artifact = _policy()
    return VideoPolicyReader(
        FakeProvider(artifact.artifact_id, rows),
        VPMPolicyLookup(artifact, action_metric_ids=("A", "B")),
        PolicyTransitionSpec(
            {"left": ("left",), "right": ("right",)},
            maximum_frame_gap=2,
        ),
        maximum_identical_frame_run=maximum_identical_frame_run,
    )


def _frame(value: int) -> np.ndarray:
    return np.full((2, 2), value, dtype=np.uint8)


def test_impossible_transition_rejects_but_retains_raw_prediction() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1)],
        clip_id="impossible-jump",
        nominal_fps=10.0,
    )
    trace = _reader(("left", "right")).read(source)

    assert trace.decisions[0].accepted
    assert not trace.decisions[1].accepted
    assert trace.decisions[1].reason == "transition_impossible"
    assert trace.decisions[1].raw_row_id == "right"
    assert trace.decisions[1].raw_action_id == "B"
    assert trace.decisions[1].policy is None


def test_rejected_frame_is_not_silently_carried_forward_and_recovery_is_explicit() -> (
    None
):
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1), _frame(2)],
        clip_id="occlusion-gap",
        nominal_fps=10.0,
    )
    trace = _reader(("left", None, "left")).read(source)

    assert trace.decisions[0].accepted
    assert not trace.decisions[1].accepted
    assert trace.decisions[1].accepted_row_id is None
    assert trace.decisions[1].temporal.current_frame_independently_supported is False
    assert trace.decisions[2].accepted
    assert trace.decisions[2].temporal.transition.status == "possible_with_gap"
    assert trace.decisions[2].temporal.previous_accepted_row_id == "left"


def test_stale_repeated_frame_is_rejected_after_declared_horizon() -> None:
    frame = _frame(3)
    source = InMemoryVideoFrameSource.from_arrays(
        [frame, frame, frame],
        clip_id="stale-repeat",
        nominal_fps=10.0,
    )
    trace = _reader(("left", "left", "left"), maximum_identical_frame_run=2).read(
        source
    )

    assert trace.decisions[0].accepted
    assert trace.decisions[1].accepted
    assert not trace.decisions[2].accepted
    assert "stale_repeated_frame" in trace.decisions[2].rejection_reasons


def test_reader_detects_reordered_frames_against_manifest() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1)],
        clip_id="reorder",
        nominal_fps=10.0,
    )

    class ReorderedSource:
        def manifest(self):
            return source.manifest()

        def frames(self):
            frames = list(source.frames())
            frames[0], frames[1] = frames[1], frames[0]
            return frames

    with pytest.raises(VPMValidationError, match="frame order"):
        _reader(("left", "right")).read(ReorderedSource())


def test_trace_identity_is_deterministic_and_json_safe() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1)],
        clip_id="determinism",
        nominal_fps=10.0,
    )
    trace_a = _reader(("left", "left")).read(source)
    trace_b = _reader(("left", "left")).read(source)

    assert trace_a.trace_id == trace_b.trace_id
    assert json.loads(json.dumps(trace_a.to_dict())) == trace_a.to_dict()


def test_reader_rejects_tampered_manifest_frame_id() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1)],
        clip_id="tamper-frame-id",
        nominal_fps=10.0,
    )

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
        _reader(("left", "left")).read(TamperedManifestSource())


def test_reader_rejects_tampered_manifest_frame_digest() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1)],
        clip_id="tamper-digest",
        nominal_fps=10.0,
    )

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

    with pytest.raises(
        VPMValidationError, match="frame digest does not match manifest"
    ):
        _reader(("left", "left")).read(TamperedManifestSource())


def test_reader_rejects_manifest_and_source_frame_count_mismatch() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1)],
        clip_id="count-mismatch",
        nominal_fps=10.0,
    )

    class MissingFrameSource:
        def manifest(self):
            return source.manifest()

        def frames(self):
            return list(source.frames())[:-1]

    with pytest.raises(VPMValidationError, match="frame count does not match manifest"):
        _reader(("left", "left")).read(MissingFrameSource())


def test_reader_rejects_source_digest_mismatch() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1)],
        clip_id="source-digest-mismatch",
        nominal_fps=10.0,
    )
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

    with pytest.raises(
        VPMValidationError, match="source digest does not match manifest"
    ):
        _reader(("left", "left")).read(TamperedSourceDigestSource())


def test_reader_never_accepts_without_independent_current_frame_evidence() -> None:
    source = InMemoryVideoFrameSource.from_arrays(
        [_frame(0), _frame(1), _frame(2)],
        clip_id="unsupported-gap",
        nominal_fps=10.0,
    )
    trace = _reader(("left", None, None)).read(source)

    assert trace.decisions[1].accepted is False
    assert trace.decisions[1].temporal.current_frame_independently_supported is False
    assert trace.decisions[2].accepted is False
    assert trace.decisions[2].temporal.current_frame_independently_supported is False
