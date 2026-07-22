from __future__ import annotations

from zeromodel.analysis.policy_transitions import (
    PolicyTransitionEvidence,
    PolicyTransitionSpec,
    ROW_UNION_TRANSITION_SCOPE,
)
from zeromodel.video.video import (
    InMemoryVideoFrameSource,
    VideoClipManifest,
    VideoFrame,
    VideoFrameSource,
)
from zeromodel.video.video_policy import (
    TemporalEvidence,
    VideoPolicyDecision,
    VideoPolicyReader,
    VideoPolicyTrace,
)


def test_video_api_is_exported_from_package_surface() -> None:
    assert VideoFrame.__name__ == "VideoFrame"
    assert VideoClipManifest.__name__ == "VideoClipManifest"
    assert VideoFrameSource.__name__ == "VideoFrameSource"
    assert InMemoryVideoFrameSource.__name__ == "InMemoryVideoFrameSource"
    assert PolicyTransitionSpec.__name__ == "PolicyTransitionSpec"
    assert PolicyTransitionEvidence.__name__ == "PolicyTransitionEvidence"
    assert TemporalEvidence.__name__ == "TemporalEvidence"
    assert VideoPolicyDecision.__name__ == "VideoPolicyDecision"
    assert VideoPolicyTrace.__name__ == "VideoPolicyTrace"
    assert VideoPolicyReader.__name__ == "VideoPolicyReader"
    assert ROW_UNION_TRANSITION_SCOPE == "row_union_over_actions"
