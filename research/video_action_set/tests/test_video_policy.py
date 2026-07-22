# noqa: E402
"""Arcade closed-world proof for VideoPolicyReader.

Split in Stage A2: this file originally held 11 tests. 10 of them asserted
generic VideoPolicyReader contracts (impossible-transition rejection,
temporal-gap bookkeeping, staleness horizon, frame-reorder/tamper detection,
trace-id determinism, independent-evidence requirement) that hold for any
policy/provider - arcade frames were only ever a convenient fixture. Those
10 moved to
packages/video/tests/test_video_policy_reader_contracts.py with a small
synthetic fixture, so packages/video/tests no longer needs to import
examples/ or research/ to cover its own reader contract. Only the test below
remains here, because it specifically depends on exhaustive, closed-world
properties of the arcade shooter action space (full symbolic row/action/trace
reproduction across the entire canonical clip) and would not generalize to
an arbitrary policy. See docs/reviews/post-split-stage-a2-test-ownership-changes.csv.
"""
from __future__ import annotations

import json
import sys

import pytest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.arcade_visual_video_baseline import run_exact_video_baseline # noqa: E402

pytestmark = pytest.mark.research


def test_exact_canonical_video_reproduces_symbolic_rows_actions_and_trace() -> None:
    result = run_exact_video_baseline()
    assert result["accepted_frames"] == result["frame_count"]
    assert result["exact_row_sequence_match"]
    assert result["action_sequence_match"]
    assert json.loads(json.dumps(result["trace"])) == result["trace"]
