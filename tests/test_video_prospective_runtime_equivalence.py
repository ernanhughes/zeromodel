from __future__ import annotations

from pathlib import Path

import pytest

from zeromodel.video_action_set_benchmark import profile_runtime, verify_provider_runtime_equivalence


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_provider_runtime_equivalence_and_profiles(tmp_path: Path) -> None:
    equivalence = verify_provider_runtime_equivalence(tmp_path, REPO_ROOT)
    assert equivalence["providers_verified"] is True
    assert equivalence["summary"]["P1"]["quantized_mismatch_count"] == 0
    assert equivalence["summary"]["P2"]["ranking_mismatch_count"] == 0
    assert equivalence["summary"]["P3"]["tie_group_mismatch_count"] == 0

    profile = profile_runtime(tmp_path, REPO_ROOT, provider="P1", frame_count=2)
    assert profile["provider_scope"] == "P1"
    assert profile["profile_frame_count"] == 2
    assert (tmp_path / "runtime-profile-reference.json").exists()
    assert (tmp_path / "runtime-profile-optimized.json").exists()
    assert (tmp_path / "runtime-comparison.json").exists()
