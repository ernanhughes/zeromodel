from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.video.domains.video_action_set import runtime_profiling as profiling


REPO_ROOT = Path(__file__).resolve().parents[1]


def _record(frame_id: str, disposition: str) -> dict[str, Any]:
    return {
        "frame_id": frame_id,
        "expected_disposition": disposition,
        "pixels": [[0]],
        "metadata": {"episode_seed": 1, "seed_digest": "sha256:test"},
        "observation_pixel_digest": "sha256:" + "0" * 64,
    }


def _profiling_fixture_records() -> list[dict[str, Any]]:
    return [
        *[
            _record(frame_id, "valid")
            for frame_id in [
                "selection:selection:valid:3b017d7e360a1086:frame-00",
                "selection:selection:valid:3b017d7e360a1086:frame-01",
                "selection:selection:valid:3b017d7e360a1086:frame-02",
                "selection:selection:valid:3b017d7e360a1086:frame-03",
                "selection:selection:valid:8397b393db8e9235:frame-00",
                "selection:selection:valid:8397b393db8e9235:frame-01",
            ]
        ],
        *[
            _record(frame_id, "distinguishable_invalid_input")
            for frame_id in [
                "selection:selection:frame_invalid:d404ad591f7282e2:frame-00",
                "selection:selection:frame_invalid:d404ad591f7282e2:frame-01",
            ]
        ],
        *[
            _record(frame_id, "information_theoretic_control")
            for frame_id in [
                "selection:selection:information_control:b333ebf38f849061:frame-00",
                "selection:selection:information_control:b333ebf38f849061:frame-01",
            ]
        ],
    ]


def test_runtime_profiling_aliases_and_benchmark_wrappers() -> None:
    assert benchmark._profile_provider is profiling.profile_provider
    assert benchmark._profiling_records is not profiling.select_profiling_records
    assert benchmark.profile_runtime is not profiling.runtime_profile_payload


def test_select_profiling_records_preserves_category_order_and_limits() -> None:
    records = _profiling_fixture_records()

    assert [
        row["frame_id"] for row in profiling.select_profiling_records(records, -1)
    ] == ["selection:selection:valid:3b017d7e360a1086:frame-00"]
    assert [
        row["frame_id"] for row in profiling.select_profiling_records(records, 0)
    ] == ["selection:selection:valid:3b017d7e360a1086:frame-00"]
    assert [
        row["frame_id"] for row in profiling.select_profiling_records(records, 1)
    ] == ["selection:selection:valid:3b017d7e360a1086:frame-00"]
    assert [
        row["frame_id"] for row in profiling.select_profiling_records(records, 8)
    ] == [
        "selection:selection:valid:3b017d7e360a1086:frame-00",
        "selection:selection:valid:3b017d7e360a1086:frame-01",
        "selection:selection:valid:3b017d7e360a1086:frame-02",
        "selection:selection:valid:3b017d7e360a1086:frame-03",
        "selection:selection:valid:8397b393db8e9235:frame-00",
        "selection:selection:valid:8397b393db8e9235:frame-01",
        "selection:selection:frame_invalid:d404ad591f7282e2:frame-00",
        "selection:selection:frame_invalid:d404ad591f7282e2:frame-01",
    ]
    assert [
        row["frame_id"] for row in profiling.select_profiling_records(records, 10)
    ] == [row["frame_id"] for row in records]
    assert [
        row["frame_id"] for row in profiling.select_profiling_records(records, 12)
    ] == [row["frame_id"] for row in records]

    reduced = [_record("invalid-0", "distinguishable_invalid_input")]
    assert [
        row["frame_id"] for row in profiling.select_profiling_records(reduced, 8)
    ] == ["invalid-0"]
    assert profiling.select_profiling_records([], 8) == []


def test_profiling_records_remains_materialization_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = _profiling_fixture_records()
    calls: list[tuple[str, Path]] = []

    def fake_materialize(split: str, repo_root: Path) -> list[dict[str, Any]]:
        calls.append((split, repo_root))
        return records

    monkeypatch.setattr(benchmark, "_materialize_records", fake_materialize)

    selected = benchmark._profiling_records(REPO_ROOT, 1)

    assert calls == [("selection", REPO_ROOT)]
    assert [row["frame_id"] for row in selected] == [
        "selection:selection:valid:3b017d7e360a1086:frame-00"
    ]


def test_profile_provider_timing_boundary_and_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []

    class FakeObservation:
        pass

    def fake_observation(record: dict[str, Any]) -> FakeObservation:
        log.append(f"observation:{record['frame_id']}")
        return FakeObservation()

    times = iter([10.0, 11.5, 20.0, 20.5])

    def fake_perf_counter() -> float:
        log.append("timer")
        return next(times)

    def fake_reference(**kwargs: object) -> object:
        log.append(f"reference:{kwargs['provider_id']}")
        return object()

    def fake_optimized(**kwargs: object) -> object:
        log.append(f"optimized:{kwargs['provider_id']}")
        return object()

    monkeypatch.setattr(profiling, "provider_observation_for_record", fake_observation)
    monkeypatch.setattr(profiling.time, "perf_counter", fake_perf_counter)
    monkeypatch.setattr(profiling, "score_all_rows_reference", fake_reference)
    monkeypatch.setattr(profiling, "score_all_rows_optimized", fake_optimized)
    records = [_record("frame-0", "valid"), _record("frame-1", "valid")]

    profile = profiling.profile_provider(
        provider_id="P1",
        records=records,
        prototypes={},
        policy_artifact_id="policy",
        implementation="reference",
    )

    assert log == [
        "observation:frame-0",
        "timer",
        "reference:P1",
        "timer",
        "observation:frame-1",
        "timer",
        "reference:P1",
        "timer",
    ]
    assert profile == {
        "provider_id": "P1",
        "implementation": "reference",
        "frame_count": 2,
        "total_seconds": 2.0,
        "mean_seconds_per_frame": 1.0,
        "mean_seconds_per_candidate": 1.0 / 112.0,
        "provider_scoring_call_count": 2,
        "candidate_comparison_count": 224,
    }

    log.clear()
    times = iter([1.0, 1.25])
    fallback = profiling.profile_provider(
        provider_id="P2",
        records=[_record("frame-2", "valid")],
        prototypes={},
        policy_artifact_id="policy",
        implementation="anything-else",
    )
    assert "optimized:P2" in log
    assert fallback["total_seconds"] == 0.25

    empty = profiling.profile_provider(
        provider_id="P3",
        records=[],
        prototypes={},
        policy_artifact_id="policy",
        implementation="reference",
    )
    assert empty == {
        "provider_id": "P3",
        "implementation": "reference",
        "frame_count": 0,
        "total_seconds": 0.0,
        "mean_seconds_per_frame": 0.0,
        "mean_seconds_per_candidate": 0.0,
        "provider_scoring_call_count": 0,
        "candidate_comparison_count": 0,
    }


def test_runtime_profile_payload_is_frozen() -> None:
    reference = [
        {
            "provider_id": "P1",
            "implementation": "reference",
            "frame_count": 3,
            "total_seconds": 0.9,
            "mean_seconds_per_frame": 0.3,
            "mean_seconds_per_candidate": 0.3 / 112.0,
            "provider_scoring_call_count": 3,
            "candidate_comparison_count": 336,
        },
        {
            "provider_id": "P2",
            "implementation": "reference",
            "frame_count": 3,
            "total_seconds": 1.5,
            "mean_seconds_per_frame": 0.5,
            "mean_seconds_per_candidate": 0.5 / 112.0,
            "provider_scoring_call_count": 3,
            "candidate_comparison_count": 336,
        },
    ]
    optimized = [
        {
            "provider_id": "P1",
            "implementation": "optimized",
            "frame_count": 3,
            "total_seconds": 0.3,
            "mean_seconds_per_frame": 0.1,
            "mean_seconds_per_candidate": 0.1 / 112.0,
            "provider_scoring_call_count": 3,
            "candidate_comparison_count": 336,
        },
        {
            "provider_id": "P2",
            "implementation": "optimized",
            "frame_count": 3,
            "total_seconds": 0.0,
            "mean_seconds_per_frame": 0.0,
            "mean_seconds_per_candidate": 0.0,
            "provider_scoring_call_count": 3,
            "candidate_comparison_count": 336,
        },
    ]

    payload = profiling.runtime_profile_payload(
        provider_scope="custom",
        provider_ids=("P1", "P2"),
        profile_frame_count=3,
        reference=reference,
        optimized=optimized,
    )

    assert list(payload) == [
        "profile_frame_count",
        "provider_scope",
        "reference",
        "optimized",
        "comparison",
        "projected_runtime_seconds",
    ]
    assert payload["comparison"]["P1"]["speedup"] == 2.9999999999999996
    assert payload["comparison"]["P2"]["speedup"] is None
    assert payload["projected_runtime_seconds"] == {
        "development": 11.200000000000001,
        "calibration": 44.800000000000004,
        "selection": 100.80000000000001,
    }
    assert benchmark._sha256(payload) == (
        "sha256:01bdb0155964b3a8b77bdb3e6ead18c7b7285a6d2f908c24e5bcc3a609322505"
    )
