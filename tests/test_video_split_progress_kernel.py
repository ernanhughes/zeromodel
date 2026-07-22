from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

import research.video_action_set.video_action_set_cli as cli
from research.video_action_set import build_orchestration as build
from research.video_action_set import provider_measurement as measurement
from zeromodel.video.domains.video_action_set.artifact_io import _sha256
from research.video_action_set.provider_measurement import SplitBuildProgress
from research.video.video_prospective_providers import PROSPECTIVE_PROVIDER_IDS


def _records() -> list[dict[str, Any]]:
    return [
        {"split": "selection", "frame_id": "frame-00", "pixels": [[0]], "metadata": {}},
        {
            "split": "selection",
            "frame_id": "frame-gap",
            "event_type": "gap_unknown",
            "gap_declaration": "declared_gap",
            "pixels": None,
            "metadata": {},
        },
        {"split": "selection", "frame_id": "frame-01", "pixels": [[1]], "metadata": {}},
    ]


def _provider_rows(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "frame_id": record["frame_id"],
            "provider_id": provider_id,
            "score_vector_digest": f"sha256:{provider_id.lower()}",
            "semantic_outcome_digest": f"sha256:semantic-{provider_id.lower()}",
        }
        for provider_id in PROSPECTIVE_PROVIDER_IDS
    ]


def _measure(
    monkeypatch: pytest.MonkeyPatch,
    *,
    progress_observer=None,
) -> list[dict[str, Any]]:
    monkeypatch.setattr(
        measurement,
        "score_record",
        lambda record, *_args, **_kwargs: _provider_rows(record),
    )
    return measurement.measure_record_collection(
        _records(),
        prototypes={},
        policy_artifact_id="policy",
        reachability_tile={},
        row_actions={},
        split="selection",
        progress_observer=progress_observer,
    )


def test_measurement_without_observer_produces_no_progress_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = _measure(monkeypatch)

    captured = capsys.readouterr()
    assert rows == _provider_rows(_records()[0]) + _provider_rows(_records()[2])
    assert captured.out == ""
    assert captured.err == ""


def test_progress_observer_receives_monotonic_counts_and_gap_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[SplitBuildProgress] = []

    _measure(monkeypatch, progress_observer=events.append)

    assert [event.processed_frame_count for event in events] == [1, 2, 3]
    assert {event.total_frame_count for event in events} == {3}
    assert [event.scoreable_frame_count_processed for event in events] == [1, 1, 2]
    assert [event.typed_gap_count_processed for event in events] == [0, 1, 1]
    assert [
        event.provider_scoring_calls_completed
        for event in events
    ] == [
        event.scoreable_frame_count_processed * len(PROSPECTIVE_PROVIDER_IDS)
        for event in events
    ]
    assert [event.percentage_complete for event in events] == pytest.approx(
        [100.0 / 3.0, 200.0 / 3.0, 100.0]
    )


def test_empty_record_collection_emits_single_final_progress_event() -> None:
    events: list[SplitBuildProgress] = []

    rows = measurement.measure_record_collection(
        [],
        prototypes={},
        policy_artifact_id="policy",
        reachability_tile={},
        row_actions={},
        split="selection",
        progress_observer=events.append,
    )

    assert rows == []
    assert len(events) == 1
    event = events[0]
    assert event.processed_frame_count == 0
    assert event.total_frame_count == 0
    assert event.scoreable_frame_count_processed == 0
    assert event.typed_gap_count_processed == 0
    assert event.provider_scoring_calls_completed == 0
    assert event.percentage_complete == 100.0


def test_empty_record_collection_without_observer_emits_no_progress_output(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rows = measurement.measure_record_collection(
        [],
        prototypes={},
        policy_artifact_id="policy",
        reachability_tile={},
        row_actions={},
        split="selection",
        progress_observer=None,
    )

    captured = capsys.readouterr()
    assert rows == []
    assert captured.out == ""
    assert captured.err == ""


def test_progress_observer_does_not_alter_evidence_or_scientific_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    without_progress = _measure(monkeypatch)
    without_digest = _sha256(without_progress)
    monkeypatch.undo()

    events: list[SplitBuildProgress] = []
    with_progress = _measure(monkeypatch, progress_observer=events.append)

    assert with_progress == without_progress
    assert _sha256(with_progress) == without_digest
    assert events
    forbidden_progress_keys = {
        "processed_frame_count",
        "total_frame_count",
        "elapsed_seconds",
        "percentage_complete",
    }
    assert all(forbidden_progress_keys.isdisjoint(row.keys()) for row in with_progress)


def test_progress_observer_exceptions_propagate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def failing_observer(_event: SplitBuildProgress) -> None:
        raise RuntimeError("progress monitor failed")

    with pytest.raises(RuntimeError, match="progress monitor failed"):
        _measure(monkeypatch, progress_observer=failing_observer)


def test_cli_progress_goes_to_stderr_not_scientific_stdout(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "freeze_benchmark",
        lambda *_args: {"command": "freeze"},
    )

    def fake_build_split(*_args: Any, progress_observer=None) -> dict[str, str]:
        assert progress_observer is not None
        progress_observer(
            SplitBuildProgress(
                split="development",
                processed_frame_count=1,
                total_frame_count=1,
                scoreable_frame_count_processed=1,
                typed_gap_count_processed=0,
                provider_scoring_calls_completed=len(PROSPECTIVE_PROVIDER_IDS),
                elapsed_seconds=0.25,
                percentage_complete=100.0,
            )
        )
        return {"command": "development"}

    monkeypatch.setattr(cli, "build_split", fake_build_split)
    monkeypatch.setattr(
        sys,
        "argv",
        ["instrument", "--build-development", "--progress"],
    )

    cli.main()
    captured = capsys.readouterr()

    assert json.loads(captured.out) == {"command": "development"}
    assert "development: 1/1 frames" in captured.err
    assert "provider_calls=3" in captured.err


def test_build_split_observer_failure_leaves_no_split_files(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    class FakeDTO:
        def __init__(self, payload: dict[str, Any]):
            self.payload = payload

        @classmethod
        def from_dict(cls, payload: dict[str, Any]):
            return cls(payload)

        def to_dict(self) -> dict[str, Any]:
            return self.payload

    class VideoService:
        def load_identity(self, _repo_root: Path):
            calls.append("identity")
            return SimpleNamespace(seed_digest="seed")

        def save_episode_plans(self, plans):
            calls.append("save-plans")
            return plans

        def save_observation_records(self, records):
            calls.append("save-observations")
            return records

        def list_observation_records(self, **_kwargs):
            calls.append("list-observations")
            return (
                {
                    "split": "development",
                    "frame_id": "frame-0",
                    "pixels": [[0]],
                    "metadata": {},
                },
            )

    monkeypatch.setattr(
        build,
        "_build_durable_runtime",
        lambda _path: SimpleNamespace(video_action_set=VideoService()),
    )
    monkeypatch.setattr(build, "canonical_prototypes", lambda: {})
    monkeypatch.setattr(
        build,
        "compile_policy_artifact",
        lambda: SimpleNamespace(
            artifact_id="policy",
            source=SimpleNamespace(row_ids=("row-0",)),
        ),
    )
    monkeypatch.setattr(
        build,
        "VPMPolicyLookup",
        lambda *_args, **_kwargs: SimpleNamespace(choose=lambda _row: "LEFT"),
    )
    monkeypatch.setattr(build, "_load_reachability_tile", lambda _repo_root: {})
    monkeypatch.setattr(
        build,
        "_episode_plans_for_split",
        lambda _identity, split, _rows, _actions: [
            {"split": split, "episode_id": "episode-0"}
        ],
    )
    monkeypatch.setattr(build, "EpisodePlanDTO", FakeDTO)
    monkeypatch.setattr(
        build,
        "_materialize_records",
        lambda *_args: [{"pixels": [[0]]}],
    )

    def fail_during_measurement(*_args, progress_observer=None, **_kwargs):
        calls.append("measure")
        progress_observer(
            SplitBuildProgress(
                split="development",
                processed_frame_count=1,
                total_frame_count=1,
                scoreable_frame_count_processed=1,
                typed_gap_count_processed=0,
                provider_scoring_calls_completed=len(PROSPECTIVE_PROVIDER_IDS),
                elapsed_seconds=0.01,
                percentage_complete=100.0,
            )
        )

    monkeypatch.setattr(build, "measure_record_collection", fail_during_measurement)
    monkeypatch.setattr(
        build,
        "_write_jsonl",
        lambda *_args, **_kwargs: calls.append("write-jsonl"),
    )
    monkeypatch.setattr(
        build,
        "_write_json",
        lambda *_args, **_kwargs: calls.append("write-json"),
    )

    def failing_observer(_event: SplitBuildProgress) -> None:
        raise RuntimeError("progress monitor failed")

    with pytest.raises(RuntimeError, match="progress monitor failed"):
        build.build_split(
            "development",
            tmp_path,
            tmp_path,
            progress_observer=failing_observer,
        )

    assert calls == [
        "identity",
        "save-plans",
        "save-observations",
        "list-observations",
        "measure",
    ]
    assert not (tmp_path / "development" / "frame-metadata.jsonl").exists()
    assert not (tmp_path / "development" / "provider-evidence.jsonl").exists()
    assert not (tmp_path / "development-manifest.json").exists()
    assert not (tmp_path / "development-family-closure-report.json").exists()
