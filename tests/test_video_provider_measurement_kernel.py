from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
from copy import deepcopy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set import provider_measurement as measurement
from zeromodel.video_prospective_providers import score_all_rows_reference


REPO_ROOT = Path(__file__).resolve().parents[1]

PROVIDER_ROW_KEYS = [
    "benchmark_version",
    "generator_version",
    "split",
    "episode_id",
    "clip_id",
    "frame_id",
    "sequence_number",
    "event_type",
    "family",
    "expected_disposition",
    "episode_family",
    "episode_disposition",
    "frame_disposition",
    "denominator_class",
    "expected_row",
    "expected_action",
    "actual_executed_action",
    "action_known",
    "gap_declaration",
    "observation_pixel_digest",
    "metadata",
    "provider_id",
    "provider_version",
    "policy_artifact_id",
    "reachability_tile_digest",
    "all_112_row_ids",
    "all_112_raw_scores",
    "all_112_quantized_scores",
    "complete_ordered_ranking",
    "tie_groups",
    "semantic_top_set_outcome",
    "semantic_status",
    "resolved_row",
    "resolved_action",
    "top_quantized_score",
    "top_row_ids",
    "top_action_ids",
    "semantic_outcome_digest",
    "reachability_composition_trace",
    "winner_row",
    "winner_action",
    "winner_quantized_score",
    "runner_up_row",
    "runner_up_quantized_score",
    "policy_row_universe_digest",
    "quantized_score_vector_digest",
    "raw_score_diagnostic_digest",
    "score_vector_digest",
    "ranking_digest",
    "observation_digest",
    "provider_observation_descriptor",
    "provider_observation_digest",
    "episode_seed",
    "generator_identity",
    "provider_diagnostics",
]


def _fake_digest(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FrozenRowScore:
    row_id: str
    raw_score: float
    quantized_score: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_id": self.row_id,
            "raw_score": self.raw_score,
            "quantized_score": self.quantized_score,
        }


@dataclass(frozen=True)
class FrozenTieGroup:
    tie_group_index: int
    quantized_score: int
    row_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tie_group_index": self.tie_group_index,
            "quantized_score": self.quantized_score,
            "row_ids": list(self.row_ids),
        }


@dataclass(frozen=True)
class FrozenRanking:
    ranked_row_ids: tuple[str, ...]
    tie_groups: tuple[FrozenTieGroup, ...]
    ranking_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "ranking_digest": self.ranking_digest,
            "ranked_row_ids": list(self.ranked_row_ids),
            "tie_groups": [group.to_dict() for group in self.tie_groups],
        }


@dataclass(frozen=True)
class FrozenEvidence:
    row_scores: tuple[FrozenRowScore, ...]
    ranking: FrozenRanking
    policy_row_universe_digest: str
    quantized_score_vector_digest: str
    raw_score_diagnostic_digest: str

    @property
    def score_vector_digest(self) -> str:
        return self.quantized_score_vector_digest

    def to_dict(self) -> dict[str, Any]:
        return {"row_scores": [item.to_dict() for item in self.row_scores]}


@dataclass(frozen=True)
class FrozenOutcome:
    provider_id: str
    provider_version: str
    top_quantized_score: int
    top_row_ids: tuple[str, ...]
    top_action_ids: tuple[str, ...]
    top_row_actions: tuple[tuple[str, str], ...]
    status: str
    resolved_row_id: str | None
    resolved_action_id: str | None
    semantic_outcome_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_id": self.provider_id,
            "provider_version": self.provider_version,
            "top_quantized_score": self.top_quantized_score,
            "top_row_ids": list(self.top_row_ids),
            "top_row_actions": [
                {"row_id": row_id, "action_id": action_id}
                for row_id, action_id in self.top_row_actions
            ],
            "top_action_ids": list(self.top_action_ids),
            "status": self.status,
            "resolved_row_id": self.resolved_row_id,
            "resolved_action_id": self.resolved_action_id,
            "semantic_outcome_digest": self.semantic_outcome_digest,
        }


@dataclass(frozen=True)
class FrozenProviderResult:
    provider_id: str
    provider_version: str
    evidence: FrozenEvidence
    winner_row_id: str
    winner_action_id: str
    maximum_tie_size: int
    semantic_top_set_outcome: FrozenOutcome
    diagnostics: dict[str, Any]


def _frozen_provider_result(
    provider_id: str,
    *,
    row_ids: list[str],
    row_actions: dict[str, str],
) -> FrozenProviderResult:
    provider_version = measurement.provider_version(provider_id)
    if provider_id == "P2":
        top_rows = tuple(row_id for row_id in row_ids if row_actions[row_id] == "STAY")[
            :2
        ]
        status = "action_unanimous_tie"
        resolved_row = None
        resolved_action = "STAY"
    else:
        top_rows = (row_ids[0],)
        status = "unique_row"
        resolved_row = row_ids[0]
        resolved_action = row_actions[row_ids[0]]
    top_actions = tuple(sorted({row_actions[row_id] for row_id in top_rows}))
    ranking_rows = (
        *top_rows,
        *(row_id for row_id in row_ids if row_id not in top_rows),
    )
    lower_rows = tuple(row_id for row_id in row_ids if row_id not in top_rows)
    score_by_row = {
        row_id: (
            0.9 if row_id in top_rows else 0.1,
            900_000 if row_id in top_rows else 100_000,
        )
        for row_id in row_ids
    }
    row_scores = tuple(
        FrozenRowScore(row_id, score_by_row[row_id][0], score_by_row[row_id][1])
        for row_id in row_ids
    )
    evidence = FrozenEvidence(
        row_scores=row_scores,
        ranking=FrozenRanking(
            ranked_row_ids=tuple(ranking_rows),
            tie_groups=(
                FrozenTieGroup(0, 900_000, top_rows),
                FrozenTieGroup(1, 100_000, lower_rows),
            ),
            ranking_digest=_fake_digest(provider_id, "ranking"),
        ),
        policy_row_universe_digest=_fake_digest(provider_id, "policy-row-universe"),
        quantized_score_vector_digest=_fake_digest(provider_id, "quantized"),
        raw_score_diagnostic_digest=_fake_digest(provider_id, "raw"),
    )
    winner_row = top_rows[0]
    outcome = FrozenOutcome(
        provider_id=provider_id,
        provider_version=provider_version,
        top_quantized_score=900_000,
        top_row_ids=top_rows,
        top_action_ids=top_actions,
        top_row_actions=tuple((row_id, row_actions[row_id]) for row_id in top_rows),
        status=status,
        resolved_row_id=resolved_row,
        resolved_action_id=resolved_action,
        semantic_outcome_digest=_fake_digest(provider_id, "semantic"),
    )
    return FrozenProviderResult(
        provider_id=provider_id,
        provider_version=provider_version,
        evidence=evidence,
        winner_row_id=winner_row,
        winner_action_id=row_actions[winner_row],
        maximum_tie_size=len(top_rows),
        semantic_top_set_outcome=outcome,
        diagnostics={"frozen_provider_result": provider_id},
    )


def _frozen_provider_results(
    *,
    row_ids: list[str],
    row_actions: dict[str, str],
) -> tuple[FrozenProviderResult, ...]:
    return tuple(
        _frozen_provider_result(provider_id, row_ids=row_ids, row_actions=row_actions)
        for provider_id in ("P1", "P2", "P3")
    )


def _score_record_with_provider_results(
    record: dict[str, Any],
    prototypes: dict[str, tuple[str, str, str, Any]],
    policy_artifact_id: str,
    provider_results: tuple[Any, ...],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    with patch.object(
        measurement, "_score_all_providers", return_value=provider_results
    ):
        return measurement.score_record(
            record,
            prototypes,
            policy_artifact_id,
            **kwargs,
        )


def _measure_collection_with_provider_results(
    records: list[dict[str, Any]],
    prototypes: dict[str, tuple[str, str, str, Any]],
    policy_artifact_id: str,
    provider_results: tuple[Any, ...],
    **kwargs: Any,
) -> list[dict[str, Any]]:
    with patch.object(
        measurement, "_score_all_providers", return_value=provider_results
    ):
        return measurement.measure_record_collection(
            records,
            prototypes,
            policy_artifact_id,
            **kwargs,
        )


@pytest.fixture(scope="module")
def measurement_context() -> dict[str, Any]:
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    plans = benchmark._episode_plans_for_split(
        identity, "selection", row_ids, row_actions
    )
    by_key: dict[str, dict[str, Any]] = {}
    for plan in plans:
        key = str(plan["family_label"])
        if key in {"frame_invalid", "temporal_negative"}:
            key = f"{key}:{plan['mutation_kind']}"
        by_key.setdefault(key, plan)
    valid_records = benchmark._materialize_plan(by_key["valid"], identity, tile)
    gap_records = benchmark._materialize_plan(
        by_key["temporal_negative:declared_gap_or_unknown_action"], identity, tile
    )
    control_records = benchmark._materialize_plan(
        by_key["information_control"], identity, tile
    )
    prototypes = benchmark.canonical_prototypes()
    first_valid = deepcopy(valid_records[0])
    frozen_provider_results = _frozen_provider_results(
        row_ids=row_ids,
        row_actions=row_actions,
    )
    score_rows = _score_record_with_provider_results(
        first_valid,
        prototypes,
        policy.artifact_id,
        frozen_provider_results,
    )
    stream_rows = _measure_collection_with_provider_results(
        [
            deepcopy(valid_records[0]),
            deepcopy(gap_records[2]),
            deepcopy(valid_records[1]),
        ],
        prototypes,
        policy.artifact_id,
        frozen_provider_results,
        reachability_tile=tile,
        row_actions=row_actions,
    )
    first_control = _score_record_with_provider_results(
        deepcopy(control_records[0]),
        prototypes,
        policy.artifact_id,
        frozen_provider_results,
    )
    second_control = _score_record_with_provider_results(
        deepcopy(control_records[1]),
        prototypes,
        policy.artifact_id,
        frozen_provider_results,
    )
    return {
        "policy": policy,
        "row_actions": row_actions,
        "tile": tile,
        "prototypes": prototypes,
        "valid_records": valid_records,
        "gap_records": gap_records,
        "control_records": control_records,
        "score_record_input": first_valid,
        "score_rows": score_rows,
        "stream_rows": stream_rows,
        "control_rows": (first_control, second_control),
    }


def _digest(value: object) -> str:
    return benchmark._sha256(value)


def test_provider_measurement_aliases_and_signatures_are_direct() -> None:
    assert benchmark._provider_version is measurement.provider_version
    assert benchmark._score_vector_to_payload is measurement.score_vector_to_payload
    assert benchmark._score_record is measurement.score_record
    assert benchmark.SOURCE_SCOPE == measurement.SOURCE_SCOPE
    assert inspect.signature(benchmark._score_record) == inspect.signature(
        measurement.score_record
    )
    assert inspect.signature(benchmark._score_vector_to_payload) == inspect.signature(
        measurement.score_vector_to_payload
    )


def test_provider_version_and_score_vector_payload_goldens(
    measurement_context: dict[str, Any],
) -> None:
    assert {
        provider_id: measurement.provider_version(provider_id)
        for provider_id in ("P1", "P2", "P3")
    } == {
        "P1": "zeromodel-video-prospective-normalized-pixel/v1",
        "P2": "zeromodel-video-prospective-local-correlation/v1",
        "P3": "zeromodel-video-prospective-b3-joint-fit/v1",
    }
    with pytest.raises(KeyError) as excinfo:
        measurement.provider_version("PX")
    assert excinfo.value.args == ("PX",)

    first_observation = next(iter(measurement_context["prototypes"].values()))[3]
    vector = score_all_rows_reference(
        provider_id="P1",
        observation=first_observation,
        prototypes=measurement_context["prototypes"],
        policy_artifact_id=measurement_context["policy"].artifact_id,
        source_scope=measurement.SOURCE_SCOPE,
    )
    payload = measurement.score_vector_to_payload(vector)

    assert list(payload) == [
        "provider_id",
        "provider_version",
        "row_ids",
        "raw_scores",
        "quantized_scores",
        "ranking",
        "tie_groups",
        "score_vector_digest",
        "ranking_digest",
    ]
    assert _digest(payload) == (
        "sha256:51d3e8ffe7b955511b78b5d0ae010c8f32d4471fb118da598ff7d95fa6d75ec4"
    )
    assert _digest(payload["row_ids"]) == (
        "sha256:10369de0f84b28852b5956a893cd92f8106f2f095c96db7525c0c363f9b9a816"
    )
    assert _digest(payload["raw_scores"]) == (
        "sha256:064def264c67cc57f4109bcaa155e1810967f7a89624be0c2cf50c3362547025"
    )
    assert _digest(payload["quantized_scores"]) == (
        "sha256:8c0a26808337d6101b202d427f2e9359409afe4b2dc3c90f2374217630e2ab15"
    )
    assert payload["ranking_digest"] == (
        "sha256:59031a77435a6711ab548aabeaaf2ae27cf8370e5a7d7c205b741970a5777bfd"
    )
    with pytest.raises(AttributeError):
        measurement.score_vector_to_payload({"provider_id": "P1"})


def test_score_record_goldens_and_reachability_state(
    measurement_context: dict[str, Any],
) -> None:
    record = measurement_context["score_record_input"]
    rows = measurement_context["score_rows"]

    assert [row["provider_id"] for row in rows] == ["P1", "P2", "P3"]
    assert [list(row) for row in rows] == [PROVIDER_ROW_KEYS] * 3
    assert [row["reachability_composition_trace"] for row in rows] == [None] * 3
    assert [row["provider_observation_digest"] for row in rows] == [
        "sha256:bb6de887ea70ccd7c410547b46967c331b19186365a4d0eac3accf7240926e7b"
    ] * 3
    assert [row["provider_observation_descriptor"] for row in rows] == [
        rows[0]["provider_observation_descriptor"]
    ] * 3
    assert (
        rows[0]["provider_observation_descriptor"]
        is rows[1]["provider_observation_descriptor"]
    )
    assert [row["score_vector_digest"] for row in rows] == [
        "sha256:7000f400486036f41341e783675b7033c8eedb7e07700c3ed5e77920daeb67e3",
        "sha256:2c0c5ee11e871b7b4323e229a64237fc0adae74ca7705e68db7a2d29bd3b8e8a",
        "sha256:50a096dc86fd28708bde7132a95750c27c31ae7d1960cb055156bf138008bfdb",
    ]
    assert [_digest(row) for row in rows] == [
        "sha256:d931cf814bbd896f5a1ac8c8d4f5caa096d8223a6e8a889f220c57bd3d5c8e66",
        "sha256:c85006dba9bc6e098883991cb91bbbdccebf3284d7348ccea2777801e75ba1a5",
        "sha256:0fce0f3da58bb19f1f67d8eed014101b48562be18a844bfc7d7f627bf3f17725",
    ]
    assert rows[1]["semantic_status"] == "action_unanimous_tie"
    assert rows[1]["resolved_action"] == "STAY"
    assert rows[1]["resolved_row"] is None
    assert rows[1]["winner_quantized_score"] is None
    assert rows[0]["metadata"] is record["metadata"]

    rows_with_reachability = measurement_context["stream_rows"][:3]
    assert [
        row["reachability_composition_trace"]["trace_digest"]
        for row in rows_with_reachability
    ] == [
        "sha256:a5aa53d1e4a339d7eba7f4076006a920ebc2a83069f21c886861faad04a89b50",
        "sha256:ac99a15deb2b185f84dcbb85e0c5d08ea9086459a3c282a66ef2ed65a5122475",
        "sha256:d5a9f5a46a2f0a9b9f0f550c1e6849b2668b07ee8b3851ae229d83cd6661408b",
    ]
    assert (
        rows_with_reachability[0]["reachability_composition_trace"]["status"]
        == "resolved"
    )
    assert rows_with_reachability[1]["resolved_action"] == "STAY"


def test_score_record_validation_and_partial_reachability_inputs(
    monkeypatch: pytest.MonkeyPatch,
    measurement_context: dict[str, Any],
) -> None:
    with pytest.raises(VPMValidationError, match="materialized record missing pixels"):
        measurement.score_record(
            {"metadata": {}},
            measurement_context["prototypes"],
            measurement_context["policy"].artifact_id,
        )
    gap = deepcopy(measurement_context["gap_records"][2])
    with pytest.raises(
        VPMValidationError,
        match="typed gap events cannot be provider-scored as ordinary frames",
    ):
        measurement.score_record(
            gap,
            measurement_context["prototypes"],
            measurement_context["policy"].artifact_id,
        )

    class FakeResult:
        provider_id = "P1"

    monkeypatch.setattr(
        measurement, "_score_all_providers", lambda **_kwargs: (FakeResult(),)
    )
    monkeypatch.setattr(
        measurement,
        "_provider_evidence_row",
        lambda **kwargs: {
            "reachability_composition_trace": kwargs["operational_trace"]
        },
    )
    state = {"P1": {"status": "sentinel", "candidate_rows": ("row",)}}
    rows = measurement.score_record(
        deepcopy(measurement_context["valid_records"][0]),
        measurement_context["prototypes"],
        measurement_context["policy"].artifact_id,
        reachability_tile=measurement_context["tile"],
        reachability_state=state,
        row_actions=None,
    )
    assert [row["reachability_composition_trace"] for row in rows] == [None]
    assert state == {"P1": {"status": "sentinel", "candidate_rows": ("row",)}}


def test_measure_record_collection_gap_stream_and_final_split_passthrough(
    monkeypatch: pytest.MonkeyPatch,
    measurement_context: dict[str, Any],
) -> None:
    rows = measurement_context["stream_rows"]

    assert len(rows) == 6
    assert [(row["frame_id"], row["provider_id"]) for row in rows] == [
        ("selection:selection:valid:3b017d7e360a1086:frame-00", "P1"),
        ("selection:selection:valid:3b017d7e360a1086:frame-00", "P2"),
        ("selection:selection:valid:3b017d7e360a1086:frame-00", "P3"),
        ("selection:selection:valid:3b017d7e360a1086:frame-01", "P1"),
        ("selection:selection:valid:3b017d7e360a1086:frame-01", "P2"),
        ("selection:selection:valid:3b017d7e360a1086:frame-01", "P3"),
    ]
    assert [
        row["reachability_composition_trace"]["rejection_reason"] for row in rows
    ] == [
        None,
        None,
        None,
        "prior_state_unresolved",
        "prior_state_unresolved",
        "prior_state_unresolved",
    ]
    assert _digest(rows) == (
        "sha256:367990ec5badbefe05e72c8880e0d4bbd884036fa2c59f7cf5f2bd2276f57c61"
    )

    final_record = deepcopy(measurement_context["valid_records"][0])
    final_record["split"] = "final"
    monkeypatch.setattr(measurement, "_score_all_providers", lambda **_kwargs: tuple())
    final_rows = measurement.score_record(
        final_record,
        measurement_context["prototypes"],
        measurement_context["policy"].artifact_id,
    )
    assert final_rows == []


def test_information_controls_share_provider_visible_measurement(
    measurement_context: dict[str, Any],
) -> None:
    first, second = measurement_context["control_rows"]

    assert {row["provider_observation_digest"] for row in first + second} == {
        "sha256:198009840efffc970b1da1c9d5fd66937ddd499897340f55980223354b917805"
    }
    assert [(row["provider_id"], row["score_vector_digest"]) for row in first] == [
        (row["provider_id"], row["score_vector_digest"]) for row in second
    ]
    assert [(row["provider_id"], row["semantic_outcome_digest"]) for row in first] == [
        (row["provider_id"], row["semantic_outcome_digest"]) for row in second
    ]
    assert (
        measurement_context["control_records"][0]["metadata"][
            "hidden_source_history_id"
        ]
        != measurement_context["control_records"][1]["metadata"][
            "hidden_source_history_id"
        ]
    )


def test_provider_call_order_and_error_boundary(
    monkeypatch: pytest.MonkeyPatch,
    measurement_context: dict[str, Any],
) -> None:
    calls: list[str] = []

    class FakeObservation:
        def to_descriptor(self) -> dict[str, object]:
            calls.append("descriptor")
            return {"raw_digest": "sha256:" + "0" * 64, "shape": [16, 28]}

    def fake_observation(_record: dict[str, Any]) -> FakeObservation:
        calls.append("observation")
        return FakeObservation()

    def fake_p1(**_kwargs: object) -> object:
        calls.append("P1")
        return object()

    def fake_p2(**_kwargs: object) -> object:
        calls.append("P2")
        raise RuntimeError("synthetic P2 failure")

    def fake_p3(**_kwargs: object) -> object:
        calls.append("P3")
        return object()

    monkeypatch.setattr(
        measurement, "provider_observation_for_record", fake_observation
    )
    monkeypatch.setattr(
        measurement,
        "provider_observation_digest",
        lambda _descriptor: "sha256:" + "1" * 64,
    )
    monkeypatch.setattr(measurement, "score_normalized_pixel", fake_p1)
    monkeypatch.setattr(measurement, "score_registered_local_correlation", fake_p2)
    monkeypatch.setattr(measurement, "score_b3_joint_fit", fake_p3)
    state = {"P1": None, "P2": None, "P3": None}

    with pytest.raises(RuntimeError, match="synthetic P2 failure"):
        measurement.score_record(
            deepcopy(measurement_context["valid_records"][0]),
            measurement_context["prototypes"],
            measurement_context["policy"].artifact_id,
            reachability_tile=measurement_context["tile"],
            reachability_state=state,
            row_actions=measurement_context["row_actions"],
        )

    assert calls == ["observation", "descriptor", "P1", "P2"]
    assert state == {"P1": None, "P2": None, "P3": None}
