from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]


def _identity_rows_actions() -> tuple[benchmark.BenchmarkIdentity, list[str], dict[str, str]]:
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    return identity, row_ids, {row_id: lookup.choose(row_id) for row_id in row_ids}


def _outcome(rows: list[str], row_actions: dict[str, str], *, status: str | None = None) -> dict[str, object]:
    status = status or ("unique_row" if len(rows) == 1 else "conflicting_action_tie")
    payload = {
        "semantic_outcome_digest": benchmark._sha256({"rows": rows, "status": status}),
        "status": status,
        "top_row_ids": rows,
        "top_row_actions": [{"row_id": row_id, "action_id": row_actions[row_id]} for row_id in rows],
    }
    return payload


def test_episode_family_registry_covers_all_declared_families() -> None:
    registry = benchmark._episode_family_registry()
    family_ids = {entry["family_id"] for entry in registry["families"]}
    assert registry["version"] == benchmark.EPISODE_FAMILY_REGISTRY_VERSION
    assert {
        "valid",
        "conflicting_action_splice",
        "critical_evidence_corruption",
        "reordered_frames",
        "stale_repeated_frame",
        "impossible_transition",
        "declared_gap_or_unknown_action",
        "information_control",
    } <= family_ids


def test_valid_transformations_accept_bounds_and_regenerate_bytes() -> None:
    frame = np.zeros(benchmark.FRAME_SHAPE, dtype=np.uint8)
    frame[7:9, 25:27] = 40
    lower = {
        "version": benchmark.TRANSFORMATION_FAMILY_VERSION,
        "family": "compound_bounded",
        "seed": 1,
        "dx": -1,
        "dy": -1,
        "scale_percent": 90,
        "offset": 0,
        "occlusion": {"top": 0, "left": 0, "height": 2, "width": 3, "value": 64},
    }
    lower["parameter_digest"] = benchmark._sha256(lower)
    upper = {**lower, "dx": 1, "dy": 1, "scale_percent": 105, "offset": 5, "occlusion": {"top": 2, "left": 2, "height": 2, "width": 3, "value": 64}}
    upper["parameter_digest"] = benchmark._sha256({key: value for key, value in upper.items() if key != "parameter_digest"})
    interior = {**lower, "dx": 0, "dy": 0, "scale_percent": 100, "offset": 3, "occlusion": {"top": 1, "left": 1, "height": 2, "width": 3, "value": 64}}
    interior["parameter_digest"] = benchmark._sha256({key: value for key, value in interior.items() if key != "parameter_digest"})
    for params in (lower, upper, interior):
        first, trace = benchmark._apply_transformation(frame, params)
        second, _ = benchmark._apply_transformation(frame, params)
        assert np.array_equal(first, second)
        assert trace["transformation_parameter_digest"] == params["parameter_digest"]


def test_transformation_plan_tamper_is_rejected() -> None:
    identity, row_ids, row_actions = _identity_rows_actions()
    plan = benchmark._episode_plans_for_split(identity, "selection", row_ids, row_actions)[0]
    tampered = copy.deepcopy(plan)
    tampered["frame_plans"][0]["transformation_parameter_digest"] = "sha256:" + "0" * 64
    with pytest.raises(VPMValidationError):
        benchmark._validate_episode_plan(identity, tampered, row_actions)


def test_conflicting_splice_rejects_same_action_and_records_contributions() -> None:
    identity, row_ids, row_actions = _identity_rows_actions()
    plan = next(item for item in benchmark._episode_plans_for_split(identity, "selection", row_ids, row_actions) if item["mutation_kind"] == "conflicting_action_splice")
    records = benchmark._materialize_plan(plan, identity, benchmark._load_reachability_tile(REPO_ROOT))
    trace = records[0]["metadata"]["family_intervention_trace"]
    assert trace["primary_contributing_pixel_count"] > 0
    assert trace["secondary_contributing_pixel_count"] > 0
    same_action_row = next(row_id for row_id in row_ids if row_actions[row_id] == row_actions[plan["source_row_id"]] and row_id != plan["source_row_id"])
    with pytest.raises(VPMValidationError):
        benchmark._apply_conflicting_splice(
            primary_pixels=benchmark._render_row_frame(plan["source_row_id"]),
            secondary_pixels=benchmark._render_row_frame(same_action_row),
            primary_row_id=plan["source_row_id"],
            secondary_row_id=same_action_row,
            primary_action_id=row_actions[plan["source_row_id"]],
            secondary_action_id=row_actions[same_action_row],
            mask_manifest=plan["family_intervention"]["splice_mask"],
        )


def test_critical_corruption_rejects_bad_coordinates_and_stale_digest() -> None:
    source = benchmark._render_row_frame("tank=0|target=0|cooldown=0")
    with pytest.raises(VPMValidationError):
        benchmark._critical_coordinate_manifest([])
    with pytest.raises(VPMValidationError):
        benchmark._critical_coordinate_manifest([(0, 0)])
    output, trace = benchmark._apply_critical_corruption(source, benchmark._critical_coordinate_manifest())
    assert trace["changed_pixel_count"] == 4
    record = {"pixels": output, "observation_pixel_digest": benchmark._array_digest(source), "metadata": {"family_intervention_trace": trace}}
    assert benchmark.validate_materialized_family_record(record) == "stale_observation_digest"


def test_temporal_families_materialize_real_sequence_events() -> None:
    identity, row_ids, row_actions = _identity_rows_actions()
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    plans = benchmark._episode_plans_for_split(identity, "selection", row_ids, row_actions)
    reordered = next(item for item in plans if item["mutation_kind"] == "reordered_frames")
    reordered_records = benchmark._materialize_plan(reordered, identity, tile)
    assert [record["metadata"]["original_frame_index"] for record in reordered_records] == [1, 0, 2, 3]
    stale = next(item for item in plans if item["mutation_kind"] == "stale_repeated_frame")
    stale_records = benchmark._materialize_plan(stale, identity, tile)
    stale_event = stale_records[1]["metadata"]["stale_repeat"]
    assert stale_event["original_destination_digest"] != stale_event["replacement_digest"]
    assert stale_records[1]["observation_pixel_digest"] == stale_records[0]["observation_pixel_digest"]
    gap = next(item for item in plans if item["mutation_kind"] == "declared_gap_or_unknown_action")
    gap_records = benchmark._materialize_plan(gap, identity, tile)
    assert gap_records[2]["event_type"] == "gap_unknown"
    assert gap_records[2]["pixels"] is None


def test_information_controls_are_byte_identical_and_denominator_excluded() -> None:
    identity, row_ids, row_actions = _identity_rows_actions()
    plan = next(item for item in benchmark._episode_plans_for_split(identity, "selection", row_ids, row_actions) if item["family_label"] == "information_control")
    records = benchmark._materialize_plan(plan, identity, benchmark._load_reachability_tile(REPO_ROOT))
    assert benchmark.validate_control_episode_records(records) == "ok"
    tampered = copy.deepcopy(records)
    tampered[1]["pixels"] = np.array(tampered[1]["pixels"], copy=True)
    tampered[1]["pixels"][0, 0] = 1
    tampered[1]["observation_pixel_digest"] = benchmark._array_digest(tampered[1]["pixels"])
    assert benchmark.validate_control_episode_records(tampered) == "control_byte_identity_mismatch"


def test_reachability_composition_mutation_failure_codes() -> None:
    _identity, row_ids, row_actions = _identity_rows_actions()
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    source = row_ids[0]
    edge = benchmark._tile_edge(tile, source, row_actions[source])
    destination = edge["reachable_row_ids"][0]
    trace = benchmark.compose_reachability_trace(
        frame_id="frame-1",
        semantic_outcome=_outcome([destination], row_actions),
        previous_state={"status": "resolved", "candidate_rows": (source,)},
        reachability_tile=tile,
        row_actions=row_actions,
    )
    assert trace["status"] == "resolved"
    assert trace["resolved_row_id"] == destination
    mutated = copy.deepcopy(tile)
    for item in mutated["edges"]:
        if item["source_row_id"] == source and item["action_id"] == row_actions[source]:
            item["reachable_row_ids"] = [row for row in item["reachable_row_ids"] if row != destination]
            break
    assert benchmark.validate_reachability_trace(
        trace,
        semantic_outcome=_outcome([destination], row_actions),
        previous_state={"status": "resolved", "candidate_rows": (source,)},
        reachability_tile=mutated,
        row_actions=row_actions,
    ) == "foreign_reachability_tile"
    tampered_trace = copy.deepcopy(trace)
    tampered_trace["consulted_edges"] = []
    tampered_trace["trace_digest"] = benchmark._trace_digest(tampered_trace)
    assert benchmark.validate_reachability_trace(
        tampered_trace,
        semantic_outcome=_outcome([destination], row_actions),
        previous_state={"status": "resolved", "candidate_rows": (source,)},
        reachability_tile=tile,
        row_actions=row_actions,
    ) == "consulted_edge_mismatch"


def test_ambiguous_reachability_never_uses_lexical_winner() -> None:
    _identity, row_ids, row_actions = _identity_rows_actions()
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    conflicting = [row for row in row_ids if row_actions[row] == "LEFT"][:1] + [row for row in row_ids if row_actions[row] == "RIGHT"][:1]
    trace = benchmark.compose_reachability_trace(
        frame_id="frame-0",
        semantic_outcome=_outcome(conflicting, row_actions, status="conflicting_action_tie"),
        previous_state=None,
        reachability_tile=tile,
        row_actions=row_actions,
    )
    assert trace["status"] == "rejected"
    assert trace["resolved_row_id"] is None
    assert trace["resolved_action_id"] is None
    same_action = [row for row in row_ids if row_actions[row] == "LEFT"][:2]
    unanimous = benchmark.compose_reachability_trace(
        frame_id="frame-0",
        semantic_outcome=_outcome(same_action, row_actions, status="action_unanimous_tie"),
        previous_state=None,
        reachability_tile=tile,
        row_actions=row_actions,
    )
    assert unanimous["retained_rows"] == sorted(same_action)
    impossible = benchmark.compose_reachability_trace(
        frame_id="frame-1",
        semantic_outcome=_outcome([row_ids[-1]], row_actions),
        previous_state={"status": "resolved", "candidate_rows": (row_ids[0],)},
        reachability_tile=tile,
        row_actions=row_actions,
    )
    assert impossible["rejection_reason"] == "no_reachable_candidate"
