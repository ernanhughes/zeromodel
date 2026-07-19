from __future__ import annotations

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


def _legacy_secondary_row_for_splice(row_ids: list[str], row_actions: dict[str, str], source_row_id: str) -> str:
    source_action = row_actions[source_row_id]
    source_tank, source_target, source_cooldown = benchmark.parse_state_row_id(source_row_id)
    for row_id in row_ids:
        if row_id == source_row_id:
            continue
        if row_actions[row_id] == source_action:
            continue
        tank, target, cooldown = benchmark.parse_state_row_id(row_id)
        if target == source_target:
            continue
        if (tank, cooldown) == (source_tank, source_cooldown):
            continue
        return row_id
    raise AssertionError("legacy fixture must find a secondary row")


def _legacy_row_band_splice(primary_row_id: str, secondary_row_id: str) -> np.ndarray:
    output = np.array(benchmark._render_row_frame(primary_row_id), copy=True)
    secondary = benchmark._render_row_frame(secondary_row_id)
    for y in range(6):
        output[y, :] = secondary[y, :]
    return output


def _first_splice_plan() -> dict[str, object]:
    identity, row_ids, row_actions = _identity_rows_actions()
    return next(
        plan
        for plan in benchmark._episode_plans_for_split(identity, "selection", row_ids, row_actions)
        if plan["mutation_kind"] == "conflicting_action_splice"
    )


def test_legacy_row_band_conflicting_splice_collided_with_canonical_universe() -> None:
    _identity, row_ids, row_actions = _identity_rows_actions()
    canonical_index = benchmark.canonical_observation_universe()["digest_to_rows"]
    collisions = []
    for source_row_id in row_ids[:28]:
        secondary_row_id = _legacy_secondary_row_for_splice(row_ids, row_actions, source_row_id)
        output = _legacy_row_band_splice(source_row_id, secondary_row_id)
        digest = benchmark._array_digest(output)
        if digest in canonical_index:
            collisions.append((source_row_id, secondary_row_id, canonical_index[digest]))
    assert len(collisions) == 28


def test_repaired_conflicting_splice_selection_frames_do_not_decode_as_canonical_valid() -> None:
    records = benchmark._materialize_records("selection", REPO_ROOT)
    canonical_digests = set(benchmark.canonical_observation_universe()["digest_to_rows"])
    splice_records = [record for record in records if record["family"] == "conflicting_action_splice"]

    assert len(splice_records) == 112
    assert all(record["observation_pixel_digest"] not in canonical_digests for record in splice_records)
    assert {benchmark.validate_materialized_family_record(record) for record in splice_records} == {"ok"}
    for record in splice_records:
        trace = record["metadata"]["family_intervention_trace"]
        counts = trace["action_relevant_region_contribution_counts"]
        assert trace["primary_contributing_pixel_count"] > 0
        assert trace["secondary_contributing_pixel_count"] > 0
        assert trace["changed_pixel_count"] > 0
        assert counts["primary_target_pixel_count"] > 0
        assert counts["secondary_additive_target_pixel_count"] > 0
        assert trace["canonical_collision_count"] == 0


def test_frame_invalid_validator_rejects_legacy_valid_state_collision() -> None:
    _identity, row_ids, row_actions = _identity_rows_actions()
    source_row_id = row_ids[0]
    secondary_row_id = _legacy_secondary_row_for_splice(row_ids, row_actions, source_row_id)
    output = _legacy_row_band_splice(source_row_id, secondary_row_id)
    record = {
        "frame_id": "legacy:collision:frame-00",
        "family": "conflicting_action_splice",
        "expected_disposition": "distinguishable_invalid_input",
        "observation_pixel_digest": benchmark._array_digest(output),
        "pixels": output,
        "metadata": {},
    }

    assert benchmark.validate_materialized_family_record(record) == "invalid_family_valid_state_collision"
    closure = benchmark._frame_invalid_closure_summary([record])
    assert closure["status"] == "failed"
    assert closure["totals"]["canonical_collision_count"] == 1
    assert closure["totals"]["valid_decode_count"] == 1


def test_conflicting_splice_rejects_same_action_and_missing_or_overlapping_target_evidence() -> None:
    _identity, row_ids, row_actions = _identity_rows_actions()
    plan = _first_splice_plan()
    source_row_id = str(plan["source_row_id"])
    mask = plan["family_intervention"]["splice_mask"]
    same_action_row = next(row_id for row_id in row_ids if row_id != source_row_id and row_actions[row_id] == row_actions[source_row_id])
    no_target_row = "tank=0|target=none|cooldown=0"
    visible_conflict_row = next(
        row_id
        for row_id in row_ids
        if benchmark.parse_state_row_id(row_id)[1] is not None and row_actions[row_id] != row_actions[no_target_row]
    )
    same_target_conflict_row = next(
        row_id
        for row_id in row_ids
        if row_id != "tank=0|target=0|cooldown=0"
        and benchmark.parse_state_row_id(row_id)[1] == 0
        and row_actions[row_id] != row_actions["tank=0|target=0|cooldown=0"]
    )

    with pytest.raises(VPMValidationError, match="different governed actions"):
        benchmark._apply_conflicting_splice(
            primary_pixels=benchmark._render_row_frame(source_row_id),
            secondary_pixels=benchmark._render_row_frame(same_action_row),
            primary_row_id=source_row_id,
            secondary_row_id=same_action_row,
            primary_action_id=row_actions[source_row_id],
            secondary_action_id=row_actions[same_action_row],
            mask_manifest=mask,
        )
    with pytest.raises(VPMValidationError, match="visible target evidence"):
        benchmark._apply_conflicting_splice(
            primary_pixels=benchmark._render_row_frame(no_target_row),
            secondary_pixels=benchmark._render_row_frame(visible_conflict_row),
            primary_row_id=no_target_row,
            secondary_row_id=visible_conflict_row,
            primary_action_id=row_actions[no_target_row],
            secondary_action_id=row_actions[visible_conflict_row],
            mask_manifest=mask,
        )
    with pytest.raises(VPMValidationError, match="distinct secondary target evidence"):
        benchmark._apply_conflicting_splice(
            primary_pixels=benchmark._render_row_frame("tank=0|target=0|cooldown=0"),
            secondary_pixels=benchmark._render_row_frame(same_target_conflict_row),
            primary_row_id="tank=0|target=0|cooldown=0",
            secondary_row_id=same_target_conflict_row,
            primary_action_id=row_actions["tank=0|target=0|cooldown=0"],
            secondary_action_id=row_actions[same_target_conflict_row],
            mask_manifest=mask,
        )


def test_changing_either_splice_source_changes_trace_identity() -> None:
    _identity, row_ids, row_actions = _identity_rows_actions()
    plan = _first_splice_plan()
    mask = plan["family_intervention"]["splice_mask"]
    primary = str(plan["source_row_id"])
    secondary = str(plan["secondary_row_id"])
    primary_target = benchmark.parse_state_row_id(primary)[1]
    secondary_target = benchmark.parse_state_row_id(secondary)[1]

    first_output, first_trace = benchmark._apply_conflicting_splice(
        primary_pixels=benchmark._render_row_frame(primary),
        secondary_pixels=benchmark._render_row_frame(secondary),
        primary_row_id=primary,
        secondary_row_id=secondary,
        primary_action_id=row_actions[primary],
        secondary_action_id=row_actions[secondary],
        mask_manifest=mask,
    )
    changed_secondary = next(
        row_id
        for row_id in row_ids
        if row_id not in {primary, secondary}
        and benchmark.parse_state_row_id(row_id)[1] not in {None, primary_target, secondary_target}
        and row_actions[row_id] != row_actions[primary]
    )
    second_output, second_trace = benchmark._apply_conflicting_splice(
        primary_pixels=benchmark._render_row_frame(primary),
        secondary_pixels=benchmark._render_row_frame(changed_secondary),
        primary_row_id=primary,
        secondary_row_id=changed_secondary,
        primary_action_id=row_actions[primary],
        secondary_action_id=row_actions[changed_secondary],
        mask_manifest=mask,
    )
    changed_primary = next(
        row_id
        for row_id in row_ids
        if row_id not in {primary, secondary}
        and benchmark.parse_state_row_id(row_id)[1] not in {None, secondary_target}
        and row_actions[row_id] != row_actions[secondary]
    )
    third_output, third_trace = benchmark._apply_conflicting_splice(
        primary_pixels=benchmark._render_row_frame(changed_primary),
        secondary_pixels=benchmark._render_row_frame(secondary),
        primary_row_id=changed_primary,
        secondary_row_id=secondary,
        primary_action_id=row_actions[changed_primary],
        secondary_action_id=row_actions[secondary],
        mask_manifest=mask,
    )

    assert benchmark._array_digest(first_output) != benchmark._array_digest(second_output)
    assert benchmark._array_digest(first_output) != benchmark._array_digest(third_output)
    assert first_trace["splice_trace_digest"] != second_trace["splice_trace_digest"]
    assert first_trace["splice_trace_digest"] != third_trace["splice_trace_digest"]
