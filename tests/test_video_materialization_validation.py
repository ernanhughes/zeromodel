from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set import episode_materialization
from zeromodel.domains.video_action_set import materialization_validation as validation
from zeromodel.domains.video_action_set.canonical_json import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]


def _identity_and_plans():
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    return identity, benchmark._episode_plans_for_split(
        identity, "selection", row_ids, row_actions
    )


def _plans_by_key(plans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for plan in plans:
        family = str(plan["family_label"])
        if family in {"frame_invalid", "temporal_negative"}:
            family = f"{family}:{plan['mutation_kind']}"
        by_key.setdefault(family, plan)
    return by_key


def _control_and_valid_records() -> tuple[
    object,
    dict[str, dict[str, Any]],
    object,
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    identity, plans = _identity_and_plans()
    by_key = _plans_by_key(plans)
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    control_records = episode_materialization.materialize_plan(
        by_key["information_control"], identity, tile
    )
    valid_records = episode_materialization.materialize_plan(
        by_key["valid"], identity, tile
    )
    return identity, by_key, tile, control_records, valid_records


def test_control_validation_status_precedence_is_frozen() -> None:
    _identity, _by_key, _tile, records, valid_records = _control_and_valid_records()

    assert validation.validate_control_episode_records(records) == "ok"
    assert validation.validate_control_episode_records(valid_records) == (
        "no_control_records"
    )

    tampered = deepcopy(records)
    tampered[0]["observation_pixel_digest"] = "sha256:" + "0" * 64
    assert validation.validate_control_episode_records(tampered) == (
        "control_byte_identity_mismatch"
    )

    tampered = deepcopy(records)
    tampered[0]["metadata"]["denominator_eligible"] = True
    assert validation.validate_control_episode_records(tampered) == (
        "control_denominator_leak"
    )

    tampered = deepcopy(records)
    tampered[0]["episode_disposition"] = "valid"
    tampered[0]["metadata"]["denominator_eligible"] = True
    assert validation.validate_control_episode_records(tampered) == (
        "control_denominator_leak"
    )

    tampered = deepcopy(records)
    tampered[0]["episode_disposition"] = "valid"
    assert validation.validate_control_episode_records(tampered) == (
        "control_disposition_mismatch"
    )

    tampered = deepcopy(records)
    for record in tampered:
        record["metadata"]["hidden_source_history_id"] = "same"
    assert validation.validate_control_episode_records(tampered) == (
        "control_hidden_history_not_ambiguous"
    )

    tampered = deepcopy(records)
    for record in tampered:
        record["metadata"]["hidden_source_label_digest"] = "same"
    assert validation.validate_control_episode_records(tampered) == (
        "control_hidden_label_not_ambiguous"
    )

    tampered = deepcopy(records)
    tampered[0]["metadata"]["control_group_id"] = "different"
    assert validation.validate_control_episode_records(tampered) == (
        "control_group_mismatch"
    )

    tampered = deepcopy(records)
    tampered[0]["metadata"]["control_current_row_id"] = "tank=1|target=none|cooldown=0"
    assert validation.validate_control_episode_records(tampered) == (
        "control_current_state_mismatch"
    )

    tampered = deepcopy(records)
    tampered[0]["metadata"]["provider_visible_fields"] = ["pixels"]
    assert validation.validate_control_episode_records(tampered) == (
        "control_provider_visible_leak"
    )

    tampered = deepcopy(records)
    tampered[0]["metadata"]["provider_visible_fields"] = ["pixels"]
    tampered[0]["metadata"]["grounded_causal_history"]["resulting_row_id"] = (
        "tank=1|target=none|cooldown=0"
    )
    assert validation.validate_control_episode_records(tampered) == (
        "control_provider_visible_leak"
    )

    tampered = deepcopy(records)
    tampered[0]["metadata"]["grounded_causal_history"]["resulting_row_id"] = (
        "tank=1|target=none|cooldown=0"
    )
    assert validation.validate_control_episode_records(tampered) == (
        "control_history_result_mismatch"
    )

    tampered = deepcopy(records)
    tampered[0]["metadata"]["grounded_causal_history"]["version"] = "unsupported"
    assert validation.validate_control_episode_records(tampered) == (
        "control_history_transition_invalid"
    )

    tampered = deepcopy(records)
    duplicate_history = deepcopy(tampered[0]["metadata"]["grounded_causal_history"])
    for record in tampered:
        record["metadata"]["grounded_causal_history"] = duplicate_history
        record["metadata"]["hidden_source_history"] = duplicate_history
    assert validation.validate_control_episode_records(tampered) == (
        "control_ambiguity_absent"
    )


def test_family_closure_report_subset_payload_is_frozen() -> None:
    identity, plans = _identity_and_plans()
    by_key = _plans_by_key(plans)
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    selected = [
        by_key["valid"],
        by_key["frame_invalid:conflicting_action_splice"],
        by_key["frame_invalid:critical_evidence_corruption"],
        by_key["temporal_negative:impossible_transition"],
        by_key["information_control"],
    ]
    records = []
    for plan in selected:
        records.extend(episode_materialization.materialize_plan(plan, identity, tile))

    closure = validation.family_closure_report(
        split="selection",
        records=records,
        plans=selected,
        identity=identity,
        reachability_tile=tile,
        provider_rows=[],
    )

    assert canonical_sha256(closure) == (
        "sha256:f6a77c084e82a0e4797b76255dbf0a6d88b72ce3d5d5db7e01e7f43ae47b0c6f"
    )
    planned = [row for row in closure["families"] if row["planned_episode_count"]]
    assert [row["family_id"] for row in planned] == [
        "valid",
        "conflicting_action_splice",
        "critical_evidence_corruption",
        "impossible_transition",
        "information_control",
    ]
    assert {row["closure_status"] for row in planned} == {"closed"}
    assert closure["frame_invalid_closure"]["totals"] == {
        "frame_count": 8,
        "malformed_count": 0,
        "no_op_count": 0,
        "canonical_collision_count": 0,
        "valid_decode_count": 0,
    }
    assert closure["negative_families_verified"] is False
    assert closure["reference_instrument_correct"] is False
    assert closure["materialization_status"] == "prospective_materialization_prohibited"


def test_materialization_validation_aliases_are_direct() -> None:
    assert benchmark._record_regeneration_view is validation.record_regeneration_view
    assert benchmark._family_closure_report is validation.family_closure_report
    assert benchmark.validate_control_episode_records is (
        validation.validate_control_episode_records
    )


def test_family_closure_exception_boundary_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, plans = _identity_and_plans()
    by_key = _plans_by_key(plans)
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    plan = by_key["valid"]

    def raise_validation_error(*_args: object, **_kwargs: object) -> None:
        raise VPMValidationError("synthetic regeneration failure")

    monkeypatch.setattr(validation, "materialize_plan", raise_validation_error)
    closure = validation.family_closure_report(
        split="selection",
        records=[],
        plans=[plan],
        identity=identity,
        reachability_tile=tile,
        provider_rows=[],
    )
    valid_row = next(row for row in closure["families"] if row["family_id"] == "valid")
    assert valid_row["malformed_count"] == 1
    assert valid_row["closure_status"] == "unresolved"

    def raise_key_error(*_args: object, **_kwargs: object) -> None:
        raise KeyError("synthetic structural failure")

    monkeypatch.setattr(validation, "materialize_plan", raise_key_error)
    with pytest.raises(KeyError, match="synthetic structural failure"):
        validation.family_closure_report(
            split="selection",
            records=[],
            plans=[plan],
            identity=identity,
            reachability_tile=tile,
            provider_rows=[],
        )
