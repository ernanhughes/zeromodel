from __future__ import annotations

from pathlib import Path
from typing import Any

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.domains.video_action_set import family_intervention_planning


REPO_ROOT = Path(__file__).resolve().parents[1]


def _identity_rows_actions() -> tuple[
    benchmark.BenchmarkIdentity, list[str], dict[str, str]
]:
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    return identity, row_ids, {row_id: lookup.choose(row_id) for row_id in row_ids}


def _contains_materialized_payload(value: object) -> bool:
    if isinstance(value, dict):
        if any(
            key in {"pixels", "ImageObservation", "score_vector", "candidate_set"}
            for key in value
        ):
            return True
        return any(_contains_materialized_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_materialized_payload(item) for item in value)
    return False


def _plan_kwargs(
    case: dict[str, Any],
    identity: benchmark.BenchmarkIdentity,
    row_ids: list[str],
    row_actions: dict[str, str],
) -> dict[str, Any]:
    return {
        "identity": identity,
        "split": "selection",
        "ordinal": int(case["ordinal"]),
        "family_label": str(case["family_label"]),
        "mutation_kind": case["mutation_kind"],
        "source_row_id": str(case["source_row_id"]),
        "secondary_row_id": case["secondary_row_id"],
        "row_ids": row_ids,
        "row_actions": row_actions,
    }


FAMILY_INTERVENTION_CASES = [
    {
        "name": "valid",
        "ordinal": 0,
        "family_label": "valid",
        "mutation_kind": None,
        "source_row_id": "tank=0|target=none|cooldown=0",
        "secondary_row_id": None,
        "intervention_digest": "sha256:c5af54ecfb6d942c82cb83c89b8db94b600850d060ae8878ef8256cc2b3cfc41",
        "sequence_digest": "sha256:938d8433571fe92ccb667207fe13eb990bb3eba922c9543d2ef0bb1b35e9256f",
        "materialized_order": [0, 1, 2, 3],
        "event_type": "frame_sequence",
    },
    {
        "name": "conflicting_action_splice",
        "ordinal": 112,
        "family_label": "frame_invalid",
        "mutation_kind": "conflicting_action_splice",
        "source_row_id": "tank=0|target=0|cooldown=0",
        "secondary_row_id": "tank=0|target=1|cooldown=0",
        "intervention_digest": "sha256:8a23f959070e66db5e40f12da0fede5a513b0ab60f1237b4e81e529a40b532f0",
        "sequence_digest": "sha256:bdb97c58424512d5cb44d47ccf87dc455728770a3a19ddb949a18bc64839a32b",
        "materialized_order": [0, 1, 2, 3],
        "event_type": "frame_sequence",
    },
    {
        "name": "critical_evidence_corruption",
        "ordinal": 140,
        "family_label": "frame_invalid",
        "mutation_kind": "critical_evidence_corruption",
        "source_row_id": "tank=1|target=5|cooldown=0",
        "secondary_row_id": None,
        "intervention_digest": "sha256:d41753a5d1fa3de145e286f10d53fcb81d29e8f90db82fed731fca7127912310",
        "sequence_digest": "sha256:ee4cc62bcaa4cdde07213db011fabff0e857f400b2b3097e5aa2715c55b4d101",
        "materialized_order": [0, 1, 2, 3],
        "event_type": "frame_sequence",
    },
    {
        "name": "reordered_frames",
        "ordinal": 168,
        "family_label": "temporal_negative",
        "mutation_kind": "reordered_frames",
        "source_row_id": "tank=0|target=none|cooldown=0",
        "secondary_row_id": None,
        "intervention_digest": "sha256:a7c5072a4a62197f2b3352a187dd454197b4f5ad34f237016259ecc5d4ca0239",
        "sequence_digest": "sha256:774355a41324457e1975a623871c6a43f6998e04653e586dfe0f9e6defa4f0ef",
        "materialized_order": [1, 0, 2, 3],
        "event_type": "frame_sequence",
    },
    {
        "name": "stale_repeated_frame",
        "ordinal": 182,
        "family_label": "temporal_negative",
        "mutation_kind": "stale_repeated_frame",
        "source_row_id": "tank=0|target=6|cooldown=0",
        "secondary_row_id": None,
        "intervention_digest": "sha256:731ed9485303856dccbb13d00044c3eb613940f5172eb2f7eaf1610bd27f7dea",
        "sequence_digest": "sha256:de8f59042fbd337f202d527ff5923ebb11f1c383191ddc31fbe67d64f3886d99",
        "materialized_order": [0, 1, 2, 3],
        "event_type": "frame_sequence",
    },
    {
        "name": "impossible_transition",
        "ordinal": 196,
        "family_label": "temporal_negative",
        "mutation_kind": "impossible_transition",
        "source_row_id": "tank=1|target=5|cooldown=0",
        "secondary_row_id": None,
        "intervention_digest": "sha256:f8705eb29cd67fa700710146c7541bd336ca046217b8cc25277f366ee0c08bfd",
        "sequence_digest": "sha256:a743029be192be7fbf6a6c3cea11baeac406fe3f8c5a9da27ae498b2826810bd",
        "materialized_order": [0, 1, 2, 3],
        "event_type": "frame_sequence",
    },
    {
        "name": "declared_gap_or_unknown_action",
        "ordinal": 210,
        "family_label": "temporal_negative",
        "mutation_kind": "declared_gap_or_unknown_action",
        "source_row_id": "tank=2|target=4|cooldown=0",
        "secondary_row_id": None,
        "intervention_digest": "sha256:e8658e3bc5330011237e6919339c8fe5a628d611ead510a5020cd91cd68582a6",
        "sequence_digest": "sha256:8aae9a1bd2378fb8a7dc8746ebf55cb905e34fd387ae023e7c44a4a993bffc9b",
        "materialized_order": [0, 1, 2, 3],
        "event_type": "typed_gap_sequence",
    },
    {
        "name": "information_control",
        "ordinal": 224,
        "family_label": "information_control",
        "mutation_kind": None,
        "source_row_id": "tank=0|target=none|cooldown=0",
        "secondary_row_id": None,
        "intervention_digest": "sha256:a41720bc236bd63fc29898878fb0a9d6d0779e7dffc0bcb9f1e6f01995f3e975",
        "sequence_digest": "sha256:5c9d7285bb886a55f94161de7cbc46ba98f8242948904a5331fa9e2266ff100b",
        "materialized_order": [0, 1, 2, 3],
        "event_type": "frame_sequence",
    },
]


def test_source_row_selection_helpers_are_frozen() -> None:
    _identity, row_ids, row_actions = _identity_rows_actions()

    assert family_intervention_planning.state_row_values(
        "tank=0|target=none|cooldown=0"
    ) == (0, None, 0)
    assert family_intervention_planning.conflicting_splice_source_rows(
        row_ids, row_actions, 5
    ) == [
        "tank=0|target=0|cooldown=0",
        "tank=0|target=0|cooldown=1",
        "tank=0|target=1|cooldown=0",
        "tank=0|target=1|cooldown=1",
        "tank=0|target=2|cooldown=0",
    ]
    assert (
        family_intervention_planning.secondary_row_for_splice(
            row_ids,
            row_actions,
            "tank=0|target=0|cooldown=0",
        )
        == "tank=0|target=1|cooldown=0"
    )
    assert (
        family_intervention_planning.impossible_destination_row(
            row_ids,
            row_actions,
            "tank=0|target=none|cooldown=0",
        )
        == "tank=6|target=6|cooldown=1"
    )


def test_family_intervention_plans_match_pre_extraction_goldens() -> None:
    identity, row_ids, row_actions = _identity_rows_actions()
    for case in FAMILY_INTERVENTION_CASES:
        plan = family_intervention_planning.family_intervention_plan(
            **_plan_kwargs(case, identity, row_ids, row_actions)
        )
        assert plan["intervention_digest"] == case["intervention_digest"]
        assert plan["sequence_digest"] == case["sequence_digest"]
        assert plan["materialized_order"] == case["materialized_order"]
        assert plan["event_type"] == case["event_type"]
        assert not _contains_materialized_payload(plan)
        if case["name"] == "information_control":
            control_group = plan["control_group"]
            assert control_group["control_group_id"] == (
                "sha256:5a343fdfa3da7f0fbd0e5a1b24492411181e883633e230e0afd79d3d05e88874"
            )
            assert control_group["hidden_source_label_digest"] == (
                "sha256:33c4730b9378b58b562e9c2ce58ac5dcd6843072729d8f89917a5b859f13d260"
            )
            assert [
                history["history_id"]
                for history in control_group["hidden_source_histories"]
            ] == [
                "sha256:c03a8f6e753cdecb1e9f94c8c50db5131726d08a0f6d54616f90e42fc2a1734e",
                "sha256:e35c5faeab0bfff99af9b629ff473149507b7f73c6137b41bc479379cac877a3",
                "sha256:0bbfa71389d9c6b97646c62edb0effce87f2229ce5cbde902cdaa6b685cd4bb0",
                "sha256:405d893d7e9456f025c932ffb468148babbc95b34d6d16b1680bb88514a6f28b",
            ]


def test_benchmark_family_intervention_aliases_are_direct() -> None:
    assert (
        benchmark._seed_int_from_digest
        is family_intervention_planning.seed_int_from_digest
    )
    assert benchmark._state_row_values is family_intervention_planning.state_row_values
    assert (
        benchmark._secondary_row_for_splice
        is family_intervention_planning.secondary_row_for_splice
    )
    assert (
        benchmark._conflicting_splice_source_rows
        is family_intervention_planning.conflicting_splice_source_rows
    )
    assert (
        benchmark._impossible_destination_row
        is family_intervention_planning.impossible_destination_row
    )
    assert (
        benchmark._family_intervention_plan
        is family_intervention_planning.family_intervention_plan
    )
