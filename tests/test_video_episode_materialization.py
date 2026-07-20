from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set import episode_materialization
from zeromodel.domains.video_action_set import materialization_reachability
from zeromodel.domains.video_action_set import materialization_validation
from zeromodel.domains.video_action_set.canonical_json import (
    canonical_json_value,
    canonical_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_MATERIALIZED = {
    "valid": (
        "sha256:424d87c22e58789875eb09d1c430d6e1cb7a45e1de1cd6f32b19400cad0bb8da",
        "sha256:ae9800725f9a0f82b280b4cb79e6481909e031e65386d78e00959f15b9398a3c",
        "sha256:dc41d560acdddbd51413181324bc2a64eaad6c32429de5d132a7f20acfbe37a1",
        "sha256:c12f15527729c69e1d01df5c5bfb7fde5947ca83a0170bc5981acc8119da9f21",
    ),
    "frame_invalid:conflicting_action_splice": (
        "sha256:ca73312136f3875a09eba69d6372425ade15dcb2bbb31abeee8da75aeeef33cc",
        "sha256:ca73312136f3875a09eba69d6372425ade15dcb2bbb31abeee8da75aeeef33cc",
        "sha256:a6e6c19033d8ec0125fbc275ad8298c640bc2cf8a2328f17ff7ff39ffebc21ca",
        "sha256:370628cd1f309e881128aa142f1d6b5639af80670506579a51258811486306f4",
    ),
    "frame_invalid:critical_evidence_corruption": (
        "sha256:9ad0537398389dadcea4e4bfc6e147926446b22542c92a19bca2fa6b4b9bfc44",
        "sha256:9ad0537398389dadcea4e4bfc6e147926446b22542c92a19bca2fa6b4b9bfc44",
        "sha256:0e992f54742340c61285fe5dcdde4a187834b1372448285dfdbe115eb57a00ee",
        "sha256:8d4e438a47eb9dab63ae3953f132b4eec49a54764a29d69d92065763073076c2",
    ),
    "temporal_negative:reordered_frames": (
        "sha256:c126740a7c5411eacb512e4dec17e667821e6eb0180162aba8964daeb9cf7fe8",
        "sha256:a51fb5c64565f09d4212d9ce1431c71525d2856cd255f65fd580aa12c9bf4b96",
        "sha256:1c1296c4c0dbbe54fd65aac402a3947c28c987eaeb757655c1a56456a3630669",
        "sha256:334623b7574604b559e5451db70340679c5ab434eeebdb4da67159855dc7cc51",
    ),
    "temporal_negative:stale_repeated_frame": (
        "sha256:08fb7c9b0be264fcdfd3ee894e1b8fa163f90450d6d0e929ee4cfe9711a78772",
        "sha256:2ae8954b8933a7deb15b59b677a05114580f191a77e382dfbde206212649a9b3",
        "sha256:3835ec0a0be2d15d79a4d4dd08a772df662d6c8ffa8ff46d1382d829f27051e2",
        "sha256:258531207d9dfbecded8bdad99011194b48b69024509a0ee7b221ad4d2178e78",
    ),
    "temporal_negative:impossible_transition": (
        "sha256:01a7f7cb13a855a5464081d9b1be53288f32de6f361b20fda158effd891e7990",
        "sha256:ea7a30f669ac3953774f7236d273c9592e962cc0c54c3d70d81f111a73fd3a50",
        "sha256:ff3f1a3cfcf535b303ce8587161354228547fab6ebca355b8b29108f31c9cd51",
        "sha256:3cd4920a1a1509051d3f3e5b1164f1e571f22bcaf4f71f7d50b021f5028b8946",
    ),
    "temporal_negative:declared_gap_or_unknown_action": (
        "sha256:3f65aff391c1005b33a239cee77abd99908867dab6aa9ab1fcc13520a5a65a04",
        "sha256:abfe325e5f7cdb9a7a00fd373204da30b15fe8cc159e8453524fcca5a9b15464",
        "sha256:e55f98827c31ecfcb29c17888ad2a1088412b29f1a4651e7f598a7ff0158f0e3",
        "sha256:a318eb8253230cf70ab3a6dd630ff6b592eb63792df20dc71f79b7138e8c5a06",
    ),
    "information_control": (
        "sha256:76d6b55ab164806fa6d621e669de37b60b9cbd8f8fff2bbb610b6b0e034461f0",
        "sha256:76d6b55ab164806fa6d621e669de37b60b9cbd8f8fff2bbb610b6b0e034461f0",
        "sha256:5589cab359fe7a99af9e103e287baf29780c7d72422460ede51728e8ab3f7782",
        "sha256:d66a5e82668852fdf9cfba0458f85a1d67eb11b67952c7c7a347394ce2d1d3ad",
    ),
}


def _identity_and_plans(split: str = "selection"):
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}
    return identity, benchmark._episode_plans_for_split(
        identity, split, row_ids, row_actions
    )


def _plans_by_key(plans: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_key: dict[str, dict[str, Any]] = {}
    for plan in plans:
        family = str(plan["family_label"])
        if family in {"frame_invalid", "temporal_negative"}:
            family = f"{family}:{plan['mutation_kind']}"
        by_key.setdefault(family, plan)
    return by_key


def _strip_pixels(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: canonical_json_value(value)
        for key, value in record.items()
        if key != "pixels"
    }


def _projection_digest(records: list[dict[str, Any]]) -> str:
    return canonical_sha256(
        [
            materialization_validation.record_regeneration_view(record)
            for record in records
        ]
    )


def test_representative_episode_materialization_outputs_are_frozen() -> None:
    identity, plans = _identity_and_plans()
    by_key = _plans_by_key(plans)
    tile = benchmark._load_reachability_tile(REPO_ROOT)

    for key, expected in EXPECTED_MATERIALIZED.items():
        records = episode_materialization.materialize_plan(by_key[key], identity, tile)
        first_digest, last_digest, record_digest, projection_digest = expected
        assert len(records) == 4
        assert records[0]["observation_pixel_digest"] == first_digest
        assert records[-1]["observation_pixel_digest"] == last_digest
        assert canonical_sha256([_strip_pixels(record) for record in records]) == (
            record_digest
        )
        assert _projection_digest(records) == projection_digest


def test_materialize_plan_collection_preserves_plan_and_frame_order() -> None:
    identity, plans = _identity_and_plans()
    by_key = _plans_by_key(plans)
    selected = [
        by_key["valid"],
        by_key["temporal_negative:declared_gap_or_unknown_action"],
        by_key["information_control"],
    ]

    records = episode_materialization.materialize_plan_collection(
        selected, identity, benchmark._load_reachability_tile(REPO_ROOT)
    )

    assert len(records) == 12
    assert [record["episode_id"] for record in records[0:4]] == [
        selected[0]["episode_id"]
    ] * 4
    assert records[6]["event_type"] == "gap_unknown"
    assert [record["episode_id"] for record in records[8:12]] == [
        selected[2]["episode_id"]
    ] * 4


def test_low_level_materialize_plan_still_executes_representative_final_plan() -> None:
    identity, plans = _identity_and_plans("final")

    records = episode_materialization.materialize_plan(
        plans[0], identity, benchmark._load_reachability_tile(REPO_ROOT)
    )

    assert plans[0]["episode_id"] == "final:valid:1058e7c00930ff46"
    assert [record["observation_pixel_digest"] for record in records] == [
        "sha256:40d63f5b3d9405d92e98066b55ce998f418072771033f6e04604b8b92a21ff96",
        "sha256:1edf9e025b5386ac4716d1db7079b4152bd2fc5f876f6ea740da7fce7e674793",
        "sha256:8e3d0060f2df4026acecdeb79b4b9ca34a016cf4ccc033f80d9bd93987d13b4c",
        "sha256:76d6b55ab164806fa6d621e669de37b60b9cbd8f8fff2bbb610b6b0e034461f0",
    ]


def test_frame_invalid_splice_requires_secondary_frame() -> None:
    identity, plans = _identity_and_plans()
    plan = deepcopy(_plans_by_key(plans)["frame_invalid:conflicting_action_splice"])
    plan["secondary_row_id"] = None

    with pytest.raises(VPMValidationError, match="missing secondary frame"):
        episode_materialization.materialize_plan(
            plan, identity, benchmark._load_reachability_tile(REPO_ROOT)
        )


def test_temporal_negative_error_order_is_frozen(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity, plans = _identity_and_plans()
    by_key = _plans_by_key(plans)
    tile = benchmark._load_reachability_tile(REPO_ROOT)

    reordered = deepcopy(by_key["temporal_negative:reordered_frames"])
    reordered["family_intervention"]["materialized_order"] = list(
        reordered["family_intervention"]["original_order"]
    )
    with pytest.raises(VPMValidationError, match="non-identity order"):
        episode_materialization.materialize_plan(reordered, identity, tile)

    bad_permutation = deepcopy(by_key["temporal_negative:reordered_frames"])
    bad_permutation["family_intervention"]["materialized_order"] = [0, 0, 1, 2]
    with pytest.raises(VPMValidationError, match="complete permutation"):
        episode_materialization.materialize_plan(bad_permutation, identity, tile)

    stale_no_op = deepcopy(by_key["temporal_negative:stale_repeated_frame"])
    repeat = stale_no_op["family_intervention"]["stale_repeat"]
    repeat["source_frame_index"] = repeat["destination_frame_index"]
    with pytest.raises(VPMValidationError, match="actual payload replacement"):
        episode_materialization.materialize_plan(stale_no_op, identity, tile)

    reachable_impossible = deepcopy(by_key["temporal_negative:impossible_transition"])
    transition = reachable_impossible["family_intervention"]["impossible_transition"]
    edge = materialization_reachability.tile_edge(
        tile, str(transition["source_row_id"]), str(transition["source_action_id"])
    )
    transition["destination_row_id"] = edge["reachable_row_ids"][0]
    with pytest.raises(VPMValidationError, match="destination is reachable"):
        episode_materialization.materialize_plan(reachable_impossible, identity, tile)

    fake_frame: dict[str, Any] = {"metadata": {}}
    called = {"valid": False}

    def fake_valid_episode(**_kwargs: object) -> list[dict[str, Any]]:
        called["valid"] = True
        return [fake_frame]

    monkeypatch.setattr(episode_materialization, "valid_episode", fake_valid_episode)
    unsupported = deepcopy(by_key["temporal_negative:reordered_frames"])
    unsupported["mutation_kind"] = "unsupported_temporal_kind"
    with pytest.raises(VPMValidationError, match="unsupported temporal-negative kind"):
        episode_materialization.temporal_negative_episode(
            plan=unsupported,
            identity=identity,
            reachability_tile=tile,
        )
    assert called["valid"] is True
    assert fake_frame["episode_family"] == "temporal_negative"
    assert (
        fake_frame["metadata"]["sequence_digest"]
        == (unsupported["family_intervention"]["sequence_digest"])
    )


def test_episode_materialization_aliases_are_direct_and_records_wrapper_stays_local() -> (
    None
):
    assert benchmark._valid_episode is episode_materialization.valid_episode
    assert benchmark._invalid_episode is episode_materialization.invalid_episode
    assert (
        benchmark._temporal_negative_episode
        is episode_materialization.temporal_negative_episode
    )
    assert benchmark._control_episode is episode_materialization.control_episode
    assert benchmark._materialize_plan is episode_materialization.materialize_plan
    assert benchmark._materialize_records is not (
        episode_materialization.materialize_plan_collection
    )

    with pytest.raises(VPMValidationError, match="final split materialization"):
        benchmark._materialize_records("final", REPO_ROOT)
