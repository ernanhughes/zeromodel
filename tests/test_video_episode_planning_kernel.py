from __future__ import annotations

from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set import (
    control_histories,
    episode_planning,
    family_intervention_planning,
)
from zeromodel.domains.video_action_set.canonical_json import canonical_sha256


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_SPLIT_SUMMARIES = {
    "development": {
        "count": 112,
        "first_episode_id": "development:valid:8d2e4b45539bc773",
        "last_episode_id": "development:valid:da2f93a150a2e789",
        "frame_count_total": 112,
        "plan_digest_rollup": "sha256:000106d8842e90698d44d07534b878cf8c580e4b80b92f2b692d57e7cb78c291",
        "episode_id_rollup": "sha256:55e1eb64193f199a6d2ea7fc01fac9dfc2b9f50781528f61e589a10c020ef61c",
        "family_counts": {"valid": 112},
        "mutation_counts": {},
    },
    "calibration": {
        "count": 112,
        "first_episode_id": "calibration:valid:171e7c1e602df153",
        "last_episode_id": "calibration:valid:0eaf29b67f6c98d3",
        "frame_count_total": 448,
        "plan_digest_rollup": "sha256:3844bff2c0eac4cfc476444177e36ad6b2b8c4800bf2f0e1716fe2484ce72716",
        "episode_id_rollup": "sha256:8b36e4badfa6d4a37d77df3b18b85fb982b433d4e82ff495b1dc7d6e6c5127b6",
        "family_counts": {"valid": 112},
        "mutation_counts": {},
    },
    "selection": {
        "count": 252,
        "first_episode_id": "selection:valid:3b017d7e360a1086",
        "last_episode_id": "selection:information_control:c63e511d702c29a4",
        "frame_count_total": 1008,
        "plan_digest_rollup": "sha256:3294f053295a6d5c634f78065fa410211296b9c0d14fdf204e6bf90b462f6612",
        "episode_id_rollup": "sha256:33d0a1122332ad29679019914e3e6be03b52f2757021eac6f26a169eec3819dc",
        "family_counts": {"frame_invalid": 56, "information_control": 28, "temporal_negative": 56, "valid": 112},
        "mutation_counts": {
            "conflicting_action_splice": 28,
            "critical_evidence_corruption": 28,
            "declared_gap_or_unknown_action": 14,
            "impossible_transition": 14,
            "reordered_frames": 14,
            "stale_repeated_frame": 14,
        },
    },
    "final": {
        "count": 252,
        "first_episode_id": "final:valid:1058e7c00930ff46",
        "last_episode_id": "final:information_control:d26f251078213b36",
        "frame_count_total": 1008,
        "plan_digest_rollup": "sha256:7ada2002b9fb8be46f910ae79337b5fdda603e3413eaa3087b436354ca9ebde0",
        "episode_id_rollup": "sha256:7f95640e9ea81edcff80c8252b7fd52960b849d9e9825b197fdbc38d31b7aa22",
        "family_counts": {"frame_invalid": 56, "information_control": 28, "temporal_negative": 56, "valid": 112},
        "mutation_counts": {
            "conflicting_action_splice": 28,
            "critical_evidence_corruption": 28,
            "declared_gap_or_unknown_action": 14,
            "impossible_transition": 14,
            "reordered_frames": 14,
            "stale_repeated_frame": 14,
        },
    },
}


def _identity_rows_actions() -> tuple[benchmark.BenchmarkIdentity, list[str], dict[str, str]]:
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    return identity, row_ids, {row_id: lookup.choose(row_id) for row_id in row_ids}


def _summary(plans: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "count": len(plans),
        "first_episode_id": plans[0]["episode_id"],
        "last_episode_id": plans[-1]["episode_id"],
        "family_counts": dict(sorted(Counter(str(plan["family_label"]) for plan in plans).items())),
        "mutation_counts": dict(sorted(Counter(str(plan["mutation_kind"]) for plan in plans if plan.get("mutation_kind") is not None).items())),
        "frame_count_total": sum(int(plan["frame_count"]) for plan in plans),
        "plan_digest_rollup": canonical_sha256([plan["plan_digest"] for plan in plans]),
        "episode_id_rollup": canonical_sha256([plan["episode_id"] for plan in plans]),
    }


def _contains_materialized_payload(value: object) -> bool:
    if isinstance(value, dict):
        if any(key in {"pixels", "ImageObservation", "score_vector", "candidate_set"} for key in value):
            return True
        return any(_contains_materialized_payload(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_materialized_payload(item) for item in value)
    return False


def test_derived_seed_and_representative_episode_plan_are_frozen() -> None:
    identity, row_ids, row_actions = _identity_rows_actions()
    seed = episode_planning.derived_seed(
        identity,
        split="selection",
        ordinal=7,
        namespace="stage5_probe",
        parent_identities=(("alpha", "one"), ("beta", "two")),
    )
    plan = episode_planning.make_episode_plan(
        identity,
        split="selection",
        ordinal=0,
        family_label="valid",
        family_ordinal=0,
        source_row_id=row_ids[0],
        row_actions=row_actions,
    )

    assert seed["seed_digest"] == "sha256:4d09890ab6c6cec5916e4d3a8872fd08016dcf7eebce4586f7965db8ce419a40"
    assert seed["seed_int64"] == 5551118694820007621
    assert plan["episode_id"] == "selection:valid:3b017d7e360a1086"
    assert plan["plan_digest"] == "sha256:711d180c93547b5f6f1489cccaf5648ffb8ab7af3e0d55c8cc9cd8416d11ff0e"
    assert [frame["transformation_family"] for frame in plan["frame_plans"]] == [
        "bounded_translation_photometric",
        "bounded_translation",
        "bounded_photometric",
        "bounded_photometric",
    ]
    assert [frame["transformation_parameter_digest"] for frame in plan["frame_plans"]] == [
        "sha256:fd10a0e5b42d0abfed681a2b8a97dbd11a20c64e26c01291ebc96110f25a5a1f",
        "sha256:0fe587bbfb1d6b93055576c4a5f896c804d7d6a51e43166fcdb9f500b628ea3c",
        "sha256:8725d2ee9a282d2c5c5a0da96f62e2c1ece3295f51e4a2968e0e7eaa865b9b28",
        "sha256:870e36708b0c8cb4cc3538f552de3bab2765e82d42e0c7ff017ee42031d5587c",
    ]


def test_split_episode_plan_collections_match_pre_extraction_goldens(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("episode planning must not materialize pixels or provider scores")

    monkeypatch.setattr(benchmark, "_render_row_frame", fail)
    monkeypatch.setattr(benchmark, "render_state_frame", fail)
    monkeypatch.setattr(benchmark, "score_normalized_pixel", fail)
    monkeypatch.setattr(benchmark, "score_registered_local_correlation", fail)
    monkeypatch.setattr(benchmark, "score_b3_joint_fit", fail)

    identity, row_ids, row_actions = _identity_rows_actions()
    plans_by_split = {
        split: episode_planning.episode_plans_for_split(identity, split, row_ids, row_actions)
        for split in ("development", "calibration", "selection", "final")
    }

    assert {split: _summary(plans) for split, plans in plans_by_split.items()} == EXPECTED_SPLIT_SUMMARIES
    assert canonical_sha256({split: [plan["plan_digest"] for plan in plans] for split, plans in plans_by_split.items()}) == (
        "sha256:a56aef694069a5bc46639834e6d323866a35b701e1085c743b4b080918a0bced"
    )
    assert all(not _contains_materialized_payload(plan) for plans in plans_by_split.values() for plan in plans)
    assert all(plan["final_observation_provenance"]["observation_payload_included"] is False for plan in plans_by_split["final"])


def test_episode_plan_validation_rejects_lineage_and_identity_tampering() -> None:
    identity, row_ids, row_actions = _identity_rows_actions()
    plan = episode_planning.episode_plans_for_split(identity, "selection", row_ids, row_actions)[0]
    tampered = deepcopy(plan)
    tampered["seed_lineage"]["concrete_episode_seed"]["seed_digest"] = "sha256:" + "0" * 64

    with pytest.raises(VPMValidationError, match="seed lineage or identity"):
        episode_planning.validate_episode_plan(identity, tampered, row_actions)
    with pytest.raises(VPMValidationError, match="duplicate concrete episode identity"):
        episode_planning.validate_episode_plan_collection(identity, {"selection": [plan, plan]}, row_actions)


def test_benchmark_episode_planning_aliases_are_direct() -> None:
    assert benchmark._derived_seed is episode_planning.derived_seed
    assert benchmark._final_observation_provenance is episode_planning.final_observation_provenance
    assert benchmark._frame_count_for_plan is episode_planning.frame_count_for_plan
    assert benchmark._frame_plans is episode_planning.frame_plans
    assert benchmark._make_episode_plan is episode_planning.make_episode_plan
    assert benchmark._episode_plans_for_split is episode_planning.episode_plans_for_split
    assert benchmark._episode_ids_by_family is episode_planning.episode_ids_by_family
    assert benchmark._validate_episode_plan is episode_planning.validate_episode_plan
    assert benchmark._validate_episode_plan_collection is episode_planning.validate_episode_plan_collection
    assert episode_planning.derived_seed is family_intervention_planning.derived_seed
    assert benchmark._family_intervention_plan is family_intervention_planning.family_intervention_plan
    assert benchmark._grounded_control_history is control_histories.grounded_control_history
