from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set import provider_observation_boundary as boundary


REPO_ROOT = Path(__file__).resolve().parents[1]


def _identity_rows_actions():
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    return identity, row_ids, {row_id: lookup.choose(row_id) for row_id in row_ids}


def _selection_plan(family: str, mutation_kind: str | None = None):
    identity, row_ids, row_actions = _identity_rows_actions()
    plans = benchmark._episode_plans_for_split(
        identity, "selection", row_ids, row_actions
    )
    for plan in plans:
        if (
            plan["family_label"] == family
            and plan.get("mutation_kind") == mutation_kind
        ):
            return identity, plan
    raise AssertionError("representative plan not found")


def test_provider_observation_descriptor_and_digest_are_frozen() -> None:
    identity, plan = _selection_plan("valid")
    record = benchmark._materialize_plan(
        plan, identity, benchmark._load_reachability_tile(REPO_ROOT)
    )[0]

    descriptor = boundary.provider_observation_descriptor_for_record(record)

    assert descriptor == {
        "version": "zeromodel-image-observation/v1",
        "raw_digest": "sha256:a7390930d37f6da91984f1d11937cfbe8c0f37a69a3a9141643fd521fc0c8d44",
        "shape": [16, 28],
        "timestamp": None,
        "source_id": "selection:selection:valid:3b017d7e360a1086:frame-00",
        "metadata": {},
    }
    assert record["observation_pixel_digest"] == (
        "sha256:424d87c22e58789875eb09d1c430d6e1cb7a45e1de1cd6f32b19400cad0bb8da"
    )
    assert boundary.provider_observation_digest(descriptor) == (
        "sha256:bb6de887ea70ccd7c410547b46967c331b19186365a4d0eac3accf7240926e7b"
    )
    assert (
        boundary.provider_observation_descriptor_for_record(
            {key: value for key, value in record.items() if key != "pixels"}
        )
        == descriptor
    )


def test_information_control_provider_boundary_is_group_visible_only() -> None:
    identity, plan = _selection_plan("information_control")
    records = benchmark._materialize_plan(
        plan, identity, benchmark._load_reachability_tile(REPO_ROOT)
    )
    descriptors = [
        boundary.provider_observation_descriptor_for_record(r) for r in records
    ]

    assert (
        len({boundary.provider_observation_digest(item) for item in descriptors}) == 1
    )
    assert descriptors[0] == {
        "version": "zeromodel-image-observation/v1",
        "raw_digest": "sha256:3e3dc2d6661c710405e00ee6448c3966fa27a875ab7430d00f6bad2ab358d668",
        "shape": [16, 28],
        "timestamp": None,
        "source_id": "control:sha256:5a343fdfa3da7f0fbd0e5a1b24492411181e883633e230e0afd79d3d05e88874",
        "metadata": {},
    }
    assert (
        boundary.control_provider_source_id(records[0]) == descriptors[0]["source_id"]
    )


def test_provider_boundary_aliases_are_direct() -> None:
    assert (
        benchmark._provider_observation_digest is boundary.provider_observation_digest
    )
    assert benchmark._control_provider_source_id is boundary.control_provider_source_id
    assert (
        benchmark.provider_observation_for_record
        is boundary.provider_observation_for_record
    )
    assert (
        benchmark.provider_observation_descriptor_for_record
        is boundary.provider_observation_descriptor_for_record
    )
    assert (
        benchmark._refresh_provider_observation_metadata
        is boundary.refresh_provider_observation_metadata
    )

    with pytest.raises(VPMValidationError, match="provider observation requires"):
        boundary.provider_observation_for_record({"pixels": None})


def test_stored_provider_descriptor_requires_raw_digest_and_shape() -> None:
    with pytest.raises(VPMValidationError, match="raw digest or shape"):
        boundary.provider_observation_descriptor_for_record(
            {"metadata": {"provider_observation_descriptor": {"shape": [16, 28]}}}
        )
    with pytest.raises(VPMValidationError, match="raw digest or shape"):
        boundary.provider_observation_descriptor_for_record(
            {"metadata": {"provider_observation_raw_digest": "sha256:" + "1" * 64}}
        )


def test_control_provider_source_id_requires_control_group() -> None:
    with pytest.raises(VPMValidationError, match="lacks a control group id"):
        boundary.control_provider_source_id(
            {
                "expected_disposition": "information_theoretic_control",
                "metadata": {},
            }
        )


def test_refresh_provider_metadata_replaces_stale_values_and_clears_gap_values() -> (
    None
):
    identity, valid_plan = _selection_plan("valid")
    tile = benchmark._load_reachability_tile(REPO_ROOT)
    ordinary = deepcopy(benchmark._materialize_plan(valid_plan, identity, tile)[0])
    ordinary["metadata"]["provider_observation_boundary_version"] = "stale"
    ordinary["metadata"]["provider_observation_descriptor"] = {"stale": True}
    ordinary["metadata"]["provider_observation_digest"] = "sha256:" + "0" * 64

    boundary.refresh_provider_observation_metadata(ordinary)

    assert ordinary["metadata"]["provider_observation_boundary_version"] == (
        benchmark.PROVIDER_OBSERVATION_BOUNDARY_VERSION
    )
    descriptor = boundary.provider_observation_descriptor_for_record(ordinary)
    assert ordinary["metadata"]["provider_observation_descriptor"] == descriptor
    assert ordinary["metadata"]["provider_observation_digest"] == (
        boundary.provider_observation_digest(descriptor)
    )

    identity, gap_plan = _selection_plan(
        "temporal_negative", "declared_gap_or_unknown_action"
    )
    gap = deepcopy(benchmark._materialize_plan(gap_plan, identity, tile)[2])
    gap["metadata"]["provider_observation_boundary_version"] = "stale"
    gap["metadata"]["provider_observation_descriptor"] = {"stale": True}
    gap["metadata"]["provider_observation_digest"] = "sha256:" + "0" * 64

    boundary.refresh_provider_observation_metadata(gap)

    assert "provider_observation_boundary_version" not in gap["metadata"]
    assert "provider_observation_descriptor" not in gap["metadata"]
    assert "provider_observation_digest" not in gap["metadata"]
