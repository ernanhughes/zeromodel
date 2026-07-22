from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    OBSERVATION_OPERATION_CHAIN_VERSION,
)
from zeromodel.video.domains.video_action_set.provider_observation_dto import (
    ProviderObservationDescriptorDTO,
)
from research.evidence.video_complete_row_evidence import (
    build_complete_row_evidence,
    build_semantic_top_set_outcome,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _fake_operation(final_digest: str | None) -> dict[str, object]:
    parameters = {"event_type": "gap_unknown" if final_digest is None else "frame"}
    payload = {
        "index": 0,
        "operation": "emit_observation",
        "operation_version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "input_digests": [final_digest],
        "parameters": parameters,
        "parameter_digest": canonical_sha256(parameters),
        "output_digest": final_digest,
    }
    return payload | {"operation_digest": canonical_sha256(payload)}


def _fake_operation_chain(final_digest: str) -> dict[str, object]:
    chain_payload = {
        "version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "operations": [_fake_operation(final_digest)],
        "final_emitted_digest": final_digest,
    }
    return chain_payload | {"operation_chain_digest": canonical_sha256(chain_payload)}


# array_digest() of a FRAME_SHAPE=(16, 28) all-zero np.uint8 frame (the shape
# required by validate_observation_matrix_blob) - this fixture's fake frame's
# pixel content/digest pair.
_FAKE_PIXELS = [[0] * 28 for _ in range(16)]
_FAKE_PIXEL_DIGEST = (
    "sha256:5c55c8f4db4010ba9203d83536d0609856af8c847ac039e37e7dde8fbd574b61"
)
# _provider_raw_digest()'s domain-separated hash of the same all-zero
# (16, 28) frame - a different digest scheme than the plain pixel digest
# above, used specifically for ProviderObservationDescriptorDTO.raw_digest.
_FAKE_PROVIDER_RAW_DIGEST = (
    "sha256:cdc08e35906d759b232d985e4f5a3a46ae4aed501f6165273a682e5b4af5c13a"
)


def _row_actions() -> tuple[list[str], dict[str, str]]:
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    return row_ids, {row_id: lookup.choose(row_id) for row_id in row_ids}


def _fake_provider_output() -> list[dict[str, object]]:
    row_ids, row_actions = _row_actions()
    evidence = build_complete_row_evidence(
        row_scores=[(row_id, 1.0) for row_id in row_ids],
        policy_artifact_id=benchmark.compile_policy_artifact().artifact_id,
        provider_id="P1",
        provider_version=benchmark.PROSPECTIVE_P1_VERSION,
        policy_row_ids=row_ids,
    )
    outcome = build_semantic_top_set_outcome(evidence=evidence, row_action=row_actions)
    return [
        {
            "provider_id": "P1",
            "provider_version": benchmark.PROSPECTIVE_P1_VERSION,
            "policy_artifact_id": evidence.policy_artifact_id,
            "all_112_row_ids": row_ids,
            "all_112_raw_scores": [1.0] * 112,
            "all_112_quantized_scores": [1_000_000] * 112,
            "complete_ordered_ranking": list(evidence.ranking.ranked_row_ids),
            "tie_groups": [group.to_dict() for group in evidence.ranking.tie_groups],
            "semantic_top_set_outcome": outcome.to_dict(),
            "semantic_status": outcome.status,
            "resolved_row": outcome.resolved_row_id,
            "resolved_action": outcome.resolved_action_id,
            "top_quantized_score": outcome.top_quantized_score,
            "top_row_ids": list(outcome.top_row_ids),
            "top_action_ids": list(outcome.top_action_ids),
            "semantic_outcome_digest": outcome.semantic_outcome_digest,
            "winner_row": outcome.resolved_row_id,
            "winner_action": outcome.resolved_action_id,
            "winner_quantized_score": None,
            "runner_up_row": evidence.ranking.ranked_row_ids[1],
            "runner_up_quantized_score": 1_000_000,
            "score_vector_digest": evidence.score_vector_digest,
            "ranking_digest": evidence.ranking.to_dict()["ranking_digest"],
            "provider_diagnostics": {},
        }
    ]


def _contains_forbidden_materialized_payload(value: object) -> bool:
    if isinstance(value, dict):
        if any(
            key in {"pixels", "ImageObservation", "score_vector", "candidate_set"}
            for key in value
        ):
            return True
        return any(
            _contains_forbidden_materialized_payload(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_materialized_payload(item) for item in value)
    return False


def test_materialized_split_counts_and_final_freeze(tmp_path: Path) -> None:
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    development = benchmark._materialize_records("development", REPO_ROOT)
    calibration = benchmark._materialize_records("calibration", REPO_ROOT)
    selection = benchmark._materialize_records("selection", REPO_ROOT)
    assert len(development) == 112
    assert len(calibration) == 448
    assert len(selection) == 1008
    split_manifest = json.loads(
        (tmp_path / "split-manifest.json").read_text(encoding="utf-8")
    )
    assert split_manifest["calibration_episode_count"] == 112
    assert split_manifest["selection_valid_episode_count"] == 112
    phase_access = json.loads(
        (tmp_path / "phase-access-audits.json").read_text(encoding="utf-8")
    )
    assert phase_access["final_materialization_count"] == 0
    assert phase_access["final_score_access_count"] == 0


def _fake_provider_descriptor(source_id: str) -> dict[str, object]:
    return {
        "version": "zeromodel-image-observation/v1",
        "raw_digest": _FAKE_PROVIDER_RAW_DIGEST,
        "shape": [16, 28],
        "timestamp": None,
        "source_id": source_id,
        "metadata": {},
    }


def _fake_record(split: str) -> dict[str, object]:
    # ObservationDTO._validate_ids() requires this exact (if unusual) doubled
    # id scheme: episode_id already carries the split prefix, and clip_id /
    # frame_id are then built as f"{split}:{episode_id}:...".
    episode_id = f"{split}:episode-001"
    frame_id = f"{split}:{episode_id}:frame-0"
    descriptor_payload = _fake_provider_descriptor(frame_id)
    descriptor = ProviderObservationDescriptorDTO.from_dict(descriptor_payload)
    return {
        "benchmark_version": benchmark.BENCHMARK_VERSION,
        "generator_version": benchmark.GENERATOR_VERSION,
        "split": split,
        "episode_id": episode_id,
        "clip_id": f"{split}:{episode_id}:clip",
        "frame_id": frame_id,
        "sequence_number": 0,
        "event_type": "frame",
        "family": "exact",
        "expected_disposition": "valid",
        "episode_family": "valid",
        "episode_disposition": "valid",
        "frame_disposition": "valid",
        "denominator_class": "valid_denominator",
        "expected_row": "row-000",
        "expected_action": "left",
        "actual_executed_action": "left",
        "action_known": True,
        "gap_declaration": None,
        "observation_pixel_digest": _FAKE_PIXEL_DIGEST,
        "metadata": {
            "episode_seed": 1,
            "seed_digest": "sha256:" + "1" * 64,
            "episode_plan_digest": "sha256:" + "2" * 64,
            "reachability_trace": {"reachable_row_ids": ["row-000"]},
            "observation_operation_chain": _fake_operation_chain(_FAKE_PIXEL_DIGEST),
            "provider_observation_descriptor": descriptor_payload,
            "provider_observation_digest": descriptor.descriptor_digest,
        },
        "pixels": _FAKE_PIXELS,
    }


def test_build_split_writes_overlap_and_observation_manifests(tmp_path: Path) -> None:
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    fake_output = _fake_provider_output()
    monkeypatch = pytest.MonkeyPatch()
    fake_records = [_fake_record("development")]
    fake_records_calibration = [_fake_record("calibration")]
    fake_records_selection = [_fake_record("selection")]
    monkeypatch.setattr(
        benchmark,
        "_materialize_records",
        lambda split, repo_root: {
            "development": fake_records,
            "calibration": fake_records_calibration,
            "selection": fake_records_selection,
        }[split],
    )
    monkeypatch.setattr(
        benchmark,
        "measure_record_collection",
        lambda records, prototypes, policy_artifact_id, **_kwargs: fake_output,
    )
    monkeypatch.setattr(benchmark, "canonical_prototypes", lambda: {})
    benchmark.build_split("development", tmp_path, REPO_ROOT)
    benchmark.build_split("calibration", tmp_path, REPO_ROOT)
    benchmark.build_split("selection", tmp_path, REPO_ROOT)
    monkeypatch.undo()
    overlap = json.loads(
        (tmp_path / "split-overlap-audit.json").read_text(encoding="utf-8")
    )
    assert overlap["development_calibration_overlap"] == 0
    assert overlap["development_selection_overlap"] == 0
    assert overlap["calibration_selection_overlap"] == 0
    observation_manifest = json.loads(
        (tmp_path / "observation-identity-manifest.json").read_text(encoding="utf-8")
    )
    assert observation_manifest["development_observation_count"] == 1
    assert observation_manifest["calibration_observation_count"] == 1
    assert observation_manifest["selection_observation_count"] == 1


def test_seed_material_determines_episode_identity_and_trace(tmp_path: Path) -> None:
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    first = benchmark._materialize_records("development", REPO_ROOT)
    second = benchmark._materialize_records("development", REPO_ROOT)
    assert [row["episode_id"] for row in first] == [row["episode_id"] for row in second]
    assert [row["observation_pixel_digest"] for row in first] == [
        row["observation_pixel_digest"] for row in second
    ]
    assert (
        first[0]["metadata"]["reachability_trace"]["executed_action"]
        == first[0]["expected_action"]
    )
    assert first[0]["metadata"]["derived_seed_identity"].startswith("sha256:")
    assert first[0]["metadata"]["episode_plan_digest"].startswith("sha256:")


def test_changing_root_seed_material_changes_sealed_episode_identities() -> None:
    row_ids, row_actions = _row_actions()
    identity = benchmark.load_identity(REPO_ROOT)
    changed_material = identity.seed_material + "|changed"
    changed = benchmark.BenchmarkIdentity(
        contract_commit=identity.contract_commit,
        seed_material=changed_material,
        seed_digest="sha256:"
        + hashlib.sha256(changed_material.encode("utf-8")).hexdigest(),
        policy_artifact_id=identity.policy_artifact_id,
        parent_audit_sha=identity.parent_audit_sha,
        parent_v3_sha=identity.parent_v3_sha,
    )
    original = benchmark._episode_plans_for_split(
        identity, "selection", row_ids, row_actions
    )
    changed_plans = benchmark._episode_plans_for_split(
        changed, "selection", row_ids, row_actions
    )
    assert [plan["episode_id"] for plan in original] != [
        plan["episode_id"] for plan in changed_plans
    ]


def test_changing_derived_seed_without_declared_parent_is_rejected() -> None:
    row_ids, row_actions = _row_actions()
    identity = benchmark.load_identity(REPO_ROOT)
    plan = copy.deepcopy(
        benchmark._episode_plans_for_split(identity, "selection", row_ids, row_actions)[
            0
        ]
    )
    plan["seed_lineage"]["concrete_episode_seed"]["seed_digest"] = "sha256:" + "0" * 64
    with pytest.raises(VPMValidationError):
        benchmark._validate_episode_plan(identity, plan, row_actions)


def test_duplicate_episode_ids_are_rejected() -> None:
    row_ids, row_actions = _row_actions()
    identity = benchmark.load_identity(REPO_ROOT)
    plan = benchmark._episode_plans_for_split(
        identity, "development", row_ids, row_actions
    )[0]
    with pytest.raises(VPMValidationError):
        benchmark._validate_episode_plan_collection(
            identity, {"development": [plan, plan]}, row_actions
        )


def test_moving_episode_id_between_splits_is_rejected() -> None:
    row_ids, row_actions = _row_actions()
    identity = benchmark.load_identity(REPO_ROOT)
    development = benchmark._episode_plans_for_split(
        identity, "development", row_ids, row_actions
    )[0]
    calibration = copy.deepcopy(
        benchmark._episode_plans_for_split(
            identity, "calibration", row_ids, row_actions
        )[0]
    )
    calibration["episode_id"] = development["episode_id"]
    with pytest.raises(VPMValidationError):
        benchmark._validate_episode_plan_collection(
            identity,
            {"development": [development], "calibration": [calibration]},
            row_actions,
        )


def test_measured_phase_access_and_final_plan_remain_zero_access(
    tmp_path: Path,
) -> None:
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    phase_access = json.loads(
        (tmp_path / "phase-access-audits.json").read_text(encoding="utf-8")
    )
    sealed = json.loads(
        (tmp_path / "final-split-sealed-plan.json").read_text(encoding="utf-8")
    )
    assert phase_access["forbidden_final_access_counter"] == 0
    assert phase_access["final_materialization_count"] == 0
    assert phase_access["final_score_access_count"] == 0
    assert sealed["plan_only"] is True
    assert sealed["materialization_prohibited"] is True
    assert len(sealed["sealed_episode_ids"]["valid"]) == 112
    assert len(sealed["episodes"]) == 252
    assert not _contains_forbidden_materialized_payload(sealed)


def test_final_plan_creation_does_not_render_score_or_access_final_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(
            "final plan creation must not materialize observations or scores"
        )

    monkeypatch.setattr(benchmark, "render_state_frame", fail)
    monkeypatch.setattr(benchmark, "score_normalized_pixel", fail)
    monkeypatch.setattr(benchmark, "score_registered_local_correlation", fail)
    monkeypatch.setattr(benchmark, "score_b3_joint_fit", fail)
    benchmark.freeze_benchmark(tmp_path, REPO_ROOT)
    phase_access = json.loads(
        (tmp_path / "phase-access-audits.json").read_text(encoding="utf-8")
    )
    assert phase_access["forbidden_final_access_counter"] == 0
