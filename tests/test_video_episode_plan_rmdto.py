from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
import hashlib
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

import zeromodel.video_action_set_benchmark as benchmark
from zeromodel import build_runtime
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.canonical_json import (
    canonical_json_bytes,
    canonical_json_text,
    canonical_sha256,
)
from zeromodel.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    EPISODE_PLAN_VERSION,
    GENERATOR_VERSION,
    SEED_DERIVATION_VERSION,
)
from zeromodel.domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    CanonicalJsonDTO,
    EpisodePlanDTO,
    SealedSplitPlanDTO,
)
from zeromodel.domains.video_action_set.store import (
    EPISODE_PLAN_CONFLICT_MESSAGE,
    SEALED_SPLIT_PLAN_CONFLICT_MESSAGE,
    UNKNOWN_BENCHMARK_IDENTITY_MESSAGE,
)
from zeromodel.stores.video_action_set_memory import InMemoryVideoActionSetStore


REPO_ROOT = Path(__file__).resolve().parents[1]


def sample_identity(
    seed_material: str = "episode-plan-rmdto-seed",
) -> BenchmarkIdentityDTO:
    seed_digest = "sha256:" + hashlib.sha256(seed_material.encode("utf-8")).hexdigest()
    return BenchmarkIdentityDTO(
        contract_commit="aed523b04c258d7e28cd9466413b49fc817b4e35",
        seed_material=seed_material,
        seed_digest=seed_digest,
        policy_artifact_id="policy-artifact",
        parent_audit_sha="parent-audit",
        parent_v3_sha="parent-v3",
    )


def _seed_int_from_digest(digest: str) -> int:
    return int(digest.removeprefix("sha256:")[:16], 16)


def _seed_node(
    identity: BenchmarkIdentityDTO,
    *,
    split: str,
    ordinal: int,
    namespace: str,
    parents: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    payload = {
        "version": SEED_DERIVATION_VERSION,
        "root_seed_digest": identity.seed_digest,
        "split": split,
        "episode_ordinal": ordinal,
        "namespace": namespace,
        "parent_identities": [
            {"name": name, "identity": value} for name, value in parents
        ],
    }
    digest = canonical_sha256(payload)
    return payload | {
        "seed_digest": digest,
        "seed_int64": _seed_int_from_digest(digest),
    }


def _observation_provenance(split: str) -> dict[str, Any]:
    if split == "final":
        return {
            "materialization_status": "prospective_materialization_prohibited",
            "observation_payload_included": False,
            "provenance": "sealed_plan_only",
        }
    return {
        "materialization_status": "materialized",
        "observation_payload_included": True,
        "provenance": "in_memory_generation",
    }


def sample_plan_payload(
    identity: BenchmarkIdentityDTO | None = None,
    *,
    split: str = "final",
    ordinal: int = 0,
    family_label: str = "valid",
    family_ordinal: int = 0,
    frame_count: int = 2,
    source_row_id: str = "row:source",
    mutation_kind: str | None = None,
    secondary_row_id: str | None = None,
) -> dict[str, Any]:
    identity = identity or sample_identity()
    episode_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="concrete_episode_seed",
        parents=(
            ("benchmark_version", BENCHMARK_VERSION),
            ("generator_version", GENERATOR_VERSION),
            ("family_label", family_label),
            ("family_ordinal", str(family_ordinal)),
            ("source_row_id", source_row_id),
            ("secondary_row_id", secondary_row_id or "none"),
            ("mutation_kind", mutation_kind or "none"),
            ("frame_count", str(frame_count)),
        ),
    )
    contract = {"family_label": family_label, "family_version": "test-family/v1"}
    intervention = {
        "source_row_id": source_row_id,
        "secondary_row_id": secondary_row_id,
        "mutation_kind": mutation_kind,
    }
    intervention = intervention | {
        "intervention_digest": canonical_sha256(intervention)
    }
    episode_id_seed = _seed_node(
        identity,
        split=split,
        ordinal=ordinal,
        namespace="concrete_episode_id",
        parents=(
            ("concrete_episode_seed", episode_seed["seed_digest"]),
            ("family_contract", contract["family_version"]),
            ("family_intervention", intervention["intervention_digest"]),
        ),
    )
    frame_plans = [
        {
            "frame_index": index,
            "frame_seed_identity": _seed_node(
                identity,
                split=split,
                ordinal=ordinal,
                namespace="frame_identity",
                parents=(
                    ("concrete_episode_seed", episode_seed["seed_digest"]),
                    ("frame_index", str(index)),
                ),
            )["seed_digest"],
            "transformation_family": "exact",
        }
        for index in range(frame_count)
    ]
    plan = {
        "version": EPISODE_PLAN_VERSION,
        "seed_derivation_version": SEED_DERIVATION_VERSION,
        "episode_id": f"{split}:{family_label}:{episode_id_seed['seed_digest'][7:23]}",
        "split": split,
        "ordinal": ordinal,
        "family_label": family_label,
        "family_ordinal": family_ordinal,
        "episode_family": family_label,
        "episode_disposition": family_label,
        "denominator_class": family_label,
        "final_observation_provenance": _observation_provenance(split),
        "mutation_kind": mutation_kind,
        "source_row_id": source_row_id,
        "secondary_row_id": secondary_row_id,
        "family_contract": contract,
        "family_intervention": intervention,
        "derived_seed_identity": episode_seed["seed_digest"],
        "episode_seed": episode_seed["seed_int64"],
        "frame_count": frame_count,
        "seed_lineage": {
            "concrete_episode_seed": episode_seed,
            "concrete_episode_id": episode_id_seed,
        },
        "frame_plans": frame_plans,
    }
    return plan | {"plan_digest": canonical_sha256(plan)}


def plan_dto(**kwargs: Any) -> EpisodePlanDTO:
    return EpisodePlanDTO.from_dict(sample_plan_payload(**kwargs))


def _redigest(
    payload: dict[str, Any], digest_key: str = "plan_digest"
) -> dict[str, Any]:
    payload[digest_key] = canonical_sha256(
        {key: value for key, value in payload.items() if key != digest_key}
    )
    return payload


def test_canonical_json_contract() -> None:
    value = {"b": 1, "a": ("é", {"z": 2})}
    text = '{"a":["é",{"z":2}],"b":1}'

    assert canonical_json_text(value) == text
    assert canonical_json_bytes(value) == text.encode("utf-8")
    assert (
        canonical_sha256(value)
        == "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    )
    with pytest.raises(ValueError):
        canonical_json_text(float("nan"))
    with pytest.raises(VPMValidationError):
        CanonicalJsonDTO.from_value(object())

    dto = CanonicalJsonDTO.from_value({"items": []})
    first = dto.to_value()
    second = dto.to_value()
    assert first == second == {"items": []}
    assert first is not second
    first["items"].append("changed")  # type: ignore[index, union-attr]
    assert dto.to_value() == {"items": []}


def test_episode_plan_round_trip_hashing_and_nested_immutability() -> None:
    payload = sample_plan_payload()
    dto = EpisodePlanDTO.from_dict(payload)

    assert dto.to_dict() == payload
    assert dto == EpisodePlanDTO.from_dict(payload)
    assert hash(dto) == hash(EpisodePlanDTO.from_dict(payload))
    with pytest.raises(FrozenInstanceError):
        dto.episode_id = "changed"  # type: ignore[misc]
    assert isinstance(dto.frame_plans, tuple)
    returned = dto.to_dict()
    returned["family_contract"]["family_label"] = "changed"  # type: ignore[index]
    returned["frame_plans"][0]["frame_seed_identity"] = "changed"  # type: ignore[index]
    assert dto.to_dict() == payload
    assert dto.benchmark_seed_digest == sample_identity().seed_digest


def test_episode_plan_rejects_digest_and_structural_tampering() -> None:
    payload = sample_plan_payload()
    scalar = deepcopy(payload)
    scalar["denominator_class"] = "changed"
    with pytest.raises(VPMValidationError, match="episode plan digest mismatch"):
        EpisodePlanDTO.from_dict(scalar)

    frame = deepcopy(payload)
    frame["frame_plans"][0]["frame_seed_identity"] = "changed"
    with pytest.raises(VPMValidationError, match="episode plan digest mismatch"):
        EpisodePlanDTO.from_dict(frame)

    count = deepcopy(payload)
    count["frame_count"] = 99
    with pytest.raises(VPMValidationError, match="episode plan frame count mismatch"):
        EpisodePlanDTO.from_dict(count)

    indexes = deepcopy(payload)
    indexes["frame_plans"][1]["frame_index"] = 3
    with pytest.raises(VPMValidationError, match="frame indexes"):
        EpisodePlanDTO.from_dict(indexes)

    lineage = deepcopy(payload)
    lineage["seed_lineage"]["concrete_episode_seed"]["root_seed_digest"] = (
        "sha256:" + "0" * 64
    )
    with pytest.raises(VPMValidationError, match="root seed lineage mismatch"):
        EpisodePlanDTO.from_dict(lineage)

    split = deepcopy(payload)
    split["episode_id"] = split["episode_id"].replace("final:", "selection:", 1)
    with pytest.raises(VPMValidationError, match="episode id does not match split"):
        EpisodePlanDTO.from_dict(split)


def test_sealed_split_plan_construction_validation_and_round_trip() -> None:
    identity = sample_identity()
    episodes = (
        plan_dto(
            identity=identity,
            ordinal=2,
            family_label="temporal_negative",
            family_ordinal=0,
        ),
        plan_dto(identity=identity, ordinal=0, family_label="valid", family_ordinal=0),
        plan_dto(
            identity=identity, ordinal=1, family_label="frame_invalid", family_ordinal=0
        ),
        plan_dto(
            identity=identity,
            ordinal=3,
            family_label="information_control",
            family_ordinal=0,
        ),
    )

    sealed = SealedSplitPlanDTO.build_final(
        episodes=episodes,
        seed_commitment=identity.seed_digest,
    )
    payload = sealed.to_dict()
    without_digest = dict(payload)
    digest = without_digest.pop("sealed_plan_digest")

    assert sealed.episodes == episodes
    assert sealed.episode_counts.to_dict() == {
        "valid": 1,
        "frame_invalid": 1,
        "temporal_negative": 1,
        "information_control": 1,
    }
    assert sealed.sealed_episode_ids.to_dict() == {
        "valid": [episodes[1].episode_id],
        "frame_invalid": [episodes[2].episode_id],
        "temporal_negative": [episodes[0].episode_id],
        "information_control": [episodes[3].episode_id],
    }
    assert sealed.frame_count == sum(episode.frame_count for episode in episodes)
    assert digest == canonical_sha256(without_digest)
    assert SealedSplitPlanDTO.from_dict(payload) == sealed

    with pytest.raises(VPMValidationError, match="duplicate episode ids"):
        SealedSplitPlanDTO.build_final(
            episodes=(episodes[0], episodes[0]), seed_commitment=identity.seed_digest
        )
    with pytest.raises(VPMValidationError, match="non-final episode"):
        SealedSplitPlanDTO.build_final(
            episodes=(plan_dto(identity=identity, split="selection"),),
            seed_commitment=identity.seed_digest,
        )
    with pytest.raises(VPMValidationError, match="seed commitment mismatch"):
        SealedSplitPlanDTO.build_final(
            episodes=episodes, seed_commitment="sha256:" + "0" * 64
        )

    tampered = dict(payload)
    tampered["sealed_plan_digest"] = "sha256:" + "0" * 64
    with pytest.raises(VPMValidationError, match="sealed plan digest mismatch"):
        SealedSplitPlanDTO.from_dict(tampered)


def test_in_memory_store_plan_and_sealed_semantics() -> None:
    identity = sample_identity()
    other_identity = sample_identity("other-episode-plan-rmdto-seed")
    store = InMemoryVideoActionSetStore()
    plan = plan_dto(identity=identity, split="selection", ordinal=1)

    with pytest.raises(VPMValidationError, match=UNKNOWN_BENCHMARK_IDENTITY_MESSAGE):
        store.save_episode_plan(plan)
    store.save_identity(identity)
    store.save_identity(other_identity)
    assert store.save_episode_plan(plan) == plan
    assert store.save_episode_plan(plan) == plan
    assert store.get_episode_plan(plan.episode_id) == plan

    conflict_payload = deepcopy(plan.to_dict())
    conflict_payload["source_row_id"] = "row:changed"
    conflict = EpisodePlanDTO.from_dict(_redigest(conflict_payload))
    with pytest.raises(VPMValidationError, match=EPISODE_PLAN_CONFLICT_MESSAGE):
        store.save_episode_plan(conflict)

    new_plan = plan_dto(
        identity=identity, split="selection", ordinal=2, source_row_id="row:new"
    )
    with pytest.raises(VPMValidationError, match=EPISODE_PLAN_CONFLICT_MESSAGE):
        store.save_episode_plans((new_plan, conflict))
    assert store.get_episode_plan(new_plan.episode_id) is None

    first = plan_dto(identity=identity, split="development", ordinal=0)
    second = plan_dto(
        identity=identity, split="development", ordinal=2, source_row_id="row:two"
    )
    third = plan_dto(
        identity=identity, split="development", ordinal=1, source_row_id="row:one"
    )
    other_seed = plan_dto(identity=other_identity, split="development", ordinal=0)
    store.save_episode_plans((second, third, first, other_seed))
    assert store.list_episode_plans(
        benchmark_seed_digest=identity.seed_digest,
        split="development",
    ) == (first, third, second)
    assert store.list_episode_plans(
        benchmark_seed_digest=other_identity.seed_digest,
        split="development",
    ) == (other_seed,)

    final_episodes = (
        plan_dto(identity=identity, ordinal=0, family_label="valid", family_ordinal=0),
        plan_dto(
            identity=identity, ordinal=1, family_label="frame_invalid", family_ordinal=0
        ),
    )
    sealed = SealedSplitPlanDTO.build_final(
        episodes=final_episodes,
        seed_commitment=identity.seed_digest,
    )
    assert store.save_sealed_split_plan(sealed) == sealed
    assert (
        store.get_sealed_split_plan(seed_commitment=identity.seed_digest, split="final")
        == sealed
    )
    assert isinstance(
        store.get_episode_plan(final_episodes[0].episode_id), EpisodePlanDTO
    )
    conflicting_sealed = SealedSplitPlanDTO.build_final(
        episodes=(
            plan_dto(
                identity=identity,
                ordinal=2,
                family_label="information_control",
                family_ordinal=0,
            ),
        ),
        seed_commitment=identity.seed_digest,
    )
    with pytest.raises(VPMValidationError, match=SEALED_SPLIT_PLAN_CONFLICT_MESSAGE):
        store.save_sealed_split_plan(conflicting_sealed)


def test_runtime_facade_delegates_with_one_shared_store() -> None:
    identity = sample_identity()
    store = InMemoryVideoActionSetStore()
    runtime = build_runtime(video_action_set_store=store)
    plan = plan_dto(identity=identity)

    assert runtime.video_action_set.engine.identity_service.store is store
    assert runtime.video_action_set.engine.episode_plan_service.store is store
    with pytest.raises(VPMValidationError, match=UNKNOWN_BENCHMARK_IDENTITY_MESSAGE):
        runtime.video_action_set.save_episode_plan(plan)

    runtime.video_action_set.save_identity(identity)
    assert runtime.video_action_set.save_episode_plan(plan) == plan
    assert runtime.video_action_set.get_episode_plan(plan.episode_id) == plan
    sealed = runtime.video_action_set.seal_final_split(
        episodes=(plan,),
        seed_commitment=identity.seed_digest,
    )
    assert (
        runtime.video_action_set.get_sealed_split_plan(
            seed_commitment=identity.seed_digest,
            split="final",
        )
        == sealed
    )


def test_core_runtime_import_remains_sqlalchemy_free() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import zeromodel; print('sqlalchemy' in sys.modules)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.strip() == "False"


def test_legacy_private_generator_returns_episode_plan_dto_payload() -> None:
    identity = benchmark.load_identity(REPO_ROOT)
    policy = benchmark.compile_policy_artifact()
    lookup = benchmark.VPMPolicyLookup(policy, action_metric_ids=benchmark.ACTIONS)
    row_ids = [str(row_id) for row_id in policy.source.row_ids]
    row_actions = {row_id: lookup.choose(row_id) for row_id in row_ids}

    legacy_plan = benchmark._make_episode_plan(
        identity,
        split="development",
        ordinal=0,
        family_label="valid",
        family_ordinal=0,
        source_row_id=row_ids[0],
        row_actions=row_actions,
    )

    assert EpisodePlanDTO.from_dict(legacy_plan).to_dict() == legacy_plan
