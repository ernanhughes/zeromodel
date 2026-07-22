from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess
import sys

import pytest

import research.benchmarks.video_action_set_benchmark as benchmark
from zeromodel.video.runtime import build_runtime
from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.contracts import (
    BENCHMARK_VERSION,
    GENERATOR_VERSION,
    REACHABILITY_TILE_DIGEST,
    REACHABILITY_TILE_VERSION,
)
from zeromodel.video.domains.video_action_set.dto import BenchmarkIdentityDTO
from zeromodel.video.domains.video_action_set.store import (
    BENCHMARK_IDENTITY_CONFLICT_MESSAGE,
)
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore


REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_MATERIAL = (
    "zeromodel-action-set-reachability-v1|aed523b04c258d7e28cd9466413b49fc817b4e35"
)
SEED_DIGEST = "sha256:22f0b8b706198c4d00df0f8e1d6e09dd324aefdd6ac1dc0768fb9b24a8b519c9"


def sample_identity() -> BenchmarkIdentityDTO:
    return BenchmarkIdentityDTO(
        contract_commit="aed523b04c258d7e28cd9466413b49fc817b4e35",
        seed_material=SEED_MATERIAL,
        seed_digest=SEED_DIGEST,
        policy_artifact_id=(
            "eb7523f406b45ac30b478fe9528db8f89a548693b0add2fc8d3e51c4badd857e"
        ),
        parent_audit_sha="e4c3f894e47e070318edc046171233cbc862aa11",
        parent_v3_sha="4790165de78557fce63d64e5f2b7ddfde04f1e98",
    )


def test_benchmark_identity_dto_is_frozen_hashable_and_serializes_exactly() -> None:
    identity = sample_identity()
    same_identity = sample_identity()

    assert identity == same_identity
    assert hash(identity) == hash(same_identity)
    with pytest.raises(FrozenInstanceError):
        identity.contract_commit = "changed"  # type: ignore[misc]

    serialized = identity.to_dict()
    assert list(serialized) == [
        "benchmark_version",
        "generator_version",
        "contract_commit",
        "seed_material",
        "seed_digest",
        "policy_artifact_id",
        "parent_audit_sha",
        "parent_v3_sha",
        "reachability_tile_version",
        "reachability_tile_digest",
    ]
    assert serialized == {
        "benchmark_version": BENCHMARK_VERSION,
        "generator_version": GENERATOR_VERSION,
        "contract_commit": identity.contract_commit,
        "seed_material": identity.seed_material,
        "seed_digest": identity.seed_digest,
        "policy_artifact_id": identity.policy_artifact_id,
        "parent_audit_sha": identity.parent_audit_sha,
        "parent_v3_sha": identity.parent_v3_sha,
        "reachability_tile_version": REACHABILITY_TILE_VERSION,
        "reachability_tile_digest": REACHABILITY_TILE_DIGEST,
    }


def test_benchmark_identity_rejects_invalid_seed_digest() -> None:
    with pytest.raises(
        VPMValidationError,
        match="benchmark seed digest is inconsistent with frozen seed material",
    ):
        BenchmarkIdentityDTO(
            contract_commit="aed523b04c258d7e28cd9466413b49fc817b4e35",
            seed_material=SEED_MATERIAL,
            seed_digest="sha256:bad",
            policy_artifact_id="policy",
            parent_audit_sha="audit",
            parent_v3_sha="v3",
        )


def test_legacy_benchmark_identity_alias_still_constructs_dto() -> None:
    identity = sample_identity()

    assert benchmark.BenchmarkIdentity is BenchmarkIdentityDTO
    assert (
        benchmark.BenchmarkIdentity(
            contract_commit=identity.contract_commit,
            seed_material=identity.seed_material,
            seed_digest=identity.seed_digest,
            policy_artifact_id=identity.policy_artifact_id,
            parent_audit_sha=identity.parent_audit_sha,
            parent_v3_sha=identity.parent_v3_sha,
        )
        == identity
    )


def test_in_memory_store_save_retrieve_idempotence_and_conflict() -> None:
    store = InMemoryVideoActionSetStore()
    identity = sample_identity()
    conflicting = replace(identity, contract_commit="different")

    assert store.get_identity(identity.seed_digest) is None
    assert store.save_identity(identity) == identity
    assert store.get_identity(identity.seed_digest) == identity
    assert store.save_identity(identity) == identity
    with pytest.raises(VPMValidationError, match=BENCHMARK_IDENTITY_CONFLICT_MESSAGE):
        store.save_identity(conflicting)

    retrieved = store.get_identity(identity.seed_digest)
    assert isinstance(retrieved, BenchmarkIdentityDTO)


def test_runtime_facade_loads_and_saves_identity_with_supplied_store() -> None:
    store = InMemoryVideoActionSetStore()
    runtime = build_runtime(video_action_set_store=store)

    loaded = runtime.video_action_set.load_identity(REPO_ROOT)

    assert loaded == sample_identity()
    assert runtime.video_action_set.get_identity(loaded.seed_digest) == loaded
    assert store.get_identity(loaded.seed_digest) == loaded


def test_runtime_uses_in_memory_store_by_default() -> None:
    runtime = build_runtime()
    identity = sample_identity()

    assert isinstance(
        runtime.video_action_set.engine.identity_service.store,
        InMemoryVideoActionSetStore,
    )
    assert runtime.video_action_set.save_identity(identity) == identity
    assert runtime.video_action_set.get_identity(identity.seed_digest) == identity


def test_legacy_load_identity_matches_runtime_result() -> None:
    runtime_identity = build_runtime().video_action_set.load_identity(REPO_ROOT)

    assert benchmark.load_identity(REPO_ROOT) == runtime_identity


@pytest.mark.integration
def test_core_import_does_not_import_sqlalchemy() -> None:
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
