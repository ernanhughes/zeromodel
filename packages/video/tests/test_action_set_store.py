from __future__ import annotations

import hashlib

import numpy as np
import pytest

from zeromodel.core.artifact import VPMValidationError
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.video import InMemoryVideoActionSetStore, build_runtime
from zeromodel.video.domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    EpisodePlanDTO,
    SealedSplitPlanDTO,
)
from zeromodel.video.domains.video_action_set.episode_planning import make_episode_plan


def _identity() -> BenchmarkIdentityDTO:
    seed = "video-package-test-seed"
    return BenchmarkIdentityDTO(
        contract_commit="contract",
        seed_material=seed,
        seed_digest="sha256:" + hashlib.sha256(seed.encode("utf-8")).hexdigest(),
        policy_artifact_id="policy",
        parent_audit_sha="audit",
        parent_v3_sha="v3",
    )


def _plan(ordinal: int = 0, row: str = "left") -> EpisodePlanDTO:
    return EpisodePlanDTO.from_dict(
        make_episode_plan(
            _identity(),
            split="final",
            ordinal=ordinal,
            family_label="valid",
            family_ordinal=ordinal,
            source_row_id=row,
            row_actions={"left": "A", "right": "B"},
        )
    )


def test_identity_plan_sealed_plan_and_matrix_blob_store_semantics() -> None:
    store = InMemoryVideoActionSetStore()
    identity = store.save_identity(_identity())
    plan = _plan()

    assert store.save_identity(_identity()) == identity
    assert store.save_episode_plan(plan) == plan
    assert store.save_episode_plan(plan) == plan
    assert store.list_episode_plans(
        benchmark_seed_digest=identity.seed_digest,
        split="final",
    ) == (plan,)

    sealed = SealedSplitPlanDTO.build_final(
        episodes=(plan,),
        seed_commitment=identity.seed_digest,
    )
    assert sealed.materialization_prohibited is True
    assert store.save_sealed_split_plan(sealed) == sealed
    assert (
        store.get_sealed_split_plan(
            seed_commitment=identity.seed_digest,
            split="final",
        )
        == sealed
    )

    blob = MatrixBlob.from_array(np.array([[1, 2]], dtype=np.uint8))
    assert store.save_matrix_blob(blob).blob_id == blob.blob_id
    assert store.get_matrix_blob(blob.blob_id).to_array().tolist() == [[1, 2]]


def test_store_rejects_conflicts_and_keeps_batch_atomic() -> None:
    store = InMemoryVideoActionSetStore()
    identity = store.save_identity(_identity())
    plan = _plan(0, "left")
    conflicting = _plan(0, "right")

    with pytest.raises(VPMValidationError, match="unknown benchmark identity"):
        InMemoryVideoActionSetStore().save_episode_plan(plan)

    with pytest.raises(VPMValidationError, match="episode plan conflict"):
        store.save_episode_plans((plan, conflicting))

    assert (
        store.list_episode_plans(
            benchmark_seed_digest=identity.seed_digest,
            split="final",
        )
        == ()
    )


def test_default_runtime_uses_in_memory_store_without_sql() -> None:
    runtime = build_runtime()
    identity = _identity()

    saved = runtime.video_action_set.save_identity(identity)

    assert saved == identity
    assert runtime.video_action_set.get_identity(identity.seed_digest) == identity
