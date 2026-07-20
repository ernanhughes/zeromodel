from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn, Protocol

from ...artifact import VPMValidationError
from .dto import BenchmarkIdentityDTO, EpisodePlanDTO, SealedSplitPlanDTO


BENCHMARK_IDENTITY_CONFLICT_MESSAGE = "benchmark identity conflict for seed digest"
EPISODE_PLAN_CONFLICT_MESSAGE = "episode plan conflict for episode id"
SEALED_SPLIT_PLAN_CONFLICT_MESSAGE = (
    "sealed split plan conflict for seed commitment and split"
)
UNKNOWN_BENCHMARK_IDENTITY_MESSAGE = (
    "episode plan references unknown benchmark identity"
)


class VideoActionSetStore(Protocol):
    def save_identity(
        self,
        identity: BenchmarkIdentityDTO,
    ) -> BenchmarkIdentityDTO: ...

    def get_identity(
        self,
        seed_digest: str,
    ) -> BenchmarkIdentityDTO | None: ...

    def save_episode_plan(
        self,
        plan: EpisodePlanDTO,
    ) -> EpisodePlanDTO: ...

    def save_episode_plans(
        self,
        plans: Sequence[EpisodePlanDTO],
    ) -> tuple[EpisodePlanDTO, ...]: ...

    def get_episode_plan(
        self,
        episode_id: str,
    ) -> EpisodePlanDTO | None: ...

    def list_episode_plans(
        self,
        *,
        benchmark_seed_digest: str,
        split: str,
    ) -> tuple[EpisodePlanDTO, ...]: ...

    def save_sealed_split_plan(
        self,
        plan: SealedSplitPlanDTO,
    ) -> SealedSplitPlanDTO: ...

    def get_sealed_split_plan(
        self,
        *,
        seed_commitment: str,
        split: str,
    ) -> SealedSplitPlanDTO | None: ...


def raise_identity_conflict() -> NoReturn:
    raise VPMValidationError(BENCHMARK_IDENTITY_CONFLICT_MESSAGE)


def raise_episode_plan_conflict() -> NoReturn:
    raise VPMValidationError(EPISODE_PLAN_CONFLICT_MESSAGE)


def raise_sealed_split_plan_conflict() -> NoReturn:
    raise VPMValidationError(SEALED_SPLIT_PLAN_CONFLICT_MESSAGE)


def raise_unknown_benchmark_identity() -> NoReturn:
    raise VPMValidationError(UNKNOWN_BENCHMARK_IDENTITY_MESSAGE)


__all__ = [
    "BENCHMARK_IDENTITY_CONFLICT_MESSAGE",
    "EPISODE_PLAN_CONFLICT_MESSAGE",
    "SEALED_SPLIT_PLAN_CONFLICT_MESSAGE",
    "UNKNOWN_BENCHMARK_IDENTITY_MESSAGE",
    "VideoActionSetStore",
    "raise_episode_plan_conflict",
    "raise_identity_conflict",
    "raise_sealed_split_plan_conflict",
    "raise_unknown_benchmark_identity",
]
