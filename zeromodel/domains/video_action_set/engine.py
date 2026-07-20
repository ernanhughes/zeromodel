from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from .dto import BenchmarkIdentityDTO, EpisodePlanDTO, SealedSplitPlanDTO
from .episode_plan_service import EpisodePlanService
from .identity_service import IdentityService


@dataclass(frozen=True, slots=True)
class VideoActionSetEngine:
    identity_service: IdentityService
    episode_plan_service: EpisodePlanService

    def load_identity(self, repo_root: Path) -> BenchmarkIdentityDTO:
        return self.identity_service.load_identity(repo_root)

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        return self.identity_service.get_identity(seed_digest)

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        return self.identity_service.save_identity(identity)

    def save_episode_plan(self, plan: EpisodePlanDTO) -> EpisodePlanDTO:
        return self.episode_plan_service.save_plan(plan)

    def save_episode_plans(
        self,
        plans: Sequence[EpisodePlanDTO],
    ) -> tuple[EpisodePlanDTO, ...]:
        return self.episode_plan_service.save_plans(plans)

    def get_episode_plan(self, episode_id: str) -> EpisodePlanDTO | None:
        return self.episode_plan_service.get_plan(episode_id)

    def list_episode_plans(
        self,
        *,
        benchmark_seed_digest: str,
        split: str,
    ) -> tuple[EpisodePlanDTO, ...]:
        return self.episode_plan_service.list_plans(
            benchmark_seed_digest=benchmark_seed_digest,
            split=split,
        )

    def seal_final_split(
        self,
        *,
        episodes: Sequence[EpisodePlanDTO],
        seed_commitment: str,
    ) -> SealedSplitPlanDTO:
        return self.episode_plan_service.seal_final_split(
            episodes=episodes,
            seed_commitment=seed_commitment,
        )

    def save_sealed_split_plan(
        self,
        plan: SealedSplitPlanDTO,
    ) -> SealedSplitPlanDTO:
        return self.episode_plan_service.save_sealed_split(plan)

    def get_sealed_split_plan(
        self,
        *,
        seed_commitment: str,
        split: str,
    ) -> SealedSplitPlanDTO | None:
        return self.episode_plan_service.get_sealed_split(
            seed_commitment=seed_commitment,
            split=split,
        )


__all__ = ["VideoActionSetEngine"]
