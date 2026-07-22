from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from zeromodel.video.domains.video_action_set.dto import EpisodePlanDTO, SealedSplitPlanDTO
from zeromodel.video.domains.video_action_set.store import VideoActionSetStore


@dataclass(frozen=True, slots=True)
class EpisodePlanService:
    store: VideoActionSetStore

    def save_plan(self, plan: EpisodePlanDTO) -> EpisodePlanDTO:
        return self.store.save_episode_plan(plan)

    def save_plans(
        self,
        plans: Sequence[EpisodePlanDTO],
    ) -> tuple[EpisodePlanDTO, ...]:
        return self.store.save_episode_plans(plans)

    def get_plan(self, episode_id: str) -> EpisodePlanDTO | None:
        return self.store.get_episode_plan(episode_id)

    def list_plans(
        self,
        *,
        benchmark_seed_digest: str,
        split: str,
    ) -> tuple[EpisodePlanDTO, ...]:
        return self.store.list_episode_plans(
            benchmark_seed_digest=benchmark_seed_digest,
            split=split,
        )

    def seal_final_split(
        self,
        *,
        episodes: Sequence[EpisodePlanDTO],
        seed_commitment: str,
    ) -> SealedSplitPlanDTO:
        plan = SealedSplitPlanDTO.build_final(
            episodes=episodes,
            seed_commitment=seed_commitment,
        )
        return self.store.save_sealed_split_plan(plan)

    def save_sealed_split(self, plan: SealedSplitPlanDTO) -> SealedSplitPlanDTO:
        return self.store.save_sealed_split_plan(plan)

    def get_sealed_split(
        self,
        *,
        seed_commitment: str,
        split: str,
    ) -> SealedSplitPlanDTO | None:
        return self.store.get_sealed_split_plan(
            seed_commitment=seed_commitment,
            split=split,
        )


__all__ = ["EpisodePlanService"]
