from __future__ import annotations

from collections.abc import Sequence

from ..domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    EpisodePlanDTO,
    SealedSplitPlanDTO,
)
from ..domains.video_action_set.store import (
    VideoActionSetStore,
    raise_episode_plan_conflict,
    raise_identity_conflict,
    raise_sealed_split_plan_conflict,
    raise_unknown_benchmark_identity,
)


class InMemoryVideoActionSetStore(VideoActionSetStore):
    def __init__(self) -> None:
        self._identities: dict[str, BenchmarkIdentityDTO] = {}
        self._episode_plans: dict[str, EpisodePlanDTO] = {}
        self._sealed_split_plans: dict[tuple[str, str], SealedSplitPlanDTO] = {}

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        existing = self._identities.get(identity.seed_digest)
        if existing is not None:
            if existing != identity:
                raise_identity_conflict()
            return existing
        self._identities[identity.seed_digest] = identity
        return identity

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        return self._identities.get(seed_digest)

    def save_episode_plan(self, plan: EpisodePlanDTO) -> EpisodePlanDTO:
        return self.save_episode_plans((plan,))[0]

    def save_episode_plans(
        self,
        plans: Sequence[EpisodePlanDTO],
    ) -> tuple[EpisodePlanDTO, ...]:
        plan_tuple = tuple(plans)
        self._preflight_episode_plans(plan_tuple)
        for plan in plan_tuple:
            self._episode_plans.setdefault(plan.episode_id, plan)
        return tuple(self._episode_plans[plan.episode_id] for plan in plan_tuple)

    def get_episode_plan(self, episode_id: str) -> EpisodePlanDTO | None:
        return self._episode_plans.get(episode_id)

    def list_episode_plans(
        self,
        *,
        benchmark_seed_digest: str,
        split: str,
    ) -> tuple[EpisodePlanDTO, ...]:
        return tuple(
            sorted(
                (
                    plan
                    for plan in self._episode_plans.values()
                    if plan.benchmark_seed_digest == benchmark_seed_digest
                    and plan.split == split
                ),
                key=lambda plan: (plan.ordinal, plan.episode_id),
            )
        )

    def save_sealed_split_plan(
        self,
        plan: SealedSplitPlanDTO,
    ) -> SealedSplitPlanDTO:
        if plan.seed_commitment not in self._identities:
            raise_unknown_benchmark_identity()
        key = (plan.seed_commitment, plan.split)
        existing = self._sealed_split_plans.get(key)
        if existing is not None:
            if existing != plan:
                raise_sealed_split_plan_conflict()
            return existing
        self._preflight_episode_plans(plan.episodes)
        for episode in plan.episodes:
            self._episode_plans.setdefault(episode.episode_id, episode)
        self._sealed_split_plans[key] = plan
        return plan

    def get_sealed_split_plan(
        self,
        *,
        seed_commitment: str,
        split: str,
    ) -> SealedSplitPlanDTO | None:
        return self._sealed_split_plans.get((seed_commitment, split))

    def _preflight_episode_plans(self, plans: Sequence[EpisodePlanDTO]) -> None:
        seen_ids: dict[str, EpisodePlanDTO] = {}
        seen_ordinals: dict[tuple[str, str, int], EpisodePlanDTO] = {}
        for plan in plans:
            if plan.benchmark_seed_digest not in self._identities:
                raise_unknown_benchmark_identity()
            existing = self._episode_plans.get(plan.episode_id)
            if existing is not None and existing != plan:
                raise_episode_plan_conflict()
            seen = seen_ids.get(plan.episode_id)
            if seen is not None and seen != plan:
                raise_episode_plan_conflict()
            seen_ids[plan.episode_id] = plan
            ordinal_key = (plan.benchmark_seed_digest, plan.split, plan.ordinal)
            existing_for_ordinal = self._plan_for_ordinal(ordinal_key)
            if existing_for_ordinal is not None and existing_for_ordinal != plan:
                raise_episode_plan_conflict()
            seen_for_ordinal = seen_ordinals.get(ordinal_key)
            if seen_for_ordinal is not None and seen_for_ordinal != plan:
                raise_episode_plan_conflict()
            seen_ordinals[ordinal_key] = plan

    def _plan_for_ordinal(
        self,
        ordinal_key: tuple[str, str, int],
    ) -> EpisodePlanDTO | None:
        benchmark_seed_digest, split, ordinal = ordinal_key
        for plan in self._episode_plans.values():
            if (
                plan.benchmark_seed_digest == benchmark_seed_digest
                and plan.split == split
                and plan.ordinal == ordinal
            ):
                return plan
        return None


__all__ = ["InMemoryVideoActionSetStore"]
