from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ...matrix_blob import MatrixBlob
from .dto import BenchmarkIdentityDTO, EpisodePlanDTO, SealedSplitPlanDTO
from .engine import VideoActionSetEngine
from .observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
    ObservationOperationChainDTO,
)


@dataclass(frozen=True, slots=True)
class VideoActionSetFacade:
    engine: VideoActionSetEngine

    def load_identity(self, repo_root: Path) -> BenchmarkIdentityDTO:
        return self.engine.load_identity(repo_root)

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        return self.engine.get_identity(seed_digest)

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        return self.engine.save_identity(identity)

    def save_episode_plan(self, plan: EpisodePlanDTO) -> EpisodePlanDTO:
        return self.engine.save_episode_plan(plan)

    def save_episode_plans(
        self,
        plans: Sequence[EpisodePlanDTO],
    ) -> tuple[EpisodePlanDTO, ...]:
        return self.engine.save_episode_plans(plans)

    def get_episode_plan(self, episode_id: str) -> EpisodePlanDTO | None:
        return self.engine.get_episode_plan(episode_id)

    def list_episode_plans(
        self,
        *,
        benchmark_seed_digest: str,
        split: str,
    ) -> tuple[EpisodePlanDTO, ...]:
        return self.engine.list_episode_plans(
            benchmark_seed_digest=benchmark_seed_digest,
            split=split,
        )

    def seal_final_split(
        self,
        *,
        episodes: Sequence[EpisodePlanDTO],
        seed_commitment: str,
    ) -> SealedSplitPlanDTO:
        return self.engine.seal_final_split(
            episodes=episodes,
            seed_commitment=seed_commitment,
        )

    def save_sealed_split_plan(
        self,
        plan: SealedSplitPlanDTO,
    ) -> SealedSplitPlanDTO:
        return self.engine.save_sealed_split_plan(plan)

    def get_sealed_split_plan(
        self,
        *,
        seed_commitment: str,
        split: str,
    ) -> SealedSplitPlanDTO | None:
        return self.engine.get_sealed_split_plan(
            seed_commitment=seed_commitment,
            split=split,
        )

    def save_matrix_blob(self, blob: MatrixBlob) -> MatrixBlob:
        return self.engine.save_matrix_blob(blob)

    def get_matrix_blob(self, blob_id: str) -> MatrixBlob | None:
        return self.engine.get_matrix_blob(blob_id)

    def save_observation_record(
        self,
        record: Mapping[str, object],
    ) -> ObservationDTO:
        return self.engine.save_observation_record(record)

    def save_observation_records(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> tuple[ObservationDTO, ...]:
        return self.engine.save_observation_records(records)

    def save_materialized_observation(
        self,
        item: MaterializedObservationDTO,
    ) -> ObservationDTO:
        return self.engine.save_materialized_observation(item)

    def get_observation(self, frame_id: str) -> ObservationDTO | None:
        return self.engine.get_observation(frame_id)

    def get_materialized_observation(
        self,
        frame_id: str,
    ) -> MaterializedObservationDTO | None:
        return self.engine.get_materialized_observation(frame_id)

    def get_observation_record(
        self,
        frame_id: str,
        *,
        include_pixels: bool = True,
    ) -> dict[str, object] | None:
        return self.engine.get_observation_record(
            frame_id,
            include_pixels=include_pixels,
        )

    def list_observations(
        self,
        *,
        benchmark_seed_digest: str | None = None,
        split: str | None = None,
        episode_id: str | None = None,
        family: str | None = None,
        event_type: str | None = None,
        denominator_class: str | None = None,
        has_pixels: bool | None = None,
    ) -> tuple[ObservationDTO, ...]:
        return self.engine.list_observations(
            benchmark_seed_digest=benchmark_seed_digest,
            split=split,
            episode_id=episode_id,
            family=family,
            event_type=event_type,
            denominator_class=denominator_class,
            has_pixels=has_pixels,
        )

    def list_observation_records(
        self,
        *,
        benchmark_seed_digest: str | None = None,
        split: str | None = None,
        episode_id: str | None = None,
        family: str | None = None,
        event_type: str | None = None,
        denominator_class: str | None = None,
        has_pixels: bool | None = None,
        include_pixels: bool = False,
    ) -> tuple[dict[str, object], ...]:
        return self.engine.list_observation_records(
            benchmark_seed_digest=benchmark_seed_digest,
            split=split,
            episode_id=episode_id,
            family=family,
            event_type=event_type,
            denominator_class=denominator_class,
            has_pixels=has_pixels,
            include_pixels=include_pixels,
        )

    def get_observation_operation_chain(
        self,
        frame_id: str,
    ) -> ObservationOperationChainDTO | None:
        return self.engine.get_observation_operation_chain(frame_id)

    def list_observations_by_operation(
        self,
        *,
        operation: str,
        split: str | None = None,
        family: str | None = None,
    ) -> tuple[ObservationDTO, ...]:
        return self.engine.list_observations_by_operation(
            operation=operation,
            split=split,
            family=family,
        )

    def list_observations_by_output_digest(
        self,
        output_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        return self.engine.list_observations_by_output_digest(output_digest)

    def list_observation_consumers_of_digest(
        self,
        input_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        return self.engine.list_observation_consumers_of_digest(input_digest)


__all__ = ["VideoActionSetFacade"]
