from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ...matrix_blob import MatrixBlob
from .dto import BenchmarkIdentityDTO, EpisodePlanDTO, SealedSplitPlanDTO
from .episode_plan_service import EpisodePlanService
from .final_access_dto import (
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionRequestDTO,
)
from .final_access_service import FinalAccessService
from .identity_service import IdentityService
from .observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
    ObservationOperationChainDTO,
)
from .observation_service import ObservationService


@dataclass(frozen=True, slots=True)
class VideoActionSetEngine:
    identity_service: IdentityService
    episode_plan_service: EpisodePlanService
    observation_service: ObservationService
    final_access_service: FinalAccessService

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

    def save_matrix_blob(self, blob: MatrixBlob) -> MatrixBlob:
        return self.observation_service.save_matrix_blob(blob)

    def get_matrix_blob(self, blob_id: str) -> MatrixBlob | None:
        return self.observation_service.get_matrix_blob(blob_id)

    def save_observation_record(
        self,
        record: Mapping[str, object],
    ) -> ObservationDTO:
        return self.observation_service.save_record(record)

    def save_observation_records(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> tuple[ObservationDTO, ...]:
        return self.observation_service.save_records(records)

    def save_materialized_observation(
        self,
        item: MaterializedObservationDTO,
    ) -> ObservationDTO:
        return self.observation_service.save_materialized(item)

    def get_observation(self, frame_id: str) -> ObservationDTO | None:
        return self.observation_service.get_observation(frame_id)

    def get_materialized_observation(
        self,
        frame_id: str,
    ) -> MaterializedObservationDTO | None:
        return self.observation_service.get_materialized(frame_id)

    def list_materialized_observations(
        self,
        *,
        benchmark_seed_digest: str | None = None,
        split: str | None = None,
        episode_id: str | None = None,
        family: str | None = None,
        event_type: str | None = None,
        denominator_class: str | None = None,
        has_pixels: bool | None = None,
    ) -> tuple[MaterializedObservationDTO, ...]:
        return self.observation_service.list_materialized(
            benchmark_seed_digest=benchmark_seed_digest,
            split=split,
            episode_id=episode_id,
            family=family,
            event_type=event_type,
            denominator_class=denominator_class,
            has_pixels=has_pixels,
        )

    def get_observation_record(
        self,
        frame_id: str,
        *,
        include_pixels: bool = True,
    ) -> dict[str, object] | None:
        return self.observation_service.get_record(
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
        return self.observation_service.list_observations(
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
        return self.observation_service.list_records(
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
        return self.observation_service.get_operation_chain(frame_id)

    def list_observations_by_operation(
        self,
        *,
        operation: str,
        split: str | None = None,
        family: str | None = None,
    ) -> tuple[ObservationDTO, ...]:
        return self.observation_service.list_by_operation(
            operation=operation,
            split=split,
            family=family,
        )

    def list_observations_by_output_digest(
        self,
        output_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        return self.observation_service.list_by_output_digest(output_digest)

    def list_observation_consumers_of_digest(
        self,
        input_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        return self.observation_service.list_consumers_of_digest(input_digest)

    def create_final_authorization(
        self,
        authorization: FinalExecutionAuthorizationDTO,
    ) -> FinalAccessRecordDTO:
        return self.final_access_service.create_authorization(authorization)

    def load_final_access_record(
        self,
        access_id: str,
    ) -> FinalAccessRecordDTO | None:
        return self.final_access_service.load_record(access_id)

    def list_final_access_events(
        self,
        access_id: str,
    ) -> tuple[FinalAccessEventDTO, ...]:
        return self.final_access_service.list_events(access_id)

    def reserve_final_access(self, access_id: str) -> FinalAccessRecordDTO:
        return self.final_access_service.reserve(access_id)

    def mark_final_access_running(self, access_id: str) -> FinalAccessRecordDTO:
        return self.final_access_service.mark_running(access_id)

    def save_final_observation_record(
        self,
        access_id: str,
        record: Mapping[str, object],
    ) -> ObservationDTO:
        return self.final_access_service.save_final_observation_record(
            access_id,
            record,
        )

    def execute_final_once(
        self,
        request: FinalExecutionRequestDTO,
    ) -> dict[str, object]:
        return self.final_access_service.execute_final_once(request)


__all__ = ["VideoActionSetEngine"]
