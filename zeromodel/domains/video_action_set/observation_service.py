from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ...matrix_blob import MatrixBlob
from .observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
    ObservationOperationChainDTO,
)
from .store import VideoActionSetStore


@dataclass(frozen=True, slots=True)
class ObservationService:
    store: VideoActionSetStore

    def save_matrix_blob(self, blob: MatrixBlob) -> MatrixBlob:
        return self.store.save_matrix_blob(blob)

    def get_matrix_blob(self, blob_id: str) -> MatrixBlob | None:
        return self.store.get_matrix_blob(blob_id)

    def save_record(self, record: Mapping[str, object]) -> ObservationDTO:
        item = MaterializedObservationDTO.from_record(record)
        return self.save_materialized(item)

    def save_records(
        self,
        records: Sequence[Mapping[str, object]],
    ) -> tuple[ObservationDTO, ...]:
        return self.store.save_observations(
            tuple(MaterializedObservationDTO.from_record(record) for record in records)
        )

    def save_materialized(self, item: MaterializedObservationDTO) -> ObservationDTO:
        return self.store.save_observation(
            item.observation,
            matrix_blob=item.matrix_blob,
        )

    def get_observation(self, frame_id: str) -> ObservationDTO | None:
        return self.store.get_observation(frame_id)

    def get_materialized(
        self,
        frame_id: str,
    ) -> MaterializedObservationDTO | None:
        return self.store.get_materialized_observation(frame_id)

    def get_record(
        self,
        frame_id: str,
        *,
        include_pixels: bool = True,
    ) -> dict[str, object] | None:
        if include_pixels:
            item = self.store.get_materialized_observation(frame_id)
            return None if item is None else item.to_record(include_pixels=True)
        observation = self.store.get_observation(frame_id)
        return (
            None if observation is None else observation.to_record(include_pixels=False)
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
        return self.store.list_observations(
            benchmark_seed_digest=benchmark_seed_digest,
            split=split,
            episode_id=episode_id,
            family=family,
            event_type=event_type,
            denominator_class=denominator_class,
            has_pixels=has_pixels,
        )

    def list_records(
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
        observations = self.list_observations(
            benchmark_seed_digest=benchmark_seed_digest,
            split=split,
            episode_id=episode_id,
            family=family,
            event_type=event_type,
            denominator_class=denominator_class,
            has_pixels=has_pixels,
        )
        return tuple(
            self._record_for_observation(
                observation,
                include_pixels=include_pixels,
            )
            for observation in observations
        )

    def get_operation_chain(
        self,
        frame_id: str,
    ) -> ObservationOperationChainDTO | None:
        return self.store.get_operation_chain(frame_id)

    def list_by_operation(
        self,
        *,
        operation: str,
        split: str | None = None,
        family: str | None = None,
    ) -> tuple[ObservationDTO, ...]:
        return self.store.list_observations_by_operation(
            operation=operation,
            split=split,
            family=family,
        )

    def list_by_output_digest(
        self,
        output_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        return self.store.list_observations_by_output_digest(output_digest)

    def list_consumers_of_digest(
        self,
        input_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        return self.store.list_observation_consumers_of_digest(input_digest)

    def _record_for_observation(
        self,
        observation: ObservationDTO,
        *,
        include_pixels: bool,
    ) -> dict[str, object]:
        if not include_pixels:
            return observation.to_record(include_pixels=False)
        item = self.store.get_materialized_observation(observation.frame_id)
        if item is None:
            return observation.to_record(include_pixels=True, matrix_blob=None)
        return item.to_record(include_pixels=True)


__all__ = ["ObservationService"]
