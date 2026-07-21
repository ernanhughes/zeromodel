from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn, Protocol

from ...artifact import VPMValidationError
from ...matrix_blob import MatrixBlob
from .dto import BenchmarkIdentityDTO, EpisodePlanDTO, SealedSplitPlanDTO
from .final_access_dto import (
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionFailureDTO,
    FinalExecutionReceiptDTO,
)
from .observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
    ObservationOperationChainDTO,
)


BENCHMARK_IDENTITY_CONFLICT_MESSAGE = "benchmark identity conflict for seed digest"
EPISODE_PLAN_CONFLICT_MESSAGE = "episode plan conflict for episode id"
SEALED_SPLIT_PLAN_CONFLICT_MESSAGE = (
    "sealed split plan conflict for seed commitment and split"
)
UNKNOWN_BENCHMARK_IDENTITY_MESSAGE = (
    "episode plan references unknown benchmark identity"
)
MATRIX_BLOB_CONFLICT_MESSAGE = "matrix blob conflict for blob id"
OBSERVATION_CONFLICT_MESSAGE = "observation conflict for frame id"
OBSERVATION_SEQUENCE_CONFLICT_MESSAGE = "observation conflict for episode sequence"
OBSERVATION_BLOB_MISMATCH_MESSAGE = (
    "observation matrix blob does not match declared pixels"
)
OPERATION_CHAIN_CONFLICT_MESSAGE = "observation operation chain conflict for frame id"
UNKNOWN_EPISODE_PLAN_MESSAGE = "observation references unknown episode plan"
FINAL_ACCESS_CONFLICT_MESSAGE = "final access record conflict"
FINAL_ACCESS_STATE_MESSAGE = "final access state transition mismatch"
FINAL_ACCESS_AUTHORIZATION_MESSAGE = "final execution authorization mismatch"


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

    def save_matrix_blob(
        self,
        blob: MatrixBlob,
    ) -> MatrixBlob: ...

    def get_matrix_blob(
        self,
        blob_id: str,
    ) -> MatrixBlob | None: ...

    def save_observation(
        self,
        observation: ObservationDTO,
        *,
        matrix_blob: MatrixBlob | None,
    ) -> ObservationDTO: ...

    def save_observations(
        self,
        observations: Sequence[MaterializedObservationDTO],
    ) -> tuple[ObservationDTO, ...]: ...

    def save_authorized_final_observations(
        self,
        access: FinalAccessRecordDTO,
        observations: Sequence[MaterializedObservationDTO],
    ) -> tuple[ObservationDTO, ...]: ...

    def get_observation(
        self,
        frame_id: str,
    ) -> ObservationDTO | None: ...

    def get_materialized_observation(
        self,
        frame_id: str,
    ) -> MaterializedObservationDTO | None: ...

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
    ) -> tuple[MaterializedObservationDTO, ...]: ...

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
    ) -> tuple[ObservationDTO, ...]: ...

    def get_operation_chain(
        self,
        frame_id: str,
    ) -> ObservationOperationChainDTO | None: ...

    def list_observations_by_operation(
        self,
        *,
        operation: str,
        split: str | None = None,
        family: str | None = None,
    ) -> tuple[ObservationDTO, ...]: ...

    def list_observations_by_output_digest(
        self,
        output_digest: str,
    ) -> tuple[ObservationDTO, ...]: ...

    def list_observation_consumers_of_digest(
        self,
        input_digest: str,
    ) -> tuple[ObservationDTO, ...]: ...

    def create_final_authorization(
        self,
        authorization: FinalExecutionAuthorizationDTO,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO: ...

    def load_final_authorization(
        self,
        authorization_id: str,
    ) -> FinalExecutionAuthorizationDTO | None: ...

    def load_final_access_record(
        self,
        access_id: str,
    ) -> FinalAccessRecordDTO | None: ...

    def list_final_access_events(
        self,
        access_id: str,
    ) -> tuple[FinalAccessEventDTO, ...]: ...

    def reserve_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO: ...

    def mark_final_access_running(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO: ...

    def complete_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
        receipt: FinalExecutionReceiptDTO,
    ) -> FinalAccessRecordDTO: ...

    def fail_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
        failure: FinalExecutionFailureDTO,
    ) -> FinalAccessRecordDTO: ...

    def interrupt_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
        failure: FinalExecutionFailureDTO,
    ) -> FinalAccessRecordDTO: ...


def raise_identity_conflict() -> NoReturn:
    raise VPMValidationError(BENCHMARK_IDENTITY_CONFLICT_MESSAGE)


def raise_episode_plan_conflict() -> NoReturn:
    raise VPMValidationError(EPISODE_PLAN_CONFLICT_MESSAGE)


def raise_sealed_split_plan_conflict() -> NoReturn:
    raise VPMValidationError(SEALED_SPLIT_PLAN_CONFLICT_MESSAGE)


def raise_unknown_benchmark_identity() -> NoReturn:
    raise VPMValidationError(UNKNOWN_BENCHMARK_IDENTITY_MESSAGE)


def raise_matrix_blob_conflict() -> NoReturn:
    raise VPMValidationError(MATRIX_BLOB_CONFLICT_MESSAGE)


def raise_observation_conflict() -> NoReturn:
    raise VPMValidationError(OBSERVATION_CONFLICT_MESSAGE)


def raise_observation_sequence_conflict() -> NoReturn:
    raise VPMValidationError(OBSERVATION_SEQUENCE_CONFLICT_MESSAGE)


def raise_observation_blob_mismatch() -> NoReturn:
    raise VPMValidationError(OBSERVATION_BLOB_MISMATCH_MESSAGE)


def raise_operation_chain_conflict() -> NoReturn:
    raise VPMValidationError(OPERATION_CHAIN_CONFLICT_MESSAGE)


def raise_unknown_episode_plan() -> NoReturn:
    raise VPMValidationError(UNKNOWN_EPISODE_PLAN_MESSAGE)


def raise_final_access_conflict() -> NoReturn:
    raise VPMValidationError(FINAL_ACCESS_CONFLICT_MESSAGE)


def raise_final_access_state() -> NoReturn:
    raise VPMValidationError(FINAL_ACCESS_STATE_MESSAGE)


def raise_final_access_authorization() -> NoReturn:
    raise VPMValidationError(FINAL_ACCESS_AUTHORIZATION_MESSAGE)


__all__ = [
    "BENCHMARK_IDENTITY_CONFLICT_MESSAGE",
    "EPISODE_PLAN_CONFLICT_MESSAGE",
    "FINAL_ACCESS_AUTHORIZATION_MESSAGE",
    "FINAL_ACCESS_CONFLICT_MESSAGE",
    "FINAL_ACCESS_STATE_MESSAGE",
    "MATRIX_BLOB_CONFLICT_MESSAGE",
    "OBSERVATION_BLOB_MISMATCH_MESSAGE",
    "OBSERVATION_CONFLICT_MESSAGE",
    "OPERATION_CHAIN_CONFLICT_MESSAGE",
    "OBSERVATION_SEQUENCE_CONFLICT_MESSAGE",
    "SEALED_SPLIT_PLAN_CONFLICT_MESSAGE",
    "UNKNOWN_BENCHMARK_IDENTITY_MESSAGE",
    "UNKNOWN_EPISODE_PLAN_MESSAGE",
    "VideoActionSetStore",
    "raise_episode_plan_conflict",
    "raise_final_access_authorization",
    "raise_final_access_conflict",
    "raise_final_access_state",
    "raise_identity_conflict",
    "raise_matrix_blob_conflict",
    "raise_observation_blob_mismatch",
    "raise_observation_conflict",
    "raise_observation_sequence_conflict",
    "raise_operation_chain_conflict",
    "raise_sealed_split_plan_conflict",
    "raise_unknown_benchmark_identity",
    "raise_unknown_episode_plan",
]
