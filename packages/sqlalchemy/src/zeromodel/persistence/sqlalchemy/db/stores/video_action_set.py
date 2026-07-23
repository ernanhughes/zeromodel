from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

from sqlalchemy import Engine, select, update
from sqlalchemy.engine import CursorResult
from sqlalchemy.orm import Session, sessionmaker

from zeromodel.core.artifact import VPMValidationError
from zeromodel.video.domains.video_action_set.canonical_json import canonical_json_text
from zeromodel.video.domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    EpisodePlanDTO,
    SealedSplitPlanDTO,
)
from zeromodel.video.domains.video_action_set.final_access_dto import (
    FINAL_ACCESS_TERMINAL_STATES,
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalEvaluationProtocolDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionFailureDTO,
    validate_final_access_event,
)
from zeromodel.video.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
    ObservationOperationChainDTO,
)
from zeromodel.video.domains.video_action_set.store import (
    VideoActionSetStore,
    raise_final_access_authorization,
    raise_final_access_conflict,
    raise_final_access_state,
    raise_matrix_blob_conflict,
    raise_observation_conflict,
    raise_episode_plan_conflict,
    raise_identity_conflict,
    raise_sealed_split_plan_conflict,
    raise_unknown_benchmark_identity,
    raise_unknown_episode_plan,
)
from zeromodel.core.matrix_blob import MatrixBlob
from zeromodel.persistence.sqlalchemy.db.orm.video_action_set import (
    BenchmarkIdentityORM,
    EpisodePlanORM,
    FinalAccessAuthorizationORM,
    FinalAccessEventORM,
    FinalAccessRecordORM,
    FinalEvaluationProtocolORM,
    MatrixBlobORM,
    ObservationORM,
    ObservationOperationInputORM,
    ObservationOperationORM,
    SealedSplitPlanORM,
)
from zeromodel.persistence.sqlalchemy.db.stores.provider_evaluation import (
    ProviderEvaluationSqlStoreMixin,
)
from zeromodel.persistence.sqlalchemy.db.stores.video_action_set_observation import (
    chain_for_frame,
    materialized_observations_from_rows,
    observation_select,
    observations_from_rows,
    operation_observation_select,
    optional_observation_predicates,
    preflight_observation_sequence,
    to_matrix_blob,
    to_matrix_blob_orm,
    to_observation_dto,
    to_observation_orm,
    to_operation_chain_orm,
    to_operation_input_orms,
    to_operation_orms,
)


class _ObservationSqlStoreMixin:
    _session_factory: sessionmaker[Session]

    def save_matrix_blob(self, blob: MatrixBlob) -> MatrixBlob:
        session = self._session_factory()
        try:
            with session.begin():
                return self._save_matrix_blob_in_session(session, blob)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_matrix_blob(self, blob_id: str) -> MatrixBlob | None:
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(MatrixBlobORM, blob_id)
                return None if row is None else to_matrix_blob(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_observation(
        self,
        observation: ObservationDTO,
        *,
        matrix_blob: MatrixBlob | None,
    ) -> ObservationDTO:
        return self.save_observations(
            (MaterializedObservationDTO(observation, matrix_blob),)
        )[0]

    def save_observations(
        self,
        observations: Sequence[MaterializedObservationDTO],
    ) -> tuple[ObservationDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                return self._save_observations_in_session(
                    session,
                    tuple(observations),
                    final_access=None,
                )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_authorized_final_observations(
        self,
        access: FinalAccessRecordDTO,
        observations: Sequence[MaterializedObservationDTO],
    ) -> tuple[ObservationDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                return self._save_observations_in_session(
                    session,
                    tuple(observations),
                    final_access=access,
                )
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_observation(self, frame_id: str) -> ObservationDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(ObservationORM, frame_id)
                return None if row is None else to_observation_dto(session, row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_materialized_observation(
        self,
        frame_id: str,
    ) -> MaterializedObservationDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(ObservationORM, frame_id)
                if row is None:
                    return None
                return materialized_observations_from_rows(session, (row,))[0]
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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
        session = self._session_factory()
        try:
            with session.begin():
                statement = observation_select(
                    benchmark_seed_digest=benchmark_seed_digest,
                    split=split,
                    episode_id=episode_id,
                    family=family,
                    event_type=event_type,
                    denominator_class=denominator_class,
                    has_pixels=has_pixels,
                )
                rows = session.scalars(statement).all()
                return observations_from_rows(session, rows)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

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
        session = self._session_factory()
        try:
            with session.begin():
                statement = observation_select(
                    benchmark_seed_digest=benchmark_seed_digest,
                    split=split,
                    episode_id=episode_id,
                    family=family,
                    event_type=event_type,
                    denominator_class=denominator_class,
                    has_pixels=has_pixels,
                )
                rows = session.scalars(statement).all()
                return materialized_observations_from_rows(session, rows)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_operation_chain(
        self,
        frame_id: str,
    ) -> ObservationOperationChainDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                if session.get(ObservationORM, frame_id) is None:
                    return None
                return chain_for_frame(session, frame_id)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_observations_by_operation(
        self,
        *,
        operation: str,
        split: str | None = None,
        family: str | None = None,
    ) -> tuple[ObservationDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                rows = session.scalars(
                    operation_observation_select()
                    .where(ObservationOperationORM.operation == operation)
                    .where(*optional_observation_predicates(split=split, family=family))
                ).all()
                return observations_from_rows(session, rows)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_observations_by_output_digest(
        self,
        output_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                rows = session.scalars(
                    operation_observation_select().where(
                        ObservationOperationORM.output_digest == output_digest
                    )
                ).all()
                return observations_from_rows(session, rows)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_observation_consumers_of_digest(
        self,
        input_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                rows = session.scalars(
                    operation_observation_select()
                    .join(
                        ObservationOperationInputORM,
                        (
                            ObservationOperationInputORM.frame_id
                            == ObservationOperationORM.frame_id
                        )
                        & (
                            ObservationOperationInputORM.operation_index
                            == ObservationOperationORM.operation_index
                        ),
                    )
                    .where(ObservationOperationInputORM.input_digest == input_digest)
                ).all()
                return observations_from_rows(session, rows)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _save_observations_in_session(
        self,
        session: Session,
        observations: Sequence[MaterializedObservationDTO],
        *,
        final_access: FinalAccessRecordDTO | None,
    ) -> tuple[ObservationDTO, ...]:
        existing = self._preflight_observations(
            session,
            observations,
            final_access=final_access,
        )
        inserted_frame_ids: set[str] = set()
        operation_inputs: list[ObservationOperationInputORM] = []
        for item in observations:
            observation = item.observation
            if (
                observation.frame_id in existing
                or observation.frame_id in inserted_frame_ids
            ):
                continue
            if item.matrix_blob is not None:
                self._save_matrix_blob_in_session(session, item.matrix_blob)
            session.add(to_observation_orm(observation))
            session.add(to_operation_chain_orm(observation))
            session.add_all(to_operation_orms(observation))
            operation_inputs.extend(to_operation_input_orms(observation))
            inserted_frame_ids.add(observation.frame_id)
        session.flush()
        session.add_all(operation_inputs)
        return tuple(
            existing.get(item.observation.frame_id, item.observation)
            for item in observations
        )

    def _preflight_observations(
        self,
        session: Session,
        observations: Sequence[MaterializedObservationDTO],
        *,
        final_access: FinalAccessRecordDTO | None,
    ) -> dict[str, ObservationDTO]:
        existing_observations: dict[str, ObservationDTO] = {}
        seen_observations: dict[str, ObservationDTO] = {}
        seen_sequences: dict[tuple[str, int], str] = {}
        seen_blobs: dict[str, MatrixBlob] = {}
        for item in observations:
            self._preflight_final_observation_access(
                session,
                item.observation,
                final_access,
            )
            self._preflight_observation_ownership(session, item.observation)
            self._preflight_observation_identity(
                session,
                item.observation,
                existing_observations,
                seen_observations,
                seen_sequences,
            )
            self._preflight_blob(session, item.matrix_blob, seen_blobs)
        return existing_observations

    @staticmethod
    def _preflight_final_observation_access(
        session: Session,
        observation: ObservationDTO,
        final_access: FinalAccessRecordDTO | None,
    ) -> None:
        if observation.split != "final":
            if observation.final_access_id is not None:
                raise_final_access_authorization()
            return
        if (
            final_access is None
            or observation.final_access_id != final_access.access_id
        ):
            raise_final_access_authorization()
        row = session.get(FinalAccessRecordORM, final_access.access_id)
        if row is None:
            raise_final_access_authorization()
        if row.state != "running":
            raise_final_access_state()
        stored = SqlAlchemyVideoActionSetStore._to_final_record_dto(row)
        if stored != final_access:
            raise_final_access_state()

    def _preflight_observation_ownership(
        self,
        session: Session,
        observation: ObservationDTO,
    ) -> None:
        if session.get(BenchmarkIdentityORM, observation.benchmark_seed_digest) is None:
            raise_unknown_benchmark_identity()
        episode = session.get(EpisodePlanORM, observation.episode_id)
        if episode is None:
            raise_unknown_episode_plan()
        if (
            episode.plan_digest != observation.episode_plan_digest
            or episode.benchmark_seed_digest != observation.benchmark_seed_digest
            or episode.split != observation.split
        ):
            raise_unknown_episode_plan()

    def _preflight_observation_identity(
        self,
        session: Session,
        observation: ObservationDTO,
        existing_observations: dict[str, ObservationDTO],
        seen_observations: dict[str, ObservationDTO],
        seen_sequences: dict[tuple[str, int], str],
    ) -> None:
        existing = session.get(ObservationORM, observation.frame_id)
        if existing is not None:
            existing_dto = to_observation_dto(session, existing)
            if existing_dto != observation:
                raise_observation_conflict()
            existing_observations[observation.frame_id] = existing_dto
        seen = seen_observations.get(observation.frame_id)
        if seen is not None and seen != observation:
            raise_observation_conflict()
        seen_observations[observation.frame_id] = observation
        preflight_observation_sequence(session, observation, seen_sequences)

    def _preflight_blob(
        self,
        session: Session,
        blob: MatrixBlob | None,
        seen_blobs: dict[str, MatrixBlob],
    ) -> None:
        if blob is None:
            return
        existing = session.get(MatrixBlobORM, blob.blob_id)
        if existing is not None and to_matrix_blob(existing) != blob:
            raise_matrix_blob_conflict()
        seen = seen_blobs.get(blob.blob_id)
        if seen is not None and seen != blob:
            raise_matrix_blob_conflict()
        seen_blobs[blob.blob_id] = blob

    @staticmethod
    def _save_matrix_blob_in_session(
        session: Session,
        blob: MatrixBlob,
    ) -> MatrixBlob:
        existing = session.get(MatrixBlobORM, blob.blob_id)
        if existing is not None:
            existing_blob = to_matrix_blob(existing)
            if existing_blob != blob:
                raise_matrix_blob_conflict()
            return existing_blob
        session.add(to_matrix_blob_orm(blob))
        return blob


class SqlAlchemyVideoActionSetStore(
    _ObservationSqlStoreMixin, ProviderEvaluationSqlStoreMixin, VideoActionSetStore
):
    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        finalization_engine: Engine | None = None,
    ) -> None:
        self._session_factory = session_factory
        self._finalization_engine = finalization_engine

    def save_identity(self, identity: BenchmarkIdentityDTO) -> BenchmarkIdentityDTO:
        session = self._session_factory()
        try:
            with session.begin():
                existing = session.get(BenchmarkIdentityORM, identity.seed_digest)
                if existing is not None:
                    existing_dto = self._to_dto(existing)
                    if existing_dto != identity:
                        raise_identity_conflict()
                    return existing_dto
                session.add(self._to_orm(identity))
            return identity
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_identity(self, seed_digest: str) -> BenchmarkIdentityDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                existing = session.get(BenchmarkIdentityORM, seed_digest)
                if existing is None:
                    return None
                return self._to_dto(existing)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_episode_plan(self, plan: EpisodePlanDTO) -> EpisodePlanDTO:
        return self.save_episode_plans((plan,))[0]

    def save_episode_plans(
        self,
        plans: Sequence[EpisodePlanDTO],
    ) -> tuple[EpisodePlanDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                return self._save_episode_plans_in_session(session, tuple(plans))
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_episode_plan(self, episode_id: str) -> EpisodePlanDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                existing = session.get(EpisodePlanORM, episode_id)
                if existing is None:
                    return None
                return self._to_episode_plan_dto(existing)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_episode_plans(
        self,
        *,
        benchmark_seed_digest: str,
        split: str,
    ) -> tuple[EpisodePlanDTO, ...]:
        session = self._session_factory()
        try:
            with session.begin():
                rows = session.scalars(
                    select(EpisodePlanORM)
                    .where(
                        EpisodePlanORM.benchmark_seed_digest == benchmark_seed_digest,
                        EpisodePlanORM.split == split,
                    )
                    .order_by(EpisodePlanORM.ordinal, EpisodePlanORM.episode_id)
                ).all()
                return tuple(self._to_episode_plan_dto(row) for row in rows)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def save_sealed_split_plan(
        self,
        plan: SealedSplitPlanDTO,
    ) -> SealedSplitPlanDTO:
        session = self._session_factory()
        try:
            with session.begin():
                if session.get(BenchmarkIdentityORM, plan.seed_commitment) is None:
                    raise_unknown_benchmark_identity()
                existing = session.get(
                    SealedSplitPlanORM,
                    (plan.seed_commitment, plan.split),
                )
                if existing is not None:
                    existing_dto = self._to_sealed_split_plan_dto(session, existing)
                    if existing_dto != plan:
                        raise_sealed_split_plan_conflict()
                    return existing_dto
                self._save_episode_plans_in_session(session, plan.episodes)
                session.add(self._to_sealed_split_plan_orm(plan))
                return plan
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_sealed_split_plan(
        self,
        *,
        seed_commitment: str,
        split: str,
    ) -> SealedSplitPlanDTO | None:
        session = self._session_factory()
        try:
            with session.begin():
                existing = session.get(SealedSplitPlanORM, (seed_commitment, split))
                if existing is None:
                    return None
                return self._to_sealed_split_plan_dto(session, existing)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def create_final_authorization(
        self,
        authorization: FinalExecutionAuthorizationDTO,
        protocol: FinalEvaluationProtocolDTO,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO:
        self.assert_finalization_authority()
        session = self._session_factory()
        try:
            with session.begin():
                if authorization.authorization_status != "authorized":
                    raise_final_access_authorization()
                if (
                    not protocol.approved
                    or protocol.protocol_digest != authorization.protocol_digest
                ):
                    raise_final_access_authorization()
                if authorization.authorization_id != record.authorization_id:
                    raise_final_access_authorization()
                if authorization.authorization_digest != record.authorization_digest:
                    raise_final_access_authorization()
                if (
                    session.get(
                        FinalAccessAuthorizationORM,
                        authorization.authorization_id,
                    )
                    is not None
                ):
                    raise_final_access_conflict()
                if session.get(FinalAccessRecordORM, record.access_id) is not None:
                    raise_final_access_conflict()
                existing_for_seed = session.scalars(
                    select(FinalAccessRecordORM).where(
                        FinalAccessRecordORM.benchmark_seed_digest
                        == record.benchmark_seed_digest,
                        FinalAccessRecordORM.sealed_plan_digest
                        == record.sealed_plan_digest,
                    )
                ).first()
                if existing_for_seed is not None:
                    raise_final_access_conflict()
                if event.ordinal != 0 or event.previous_event_digest is not None:
                    raise_final_access_state()
                self._validate_final_record_event(
                    session,
                    record,
                    event,
                    expected_previous=None,
                )
                existing_protocol = session.get(
                    FinalEvaluationProtocolORM,
                    protocol.protocol_digest,
                )
                if existing_protocol is None:
                    session.add(self._to_final_protocol_orm(protocol))
                    session.flush()
                elif self._to_final_protocol_dto(existing_protocol) != protocol:
                    raise_final_access_conflict()
                session.add(self._to_final_authorization_orm(authorization))
                session.add(self._to_final_record_orm(record))
                session.flush()
                session.add(self._to_final_event_orm(event))
                return record
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def assert_finalization_authority(self) -> None:
        if self._finalization_engine is None:
            raise VPMValidationError(
                "final access requires a dedicated finalization database"
            )
        from zeromodel.persistence.sqlalchemy.db.session import (
            verify_finalization_authority,
        )

        verify_finalization_authority(self._finalization_engine)

    def load_final_evaluation_protocol(
        self,
        protocol_digest: str,
    ) -> FinalEvaluationProtocolDTO | None:
        self.assert_finalization_authority()
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(FinalEvaluationProtocolORM, protocol_digest)
                return None if row is None else self._to_final_protocol_dto(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_final_authorization(
        self,
        authorization_id: str,
    ) -> FinalExecutionAuthorizationDTO | None:
        self.assert_finalization_authority()
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(FinalAccessAuthorizationORM, authorization_id)
                return None if row is None else self._to_final_authorization_dto(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def load_final_access_record(
        self,
        access_id: str,
    ) -> FinalAccessRecordDTO | None:
        self.assert_finalization_authority()
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(FinalAccessRecordORM, access_id)
                return None if row is None else self._to_final_record_dto(row)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def list_final_access_events(
        self,
        access_id: str,
    ) -> tuple[FinalAccessEventDTO, ...]:
        self.assert_finalization_authority()
        session = self._session_factory()
        try:
            with session.begin():
                return self._final_events_for_session(session, access_id)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def reserve_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO:
        return self._transition_final_access(record, event)

    def mark_final_access_running(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO:
        return self._transition_final_access(record, event)

    def append_final_access_event(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO:
        return self._transition_final_access(record, event)

    def complete_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO:
        if record.state != "completed":
            raise_final_access_state()
        return self._transition_final_access(record, event)

    def fail_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
        failure: FinalExecutionFailureDTO,
    ) -> FinalAccessRecordDTO:
        if failure.access_id != record.access_id or failure.state != "failed":
            raise_final_access_state()
        return self._transition_final_access(record, event)

    def interrupt_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
        failure: FinalExecutionFailureDTO,
    ) -> FinalAccessRecordDTO:
        if failure.access_id != record.access_id or failure.state != "interrupted":
            raise_final_access_state()
        return self._transition_final_access(record, event)

    @staticmethod
    def _to_dto(identity: BenchmarkIdentityORM) -> BenchmarkIdentityDTO:
        return BenchmarkIdentityDTO(
            contract_commit=identity.contract_commit,
            seed_material=identity.seed_material,
            seed_digest=identity.seed_digest,
            policy_artifact_id=identity.policy_artifact_id,
            parent_audit_sha=identity.parent_audit_sha,
            parent_v3_sha=identity.parent_v3_sha,
        )

    @staticmethod
    def _to_orm(identity: BenchmarkIdentityDTO) -> BenchmarkIdentityORM:
        return BenchmarkIdentityORM(
            contract_commit=identity.contract_commit,
            seed_material=identity.seed_material,
            seed_digest=identity.seed_digest,
            policy_artifact_id=identity.policy_artifact_id,
            parent_audit_sha=identity.parent_audit_sha,
            parent_v3_sha=identity.parent_v3_sha,
        )

    def _save_episode_plans_in_session(
        self,
        session: Session,
        plans: Sequence[EpisodePlanDTO],
    ) -> tuple[EpisodePlanDTO, ...]:
        existing_dtos: dict[str, EpisodePlanDTO] = {}
        self._preflight_episode_plans(session, plans, existing_dtos)
        added_ids: set[str] = set()
        for plan in plans:
            if (
                plan.episode_id not in existing_dtos
                and plan.episode_id not in added_ids
            ):
                session.add(self._to_episode_plan_orm(plan))
                added_ids.add(plan.episode_id)
        return tuple(existing_dtos.get(plan.episode_id, plan) for plan in plans)

    def _preflight_episode_plans(
        self,
        session: Session,
        plans: Sequence[EpisodePlanDTO],
        existing_dtos: dict[str, EpisodePlanDTO],
    ) -> None:
        seen_ids: dict[str, EpisodePlanDTO] = {}
        seen_ordinals: dict[tuple[str, str, int], EpisodePlanDTO] = {}
        for plan in plans:
            if session.get(BenchmarkIdentityORM, plan.benchmark_seed_digest) is None:
                raise_unknown_benchmark_identity()
            existing = session.get(EpisodePlanORM, plan.episode_id)
            if existing is not None:
                existing_dto = self._to_episode_plan_dto(existing)
                if existing_dto != plan:
                    raise_episode_plan_conflict()
                existing_dtos[plan.episode_id] = existing_dto
            ordinal_key = (plan.benchmark_seed_digest, plan.split, plan.ordinal)
            existing_for_ordinal = self._episode_plan_for_ordinal(
                session,
                ordinal_key,
            )
            if existing_for_ordinal is not None:
                ordinal_dto = self._to_episode_plan_dto(existing_for_ordinal)
                if ordinal_dto != plan:
                    raise_episode_plan_conflict()
            self._preflight_batch_keys(plan, seen_ids, seen_ordinals, ordinal_key)

    @staticmethod
    def _preflight_batch_keys(
        plan: EpisodePlanDTO,
        seen_ids: dict[str, EpisodePlanDTO],
        seen_ordinals: dict[tuple[str, str, int], EpisodePlanDTO],
        ordinal_key: tuple[str, str, int],
    ) -> None:
        seen = seen_ids.get(plan.episode_id)
        if seen is not None and seen != plan:
            raise_episode_plan_conflict()
        seen_ids[plan.episode_id] = plan
        seen_for_ordinal = seen_ordinals.get(ordinal_key)
        if seen_for_ordinal is not None and seen_for_ordinal != plan:
            raise_episode_plan_conflict()
        seen_ordinals[ordinal_key] = plan

    @staticmethod
    def _episode_plan_for_ordinal(
        session: Session,
        ordinal_key: tuple[str, str, int],
    ) -> EpisodePlanORM | None:
        benchmark_seed_digest, split, ordinal = ordinal_key
        return session.scalars(
            select(EpisodePlanORM).where(
                EpisodePlanORM.benchmark_seed_digest == benchmark_seed_digest,
                EpisodePlanORM.split == split,
                EpisodePlanORM.ordinal == ordinal,
            )
        ).first()

    @staticmethod
    def _to_episode_plan_dto(row: EpisodePlanORM) -> EpisodePlanDTO:
        payload = _json_mapping(row.payload_json, "episode plan digest mismatch")
        dto = EpisodePlanDTO.from_dict(payload)
        if (
            row.episode_id != dto.episode_id
            or row.benchmark_seed_digest != dto.benchmark_seed_digest
            or row.plan_digest != dto.plan_digest
            or row.version != dto.version
            or row.seed_derivation_version != dto.seed_derivation_version
            or row.split != dto.split
            or row.ordinal != dto.ordinal
            or row.family_label != dto.family_label
            or row.family_ordinal != dto.family_ordinal
            or row.episode_disposition != dto.episode_disposition
            or row.denominator_class != dto.denominator_class
            or row.mutation_kind != dto.mutation_kind
            or row.source_row_id != dto.source_row_id
            or row.secondary_row_id != dto.secondary_row_id
            or row.derived_seed_identity != dto.derived_seed_identity
            or _episode_seed_from_hex(row.episode_seed_hex) != dto.episode_seed
            or row.frame_count != dto.frame_count
        ):
            raise VPMValidationError("episode plan digest mismatch")
        return dto

    @staticmethod
    def _to_episode_plan_orm(plan: EpisodePlanDTO) -> EpisodePlanORM:
        return EpisodePlanORM(
            episode_id=plan.episode_id,
            benchmark_seed_digest=plan.benchmark_seed_digest,
            plan_digest=plan.plan_digest,
            version=plan.version,
            seed_derivation_version=plan.seed_derivation_version,
            split=plan.split,
            ordinal=plan.ordinal,
            family_label=plan.family_label,
            family_ordinal=plan.family_ordinal,
            episode_disposition=plan.episode_disposition,
            denominator_class=plan.denominator_class,
            mutation_kind=plan.mutation_kind,
            source_row_id=plan.source_row_id,
            secondary_row_id=plan.secondary_row_id,
            derived_seed_identity=plan.derived_seed_identity,
            episode_seed_hex=_episode_seed_hex(plan.episode_seed),
            frame_count=plan.frame_count,
            payload_json=canonical_json_text(plan.to_dict()),
        )

    def _to_sealed_split_plan_dto(
        self,
        session: Session,
        row: SealedSplitPlanORM,
    ) -> SealedSplitPlanDTO:
        episodes = self.list_episode_plans_for_session(
            session,
            benchmark_seed_digest=row.seed_commitment,
            split=row.split,
        )
        payload = {
            "version": row.version,
            "seed_derivation_version": row.seed_derivation_version,
            "split": row.split,
            "plan_only": row.plan_only,
            "materialization_prohibited": row.materialization_prohibited,
            "episode_counts": _json_mapping(
                row.episode_counts_json,
                "sealed plan episode counts mismatch",
            ),
            "frame_count": row.frame_count,
            "sealed_episode_ids": _json_mapping(
                row.sealed_episode_ids_json,
                "sealed plan episode id manifest mismatch",
            ),
            "episodes": [episode.to_dict() for episode in episodes],
            "seed_commitment": row.seed_commitment,
            "sealed_plan_digest": row.sealed_plan_digest,
        }
        return SealedSplitPlanDTO.from_dict(payload)

    def list_episode_plans_for_session(
        self,
        session: Session,
        *,
        benchmark_seed_digest: str,
        split: str,
    ) -> tuple[EpisodePlanDTO, ...]:
        rows = session.scalars(
            select(EpisodePlanORM)
            .where(
                EpisodePlanORM.benchmark_seed_digest == benchmark_seed_digest,
                EpisodePlanORM.split == split,
            )
            .order_by(EpisodePlanORM.ordinal, EpisodePlanORM.episode_id)
        ).all()
        return tuple(self._to_episode_plan_dto(row) for row in rows)

    @staticmethod
    def _to_sealed_split_plan_orm(plan: SealedSplitPlanDTO) -> SealedSplitPlanORM:
        return SealedSplitPlanORM(
            seed_commitment=plan.seed_commitment,
            split=plan.split,
            version=plan.version,
            seed_derivation_version=plan.seed_derivation_version,
            plan_only=plan.plan_only,
            materialization_prohibited=plan.materialization_prohibited,
            frame_count=plan.frame_count,
            episode_counts_json=canonical_json_text(plan.episode_counts.to_dict()),
            sealed_episode_ids_json=canonical_json_text(
                plan.sealed_episode_ids.to_dict()
            ),
            sealed_plan_digest=plan.sealed_plan_digest,
        )

    def _transition_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO:
        self.assert_finalization_authority()
        session = self._session_factory()
        try:
            with session.begin():
                row = session.get(FinalAccessRecordORM, record.access_id)
                if row is None:
                    raise_final_access_authorization()
                existing = self._to_final_record_dto(row)
                self._validate_final_identity(existing, record)
                self._validate_final_record_event(
                    session,
                    record,
                    event,
                    expected_previous=existing,
                )
                result = session.execute(
                    update(FinalAccessRecordORM)
                    .where(
                        FinalAccessRecordORM.access_id == record.access_id,
                        FinalAccessRecordORM.state == existing.state,
                        FinalAccessRecordORM.current_event_ordinal
                        == existing.current_event_ordinal,
                        FinalAccessRecordORM.last_event_digest
                        == existing.last_event_digest,
                        FinalAccessRecordORM.record_digest == existing.record_digest,
                    )
                    .values(
                        state=record.state,
                        updated_utc=record.updated_utc,
                        process_identity=record.process_identity,
                        current_event_ordinal=record.current_event_ordinal,
                        last_event_digest=record.last_event_digest,
                        record_digest=record.record_digest,
                        payload_json=canonical_json_text(record.to_dict()),
                    )
                )
                # session.execute() on a Core UPDATE statement always returns a
                # CursorResult at runtime, but its static return type is the
                # more generic Result[Any], which does not declare `rowcount`.
                # Narrow it explicitly instead of asserting the attribute exists.
                if not isinstance(result, CursorResult) or result.rowcount != 1:
                    raise_final_access_state()
                session.add(self._to_final_event_orm(event))
                return record
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _validate_final_record_event(
        self,
        session: Session,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
        *,
        expected_previous: FinalAccessRecordDTO | None,
    ) -> None:
        previous_state = None if expected_previous is None else expected_previous.state
        previous_event_digest = (
            None if expected_previous is None else expected_previous.last_event_digest
        )
        if (
            event.access_id != record.access_id
            or event.authorization_id != record.authorization_id
            or event.previous_state != previous_state
            or event.new_state != record.state
            or event.previous_event_digest != previous_event_digest
            or record.last_event_digest != event.event_digest
        ):
            raise_final_access_state()
        payload = event.event_payload.to_value()
        kind = payload.get("kind") if isinstance(payload, Mapping) else None
        validate_final_access_event(previous_state, record.state, kind)
        events = self._final_events_for_session(session, record.access_id)
        expected_ordinal = (
            -1 if expected_previous is None else expected_previous.current_event_ordinal
        )
        if (
            event.ordinal != len(events)
            or event.ordinal != expected_ordinal + 1
            or record.current_event_ordinal != event.ordinal
        ):
            raise_final_access_state()
        if (
            expected_previous is not None
            and previous_state in FINAL_ACCESS_TERMINAL_STATES
            and record.state != previous_state
        ):
            raise_final_access_state()

    @staticmethod
    def _validate_final_identity(
        existing: FinalAccessRecordDTO,
        record: FinalAccessRecordDTO,
    ) -> None:
        if (
            existing.access_id != record.access_id
            or existing.authorization_id != record.authorization_id
            or existing.benchmark_seed_digest != record.benchmark_seed_digest
            or existing.sealed_plan_digest != record.sealed_plan_digest
            or existing.protocol_digest != record.protocol_digest
            or existing.authorization_digest != record.authorization_digest
            or existing.created_utc != record.created_utc
        ):
            raise_final_access_conflict()

    @staticmethod
    def _to_final_protocol_orm(
        protocol: FinalEvaluationProtocolDTO,
    ) -> FinalEvaluationProtocolORM:
        return FinalEvaluationProtocolORM(
            protocol_digest=protocol.protocol_digest,
            protocol_id=protocol.protocol_id,
            protocol_status=protocol.protocol_status,
            benchmark_seed_digest=protocol.benchmark_seed_digest,
            sealed_plan_digest=protocol.sealed_plan_digest,
            payload_json=canonical_json_text(protocol.to_dict()),
        )

    @staticmethod
    def _to_final_protocol_dto(
        row: FinalEvaluationProtocolORM,
    ) -> FinalEvaluationProtocolDTO:
        dto = FinalEvaluationProtocolDTO.from_dict(
            _json_mapping(row.payload_json, "final protocol digest mismatch")
        )
        if (
            row.protocol_digest != dto.protocol_digest
            or row.protocol_id != dto.protocol_id
            or row.protocol_status != dto.protocol_status
            or row.benchmark_seed_digest != dto.benchmark_seed_digest
            or row.sealed_plan_digest != dto.sealed_plan_digest
        ):
            raise VPMValidationError("final protocol digest mismatch")
        return dto

    @staticmethod
    def _to_final_authorization_orm(
        authorization: FinalExecutionAuthorizationDTO,
    ) -> FinalAccessAuthorizationORM:
        return FinalAccessAuthorizationORM(
            authorization_id=authorization.authorization_id,
            authorization_status=authorization.authorization_status,
            authorization_digest=authorization.authorization_digest,
            protocol_digest=authorization.protocol_digest,
            benchmark_seed_digest=authorization.expected_benchmark_seed_digest,
            sealed_plan_digest=authorization.expected_sealed_plan_digest,
            created_utc=authorization.created_utc,
            payload_json=canonical_json_text(authorization.to_dict()),
        )

    @staticmethod
    def _to_final_authorization_dto(
        row: FinalAccessAuthorizationORM,
    ) -> FinalExecutionAuthorizationDTO:
        dto = FinalExecutionAuthorizationDTO.from_dict(
            _json_mapping(
                row.payload_json,
                "final authorization digest mismatch",
            )
        )
        if (
            row.authorization_id != dto.authorization_id
            or row.authorization_status != dto.authorization_status
            or row.authorization_digest != dto.authorization_digest
            or row.protocol_digest != dto.protocol_digest
            or row.benchmark_seed_digest != dto.expected_benchmark_seed_digest
            or row.sealed_plan_digest != dto.expected_sealed_plan_digest
            or row.created_utc != dto.created_utc
        ):
            raise VPMValidationError("final authorization digest mismatch")
        return dto

    @staticmethod
    def _to_final_record_orm(record: FinalAccessRecordDTO) -> FinalAccessRecordORM:
        return FinalAccessRecordORM(
            access_id=record.access_id,
            authorization_id=record.authorization_id,
            state=record.state,
            benchmark_seed_digest=record.benchmark_seed_digest,
            sealed_plan_digest=record.sealed_plan_digest,
            protocol_digest=record.protocol_digest,
            authorization_digest=record.authorization_digest,
            created_utc=record.created_utc,
            updated_utc=record.updated_utc,
            process_identity=record.process_identity,
            current_event_ordinal=record.current_event_ordinal,
            last_event_digest=record.last_event_digest,
            record_digest=record.record_digest,
            payload_json=canonical_json_text(record.to_dict()),
        )

    @staticmethod
    def _to_final_record_dto(row: FinalAccessRecordORM) -> FinalAccessRecordDTO:
        dto = FinalAccessRecordDTO.from_dict(
            _json_mapping(row.payload_json, "final access record digest mismatch")
        )
        if (
            row.access_id != dto.access_id
            or row.authorization_id != dto.authorization_id
            or row.state != dto.state
            or row.benchmark_seed_digest != dto.benchmark_seed_digest
            or row.sealed_plan_digest != dto.sealed_plan_digest
            or row.protocol_digest != dto.protocol_digest
            or row.authorization_digest != dto.authorization_digest
            or row.created_utc != dto.created_utc
            or row.updated_utc != dto.updated_utc
            or row.process_identity != dto.process_identity
            or row.current_event_ordinal != dto.current_event_ordinal
            or row.last_event_digest != dto.last_event_digest
            or row.record_digest != dto.record_digest
        ):
            raise VPMValidationError("final access record digest mismatch")
        return dto

    @staticmethod
    def _update_final_record_row(
        row: FinalAccessRecordORM,
        record: FinalAccessRecordDTO,
    ) -> None:
        row.state = record.state
        row.updated_utc = record.updated_utc
        row.process_identity = record.process_identity
        row.current_event_ordinal = record.current_event_ordinal
        row.last_event_digest = record.last_event_digest
        row.record_digest = record.record_digest
        row.payload_json = canonical_json_text(record.to_dict())

    @staticmethod
    def _to_final_event_orm(event: FinalAccessEventDTO) -> FinalAccessEventORM:
        return FinalAccessEventORM(
            event_digest=event.event_digest,
            access_id=event.access_id,
            authorization_id=event.authorization_id,
            ordinal=event.ordinal,
            previous_state=event.previous_state,
            new_state=event.new_state,
            utc=event.utc,
            process_identity=event.process_identity,
            previous_event_digest=event.previous_event_digest,
            payload_json=canonical_json_text(event.to_dict()),
        )

    @staticmethod
    def _to_final_event_dto(row: FinalAccessEventORM) -> FinalAccessEventDTO:
        dto = FinalAccessEventDTO.from_dict(
            _json_mapping(row.payload_json, "final access event digest mismatch")
        )
        if (
            row.event_digest != dto.event_digest
            or row.access_id != dto.access_id
            or row.authorization_id != dto.authorization_id
            or row.ordinal != dto.ordinal
            or row.previous_state != dto.previous_state
            or row.new_state != dto.new_state
            or row.utc != dto.utc
            or row.process_identity != dto.process_identity
            or row.previous_event_digest != dto.previous_event_digest
        ):
            raise VPMValidationError("final access event digest mismatch")
        return dto

    def _final_events_for_session(
        self,
        session: Session,
        access_id: str,
    ) -> tuple[FinalAccessEventDTO, ...]:
        rows = session.scalars(
            select(FinalAccessEventORM)
            .where(FinalAccessEventORM.access_id == access_id)
            .order_by(FinalAccessEventORM.ordinal)
        ).all()
        return tuple(self._to_final_event_dto(row) for row in rows)


def _json_mapping(text: str, message: str) -> Mapping[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VPMValidationError(message) from exc
    if not isinstance(value, Mapping):
        raise VPMValidationError(message)
    return cast(Mapping[str, object], value)


def _episode_seed_hex(seed: int) -> str:
    if seed < 0 or seed >= 2**64:
        raise VPMValidationError("episode plan root seed lineage mismatch")
    return f"{seed:016x}"


def _episode_seed_from_hex(value: str) -> int:
    if len(value) != 16 or any(item not in "0123456789abcdef" for item in value):
        raise VPMValidationError("episode plan digest mismatch")
    return int(value, 16)


__all__ = ["SqlAlchemyVideoActionSetStore"]
