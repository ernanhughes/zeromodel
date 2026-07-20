from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import cast

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from ...artifact import VPMValidationError
from ...domains.video_action_set.canonical_json import canonical_json_text
from ...domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    EpisodePlanDTO,
    SealedSplitPlanDTO,
)
from ...domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
    ObservationOperationChainDTO,
)
from ...domains.video_action_set.store import (
    VideoActionSetStore,
    raise_matrix_blob_conflict,
    raise_observation_conflict,
    raise_episode_plan_conflict,
    raise_identity_conflict,
    raise_sealed_split_plan_conflict,
    raise_unknown_benchmark_identity,
    raise_unknown_episode_plan,
)
from ...matrix_blob import MatrixBlob
from ..orm.video_action_set import (
    BenchmarkIdentityORM,
    EpisodePlanORM,
    MatrixBlobORM,
    ObservationORM,
    ObservationOperationORM,
    SealedSplitPlanORM,
)
from .video_action_set_observation import (
    chain_for_frame,
    matrix_blob_for_observation,
    observation_select,
    operation_observation_select,
    optional_observation_predicates,
    preflight_observation_sequence,
    to_matrix_blob,
    to_matrix_blob_orm,
    to_observation_dto,
    to_observation_orm,
    to_operation_chain_orm,
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
                return self._save_observations_in_session(session, tuple(observations))
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
                observation = to_observation_dto(session, row)
                blob = matrix_blob_for_observation(session, observation)
                return MaterializedObservationDTO(observation, blob)
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
                return tuple(to_observation_dto(session, row) for row in rows)
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
                return tuple(to_observation_dto(session, row) for row in rows)
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
                return tuple(to_observation_dto(session, row) for row in rows)
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
                    operation_observation_select().where(
                        ObservationOperationORM.input_digests_json.contains(
                            f'"{input_digest}"'
                        )
                    )
                ).all()
                return tuple(to_observation_dto(session, row) for row in rows)
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def _save_observations_in_session(
        self,
        session: Session,
        observations: Sequence[MaterializedObservationDTO],
    ) -> tuple[ObservationDTO, ...]:
        existing = self._preflight_observations(session, observations)
        for item in observations:
            observation = item.observation
            if observation.frame_id in existing:
                continue
            if item.matrix_blob is not None:
                self._save_matrix_blob_in_session(session, item.matrix_blob)
            session.add(to_observation_orm(observation))
            session.add(to_operation_chain_orm(observation))
            session.add_all(to_operation_orms(observation))
        return tuple(
            existing.get(item.observation.frame_id, item.observation)
            for item in observations
        )

    def _preflight_observations(
        self,
        session: Session,
        observations: Sequence[MaterializedObservationDTO],
    ) -> dict[str, ObservationDTO]:
        existing_observations: dict[str, ObservationDTO] = {}
        seen_observations: dict[str, ObservationDTO] = {}
        seen_sequences: dict[tuple[str, int], str] = {}
        seen_blobs: dict[str, MatrixBlob] = {}
        for item in observations:
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


class SqlAlchemyVideoActionSetStore(_ObservationSqlStoreMixin, VideoActionSetStore):
    def __init__(self, session_factory: sessionmaker[Session]) -> None:
        self._session_factory = session_factory

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
            or row.episode_seed != dto.episode_seed
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
            episode_seed=plan.episode_seed,
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


def _json_mapping(text: str, message: str) -> Mapping[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise VPMValidationError(message) from exc
    if not isinstance(value, Mapping):
        raise VPMValidationError(message)
    return cast(Mapping[str, object], value)


__all__ = ["SqlAlchemyVideoActionSetStore"]
