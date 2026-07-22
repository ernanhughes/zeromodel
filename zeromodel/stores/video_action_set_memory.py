from __future__ import annotations

from collections.abc import Sequence
from threading import RLock

from ..domains.video_action_set.dto import (
    BenchmarkIdentityDTO,
    EpisodePlanDTO,
    SealedSplitPlanDTO,
)
from ..domains.video_action_set.final_access_dto import (
    FINAL_ACCESS_TERMINAL_STATES,
    FinalAccessEventDTO,
    FinalAccessRecordDTO,
    FinalEvaluationProtocolDTO,
    FinalExecutionAuthorizationDTO,
    FinalExecutionFailureDTO,
    validate_final_access_event,
)
from ..domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
    ObservationDTO,
    ObservationOperationChainDTO,
)
from ..domains.video_action_set.store import (
    VideoActionSetStore,
    raise_episode_plan_conflict,
    raise_final_access_authorization,
    raise_final_access_conflict,
    raise_final_access_state,
    raise_identity_conflict,
    raise_matrix_blob_conflict,
    raise_observation_conflict,
    raise_observation_sequence_conflict,
    raise_sealed_split_plan_conflict,
    raise_unknown_benchmark_identity,
    raise_unknown_episode_plan,
)
from ..matrix_blob import MatrixBlob


class InMemoryVideoActionSetStore(VideoActionSetStore):
    def __init__(self) -> None:
        self._identities: dict[str, BenchmarkIdentityDTO] = {}
        self._episode_plans: dict[str, EpisodePlanDTO] = {}
        self._sealed_split_plans: dict[tuple[str, str], SealedSplitPlanDTO] = {}
        self._matrix_blobs: dict[str, MatrixBlob] = {}
        self._observations: dict[str, ObservationDTO] = {}
        self._observation_sequence_index: dict[tuple[str, int], str] = {}
        self._final_authorizations: dict[str, FinalExecutionAuthorizationDTO] = {}
        self._final_protocols: dict[str, FinalEvaluationProtocolDTO] = {}
        self._final_access_records: dict[str, FinalAccessRecordDTO] = {}
        self._final_events: dict[str, list[FinalAccessEventDTO]] = {}
        self._final_seed_sealed_index: dict[tuple[str, str], str] = {}
        self._final_lock = RLock()

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

    def save_matrix_blob(self, blob: MatrixBlob) -> MatrixBlob:
        existing = self._matrix_blobs.get(blob.blob_id)
        if existing is not None:
            if existing != blob:
                raise_matrix_blob_conflict()
            return existing
        self._matrix_blobs[blob.blob_id] = blob
        return blob

    def get_matrix_blob(self, blob_id: str) -> MatrixBlob | None:
        return self._matrix_blobs.get(blob_id)

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
        return self._save_observations(tuple(observations), final_access=None)

    def save_authorized_final_observations(
        self,
        access: FinalAccessRecordDTO,
        observations: Sequence[MaterializedObservationDTO],
    ) -> tuple[ObservationDTO, ...]:
        return self._save_observations(tuple(observations), final_access=access)

    def _save_observations(
        self,
        observations: Sequence[MaterializedObservationDTO],
        *,
        final_access: FinalAccessRecordDTO | None,
    ) -> tuple[ObservationDTO, ...]:
        observation_tuple = tuple(observations)
        existing = self._preflight_observations(
            observation_tuple,
            final_access=final_access,
        )
        inserted_frame_ids: set[str] = set()
        for item in observation_tuple:
            if (
                item.observation.frame_id in existing
                or item.observation.frame_id in inserted_frame_ids
            ):
                continue
            if item.matrix_blob is not None:
                self._matrix_blobs.setdefault(
                    item.matrix_blob.blob_id,
                    item.matrix_blob,
                )
            self._observations[item.observation.frame_id] = item.observation
            self._observation_sequence_index[_sequence_key(item.observation)] = (
                item.observation.frame_id
            )
            inserted_frame_ids.add(item.observation.frame_id)
        return tuple(
            existing.get(item.observation.frame_id, item.observation)
            for item in observation_tuple
        )

    def get_observation(self, frame_id: str) -> ObservationDTO | None:
        return self._observations.get(frame_id)

    def get_materialized_observation(
        self,
        frame_id: str,
    ) -> MaterializedObservationDTO | None:
        observation = self._observations.get(frame_id)
        if observation is None:
            return None
        blob = (
            None
            if observation.matrix_blob_id is None
            else self._matrix_blobs.get(observation.matrix_blob_id)
        )
        return MaterializedObservationDTO(
            observation,
            blob,
            final_access_id=observation.final_access_id,
        )

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
        return tuple(
            MaterializedObservationDTO(
                observation,
                None
                if observation.matrix_blob_id is None
                else self._matrix_blobs.get(observation.matrix_blob_id),
                final_access_id=observation.final_access_id,
            )
            for observation in self.list_observations(
                benchmark_seed_digest=benchmark_seed_digest,
                split=split,
                episode_id=episode_id,
                family=family,
                event_type=event_type,
                denominator_class=denominator_class,
                has_pixels=has_pixels,
            )
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
        return tuple(
            sorted(
                (
                    observation
                    for observation in self._observations.values()
                    if _matches_observation(
                        observation,
                        benchmark_seed_digest=benchmark_seed_digest,
                        split=split,
                        episode_id=episode_id,
                        family=family,
                        event_type=event_type,
                        denominator_class=denominator_class,
                        has_pixels=has_pixels,
                    )
                ),
                key=_observation_sort_key,
            )
        )

    def get_operation_chain(
        self,
        frame_id: str,
    ) -> ObservationOperationChainDTO | None:
        observation = self._observations.get(frame_id)
        return None if observation is None else observation.operation_chain

    def list_observations_by_operation(
        self,
        *,
        operation: str,
        split: str | None = None,
        family: str | None = None,
    ) -> tuple[ObservationDTO, ...]:
        return tuple(
            observation
            for observation in self.list_observations(split=split, family=family)
            if any(
                item.operation == operation
                for item in observation.operation_chain.operations
            )
        )

    def list_observations_by_output_digest(
        self,
        output_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        return tuple(
            observation
            for observation in self.list_observations()
            if any(
                item.output_digest == output_digest
                for item in observation.operation_chain.operations
            )
        )

    def list_observation_consumers_of_digest(
        self,
        input_digest: str,
    ) -> tuple[ObservationDTO, ...]:
        return tuple(
            observation
            for observation in self.list_observations()
            if any(
                input_digest in item.input_digests
                for item in observation.operation_chain.operations
            )
        )

    def create_final_authorization(
        self,
        authorization: FinalExecutionAuthorizationDTO,
        protocol: FinalEvaluationProtocolDTO,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO:
        with self._final_lock:
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
            seed_key = (record.benchmark_seed_digest, record.sealed_plan_digest)
            if authorization.authorization_id in self._final_authorizations:
                raise_final_access_conflict()
            if record.access_id in self._final_access_records:
                raise_final_access_conflict()
            if seed_key in self._final_seed_sealed_index:
                raise_final_access_conflict()
            if event.ordinal != 0 or event.previous_event_digest is not None:
                raise_final_access_state()
            self._validate_record_event(record, event, expected_previous=None)
            self._final_protocols[protocol.protocol_digest] = protocol
            self._final_authorizations[authorization.authorization_id] = authorization
            self._final_access_records[record.access_id] = record
            self._final_events[record.access_id] = [event]
            self._final_seed_sealed_index[seed_key] = record.access_id
            return record

    def assert_finalization_authority(self) -> None:
        return None

    def load_final_evaluation_protocol(
        self,
        protocol_digest: str,
    ) -> FinalEvaluationProtocolDTO | None:
        with self._final_lock:
            return self._final_protocols.get(protocol_digest)

    def load_final_authorization(
        self,
        authorization_id: str,
    ) -> FinalExecutionAuthorizationDTO | None:
        with self._final_lock:
            return self._final_authorizations.get(authorization_id)

    def load_final_access_record(
        self,
        access_id: str,
    ) -> FinalAccessRecordDTO | None:
        with self._final_lock:
            return self._final_access_records.get(access_id)

    def list_final_access_events(
        self,
        access_id: str,
    ) -> tuple[FinalAccessEventDTO, ...]:
        with self._final_lock:
            return tuple(self._final_events.get(access_id, ()))

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

    def _transition_final_access(
        self,
        record: FinalAccessRecordDTO,
        event: FinalAccessEventDTO,
    ) -> FinalAccessRecordDTO:
        with self._final_lock:
            existing = self._final_access_records.get(record.access_id)
            if existing is None:
                raise_final_access_authorization()
            self._validate_final_identity(existing, record)
            self._validate_record_event(record, event, expected_previous=existing)
            self._final_access_records[record.access_id] = record
            self._final_events[record.access_id].append(event)
            return record

    def _validate_record_event(
        self,
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
        kind = payload.get("kind") if isinstance(payload, dict) else None
        validate_final_access_event(previous_state, record.state, kind)
        existing_events = self._final_events.get(record.access_id, [])
        expected_ordinal = (
            -1 if expected_previous is None else expected_previous.current_event_ordinal
        )
        if (
            event.ordinal != len(existing_events)
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

    def _preflight_observations(
        self,
        observations: Sequence[MaterializedObservationDTO],
        *,
        final_access: FinalAccessRecordDTO | None,
    ) -> dict[str, ObservationDTO]:
        existing_observations: dict[str, ObservationDTO] = {}
        seen_observations: dict[str, ObservationDTO] = {}
        seen_sequences: dict[tuple[str, int], str] = {}
        seen_blobs: dict[str, MatrixBlob] = {}
        for item in observations:
            observation = item.observation
            self._preflight_final_observation_access(observation, final_access)
            self._preflight_observation_ownership(observation)
            self._preflight_observation_identity(
                observation,
                existing_observations,
                seen_observations,
                seen_sequences,
            )
            self._preflight_blob(item.matrix_blob, seen_blobs)
        return existing_observations

    def _preflight_final_observation_access(
        self,
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
        stored = self._final_access_records.get(final_access.access_id)
        if stored != final_access or final_access.state != "running":
            raise_final_access_state()

    def _preflight_observation_ownership(
        self,
        observation: ObservationDTO,
    ) -> None:
        if observation.benchmark_seed_digest not in self._identities:
            raise_unknown_benchmark_identity()
        episode = self._episode_plans.get(observation.episode_id)
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
        observation: ObservationDTO,
        existing_observations: dict[str, ObservationDTO],
        seen_observations: dict[str, ObservationDTO],
        seen_sequences: dict[tuple[str, int], str],
    ) -> None:
        existing = self._observations.get(observation.frame_id)
        if existing is not None:
            if existing != observation:
                raise_observation_conflict()
            existing_observations[observation.frame_id] = existing
        seen = seen_observations.get(observation.frame_id)
        if seen is not None and seen != observation:
            raise_observation_conflict()
        seen_observations[observation.frame_id] = observation
        self._preflight_sequence(observation, seen_sequences)

    def _preflight_sequence(
        self,
        observation: ObservationDTO,
        seen_sequences: dict[tuple[str, int], str],
    ) -> None:
        sequence_key = _sequence_key(observation)
        existing_frame_id = self._observation_sequence_index.get(sequence_key)
        if existing_frame_id is not None and existing_frame_id != observation.frame_id:
            raise_observation_sequence_conflict()
        seen_frame_id = seen_sequences.get(sequence_key)
        if seen_frame_id is not None and seen_frame_id != observation.frame_id:
            raise_observation_sequence_conflict()
        seen_sequences[sequence_key] = observation.frame_id

    def _preflight_blob(
        self,
        blob: MatrixBlob | None,
        seen_blobs: dict[str, MatrixBlob],
    ) -> None:
        if blob is None:
            return
        existing = self._matrix_blobs.get(blob.blob_id)
        if existing is not None and existing != blob:
            raise_matrix_blob_conflict()
        seen = seen_blobs.get(blob.blob_id)
        if seen is not None and seen != blob:
            raise_matrix_blob_conflict()
        seen_blobs[blob.blob_id] = blob


def _sequence_key(observation: ObservationDTO) -> tuple[str, int]:
    return (observation.episode_id, observation.sequence_number)


def _observation_sort_key(observation: ObservationDTO) -> tuple[str, str, int, str]:
    return (
        observation.split,
        observation.episode_id,
        observation.sequence_number,
        observation.frame_id,
    )


def _matches_observation(
    observation: ObservationDTO,
    *,
    benchmark_seed_digest: str | None,
    split: str | None,
    episode_id: str | None,
    family: str | None,
    event_type: str | None,
    denominator_class: str | None,
    has_pixels: bool | None,
) -> bool:
    if benchmark_seed_digest is not None and (
        observation.benchmark_seed_digest != benchmark_seed_digest
    ):
        return False
    if split is not None and observation.split != split:
        return False
    if episode_id is not None and observation.episode_id != episode_id:
        return False
    if family is not None and observation.family != family:
        return False
    if event_type is not None and observation.event_type != event_type:
        return False
    if (
        denominator_class is not None
        and observation.denominator_class != denominator_class
    ):
        return False
    if has_pixels is not None and observation.has_pixels != has_pixels:
        return False
    return True


__all__ = ["InMemoryVideoActionSetStore"]
