from __future__ import annotations

from copy import deepcopy

import pytest
from sqlalchemy import func, inspect, select

from test_video_episode_plan_sql_store import sql_safe_identity
from test_video_observation_rmdto import (
    _pixels,
    assert_records_equivalent,
    sample_gap_record,
    sample_record,
)
from zeromodel.artifact import VPMValidationError
from zeromodel.db.orm.video_action_set import (
    MatrixBlobORM,
    ObservationORM,
    ObservationOperationChainORM,
    ObservationOperationORM,
)
from zeromodel.db.runtime import build_sqlite_runtime
from zeromodel.db.session import (
    create_database_engine,
    create_schema,
    create_session_factory,
)
from zeromodel.db.stores.video_action_set import SqlAlchemyVideoActionSetStore
from zeromodel.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
)
from zeromodel.domains.video_action_set.store import (
    OBSERVATION_CONFLICT_MESSAGE,
)

from test_video_episode_plan_rmdto import plan_dto


pytestmark = pytest.mark.integration


def build_store():
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    return SqlAlchemyVideoActionSetStore(session_factory), session_factory, engine


def _save_identity_plan(store: SqlAlchemyVideoActionSetStore, frame_count: int = 3):
    identity = sql_safe_identity("sql-observation-rmdto-seed")
    plan = plan_dto(identity=identity, split="development", frame_count=frame_count)
    store.save_identity(identity)
    store.save_episode_plan(plan)
    return identity, plan


def test_sql_schema_creation_registers_observation_tables() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    assert "video_action_set_observation" not in inspect(engine).get_table_names()

    create_schema(engine)
    table_names = set(inspect(engine).get_table_names())

    assert "matrix_blob" in table_names
    assert "video_action_set_observation" in table_names
    assert "video_action_set_observation_operation_chain" in table_names
    assert "video_action_set_observation_operation" in table_names


def test_sql_observation_round_trip_dedupes_blob_and_reconstructs_rows() -> None:
    store, session_factory, _engine = build_store()
    identity, plan = _save_identity_plan(store)
    first_record = sample_record(plan=plan, sequence_number=0, pixels=_pixels())
    second_record = sample_record(plan=plan, sequence_number=1, pixels=_pixels())
    gap_record = sample_gap_record(plan=plan, sequence_number=2)

    saved = store.save_observations(
        tuple(
            MaterializedObservationDTO.from_record(record)
            for record in (first_record, second_record, gap_record)
        )
    )

    assert len(saved) == 3
    read_store = SqlAlchemyVideoActionSetStore(session_factory)
    first_materialized = read_store.get_materialized_observation(saved[0].frame_id)
    assert first_materialized is not None
    assert_records_equivalent(first_record, first_materialized.to_record())
    assert read_store.get_operation_chain(saved[0].frame_id) == saved[0].operation_chain

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(MatrixBlobORM)) == 1
        assert session.scalar(select(func.count()).select_from(ObservationORM)) == 3
        assert (
            session.scalar(
                select(func.count()).select_from(ObservationOperationChainORM)
            )
            == 3
        )
        assert (
            session.scalar(select(func.count()).select_from(ObservationOperationORM))
            == 3
        )
        blob = session.get(MatrixBlobORM, saved[0].matrix_blob_id)
        assert blob is not None
        assert isinstance(blob.data, bytes)
        assert blob.byte_length == len(blob.data)

    assert (
        read_store.list_observations(
            benchmark_seed_digest=identity.seed_digest,
            split="development",
        )
        == saved
    )
    assert read_store.list_observations(
        family="bounded_translation", has_pixels=True
    ) == (
        saved[0],
        saved[1],
    )
    assert read_store.list_observations(event_type="gap_unknown", has_pixels=False) == (
        saved[2],
    )
    assert (
        read_store.list_observations_by_operation(operation="emit_observation") == saved
    )
    assert read_store.list_observations_by_output_digest(
        str(saved[0].observation_pixel_digest)
    ) == (saved[0], saved[1])
    assert read_store.list_observation_consumers_of_digest(
        str(saved[0].observation_pixel_digest)
    ) == (saved[0], saved[1])


def test_sql_observation_batch_rollback_leaves_no_orphan_rows() -> None:
    store, session_factory, _engine = build_store()
    _identity, plan = _save_identity_plan(store)
    first = MaterializedObservationDTO.from_record(
        sample_record(plan=plan, sequence_number=0, pixels=_pixels())
    )
    store.save_observations((first,))

    new_record = sample_record(plan=plan, sequence_number=1, pixels=_pixels(9))
    conflict_record = deepcopy(
        sample_record(plan=plan, sequence_number=0, pixels=_pixels())
    )
    conflict_record["expected_action"] = "right"
    batch = (
        MaterializedObservationDTO.from_record(new_record),
        MaterializedObservationDTO.from_record(conflict_record),
    )

    with pytest.raises(VPMValidationError, match=OBSERVATION_CONFLICT_MESSAGE):
        store.save_observations(batch)

    assert store.get_observation(batch[0].observation.frame_id) is None
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(ObservationORM)) == 1
        assert session.scalar(select(func.count()).select_from(MatrixBlobORM)) == 1
        assert (
            session.scalar(select(func.count()).select_from(ObservationOperationORM))
            == 1
        )


def test_sqlite_runtime_composes_observation_store() -> None:
    runtime = build_sqlite_runtime("sqlite:///:memory:", initialize_schema=True)
    identity = sql_safe_identity("sql-runtime-observation-rmdto-seed")
    plan = plan_dto(identity=identity, split="development", frame_count=1)
    record = sample_record(plan=plan, pixels=_pixels())

    runtime.video_action_set.save_identity(identity)
    runtime.video_action_set.save_episode_plan(plan)
    observation = runtime.video_action_set.save_observation_record(record)

    assert runtime.video_action_set.get_observation(observation.frame_id) == observation
    assert_records_equivalent(
        record,
        runtime.video_action_set.get_observation_record(observation.frame_id),
    )
