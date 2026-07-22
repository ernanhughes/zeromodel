from __future__ import annotations

from copy import deepcopy

import pytest
from sqlalchemy import event, func, inspect, select
from sqlalchemy.exc import IntegrityError

from test_video_episode_plan_sql_store import sql_identity
from test_video_observation_rmdto import (
    _operation,
    _pixels,
    assert_records_equivalent,
    sample_gap_record,
    sample_record,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.persistence.sqlalchemy.db.orm.video_action_set import (
    MatrixBlobORM,
    ObservationORM,
    ObservationOperationChainORM,
    ObservationOperationInputORM,
    ObservationOperationORM,
)
from zeromodel.persistence.sqlalchemy.db.runtime import build_sqlite_runtime
from zeromodel.persistence.sqlalchemy.db.session import (
    create_database_engine,
    create_schema,
    create_session_factory,
)
from zeromodel.persistence.sqlalchemy.db.stores.video_action_set import SqlAlchemyVideoActionSetStore
from zeromodel.video.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.video.domains.video_action_set.contracts import (
    OBSERVATION_OPERATION_CHAIN_VERSION,
)
from zeromodel.video.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
)
from zeromodel.video.domains.video_action_set.store import (
    OBSERVATION_CONFLICT_MESSAGE,
    OBSERVATION_SEQUENCE_CONFLICT_MESSAGE,
)
from zeromodel.video.stores.video_action_set_memory import InMemoryVideoActionSetStore

from test_video_episode_plan_rmdto import plan_dto


pytestmark = pytest.mark.integration


def build_store():
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    return SqlAlchemyVideoActionSetStore(session_factory), session_factory, engine


def _save_identity_plan(store: SqlAlchemyVideoActionSetStore, frame_count: int = 3):
    identity = sql_identity("sql-observation-rmdto-seed")
    plan = plan_dto(identity=identity, split="development", frame_count=frame_count)
    store.save_identity(identity)
    store.save_episode_plan(plan)
    return identity, plan


def _multi_operation_record(*, plan, sequence_number: int) -> dict[str, object]:
    record = sample_record(
        plan=plan,
        sequence_number=sequence_number,
        pixels=_pixels(sequence_number),
    )
    pixel_digest = record["observation_pixel_digest"]
    assert isinstance(pixel_digest, str)
    render = _operation(
        index=0,
        operation="render_canonical_row",
        operation_version="zeromodel-arcade-shooter-render-state-frame/v1",
        input_digests=(),
        parameters={"row_id": record["expected_row"]},
        output_digest=pixel_digest,
    )
    emit = _operation(
        index=1,
        operation="emit_observation",
        operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
        input_digests=(pixel_digest,),
        parameters={"event_type": "frame"},
        output_digest=pixel_digest,
    )
    chain = {
        "version": OBSERVATION_OPERATION_CHAIN_VERSION,
        "operations": [render, emit],
        "final_emitted_digest": pixel_digest,
    }
    chain["operation_chain_digest"] = canonical_sha256(chain)
    record["metadata"]["observation_operation_chain"] = chain
    return record


def _save_single_observation():
    store, session_factory, engine = build_store()
    _identity, plan = _save_identity_plan(store, frame_count=1)
    item = MaterializedObservationDTO.from_record(
        sample_record(plan=plan, sequence_number=0, pixels=_pixels())
    )
    store.save_observations((item,))
    return SqlAlchemyVideoActionSetStore(session_factory), session_factory, engine, item


def test_sql_schema_creation_registers_observation_tables() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    assert "video_action_set_observation" not in inspect(engine).get_table_names()

    create_schema(engine)
    table_names = set(inspect(engine).get_table_names())

    assert "matrix_blob" in table_names
    assert "video_action_set_observation" in table_names
    assert "video_action_set_observation_operation_chain" in table_names
    assert "video_action_set_observation_operation" in table_names
    assert "video_action_set_observation_operation_input" in table_names


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
    assert read_store.list_materialized_observations(split="development") == (
        first_materialized,
        read_store.get_materialized_observation(saved[1].frame_id),
        read_store.get_materialized_observation(saved[2].frame_id),
    )

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
        assert (
            session.scalar(
                select(func.count()).select_from(ObservationOperationInputORM)
            )
            == 3
        )
        blob = session.get(MatrixBlobORM, saved[0].matrix_blob_id)
        assert blob is not None
        assert isinstance(blob.data, bytes)
        assert blob.byte_length == len(blob.data)

    assert first_materialized.observation.provider_observation_descriptor is not None
    assert (
        first_materialized.observation.provider_observation_descriptor.descriptor_digest
        == first_materialized.observation.provider_observation_digest
    )
    gap_materialized = read_store.get_materialized_observation(saved[2].frame_id)
    assert gap_materialized is not None
    assert gap_materialized.to_record(include_pixels=True)["pixels"] is None

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


def test_sql_observation_duplicate_same_batch_is_idempotent() -> None:
    store, session_factory, _engine = build_store()
    _identity, plan = _save_identity_plan(store, frame_count=1)
    first = MaterializedObservationDTO.from_record(
        sample_record(plan=plan, sequence_number=0, pixels=_pixels())
    )

    saved = store.save_observations((first, first))

    assert saved == (first.observation, first.observation)
    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(MatrixBlobORM)) == 1
        assert session.scalar(select(func.count()).select_from(ObservationORM)) == 1
        assert (
            session.scalar(
                select(func.count()).select_from(ObservationOperationChainORM)
            )
            == 1
        )
        assert (
            session.scalar(select(func.count()).select_from(ObservationOperationORM))
            == 1
        )
        assert (
            session.scalar(
                select(func.count()).select_from(ObservationOperationInputORM)
            )
            == 1
        )


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
        assert (
            session.scalar(
                select(func.count()).select_from(ObservationOperationInputORM)
            )
            == 1
        )


def test_sqlite_foreign_keys_reject_orphan_observation_operation() -> None:
    _store, session_factory, engine = build_store()

    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1

    orphan = ObservationOperationORM(
        frame_id="missing-frame",
        operation_index=0,
        operation="emit_observation",
        operation_version=OBSERVATION_OPERATION_CHAIN_VERSION,
        parameters_json='{"event_type":"frame"}',
        parameter_digest="sha256:" + "1" * 64,
        output_digest=None,
        operation_digest="sha256:" + "2" * 64,
    )
    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            session.add(orphan)


def test_sql_observation_sequence_uniqueness_is_preflighted() -> None:
    store, _session_factory, _engine = build_store()
    _identity, plan = _save_identity_plan(store, frame_count=4)
    first = MaterializedObservationDTO.from_record(
        sample_record(plan=plan, sequence_number=0, pixels=_pixels())
    )
    conflict = MaterializedObservationDTO.from_record(
        sample_record(
            plan=plan,
            sequence_number=0,
            frame_index=3,
            pixels=_pixels(4),
            metadata_extra={"original_frame_index": 3, "materialized_order": [3, 0]},
        )
    )

    store.save_observations((first,))

    with pytest.raises(VPMValidationError, match=OBSERVATION_SEQUENCE_CONFLICT_MESSAGE):
        store.save_observations((conflict,))


def test_sql_observation_multi_operation_ordering_and_inputs_round_trip() -> None:
    store, session_factory, _engine = build_store()
    _identity, plan = _save_identity_plan(store, frame_count=2)
    record = _multi_operation_record(plan=plan, sequence_number=1)
    item = MaterializedObservationDTO.from_record(record)

    store.save_observations((item,))
    read_store = SqlAlchemyVideoActionSetStore(session_factory)
    chain = read_store.get_operation_chain(item.observation.frame_id)

    assert chain is not None
    assert [operation.operation for operation in chain.operations] == [
        "render_canonical_row",
        "emit_observation",
    ]
    assert chain.operations[0].input_digests == ()
    assert chain.operations[1].input_digests == (
        item.observation.observation_pixel_digest,
    )
    assert_records_equivalent(
        record,
        read_store.get_materialized_observation(item.observation.frame_id).to_record(),
    )
    with session_factory() as session:
        assert (
            session.scalar(select(func.count()).select_from(ObservationOperationORM))
            == 2
        )
        assert (
            session.scalar(
                select(func.count()).select_from(ObservationOperationInputORM)
            )
            == 1
        )


def test_sql_read_rejects_tampered_observation_chain_digest() -> None:
    read_store, session_factory, _engine, item = _save_single_observation()

    with session_factory.begin() as session:
        row = session.get(ObservationORM, item.observation.frame_id)
        assert row is not None
        row.operation_chain_digest = "sha256:" + "0" * 64

    with pytest.raises(VPMValidationError, match="operation chain mismatch"):
        read_store.get_observation(item.observation.frame_id)


def test_sql_read_rejects_tampered_observation_plan_ownership() -> None:
    read_store, session_factory, _engine, item = _save_single_observation()

    with session_factory.begin() as session:
        row = session.get(ObservationORM, item.observation.frame_id)
        assert row is not None
        row.episode_plan_digest = "sha256:" + "0" * 64

    with pytest.raises(VPMValidationError, match="episode plan mismatch"):
        read_store.get_observation(item.observation.frame_id)


def test_sql_read_rejects_tampered_matrix_blob_length() -> None:
    read_store, session_factory, _engine, item = _save_single_observation()

    with session_factory.begin() as session:
        row = session.get(MatrixBlobORM, item.observation.matrix_blob_id)
        assert row is not None
        row.byte_length += 1

    with pytest.raises(VPMValidationError, match="matrix blob byte length mismatch"):
        read_store.get_materialized_observation(item.observation.frame_id)


def test_sql_read_rejects_tampered_operation_chain_count_and_digest() -> None:
    read_store, session_factory, _engine, item = _save_single_observation()

    with session_factory.begin() as session:
        row = session.get(ObservationOperationChainORM, item.observation.frame_id)
        assert row is not None
        row.operation_count += 1

    with pytest.raises(VPMValidationError, match="operation chain mismatch"):
        read_store.get_operation_chain(item.observation.frame_id)

    read_store, session_factory, _engine, item = _save_single_observation()
    with session_factory.begin() as session:
        row = session.get(
            ObservationOperationORM,
            (item.observation.frame_id, 0),
        )
        assert row is not None
        row.operation_digest = "sha256:" + "0" * 64

    with pytest.raises(VPMValidationError, match="operation digest mismatch"):
        read_store.get_operation_chain(item.observation.frame_id)


def test_sql_final_split_observation_materialization_remains_prohibited() -> None:
    store, _session_factory, _engine = build_store()
    identity = sql_identity("sql-final-observation-rmdto-seed")
    plan = plan_dto(identity=identity, split="final", frame_count=1)
    store.save_identity(identity)
    store.save_episode_plan(plan)

    with pytest.raises(VPMValidationError, match="final split observation"):
        MaterializedObservationDTO.from_record(
            sample_record(split="final", plan=plan, pixels=_pixels())
        )


def test_sql_materialized_batch_loader_uses_set_based_queries() -> None:
    store, session_factory, engine = build_store()
    _identity, plan = _save_identity_plan(store, frame_count=5)
    records = tuple(
        _multi_operation_record(plan=plan, sequence_number=index) for index in range(5)
    )
    store.save_observations(
        tuple(MaterializedObservationDTO.from_record(record) for record in records)
    )
    statements: list[str] = []

    def count_statement(
        _connection,
        _cursor,
        statement,
        _parameters,
        _context,
        _executemany,
    ) -> None:
        statements.append(statement)

    event.listen(engine, "before_cursor_execute", count_statement)
    try:
        materialized = SqlAlchemyVideoActionSetStore(
            session_factory
        ).list_materialized_observations(split="development")
    finally:
        event.remove(engine, "before_cursor_execute", count_statement)

    assert len(materialized) == 5
    assert len(statements) <= 6
    for record, item in zip(records, materialized, strict=True):
        assert_records_equivalent(record, item.to_record())


def test_observation_store_query_parity_between_memory_and_sql() -> None:
    sql_store, _session_factory, _engine = build_store()
    memory_store = InMemoryVideoActionSetStore()
    identity = sql_identity("sql-memory-parity-observation-rmdto-seed")
    plan = plan_dto(identity=identity, split="development", frame_count=3)
    records = (
        sample_record(plan=plan, sequence_number=0, pixels=_pixels()),
        sample_record(plan=plan, sequence_number=1, pixels=_pixels(3)),
        sample_gap_record(plan=plan, sequence_number=2),
    )
    materialized = tuple(MaterializedObservationDTO.from_record(row) for row in records)

    for store in (sql_store, memory_store):
        store.save_identity(identity)
        store.save_episode_plan(plan)
        store.save_observations((*materialized, materialized[0]))

    assert sql_store.list_observations(
        split="development"
    ) == memory_store.list_observations(split="development")
    assert sql_store.list_observations(
        has_pixels=False
    ) == memory_store.list_observations(has_pixels=False)
    assert sql_store.list_observations_by_operation(
        operation="emit_observation"
    ) == memory_store.list_observations_by_operation(operation="emit_observation")
    assert sql_store.list_observations_by_output_digest(
        str(materialized[0].observation.observation_pixel_digest)
    ) == memory_store.list_observations_by_output_digest(
        str(materialized[0].observation.observation_pixel_digest)
    )
    assert sql_store.list_observation_consumers_of_digest(
        str(materialized[1].observation.observation_pixel_digest)
    ) == memory_store.list_observation_consumers_of_digest(
        str(materialized[1].observation.observation_pixel_digest)
    )
    for sql_item, memory_item in zip(
        sql_store.list_materialized_observations(split="development"),
        memory_store.list_materialized_observations(split="development"),
        strict=True,
    ):
        assert_records_equivalent(sql_item.to_record(), memory_item.to_record())


def test_sqlite_runtime_composes_observation_store() -> None:
    runtime = build_sqlite_runtime("sqlite:///:memory:", initialize_schema=True)
    identity = sql_identity("sql-runtime-observation-rmdto-seed")
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
    assert_records_equivalent(
        record,
        runtime.video_action_set.list_observation_records(
            split="development",
            include_pixels=True,
        )[0],
    )
