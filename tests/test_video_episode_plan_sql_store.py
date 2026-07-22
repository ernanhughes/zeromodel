from __future__ import annotations

from copy import deepcopy

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy import inspect

from test_video_episode_plan_rmdto import (
    _redigest,
    plan_dto,
    sample_identity,
)
from zeromodel.core.artifact import VPMValidationError
from zeromodel.persistence.sqlalchemy.db.orm.video_action_set import EpisodePlanORM, SealedSplitPlanORM
from zeromodel.persistence.sqlalchemy.db.runtime import build_sqlite_runtime
from zeromodel.persistence.sqlalchemy.db.session import (
    create_database_engine,
    create_schema,
    create_session_factory,
)
from zeromodel.persistence.sqlalchemy.db.stores.video_action_set import SqlAlchemyVideoActionSetStore
from zeromodel.video.domains.video_action_set.canonical_json import canonical_json_text
from zeromodel.video.domains.video_action_set.dto import EpisodePlanDTO, SealedSplitPlanDTO
from zeromodel.video.domains.video_action_set.store import (
    EPISODE_PLAN_CONFLICT_MESSAGE,
    UNKNOWN_BENCHMARK_IDENTITY_MESSAGE,
)


pytestmark = pytest.mark.integration


def sql_identity(prefix: str = "sql-episode-plan-rmdto-seed"):
    return sample_identity(prefix)


def unsigned_seed_identity(prefix: str = "sql-unsigned-episode-plan-seed"):
    for index in range(10_000):
        identity = sample_identity(f"{prefix}-{index}")
        if plan_dto(identity=identity).episode_seed >= 2**63:
            return identity
    raise AssertionError("could not find unsigned 64-bit episode seed")


def build_store():
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    return SqlAlchemyVideoActionSetStore(session_factory), session_factory, engine


def test_sql_schema_creation_registers_episode_plan_tables() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    assert "video_action_set_episode_plan" not in inspect(engine).get_table_names()

    create_schema(engine)
    table_names = set(inspect(engine).get_table_names())

    assert "video_action_set_benchmark_identity" in table_names
    assert "video_action_set_episode_plan" in table_names
    assert "video_action_set_sealed_split_plan" in table_names


def test_sql_plan_identity_ownership_and_canonical_round_trip() -> None:
    store, session_factory, _engine = build_store()
    identity = sql_identity()
    plan = plan_dto(identity=identity)

    with pytest.raises(VPMValidationError, match=UNKNOWN_BENCHMARK_IDENTITY_MESSAGE):
        store.save_episode_plan(plan)

    store.save_identity(identity)
    assert store.save_episode_plan(plan) == plan
    assert store.get_episode_plan(plan.episode_id) == plan
    with session_factory() as session:
        row = session.get(EpisodePlanORM, plan.episode_id)
        assert row is not None
        assert row.payload_json == canonical_json_text(plan.to_dict())
        assert row.episode_seed_hex == f"{plan.episode_seed:016x}"


def test_sql_plan_persists_unsigned_episode_seed() -> None:
    store, session_factory, _engine = build_store()
    identity = unsigned_seed_identity()
    plan = plan_dto(identity=identity)

    assert plan.episode_seed >= 2**63
    store.save_identity(identity)
    store.save_episode_plan(plan)

    with session_factory() as session:
        row = session.get(EpisodePlanORM, plan.episode_id)
        assert row is not None
        assert row.episode_seed_hex == f"{plan.episode_seed:016x}"

    assert (
        SqlAlchemyVideoActionSetStore(session_factory).get_episode_plan(plan.episode_id)
        == plan
    )


def test_sql_plan_retrieval_across_sessions_idempotence_and_conflict() -> None:
    _store, session_factory, _engine = build_store()
    identity = sql_identity()
    plan = plan_dto(identity=identity)
    save_store = SqlAlchemyVideoActionSetStore(session_factory)
    read_store = SqlAlchemyVideoActionSetStore(session_factory)

    save_store.save_identity(identity)
    assert save_store.save_episode_plan(plan) == plan
    assert read_store.get_episode_plan(plan.episode_id) == plan
    assert read_store.save_episode_plan(plan) == plan

    conflict_payload = deepcopy(plan.to_dict())
    conflict_payload["source_row_id"] = "row:changed"
    conflict = EpisodePlanDTO.from_dict(_redigest(conflict_payload))
    with pytest.raises(VPMValidationError, match=EPISODE_PLAN_CONFLICT_MESSAGE):
        read_store.save_episode_plan(conflict)
    assert not isinstance(read_store.get_episode_plan(plan.episode_id), EpisodePlanORM)


def test_sql_batch_rollback_ordering_and_filters() -> None:
    store, _session_factory, _engine = build_store()
    identity = sql_identity()
    other_identity = sql_identity("sql-episode-plan-other-seed")
    first = plan_dto(identity=identity, split="development", ordinal=0)
    second = plan_dto(
        identity=identity,
        split="development",
        ordinal=2,
        source_row_id="row:two",
    )
    third = plan_dto(
        identity=identity,
        split="development",
        ordinal=1,
        source_row_id="row:one",
    )
    other_seed = plan_dto(identity=other_identity, split="development", ordinal=0)

    store.save_identity(identity)
    store.save_identity(other_identity)
    store.save_episode_plan(first)
    conflict_payload = deepcopy(first.to_dict())
    conflict_payload["source_row_id"] = "row:changed"
    conflict = EpisodePlanDTO.from_dict(_redigest(conflict_payload))
    with pytest.raises(VPMValidationError, match=EPISODE_PLAN_CONFLICT_MESSAGE):
        store.save_episode_plans((second, conflict))
    assert store.get_episode_plan(second.episode_id) is None

    store.save_episode_plans((second, third, other_seed))
    assert store.list_episode_plans(
        benchmark_seed_digest=identity.seed_digest,
        split="development",
    ) == (first, third, second)
    assert store.list_episode_plans(
        benchmark_seed_digest=other_identity.seed_digest,
        split="development",
    ) == (other_seed,)


def test_sql_sealed_envelope_reconstructs_from_episode_rows() -> None:
    store, session_factory, _engine = build_store()
    identity = sql_identity()
    episodes = (
        plan_dto(identity=identity, ordinal=0, family_label="valid", family_ordinal=0),
        plan_dto(
            identity=identity,
            ordinal=1,
            family_label="frame_invalid",
            family_ordinal=0,
        ),
    )
    sealed = SealedSplitPlanDTO.build_final(
        episodes=episodes,
        seed_commitment=identity.seed_digest,
    )

    store.save_identity(identity)
    assert store.save_sealed_split_plan(sealed) == sealed
    with session_factory() as session:
        row = session.get(SealedSplitPlanORM, (identity.seed_digest, "final"))
        assert row is not None
        assert row.sealed_plan_digest == sealed.sealed_plan_digest
        assert (
            canonical_json_text(sealed.episode_counts.to_dict())
            == row.episode_counts_json
        )
        assert "episodes" not in inspect(SealedSplitPlanORM).columns

    read_store = SqlAlchemyVideoActionSetStore(session_factory)
    assert (
        read_store.get_sealed_split_plan(
            seed_commitment=identity.seed_digest,
            split="final",
        )
        == sealed
    )


def test_sql_sealed_digest_validation_rejects_corrupted_envelope() -> None:
    store, session_factory, _engine = build_store()
    identity = sql_identity()
    plan = plan_dto(identity=identity)
    sealed = SealedSplitPlanDTO.build_final(
        episodes=(plan,),
        seed_commitment=identity.seed_digest,
    )
    store.save_identity(identity)
    store.save_sealed_split_plan(sealed)
    with session_factory.begin() as session:
        row = session.get(SealedSplitPlanORM, (identity.seed_digest, "final"))
        assert row is not None
        row.sealed_plan_digest = "sha256:" + "0" * 64

    with pytest.raises(VPMValidationError, match="sealed plan digest mismatch"):
        store.get_sealed_split_plan(seed_commitment=identity.seed_digest, split="final")


def test_sqlite_runtime_composes_episode_plan_store() -> None:
    runtime = build_sqlite_runtime("sqlite:///:memory:", initialize_schema=True)
    identity = sql_identity()
    plan = plan_dto(identity=identity)

    runtime.video_action_set.save_identity(identity)
    assert runtime.video_action_set.save_episode_plan(plan) == plan
    sealed = runtime.video_action_set.seal_final_split(
        episodes=(plan,),
        seed_commitment=identity.seed_digest,
    )

    assert (
        runtime.video_action_set.get_sealed_split_plan(
            seed_commitment=identity.seed_digest,
            split="final",
        )
        == sealed
    )


def test_sqlite_foreign_keys_are_enabled_and_reject_orphan_plan() -> None:
    _store, session_factory, engine = build_store()

    with engine.connect() as connection:
        assert connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1

    identity = sql_identity("sql-orphan-plan-seed")
    plan = plan_dto(identity=identity)
    orphan = SqlAlchemyVideoActionSetStore._to_episode_plan_orm(plan)
    with pytest.raises(IntegrityError):
        with session_factory.begin() as session:
            session.add(orphan)
