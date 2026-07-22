from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import inspect

from zeromodel.core.artifact import VPMValidationError
from zeromodel.persistence.sqlalchemy.db.orm.video_action_set import BenchmarkIdentityORM
from zeromodel.persistence.sqlalchemy.db.runtime import build_sqlite_runtime
from zeromodel.persistence.sqlalchemy.db.session import (
    create_database_engine,
    create_schema,
    create_session_factory,
)
from zeromodel.persistence.sqlalchemy.db.stores.video_action_set import SqlAlchemyVideoActionSetStore
from zeromodel.video.domains.video_action_set.dto import BenchmarkIdentityDTO
from zeromodel.video.domains.video_action_set.store import (
    BENCHMARK_IDENTITY_CONFLICT_MESSAGE,
)


pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[1]
SEED_MATERIAL = (
    "zeromodel-action-set-reachability-v1|aed523b04c258d7e28cd9466413b49fc817b4e35"
)
SEED_DIGEST = "sha256:22f0b8b706198c4d00df0f8e1d6e09dd324aefdd6ac1dc0768fb9b24a8b519c9"


def sample_identity() -> BenchmarkIdentityDTO:
    return BenchmarkIdentityDTO(
        contract_commit="aed523b04c258d7e28cd9466413b49fc817b4e35",
        seed_material=SEED_MATERIAL,
        seed_digest=SEED_DIGEST,
        policy_artifact_id=(
            "eb7523f406b45ac30b478fe9528db8f89a548693b0add2fc8d3e51c4badd857e"
        ),
        parent_audit_sha="e4c3f894e47e070318edc046171233cbc862aa11",
        parent_v3_sha="4790165de78557fce63d64e5f2b7ddfde04f1e98",
    )


def build_store() -> SqlAlchemyVideoActionSetStore:
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    return SqlAlchemyVideoActionSetStore(session_factory)


def test_sql_store_requires_explicit_schema_creation() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    assert (
        "video_action_set_benchmark_identity" not in inspect(engine).get_table_names()
    )

    create_schema(engine)

    assert "video_action_set_benchmark_identity" in inspect(engine).get_table_names()


def test_sql_store_save_and_retrieve_through_separate_sessions() -> None:
    engine = create_database_engine("sqlite:///:memory:")
    create_schema(engine)
    session_factory = create_session_factory(engine)
    identity = sample_identity()

    save_store = SqlAlchemyVideoActionSetStore(session_factory)
    read_store = SqlAlchemyVideoActionSetStore(session_factory)

    assert save_store.save_identity(identity) == identity
    assert read_store.get_identity(identity.seed_digest) == identity


def test_sql_store_idempotence_conflict_and_dto_return_type() -> None:
    store = build_store()
    identity = sample_identity()
    conflicting = replace(identity, contract_commit="different")

    assert store.get_identity(identity.seed_digest) is None
    assert store.save_identity(identity) == identity
    assert store.save_identity(identity) == identity
    with pytest.raises(VPMValidationError, match=BENCHMARK_IDENTITY_CONFLICT_MESSAGE):
        store.save_identity(conflicting)

    retrieved = store.get_identity(identity.seed_digest)
    assert isinstance(retrieved, BenchmarkIdentityDTO)
    assert not isinstance(retrieved, BenchmarkIdentityORM)


def test_sqlite_runtime_composes_persistent_identity_store() -> None:
    runtime = build_sqlite_runtime("sqlite:///:memory:", initialize_schema=True)

    loaded = runtime.video_action_set.load_identity(REPO_ROOT)

    assert loaded == sample_identity()
    assert runtime.video_action_set.get_identity(loaded.seed_digest) == loaded


def test_sqlite_runtime_does_not_create_schema_implicitly(tmp_path: Path) -> None:
    database_path = tmp_path / "identity.sqlite"

    build_sqlite_runtime(
        f"sqlite:///{database_path.as_posix()}",
        initialize_schema=False,
    )

    assert not database_path.exists()
