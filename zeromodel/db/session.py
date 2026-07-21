from __future__ import annotations

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .orm.base import Base
from .orm import video_action_set as _video_action_set_orm


def create_database_engine(database_url: str) -> Engine:
    if database_url in {"sqlite://", "sqlite:///:memory:"}:
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    else:
        engine = create_engine(database_url)
    _enable_sqlite_foreign_keys(engine)
    return engine


def _enable_sqlite_foreign_keys(engine: Engine) -> None:
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_schema(engine: Engine) -> None:
    _ = _video_action_set_orm.BenchmarkIdentityORM
    _ = _video_action_set_orm.EpisodePlanORM
    _ = _video_action_set_orm.FinalAccessAuthorizationORM
    _ = _video_action_set_orm.FinalAccessEventORM
    _ = _video_action_set_orm.FinalAccessRecordORM
    _ = _video_action_set_orm.MatrixBlobORM
    _ = _video_action_set_orm.ObservationORM
    _ = _video_action_set_orm.ObservationOperationChainORM
    _ = _video_action_set_orm.ObservationOperationInputORM
    _ = _video_action_set_orm.ObservationOperationORM
    _ = _video_action_set_orm.SealedSplitPlanORM
    Base.metadata.create_all(engine)


__all__ = [
    "create_database_engine",
    "create_schema",
    "create_session_factory",
]
