from __future__ import annotations

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .orm.base import Base
from .orm import video_action_set as _video_action_set_orm


def create_database_engine(database_url: str) -> Engine:
    if database_url in {"sqlite://", "sqlite:///:memory:"}:
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
    return create_engine(database_url)


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


def create_schema(engine: Engine) -> None:
    _ = _video_action_set_orm.BenchmarkIdentityORM
    _ = _video_action_set_orm.EpisodePlanORM
    _ = _video_action_set_orm.SealedSplitPlanORM
    Base.metadata.create_all(engine)


__all__ = [
    "create_database_engine",
    "create_schema",
    "create_session_factory",
]
