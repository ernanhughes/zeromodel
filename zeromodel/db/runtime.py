from __future__ import annotations

from ..runtime import ZeroModelRuntime, build_runtime
from .session import create_database_engine, create_schema, create_session_factory
from .stores.video_action_set import SqlAlchemyVideoActionSetStore


def build_sqlite_runtime(
    database_url: str,
    *,
    initialize_schema: bool = False,
) -> ZeroModelRuntime:
    engine = create_database_engine(database_url)
    if initialize_schema:
        create_schema(engine)
    session_factory = create_session_factory(engine)
    store = SqlAlchemyVideoActionSetStore(session_factory)
    return build_runtime(video_action_set_store=store)


__all__ = ["build_sqlite_runtime"]
