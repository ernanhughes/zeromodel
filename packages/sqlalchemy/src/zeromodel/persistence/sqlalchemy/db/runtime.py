from __future__ import annotations

from zeromodel.video.runtime import ZeroModelRuntime, build_runtime
from zeromodel.persistence.sqlalchemy.db.session import (
    create_database_engine,
    create_schema,
    create_session_factory,
    initialize_finalization_authority,
    verify_finalization_authority,
)
from zeromodel.persistence.sqlalchemy.db.stores.video_action_set import (
    SqlAlchemyVideoActionSetStore,
)


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


def build_finalization_sqlite_runtime(
    database_url: str,
    *,
    initialize_authority: bool = False,
) -> ZeroModelRuntime:
    engine = create_database_engine(database_url)
    if initialize_authority:
        initialize_finalization_authority(engine)
    else:
        verify_finalization_authority(engine)
    session_factory = create_session_factory(engine)
    store = SqlAlchemyVideoActionSetStore(
        session_factory,
        finalization_engine=engine,
    )
    return build_runtime(video_action_set_store=store)


__all__ = ["build_finalization_sqlite_runtime", "build_sqlite_runtime"]
