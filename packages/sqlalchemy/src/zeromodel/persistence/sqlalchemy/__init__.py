"""ZeroModel SQLAlchemy persistence adapter public API."""

from __future__ import annotations

from .db.runtime import (
    build_finalization_sqlite_runtime,
    build_sqlite_runtime,
)
from .db.session import (
    FINALIZATION_SCHEMA_VERSION,
    create_database_engine,
    create_schema,
    create_session_factory,
    sqlite_database_url,
    initialize_finalization_authority,
    verify_finalization_authority,
)
from .db.stores.video_action_set import (
    SqlAlchemyVideoActionSetStore,
)

__all__ = [
    "FINALIZATION_SCHEMA_VERSION",
    "SqlAlchemyVideoActionSetStore",
    "build_finalization_sqlite_runtime",
    "build_sqlite_runtime",
    "create_database_engine",
    "create_schema",
    "create_session_factory",
    "initialize_finalization_authority",
    "sqlite_database_url",
    "verify_finalization_authority",
]
