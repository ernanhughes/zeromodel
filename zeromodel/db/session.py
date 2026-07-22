from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, create_engine, event, inspect, select, text
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from .orm.base import Base
from .orm import video_action_set as _video_action_set_orm


FINALIZATION_SCHEMA_VERSION = "zeromodel-video-finalization-schema/v1"
FINALIZATION_AUTHORITY_ID = "local-finalization-authority"
FINALIZATION_AUTHORITY_KIND = "separate-finalization-authority"
_FINALIZATION_REQUIRED_COLUMNS = {
    "video_action_set_finalization_schema": {
        "authority_id",
        "schema_version",
        "authority_kind",
        "created_utc",
    },
    "video_action_set_final_evaluation_protocol": {
        "protocol_digest",
        "protocol_id",
        "protocol_status",
        "benchmark_seed_digest",
        "sealed_plan_digest",
        "payload_json",
    },
    "video_action_set_final_access_authorization": {
        "authorization_id",
        "authorization_status",
        "authorization_digest",
        "protocol_digest",
        "benchmark_seed_digest",
        "sealed_plan_digest",
        "created_utc",
        "payload_json",
    },
    "video_action_set_final_access_record": {
        "access_id",
        "authorization_id",
        "state",
        "current_event_ordinal",
        "last_event_digest",
        "record_digest",
        "payload_json",
    },
    "video_action_set_final_access_event": {
        "event_digest",
        "access_id",
        "ordinal",
        "previous_state",
        "new_state",
        "previous_event_digest",
        "payload_json",
    },
    "video_action_set_observation": {"frame_id", "final_access_id"},
}


def sqlite_database_url(path: Path) -> str:
    """Build a SQLite URL without URI-percent-encoding filesystem characters."""
    return f"sqlite:///{path.resolve().as_posix()}"


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
    _ = _video_action_set_orm.FinalEvaluationProtocolORM
    _ = _video_action_set_orm.FinalizationSchemaORM
    _ = _video_action_set_orm.MatrixBlobORM
    _ = _video_action_set_orm.ObservationORM
    _ = _video_action_set_orm.ObservationOperationChainORM
    _ = _video_action_set_orm.ObservationOperationInputORM
    _ = _video_action_set_orm.ObservationOperationORM
    _ = _video_action_set_orm.SealedSplitPlanORM
    Base.metadata.create_all(engine)


def initialize_finalization_authority(engine: Engine) -> None:
    existing_tables = set(inspect(engine).get_table_names())
    if existing_tables:
        verify_finalization_authority(engine)
        return
    create_schema(engine)
    with Session(engine) as session, session.begin():
        session.add(
            _video_action_set_orm.FinalizationSchemaORM(
                authority_id=FINALIZATION_AUTHORITY_ID,
                schema_version=FINALIZATION_SCHEMA_VERSION,
                authority_kind=FINALIZATION_AUTHORITY_KIND,
                created_utc=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
            )
        )
    verify_finalization_authority(engine)


def verify_finalization_authority(engine: Engine) -> None:
    from ..artifact import VPMValidationError

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    for table_name, required_columns in _FINALIZATION_REQUIRED_COLUMNS.items():
        if table_name not in tables:
            raise VPMValidationError("finalization database schema mismatch")
        actual_columns = {item["name"] for item in inspector.get_columns(table_name)}
        if not required_columns.issubset(actual_columns):
            raise VPMValidationError("finalization database schema mismatch")
    with Session(engine) as session, session.begin():
        rows = session.scalars(
            select(_video_action_set_orm.FinalizationSchemaORM)
        ).all()
        if len(rows) != 1:
            raise VPMValidationError("finalization database authority marker mismatch")
        marker = rows[0]
        if (
            marker.authority_id != FINALIZATION_AUTHORITY_ID
            or marker.schema_version != FINALIZATION_SCHEMA_VERSION
            or marker.authority_kind != FINALIZATION_AUTHORITY_KIND
        ):
            raise VPMValidationError("finalization database schema version mismatch")
        if engine.dialect.name == "sqlite":
            foreign_keys = session.execute(text("PRAGMA foreign_keys")).scalar_one()
            if foreign_keys != 1:
                raise VPMValidationError("finalization database foreign keys disabled")


__all__ = [
    "create_database_engine",
    "create_schema",
    "create_session_factory",
    "FINALIZATION_SCHEMA_VERSION",
    "initialize_finalization_authority",
    "verify_finalization_authority",
]
