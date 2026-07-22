"""ZeroModel sqlalchemy public API."""
from __future__ import annotations

from .db.orm import (
    Base,
    BenchmarkIdentityORM,
    EpisodePlanORM,
    FinalAccessAuthorizationORM,
    FinalAccessEventORM,
    FinalAccessRecordORM,
    FinalEvaluationProtocolORM,
    FinalizationSchemaORM,
    MatrixBlobORM,
    ObservationORM,
    ObservationOperationChainORM,
    ObservationOperationInputORM,
    ObservationOperationORM,
    SealedSplitPlanORM,
)
from .db.orm.base import (
    Base,
)
from .db.orm.video_action_set import (
    BenchmarkIdentityORM,
    EpisodePlanORM,
    FinalAccessAuthorizationORM,
    FinalAccessEventORM,
    FinalAccessRecordORM,
    MatrixBlobORM,
    ObservationORM,
    ObservationOperationChainORM,
    ObservationOperationInputORM,
    ObservationOperationORM,
    SealedSplitPlanORM,
)
from .db.runtime import (
    build_finalization_sqlite_runtime,
    build_sqlite_runtime,
)
from .db.session import (
    FINALIZATION_SCHEMA_VERSION,
    create_database_engine,
    create_schema,
    create_session_factory,
    initialize_finalization_authority,
    verify_finalization_authority,
)
from .db.stores import (
    SqlAlchemyVideoActionSetStore,
)
from .db.stores.video_action_set import (
    SqlAlchemyVideoActionSetStore,
)
from .db.stores.video_action_set_observation import (
    chain_for_frame,
    chains_for_frames,
    materialized_observations_from_rows,
    matrix_blob_for_observation,
    observation_select,
    observations_from_rows,
    operation_observation_select,
    optional_observation_predicates,
    preflight_observation_sequence,
    to_matrix_blob,
    to_matrix_blob_orm,
    to_observation_dto,
    to_observation_orm,
    to_operation_chain_dto,
    to_operation_chain_orm,
    to_operation_input_orms,
    to_operation_orms,
    validate_observation_row,
)
from .video_action_set_final_admin_cli import (
    build_argument_parser,
    main,
)
from .video_action_set_final_cli import (
    build_argument_parser,
    main,
)

__all__ = [
    "Base",
    "BenchmarkIdentityORM",
    "EpisodePlanORM",
    "FINALIZATION_SCHEMA_VERSION",
    "FinalAccessAuthorizationORM",
    "FinalAccessEventORM",
    "FinalAccessRecordORM",
    "FinalEvaluationProtocolORM",
    "FinalizationSchemaORM",
    "MatrixBlobORM",
    "ObservationORM",
    "ObservationOperationChainORM",
    "ObservationOperationInputORM",
    "ObservationOperationORM",
    "SealedSplitPlanORM",
    "SqlAlchemyVideoActionSetStore",
    "build_argument_parser",
    "build_finalization_sqlite_runtime",
    "build_sqlite_runtime",
    "chain_for_frame",
    "chains_for_frames",
    "create_database_engine",
    "create_schema",
    "create_session_factory",
    "initialize_finalization_authority",
    "main",
    "materialized_observations_from_rows",
    "matrix_blob_for_observation",
    "observation_select",
    "observations_from_rows",
    "operation_observation_select",
    "optional_observation_predicates",
    "preflight_observation_sequence",
    "to_matrix_blob",
    "to_matrix_blob_orm",
    "to_observation_dto",
    "to_observation_orm",
    "to_operation_chain_dto",
    "to_operation_chain_orm",
    "to_operation_input_orms",
    "to_operation_orms",
    "validate_observation_row",
    "verify_finalization_authority",
]
