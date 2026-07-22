from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from zeromodel.persistence.sqlalchemy.db.orm.base import Base


class BenchmarkIdentityORM(Base):
    __tablename__ = "video_action_set_benchmark_identity"

    seed_digest: Mapped[str] = mapped_column(String, primary_key=True)
    contract_commit: Mapped[str] = mapped_column(String, nullable=False)
    seed_material: Mapped[str] = mapped_column(String, nullable=False)
    policy_artifact_id: Mapped[str] = mapped_column(String, nullable=False)
    parent_audit_sha: Mapped[str] = mapped_column(String, nullable=False)
    parent_v3_sha: Mapped[str] = mapped_column(String, nullable=False)


class EpisodePlanORM(Base):
    __tablename__ = "video_action_set_episode_plan"
    __table_args__ = (
        UniqueConstraint(
            "benchmark_seed_digest",
            "split",
            "ordinal",
            name="uq_video_action_set_episode_plan_seed_split_ordinal",
        ),
        UniqueConstraint(
            "plan_digest",
            name="uq_video_action_set_episode_plan_digest",
        ),
    )

    episode_id: Mapped[str] = mapped_column(String, primary_key=True)
    benchmark_seed_digest: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_benchmark_identity.seed_digest"),
        index=True,
        nullable=False,
    )
    plan_digest: Mapped[str] = mapped_column(String, index=True, nullable=False)
    version: Mapped[str] = mapped_column(String, nullable=False)
    seed_derivation_version: Mapped[str] = mapped_column(String, nullable=False)
    split: Mapped[str] = mapped_column(String, index=True, nullable=False)
    ordinal: Mapped[int] = mapped_column(nullable=False)
    family_label: Mapped[str] = mapped_column(String, index=True, nullable=False)
    family_ordinal: Mapped[int] = mapped_column(nullable=False)
    episode_disposition: Mapped[str] = mapped_column(String, nullable=False)
    denominator_class: Mapped[str] = mapped_column(String, nullable=False)
    mutation_kind: Mapped[str | None] = mapped_column(String, nullable=True)
    source_row_id: Mapped[str] = mapped_column(String, nullable=False)
    secondary_row_id: Mapped[str | None] = mapped_column(String, nullable=True)
    derived_seed_identity: Mapped[str] = mapped_column(String, nullable=False)
    episode_seed_hex: Mapped[str] = mapped_column(String(16), nullable=False)
    frame_count: Mapped[int] = mapped_column(nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class SealedSplitPlanORM(Base):
    __tablename__ = "video_action_set_sealed_split_plan"
    __table_args__ = (
        UniqueConstraint(
            "sealed_plan_digest",
            name="uq_video_action_set_sealed_split_plan_digest",
        ),
    )

    seed_commitment: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_benchmark_identity.seed_digest"),
        primary_key=True,
    )
    split: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String, nullable=False)
    seed_derivation_version: Mapped[str] = mapped_column(String, nullable=False)
    plan_only: Mapped[bool] = mapped_column(Boolean, nullable=False)
    materialization_prohibited: Mapped[bool] = mapped_column(Boolean, nullable=False)
    frame_count: Mapped[int] = mapped_column(nullable=False)
    episode_counts_json: Mapped[str] = mapped_column(Text, nullable=False)
    sealed_episode_ids_json: Mapped[str] = mapped_column(Text, nullable=False)
    sealed_plan_digest: Mapped[str] = mapped_column(String, nullable=False)


class MatrixBlobORM(Base):
    __tablename__ = "matrix_blob"
    __table_args__ = (CheckConstraint("byte_length >= 1", name="ck_matrix_blob_bytes"),)

    blob_id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[str] = mapped_column(String, nullable=False)
    dtype: Mapped[str] = mapped_column(String, nullable=False)
    shape_json: Mapped[str] = mapped_column(Text, nullable=False)
    scale: Mapped[float | None] = mapped_column(Float, nullable=True)
    zero_point: Mapped[int | None] = mapped_column(nullable=True)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    data: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    byte_length: Mapped[int] = mapped_column(nullable=False)


class FinalizationSchemaORM(Base):
    __tablename__ = "video_action_set_finalization_schema"

    authority_id: Mapped[str] = mapped_column(String, primary_key=True)
    schema_version: Mapped[str] = mapped_column(String, nullable=False)
    authority_kind: Mapped[str] = mapped_column(String, nullable=False)
    created_utc: Mapped[str] = mapped_column(String, nullable=False)


class FinalEvaluationProtocolORM(Base):
    __tablename__ = "video_action_set_final_evaluation_protocol"

    protocol_digest: Mapped[str] = mapped_column(String, primary_key=True)
    protocol_id: Mapped[str] = mapped_column(String, nullable=False)
    protocol_status: Mapped[str] = mapped_column(String, nullable=False)
    benchmark_seed_digest: Mapped[str] = mapped_column(String, nullable=False)
    sealed_plan_digest: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class FinalAccessAuthorizationORM(Base):
    __tablename__ = "video_action_set_final_access_authorization"
    __table_args__ = (
        UniqueConstraint(
            "authorization_digest",
            name="uq_video_action_set_final_authorization_digest",
        ),
    )

    authorization_id: Mapped[str] = mapped_column(String, primary_key=True)
    authorization_status: Mapped[str] = mapped_column(String, nullable=False)
    authorization_digest: Mapped[str] = mapped_column(String, nullable=False)
    protocol_digest: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_final_evaluation_protocol.protocol_digest"),
        index=True,
        nullable=False,
    )
    benchmark_seed_digest: Mapped[str] = mapped_column(
        String, index=True, nullable=False
    )
    sealed_plan_digest: Mapped[str] = mapped_column(String, index=True, nullable=False)
    created_utc: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class FinalAccessRecordORM(Base):
    __tablename__ = "video_action_set_final_access_record"
    __table_args__ = (
        UniqueConstraint(
            "authorization_id",
            name="uq_video_action_set_final_record_authorization",
        ),
        UniqueConstraint(
            "benchmark_seed_digest",
            "sealed_plan_digest",
            name="uq_video_action_set_final_record_seed_sealed",
        ),
        UniqueConstraint(
            "record_digest",
            name="uq_video_action_set_final_record_digest",
        ),
        CheckConstraint(
            "state in ('authorized', 'reserved', 'running', 'completed', "
            "'failed', 'interrupted')",
            name="ck_video_action_set_final_record_state",
        ),
    )

    access_id: Mapped[str] = mapped_column(String, primary_key=True)
    authorization_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_final_access_authorization.authorization_id"),
        nullable=False,
    )
    state: Mapped[str] = mapped_column(String, index=True, nullable=False)
    benchmark_seed_digest: Mapped[str] = mapped_column(
        String, index=True, nullable=False
    )
    sealed_plan_digest: Mapped[str] = mapped_column(String, index=True, nullable=False)
    protocol_digest: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_final_evaluation_protocol.protocol_digest"),
        index=True,
        nullable=False,
    )
    authorization_digest: Mapped[str] = mapped_column(String, nullable=False)
    created_utc: Mapped[str] = mapped_column(String, nullable=False)
    updated_utc: Mapped[str] = mapped_column(String, nullable=False)
    process_identity: Mapped[str] = mapped_column(String, nullable=False)
    current_event_ordinal: Mapped[int] = mapped_column(nullable=False)
    last_event_digest: Mapped[str | None] = mapped_column(String, nullable=True)
    record_digest: Mapped[str] = mapped_column(String, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class FinalAccessEventORM(Base):
    __tablename__ = "video_action_set_final_access_event"
    __table_args__ = (
        UniqueConstraint(
            "access_id",
            "ordinal",
            name="uq_video_action_set_final_event_access_ordinal",
        ),
        CheckConstraint(
            "ordinal >= 0",
            name="ck_video_action_set_final_event_ordinal",
        ),
        CheckConstraint(
            "new_state in ('authorized', 'reserved', 'running', 'completed', "
            "'failed', 'interrupted')",
            name="ck_video_action_set_final_event_new_state",
        ),
    )

    event_digest: Mapped[str] = mapped_column(String, primary_key=True)
    access_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_final_access_record.access_id"),
        index=True,
        nullable=False,
    )
    authorization_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    ordinal: Mapped[int] = mapped_column(nullable=False)
    previous_state: Mapped[str | None] = mapped_column(String, nullable=True)
    new_state: Mapped[str] = mapped_column(String, nullable=False)
    utc: Mapped[str] = mapped_column(String, nullable=False)
    process_identity: Mapped[str] = mapped_column(String, nullable=False)
    previous_event_digest: Mapped[str | None] = mapped_column(String, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class ObservationORM(Base):
    __tablename__ = "video_action_set_observation"
    __table_args__ = (
        UniqueConstraint(
            "episode_id",
            "sequence_number",
            name="uq_video_action_set_observation_episode_sequence",
        ),
        CheckConstraint(
            "sequence_number >= 0",
            name="ck_video_action_set_observation_sequence_nonnegative",
        ),
        CheckConstraint(
            "action_known = (actual_executed_action IS NOT NULL)",
            name="ck_video_action_set_observation_action_known",
        ),
        CheckConstraint(
            "(matrix_blob_id IS NULL) = (observation_pixel_digest IS NULL)",
            name="ck_video_action_set_observation_blob_digest_nullity",
        ),
        CheckConstraint(
            "(provider_descriptor_json IS NULL) = "
            "(provider_observation_digest IS NULL)",
            name="ck_video_action_set_observation_provider_nullity",
        ),
    )

    frame_id: Mapped[str] = mapped_column(String, primary_key=True)
    benchmark_seed_digest: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_benchmark_identity.seed_digest"),
        index=True,
        nullable=False,
    )
    episode_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_episode_plan.episode_id"),
        index=True,
        nullable=False,
    )
    episode_plan_digest: Mapped[str] = mapped_column(
        String,
        index=True,
        nullable=False,
    )
    split: Mapped[str] = mapped_column(String, index=True, nullable=False)
    clip_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    sequence_number: Mapped[int] = mapped_column(nullable=False)
    event_type: Mapped[str] = mapped_column(String, index=True, nullable=False)
    family: Mapped[str] = mapped_column(String, index=True, nullable=False)
    expected_disposition: Mapped[str] = mapped_column(String, nullable=False)
    episode_family: Mapped[str] = mapped_column(String, nullable=False)
    episode_disposition: Mapped[str] = mapped_column(String, nullable=False)
    frame_disposition: Mapped[str] = mapped_column(String, nullable=False)
    denominator_class: Mapped[str] = mapped_column(String, index=True, nullable=False)
    expected_row: Mapped[str | None] = mapped_column(String, nullable=True)
    expected_action: Mapped[str | None] = mapped_column(String, nullable=True)
    actual_executed_action: Mapped[str | None] = mapped_column(String, nullable=True)
    action_known: Mapped[bool] = mapped_column(Boolean, nullable=False)
    gap_declaration_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    observation_pixel_digest: Mapped[str | None] = mapped_column(
        String,
        index=True,
        nullable=True,
    )
    matrix_blob_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("matrix_blob.blob_id"),
        index=True,
        nullable=True,
    )
    final_access_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("video_action_set_final_access_record.access_id"),
        index=True,
        nullable=True,
    )
    provider_descriptor_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    provider_observation_digest: Mapped[str | None] = mapped_column(
        String,
        index=True,
        nullable=True,
    )
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    operation_chain_digest: Mapped[str] = mapped_column(
        String,
        index=True,
        nullable=False,
    )


class ObservationOperationChainORM(Base):
    __tablename__ = "video_action_set_observation_operation_chain"
    __table_args__ = (
        CheckConstraint(
            "operation_count >= 1",
            name="ck_video_action_set_operation_chain_count",
        ),
    )

    frame_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_observation.frame_id"),
        primary_key=True,
    )
    version: Mapped[str] = mapped_column(String, nullable=False)
    final_emitted_digest: Mapped[str | None] = mapped_column(String, nullable=True)
    operation_chain_digest: Mapped[str] = mapped_column(
        String,
        index=True,
        nullable=False,
    )
    operation_count: Mapped[int] = mapped_column(nullable=False)


class ObservationOperationORM(Base):
    __tablename__ = "video_action_set_observation_operation"
    __table_args__ = (
        CheckConstraint(
            "operation_index >= 0",
            name="ck_video_action_set_operation_index",
        ),
    )

    frame_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_observation.frame_id"),
        primary_key=True,
    )
    operation_index: Mapped[int] = mapped_column(primary_key=True)
    operation: Mapped[str] = mapped_column(String, index=True, nullable=False)
    operation_version: Mapped[str] = mapped_column(String, index=True, nullable=False)
    parameters_json: Mapped[str] = mapped_column(Text, nullable=False)
    parameter_digest: Mapped[str] = mapped_column(String, nullable=False)
    output_digest: Mapped[str | None] = mapped_column(String, index=True, nullable=True)
    operation_digest: Mapped[str] = mapped_column(String, index=True, nullable=False)


class ObservationOperationInputORM(Base):
    __tablename__ = "video_action_set_observation_operation_input"
    __table_args__ = (
        ForeignKeyConstraint(
            ["frame_id", "operation_index"],
            [
                "video_action_set_observation_operation.frame_id",
                "video_action_set_observation_operation.operation_index",
            ],
        ),
        CheckConstraint(
            "operation_index >= 0",
            name="ck_video_action_set_operation_input_operation_index",
        ),
        CheckConstraint(
            "input_index >= 0",
            name="ck_video_action_set_operation_input_index",
        ),
    )

    frame_id: Mapped[str] = mapped_column(String, primary_key=True)
    operation_index: Mapped[int] = mapped_column(primary_key=True)
    input_index: Mapped[int] = mapped_column(primary_key=True)
    input_digest: Mapped[str | None] = mapped_column(String, index=True, nullable=True)


__all__ = [
    "BenchmarkIdentityORM",
    "EpisodePlanORM",
    "FinalAccessAuthorizationORM",
    "FinalAccessEventORM",
    "FinalAccessRecordORM",
    "MatrixBlobORM",
    "ObservationORM",
    "ObservationOperationChainORM",
    "ObservationOperationInputORM",
    "ObservationOperationORM",
    "SealedSplitPlanORM",
]
