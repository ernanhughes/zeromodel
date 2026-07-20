from __future__ import annotations

from sqlalchemy import BigInteger, Boolean, ForeignKey, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


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
    episode_seed: Mapped[int] = mapped_column(BigInteger, nullable=False)
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


__all__ = ["BenchmarkIdentityORM", "EpisodePlanORM", "SealedSplitPlanORM"]
