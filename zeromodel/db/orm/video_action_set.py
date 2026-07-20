from __future__ import annotations

from sqlalchemy import String
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


__all__ = ["BenchmarkIdentityORM"]
