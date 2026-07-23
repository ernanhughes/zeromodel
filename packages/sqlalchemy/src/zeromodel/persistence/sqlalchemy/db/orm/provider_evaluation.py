from __future__ import annotations

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Float,
    ForeignKey,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from zeromodel.persistence.sqlalchemy.db.orm.base import Base


class ProviderEvaluationConfigurationORM(Base):
    __tablename__ = "provider_evaluation_configurations"

    provider_configuration_id: Mapped[str] = mapped_column(String, primary_key=True)
    provider_kind: Mapped[str] = mapped_column(String, index=True, nullable=False)
    model_digest: Mapped[str] = mapped_column(String, index=True, nullable=False)
    payload_json: Mapped[str] = mapped_column(Text, nullable=False)


class ProviderEvaluationRunORM(Base):
    __tablename__ = "provider_evaluation_runs"

    run_id: Mapped[str] = mapped_column(String, primary_key=True)
    provider_configuration_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("provider_evaluation_configurations.provider_configuration_id"),
        index=True,
        nullable=False,
    )
    fixture_identity: Mapped[str] = mapped_column(String, index=True, nullable=False)
    policy_artifact_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    case_mode: Mapped[str] = mapped_column(String, index=True, nullable=False)
    representation_mode: Mapped[str] = mapped_column(String, index=True, nullable=False)
    case_count: Mapped[int] = mapped_column(nullable=False)
    attempted_count: Mapped[int] = mapped_column(nullable=False)
    accepted_count: Mapped[int] = mapped_column(nullable=False)
    rejected_count: Mapped[int] = mapped_column(nullable=False)
    exact_count: Mapped[int] = mapped_column(nullable=False)
    action_equivalent_count: Mapped[int] = mapped_column(nullable=False)
    action_changing_count: Mapped[int] = mapped_column(nullable=False)
    action_correct_count: Mapped[int] = mapped_column(nullable=False)
    latency_sample_count: Mapped[int] = mapped_column(nullable=False)
    latency_min_us: Mapped[int | None] = mapped_column(nullable=True)
    latency_max_us: Mapped[int | None] = mapped_column(nullable=True)
    latency_total_us: Mapped[int | None] = mapped_column(nullable=True)
    latency_median_us: Mapped[int | None] = mapped_column(nullable=True)
    latency_p95_us: Mapped[int | None] = mapped_column(nullable=True)
    summary_json: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


class ProviderEvaluationCaseORM(Base):
    __tablename__ = "provider_evaluation_cases"
    __table_args__ = (
        UniqueConstraint(
            "run_id",
            "case_ordinal",
            name="uq_provider_evaluation_case_run_ordinal",
        ),
        CheckConstraint(
            "case_ordinal >= 0",
            name="ck_provider_evaluation_case_ordinal",
        ),
        CheckConstraint(
            "accepted = (rejection_reason IS NULL)",
            name="ck_provider_evaluation_case_rejection_nullity",
        ),
        CheckConstraint(
            "accepted = (predicted_action IS NOT NULL)",
            name="ck_provider_evaluation_case_predicted_action_nullity",
        ),
        CheckConstraint(
            "accepted = (predicted_state_json IS NOT NULL)",
            name="ck_provider_evaluation_case_predicted_state_nullity",
        ),
    )

    case_id: Mapped[str] = mapped_column(String, primary_key=True)
    run_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("provider_evaluation_runs.run_id"),
        index=True,
        nullable=False,
    )
    case_ordinal: Mapped[int] = mapped_column(nullable=False)
    frame_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("video_action_set_observation.frame_id"),
        index=True,
        nullable=False,
    )
    policy_artifact_id: Mapped[str] = mapped_column(String, index=True, nullable=False)
    provider_configuration_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("provider_evaluation_configurations.provider_configuration_id"),
        index=True,
        nullable=False,
    )
    accepted: Mapped[bool] = mapped_column(Boolean, index=True, nullable=False)
    exact_state_match: Mapped[bool] = mapped_column(Boolean, index=True, nullable=False)
    action_match: Mapped[bool] = mapped_column(Boolean, index=True, nullable=False)
    outcome: Mapped[str] = mapped_column(String, index=True, nullable=False)
    expected_row_id: Mapped[str] = mapped_column(String, nullable=False)
    expected_action: Mapped[str] = mapped_column(String, nullable=False)
    predicted_row_id: Mapped[str | None] = mapped_column(String, nullable=True)
    predicted_action: Mapped[str | None] = mapped_column(String, nullable=True)
    rejection_reason: Mapped[str | None] = mapped_column(String, nullable=True)
    provider_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    provider_latency_us: Mapped[int | None] = mapped_column(nullable=True)
    provider_raw_response_digest: Mapped[str | None] = mapped_column(
        String, nullable=True
    )
    provider_raw_response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_state_json: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_state_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    expected_decision_trace_json: Mapped[str] = mapped_column(Text, nullable=False)
    predicted_decision_trace_json: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )
    factor_matches_json: Mapped[str] = mapped_column(Text, nullable=False)
    provider_response_metadata_json: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[str] = mapped_column(Text, nullable=False)


__all__ = [
    "ProviderEvaluationCaseORM",
    "ProviderEvaluationConfigurationORM",
    "ProviderEvaluationRunORM",
]
