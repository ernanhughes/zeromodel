"""Manifest-owned calibration, promotion, and final-test governance for P16D.

Low-level comparison and inference functions remain reusable numerical primitives. This
module is the governed entry point: validation and test ownership must be proven by a
content-addressed DatasetPartitionDTO rather than asserted with a split-name string.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .fields import VPMFieldSchemaDTO
from .partitions import DatasetPartitionDTO
from .promoted_inference import (
    PromotedTestEvaluationReportDTO,
    evaluate_promoted_model_on_test,
)
from .promotion import (
    ModelCalibrationDTO,
    PromotedPerceptionModelDTO,
    PromotionDecisionDTO,
    PromotionPolicyDTO,
    calibrate_comparison_candidates,
    promote_perception_model,
)
from .representation import DiscreteActionSchemaDTO, SourceVPMDTO
from .temporal import TemporalSourceVPMDTO
from .temporal_inference import (
    TemporalInferenceComparisonReportDTO,
    TemporalTranslatorDTO,
)
from .translator import SourceTargetTranslatorDTO

PARTITION_OWNED_COMPARISON_VERSION: Final = "perception-partition-owned-comparison/1"
PARTITION_OWNED_TEST_VERSION: Final = "perception-partition-owned-test-evaluation/1"
PARTITION_GOVERNANCE_SEMANTICS: Final = (
    "manifest_partition_identity_required_for_validation_calibration_promotion_and_test"
)


class PerceptionPartitionGovernanceError(ValueError):
    """Raised when model governance is not backed by manifest-owned evidence."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(payload: Mapping[str, object]) -> str:
    return f"sha256:{hashlib.sha256(_canonical_json(payload)).hexdigest()}"


@dataclass(frozen=True)
class PartitionOwnedComparisonReportDTO:
    owned_report_id: str
    partition_id: str
    dataset_id: str
    action_schema_id: str
    split: str
    comparison_report: TemporalInferenceComparisonReportDTO
    semantics: str = PARTITION_GOVERNANCE_SEMANTICS
    version: str = PARTITION_OWNED_COMPARISON_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.owned_report_id,
                self.partition_id,
                self.dataset_id,
                self.action_schema_id,
            )
        ):
            raise PerceptionPartitionGovernanceError(
                "owned comparison identities must be non-empty"
            )
        if self.split not in {"validation", "test"}:
            raise PerceptionPartitionGovernanceError(
                "owned comparison split must be validation or test"
            )
        if self.comparison_report.split != self.split:
            raise PerceptionPartitionGovernanceError(
                "comparison split does not match partition ownership"
            )
        if self.semantics != PARTITION_GOVERNANCE_SEMANTICS:
            raise PerceptionPartitionGovernanceError(
                "unsupported partition governance semantics"
            )
        if self.version != PARTITION_OWNED_COMPARISON_VERSION:
            raise PerceptionPartitionGovernanceError(
                "unsupported owned comparison version"
            )


@dataclass(frozen=True)
class PartitionOwnedTestEvaluationReportDTO:
    owned_report_id: str
    partition_id: str
    dataset_id: str
    action_schema_id: str
    test_report: PromotedTestEvaluationReportDTO
    semantics: str = PARTITION_GOVERNANCE_SEMANTICS
    version: str = PARTITION_OWNED_TEST_VERSION

    def __post_init__(self) -> None:
        if not all(
            (
                self.owned_report_id,
                self.partition_id,
                self.dataset_id,
                self.action_schema_id,
            )
        ):
            raise PerceptionPartitionGovernanceError(
                "owned test identities must be non-empty"
            )
        if self.test_report.split != "test":
            raise PerceptionPartitionGovernanceError(
                "owned final evaluation must be test-owned"
            )
        if self.semantics != PARTITION_GOVERNANCE_SEMANTICS:
            raise PerceptionPartitionGovernanceError(
                "unsupported partition governance semantics"
            )
        if self.version != PARTITION_OWNED_TEST_VERSION:
            raise PerceptionPartitionGovernanceError(
                "unsupported owned test version"
            )


def bind_comparison_report_to_partition(
    report: TemporalInferenceComparisonReportDTO,
    partition: DatasetPartitionDTO,
) -> PartitionOwnedComparisonReportDTO:
    """Bind a comparison report to exact manifest-owned validation or test evidence."""

    if partition.split not in {"validation", "test"}:
        raise PerceptionPartitionGovernanceError(
            "comparison ownership requires a validation or test partition"
        )
    if report.split != partition.split:
        raise PerceptionPartitionGovernanceError(
            "comparison report split does not match partition"
        )
    report_ids = tuple(sorted(item.interaction_id for item in report.examples))
    unowned = tuple(
        value for value in report_ids if not partition.owns_interaction(value)
    )
    if unowned:
        raise PerceptionPartitionGovernanceError(
            f"comparison contains interactions outside partition: {list(unowned)}"
        )
    payload: Mapping[str, object] = {
        "action_schema_id": partition.action_schema_id,
        "comparison_report_id": report.report_id,
        "dataset_id": partition.dataset_id,
        "partition_id": partition.partition_id,
        "report_interaction_ids": list(report_ids),
        "semantics": PARTITION_GOVERNANCE_SEMANTICS,
        "split": partition.split,
        "version": PARTITION_OWNED_COMPARISON_VERSION,
    }
    return PartitionOwnedComparisonReportDTO(
        owned_report_id=_digest(payload),
        partition_id=partition.partition_id,
        dataset_id=partition.dataset_id,
        action_schema_id=partition.action_schema_id,
        split=partition.split,
        comparison_report=report,
    )


def calibrate_partition_owned_candidates(
    owned_report: PartitionOwnedComparisonReportDTO,
    *,
    policy: PromotionPolicyDTO | None = None,
) -> tuple[ModelCalibrationDTO, ModelCalibrationDTO]:
    """Calibrate only from a manifest-owned validation partition."""

    if owned_report.split != "validation":
        raise PerceptionPartitionGovernanceError(
            "calibration requires validation partition ownership"
        )
    return calibrate_comparison_candidates(owned_report.comparison_report, policy=policy)


def promote_partition_owned_model(
    owned_report: PartitionOwnedComparisonReportDTO,
    *,
    policy: PromotionPolicyDTO | None = None,
) -> tuple[PromotionDecisionDTO, PromotedPerceptionModelDTO]:
    """Promote only from a manifest-owned validation partition."""

    if owned_report.split != "validation":
        raise PerceptionPartitionGovernanceError(
            "promotion requires validation partition ownership"
        )
    return promote_perception_model(owned_report.comparison_report, policy=policy)


def evaluate_partition_owned_model_on_test(
    partition: DatasetPartitionDTO,
    promoted: PromotedPerceptionModelDTO,
    action_schema: DiscreteActionSchemaDTO,
    test_temporal_sources: tuple[TemporalSourceVPMDTO, ...],
    current_sources: Mapping[str, SourceVPMDTO],
    *,
    single_translator: SourceTargetTranslatorDTO | None = None,
    temporal_translator: TemporalTranslatorDTO | None = None,
    single_field_schema: VPMFieldSchemaDTO | None = None,
    temporal_field_schema: VPMFieldSchemaDTO | None = None,
) -> PartitionOwnedTestEvaluationReportDTO:
    """Evaluate the frozen model against an exact manifest-owned test partition."""

    if partition.split != "test":
        raise PerceptionPartitionGovernanceError(
            "final evaluation requires test partition ownership"
        )
    if partition.action_schema_id != action_schema.action_schema_id:
        raise PerceptionPartitionGovernanceError(
            "test partition action schema mismatch"
        )
    target_ids = tuple(
        sorted(item.target_interaction_id for item in test_temporal_sources)
    )
    unowned = tuple(
        value for value in target_ids if not partition.owns_interaction(value)
    )
    if unowned:
        raise PerceptionPartitionGovernanceError(
            f"test evaluation contains interactions outside partition: {list(unowned)}"
        )
    report = evaluate_promoted_model_on_test(
        promoted,
        action_schema,
        test_temporal_sources,
        current_sources,
        single_translator=single_translator,
        temporal_translator=temporal_translator,
        single_field_schema=single_field_schema,
        temporal_field_schema=temporal_field_schema,
        split="test",
    )
    payload: Mapping[str, object] = {
        "action_schema_id": partition.action_schema_id,
        "dataset_id": partition.dataset_id,
        "partition_id": partition.partition_id,
        "semantics": PARTITION_GOVERNANCE_SEMANTICS,
        "test_report_id": report.report_id,
        "version": PARTITION_OWNED_TEST_VERSION,
    }
    return PartitionOwnedTestEvaluationReportDTO(
        owned_report_id=_digest(payload),
        partition_id=partition.partition_id,
        dataset_id=partition.dataset_id,
        action_schema_id=partition.action_schema_id,
        test_report=report,
    )
