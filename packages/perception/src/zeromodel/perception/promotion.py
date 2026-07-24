"""Validation-owned rejection calibration and immutable model promotion for Stage P10.

P10 converts P9 comparison artifacts into an explicit operating decision. Thresholds
are selected only from validation examples. Promotion is deterministic, preserves all
candidate identities and metrics, and does not retrain or mutate either candidate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .temporal_inference import TemporalInferenceComparisonReportDTO

PROMOTION_POLICY_VERSION: Final = "perception-promotion-policy/1"
MODEL_CALIBRATION_VERSION: Final = "perception-model-calibration/1"
MODEL_PROMOTION_VERSION: Final = "perception-promoted-model/1"
PROMOTION_DECISION_VERSION: Final = "perception-promotion-decision/1"
CALIBRATION_SEMANTICS: Final = (
    "validation_margin_threshold_maximizing_accepted_accuracy_then_coverage"
)
PROMOTION_SEMANTICS: Final = (
    "validation_candidate_selection_by_accepted_accuracy_accuracy_coverage_then_simplicity"
)
PROMOTED_MODEL_KINDS: Final = {"single_frame", "temporal"}


class PerceptionPromotionError(ValueError):
    """Raised when P10 calibration or promotion contracts are invalid."""


def _canonical_json(payload: Mapping[str, object]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _digest(*parts: bytes) -> str:
    hasher = hashlib.sha256()
    for part in parts:
        hasher.update(len(part).to_bytes(8, "big"))
        hasher.update(part)
    return f"sha256:{hasher.hexdigest()}"


@dataclass(frozen=True)
class PromotionPolicyDTO:
    """Explicit deterministic operating constraints for calibration and promotion."""

    minimum_coverage: float = 0.5
    minimum_accuracy_gain: float = 0.0
    prefer_simpler_on_tie: bool = True
    version: str = PROMOTION_POLICY_VERSION

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_coverage <= 1.0:
            raise PerceptionPromotionError("minimum_coverage must be in [0, 1]")
        if not -1.0 <= self.minimum_accuracy_gain <= 1.0:
            raise PerceptionPromotionError("minimum_accuracy_gain must be in [-1, 1]")

    def canonical_payload(self) -> Mapping[str, object]:
        return {
            "minimum_accuracy_gain": self.minimum_accuracy_gain,
            "minimum_coverage": self.minimum_coverage,
            "prefer_simpler_on_tie": self.prefer_simpler_on_tie,
            "version": self.version,
        }


@dataclass(frozen=True)
class ModelCalibrationDTO:
    calibration_id: str
    comparison_report_id: str
    model_kind: str
    model_id: str
    validation_split: str
    rejection_threshold: float
    accepted_count: int
    rejected_count: int
    coverage: float
    accepted_accuracy: float
    raw_accuracy: float
    candidate_thresholds: tuple[float, ...]
    semantics: str = CALIBRATION_SEMANTICS
    version: str = MODEL_CALIBRATION_VERSION

    def __post_init__(self) -> None:
        if self.model_kind not in PROMOTED_MODEL_KINDS:
            raise PerceptionPromotionError("unsupported calibration model_kind")
        if self.validation_split != "validation":
            raise PerceptionPromotionError("calibration must be validation-owned")
        if not all((self.calibration_id, self.comparison_report_id, self.model_id)):
            raise PerceptionPromotionError("calibration identities must be non-empty")
        for value in (
            self.rejection_threshold,
            self.coverage,
            self.accepted_accuracy,
            self.raw_accuracy,
        ):
            if not 0.0 <= value <= 1.0:
                raise PerceptionPromotionError("calibration metric outside [0, 1]")
        if self.accepted_count <= 0 or self.rejected_count < 0:
            raise PerceptionPromotionError("calibration counts are invalid")
        if self.candidate_thresholds != tuple(sorted(set(self.candidate_thresholds))):
            raise PerceptionPromotionError("candidate_thresholds must be unique and sorted")
        if self.semantics != CALIBRATION_SEMANTICS:
            raise PerceptionPromotionError("unsupported calibration semantics")


@dataclass(frozen=True)
class PromotionDecisionDTO:
    decision_id: str
    comparison_report_id: str
    selected_model_kind: str
    selected_model_id: str
    single_calibration_id: str
    temporal_calibration_id: str
    selection_reason: str
    policy: PromotionPolicyDTO
    semantics: str = PROMOTION_SEMANTICS
    version: str = PROMOTION_DECISION_VERSION

    def __post_init__(self) -> None:
        if self.selected_model_kind not in PROMOTED_MODEL_KINDS:
            raise PerceptionPromotionError("unsupported selected model kind")
        if not all(
            (
                self.decision_id,
                self.comparison_report_id,
                self.selected_model_id,
                self.single_calibration_id,
                self.temporal_calibration_id,
                self.selection_reason,
            )
        ):
            raise PerceptionPromotionError("promotion decision identities must be non-empty")
        if self.semantics != PROMOTION_SEMANTICS:
            raise PerceptionPromotionError("unsupported promotion semantics")


@dataclass(frozen=True)
class PromotedPerceptionModelDTO:
    promoted_model_id: str
    model_kind: str
    model_id: str
    rejection_threshold: float
    calibration_id: str
    promotion_decision_id: str
    validation_comparison_report_id: str
    training_split: str
    evaluation_split: str
    temporal_window_spec_id: str | None = None
    version: str = MODEL_PROMOTION_VERSION

    def __post_init__(self) -> None:
        if self.model_kind not in PROMOTED_MODEL_KINDS:
            raise PerceptionPromotionError("unsupported promoted model kind")
        if not all(
            (
                self.promoted_model_id,
                self.model_id,
                self.calibration_id,
                self.promotion_decision_id,
                self.validation_comparison_report_id,
            )
        ):
            raise PerceptionPromotionError("promoted model identities must be non-empty")
        if not 0.0 <= self.rejection_threshold <= 1.0:
            raise PerceptionPromotionError("rejection_threshold must be in [0, 1]")
        if self.training_split != "train" or self.evaluation_split != "validation":
            raise PerceptionPromotionError("promotion requires train/validation provenance")
        if self.model_kind == "temporal" and not self.temporal_window_spec_id:
            raise PerceptionPromotionError("temporal promotion requires window identity")
        if self.model_kind == "single_frame" and self.temporal_window_spec_id is not None:
            raise PerceptionPromotionError("single-frame promotion cannot carry temporal window")


def _calibrate_one(
    report: TemporalInferenceComparisonReportDTO,
    model_kind: str,
    policy: PromotionPolicyDTO,
) -> ModelCalibrationDTO:
    if report.split != "validation":
        raise PerceptionPromotionError("calibration requires a validation comparison report")
    margins = tuple(
        item.single_margin if model_kind == "single_frame" else item.temporal_margin
        for item in report.examples
    )
    correct = tuple(
        item.single_correct if model_kind == "single_frame" else item.temporal_correct
        for item in report.examples
    )
    thresholds = tuple(sorted(set((0.0, *margins))))
    candidates: list[tuple[float, float, float, int, int]] = []
    for threshold in thresholds:
        accepted = tuple(index for index, margin in enumerate(margins) if margin >= threshold)
        if not accepted:
            continue
        coverage = len(accepted) / len(margins)
        if coverage < policy.minimum_coverage:
            continue
        accuracy = sum(1 for index in accepted if correct[index]) / len(accepted)
        candidates.append((accuracy, coverage, -threshold, len(accepted), threshold))
    if not candidates:
        raise PerceptionPromotionError("no calibration threshold satisfies minimum coverage")
    _, coverage, _, accepted_count, threshold = max(candidates)
    accepted_indices = tuple(index for index, margin in enumerate(margins) if margin >= threshold)
    accepted_accuracy = sum(1 for index in accepted_indices if correct[index]) / len(accepted_indices)
    raw_accuracy = sum(1 for value in correct if value) / len(correct)
    model_id = report.single_translator_id if model_kind == "single_frame" else report.temporal_translator_id
    payload: Mapping[str, object] = {
        "accepted_accuracy": accepted_accuracy,
        "accepted_count": accepted_count,
        "candidate_thresholds": list(thresholds),
        "comparison_report_id": report.report_id,
        "coverage": coverage,
        "model_id": model_id,
        "model_kind": model_kind,
        "raw_accuracy": raw_accuracy,
        "rejected_count": len(margins) - accepted_count,
        "rejection_threshold": threshold,
        "semantics": CALIBRATION_SEMANTICS,
        "validation_split": report.split,
        "version": MODEL_CALIBRATION_VERSION,
    }
    return ModelCalibrationDTO(
        calibration_id=_digest(_canonical_json(payload)),
        comparison_report_id=report.report_id,
        model_kind=model_kind,
        model_id=model_id,
        validation_split=report.split,
        rejection_threshold=threshold,
        accepted_count=accepted_count,
        rejected_count=len(margins) - accepted_count,
        coverage=coverage,
        accepted_accuracy=accepted_accuracy,
        raw_accuracy=raw_accuracy,
        candidate_thresholds=thresholds,
    )


def calibrate_comparison_candidates(
    report: TemporalInferenceComparisonReportDTO,
    *,
    policy: PromotionPolicyDTO | None = None,
) -> tuple[ModelCalibrationDTO, ModelCalibrationDTO]:
    """Calibrate single-frame and temporal rejection using validation examples only."""

    resolved = policy or PromotionPolicyDTO()
    return (
        _calibrate_one(report, "single_frame", resolved),
        _calibrate_one(report, "temporal", resolved),
    )


def promote_perception_model(
    report: TemporalInferenceComparisonReportDTO,
    *,
    policy: PromotionPolicyDTO | None = None,
) -> tuple[PromotionDecisionDTO, PromotedPerceptionModelDTO]:
    """Select and promote one immutable candidate from a validation comparison."""

    resolved = policy or PromotionPolicyDTO()
    single, temporal = calibrate_comparison_candidates(report, policy=resolved)
    temporal_gain = temporal.accepted_accuracy - single.accepted_accuracy
    if temporal_gain > resolved.minimum_accuracy_gain:
        selected = temporal
        reason = "temporal_exceeds_required_accepted_accuracy_gain"
    elif temporal_gain < resolved.minimum_accuracy_gain:
        selected = single
        reason = "temporal_does_not_meet_required_accepted_accuracy_gain"
    else:
        single_rank = (single.raw_accuracy, single.coverage)
        temporal_rank = (temporal.raw_accuracy, temporal.coverage)
        if temporal_rank > single_rank:
            selected = temporal
            reason = "temporal_wins_raw_accuracy_or_coverage_tiebreak"
        elif temporal_rank < single_rank:
            selected = single
            reason = "single_frame_wins_raw_accuracy_or_coverage_tiebreak"
        elif resolved.prefer_simpler_on_tie:
            selected = single
            reason = "exact_tie_prefers_simpler_single_frame_model"
        else:
            selected = temporal
            reason = "exact_tie_policy_prefers_temporal_model"
    decision_payload: Mapping[str, object] = {
        "comparison_report_id": report.report_id,
        "policy": resolved.canonical_payload(),
        "selected_model_id": selected.model_id,
        "selected_model_kind": selected.model_kind,
        "selection_reason": reason,
        "semantics": PROMOTION_SEMANTICS,
        "single_calibration_id": single.calibration_id,
        "temporal_calibration_id": temporal.calibration_id,
        "version": PROMOTION_DECISION_VERSION,
    }
    decision = PromotionDecisionDTO(
        decision_id=_digest(_canonical_json(decision_payload)),
        comparison_report_id=report.report_id,
        selected_model_kind=selected.model_kind,
        selected_model_id=selected.model_id,
        single_calibration_id=single.calibration_id,
        temporal_calibration_id=temporal.calibration_id,
        selection_reason=reason,
        policy=resolved,
    )
    promoted_payload: Mapping[str, object] = {
        "calibration_id": selected.calibration_id,
        "evaluation_split": "validation",
        "model_id": selected.model_id,
        "model_kind": selected.model_kind,
        "promotion_decision_id": decision.decision_id,
        "rejection_threshold": selected.rejection_threshold,
        "temporal_window_spec_id": (
            report.temporal_window_spec_id if selected.model_kind == "temporal" else None
        ),
        "training_split": "train",
        "validation_comparison_report_id": report.report_id,
        "version": MODEL_PROMOTION_VERSION,
    }
    promoted = PromotedPerceptionModelDTO(
        promoted_model_id=_digest(_canonical_json(promoted_payload)),
        model_kind=selected.model_kind,
        model_id=selected.model_id,
        rejection_threshold=selected.rejection_threshold,
        calibration_id=selected.calibration_id,
        promotion_decision_id=decision.decision_id,
        validation_comparison_report_id=report.report_id,
        training_split="train",
        evaluation_split="validation",
        temporal_window_spec_id=(
            report.temporal_window_spec_id if selected.model_kind == "temporal" else None
        ),
    )
    return decision, promoted
