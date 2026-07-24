"""Unified promoted inference and untouched test evaluation for Stage P11.

P11 executes exactly the candidate and rejection threshold frozen by P10. The runtime
accepts either a current Source VPM or a P8 temporal source according to the promoted
model kind, preserves full prediction provenance, and evaluates the unchanged operating
point on a caller-declared untouched test split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping

from .fields import VPMFieldSchemaDTO
from .promotion import PromotedPerceptionModelDTO
from .representation import DiscreteActionSchemaDTO, SourceVPMDTO
from .temporal import TemporalSourceVPMDTO
from .temporal_inference import (
    TemporalTranslatorDTO,
    predict_temporal_action,
)
from .translator import (
    SourceTargetTranslatorDTO,
    TargetActionScoreDTO,
    predict_target_vpm,
)

PROMOTED_INFERENCE_VERSION: Final = "perception-promoted-inference/1"
PROMOTED_TEST_EVALUATION_VERSION: Final = "perception-promoted-test-evaluation/1"
PROMOTED_INFERENCE_SEMANTICS: Final = (
    "dispatch_promoted_model_and_apply_frozen_validation_rejection_threshold"
)
PROMOTED_TEST_EVALUATION_SEMANTICS: Final = (
    "untouched_test_evaluation_of_exact_promoted_operating_point"
)


class PerceptionPromotedInferenceError(ValueError):
    """Raised when P11 runtime or final-evaluation contracts are violated."""


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
class PromotedInferenceResultDTO:
    result_id: str
    promoted_model_id: str
    model_kind: str
    model_id: str
    input_id: str
    interaction_id: str | None
    scores: tuple[TargetActionScoreDTO, ...]
    selected_action: str
    margin: float
    status: str
    rejection_threshold: float
    calibration_id: str
    promotion_decision_id: str
    semantics: str = PROMOTED_INFERENCE_SEMANTICS
    version: str = PROMOTED_INFERENCE_VERSION

    def __post_init__(self) -> None:
        if self.model_kind not in {"single_frame", "temporal"}:
            raise PerceptionPromotedInferenceError("unsupported promoted model kind")
        if self.status not in {"accepted", "rejected_ambiguous"}:
            raise PerceptionPromotedInferenceError("unsupported promoted inference status")
        if not all(
            (
                self.result_id,
                self.promoted_model_id,
                self.model_id,
                self.input_id,
                self.selected_action,
                self.calibration_id,
                self.promotion_decision_id,
            )
        ):
            raise PerceptionPromotedInferenceError("promoted inference identities must be non-empty")
        if not self.scores or self.scores[0].action_label != self.selected_action:
            raise PerceptionPromotedInferenceError("selected action must match top-ranked score")
        if not 0.0 <= self.margin <= 1.0:
            raise PerceptionPromotedInferenceError("margin must be in [0, 1]")
        if not 0.0 <= self.rejection_threshold <= 1.0:
            raise PerceptionPromotedInferenceError("rejection_threshold must be in [0, 1]")
        if self.semantics != PROMOTED_INFERENCE_SEMANTICS:
            raise PerceptionPromotedInferenceError("unsupported promoted inference semantics")


@dataclass(frozen=True)
class PromotedTestExampleDTO:
    interaction_id: str
    expected_action: str
    result_id: str
    selected_action: str
    margin: float
    status: str
    correct: bool

    def __post_init__(self) -> None:
        if not all((self.interaction_id, self.expected_action, self.result_id, self.selected_action)):
            raise PerceptionPromotedInferenceError("test example identities must be non-empty")
        if self.status not in {"accepted", "rejected_ambiguous"}:
            raise PerceptionPromotedInferenceError("unsupported test example status")
        if not 0.0 <= self.margin <= 1.0:
            raise PerceptionPromotedInferenceError("test example margin must be in [0, 1]")


@dataclass(frozen=True)
class PromotedTestEvaluationReportDTO:
    report_id: str
    promoted_model_id: str
    model_kind: str
    model_id: str
    calibration_id: str
    promotion_decision_id: str
    validation_comparison_report_id: str
    split: str
    example_count: int
    accepted_count: int
    rejected_count: int
    raw_accuracy: float
    accepted_accuracy: float | None
    coverage: float
    mean_margin: float
    rejection_threshold: float
    examples: tuple[PromotedTestExampleDTO, ...]
    semantics: str = PROMOTED_TEST_EVALUATION_SEMANTICS
    version: str = PROMOTED_TEST_EVALUATION_VERSION

    def __post_init__(self) -> None:
        if self.split != "test":
            raise PerceptionPromotedInferenceError("final promoted evaluation must use test split")
        if self.model_kind not in {"single_frame", "temporal"}:
            raise PerceptionPromotedInferenceError("unsupported test model kind")
        if not all(
            (
                self.report_id,
                self.promoted_model_id,
                self.model_id,
                self.calibration_id,
                self.promotion_decision_id,
                self.validation_comparison_report_id,
            )
        ):
            raise PerceptionPromotedInferenceError("test report identities must be non-empty")
        if self.example_count <= 0 or self.example_count != len(self.examples):
            raise PerceptionPromotedInferenceError("test example count is invalid")
        if self.accepted_count + self.rejected_count != self.example_count:
            raise PerceptionPromotedInferenceError("accepted and rejected counts must exhaust examples")
        if self.examples != tuple(sorted(self.examples, key=lambda item: item.interaction_id)):
            raise PerceptionPromotedInferenceError("test examples must be sorted")
        for value in (self.raw_accuracy, self.coverage, self.mean_margin, self.rejection_threshold):
            if not 0.0 <= value <= 1.0:
                raise PerceptionPromotedInferenceError("bounded test metric outside [0, 1]")
        if self.accepted_accuracy is not None and not 0.0 <= self.accepted_accuracy <= 1.0:
            raise PerceptionPromotedInferenceError("accepted_accuracy outside [0, 1]")
        if self.semantics != PROMOTED_TEST_EVALUATION_SEMANTICS:
            raise PerceptionPromotedInferenceError("unsupported test evaluation semantics")


def run_promoted_inference(
    promoted: PromotedPerceptionModelDTO,
    action_schema: DiscreteActionSchemaDTO,
    *,
    single_translator: SourceTargetTranslatorDTO | None = None,
    temporal_translator: TemporalTranslatorDTO | None = None,
    single_field_schema: VPMFieldSchemaDTO | None = None,
    temporal_field_schema: VPMFieldSchemaDTO | None = None,
    source: SourceVPMDTO | None = None,
    temporal_source: TemporalSourceVPMDTO | None = None,
) -> PromotedInferenceResultDTO:
    """Execute the exact promoted candidate with its frozen validation threshold."""

    if promoted.model_kind == "single_frame":
        if any(value is None for value in (single_translator, single_field_schema, source)):
            raise PerceptionPromotedInferenceError(
                "single-frame promotion requires translator, field schema, and source"
            )
        if temporal_source is not None:
            raise PerceptionPromotedInferenceError("single-frame promotion cannot consume temporal source")
        assert single_translator is not None
        assert single_field_schema is not None
        assert source is not None
        if single_translator.translator_id != promoted.model_id:
            raise PerceptionPromotedInferenceError("single-frame translator does not match promoted model")
        prediction = predict_target_vpm(single_translator, source, single_field_schema, action_schema)
        input_id = source.source_vpm_id
        interaction_id = None
        scores = prediction.scores
        selected_action = prediction.selected_action
        margin = prediction.margin
    else:
        if any(value is None for value in (temporal_translator, temporal_field_schema, temporal_source)):
            raise PerceptionPromotedInferenceError(
                "temporal promotion requires translator, field schema, and temporal source"
            )
        if source is not None:
            raise PerceptionPromotedInferenceError("temporal promotion cannot consume standalone source")
        assert temporal_translator is not None
        assert temporal_field_schema is not None
        assert temporal_source is not None
        if temporal_translator.temporal_translator_id != promoted.model_id:
            raise PerceptionPromotedInferenceError("temporal translator does not match promoted model")
        if temporal_source.temporal_window_spec_id != promoted.temporal_window_spec_id:
            raise PerceptionPromotedInferenceError("temporal source does not match promoted window")
        prediction = predict_temporal_action(
            temporal_translator,
            temporal_source,
            temporal_field_schema,
            rejection_threshold=promoted.rejection_threshold,
        )
        input_id = temporal_source.temporal_source_id
        interaction_id = temporal_source.target_interaction_id
        scores = prediction.scores
        selected_action = prediction.selected_action
        margin = prediction.margin

    status = "accepted" if margin >= promoted.rejection_threshold else "rejected_ambiguous"
    payload: Mapping[str, object] = {
        "calibration_id": promoted.calibration_id,
        "input_id": input_id,
        "interaction_id": interaction_id,
        "margin": margin,
        "model_id": promoted.model_id,
        "model_kind": promoted.model_kind,
        "promoted_model_id": promoted.promoted_model_id,
        "promotion_decision_id": promoted.promotion_decision_id,
        "rejection_threshold": promoted.rejection_threshold,
        "scores": [
            {"action_label": item.action_label, "rank": item.rank, "score": item.score}
            for item in scores
        ],
        "selected_action": selected_action,
        "semantics": PROMOTED_INFERENCE_SEMANTICS,
        "status": status,
        "version": PROMOTED_INFERENCE_VERSION,
    }
    return PromotedInferenceResultDTO(
        result_id=_digest(_canonical_json(payload)),
        promoted_model_id=promoted.promoted_model_id,
        model_kind=promoted.model_kind,
        model_id=promoted.model_id,
        input_id=input_id,
        interaction_id=interaction_id,
        scores=scores,
        selected_action=selected_action,
        margin=margin,
        status=status,
        rejection_threshold=promoted.rejection_threshold,
        calibration_id=promoted.calibration_id,
        promotion_decision_id=promoted.promotion_decision_id,
    )


def evaluate_promoted_model_on_test(
    promoted: PromotedPerceptionModelDTO,
    action_schema: DiscreteActionSchemaDTO,
    test_temporal_sources: tuple[TemporalSourceVPMDTO, ...],
    current_sources: Mapping[str, SourceVPMDTO],
    *,
    single_translator: SourceTargetTranslatorDTO | None = None,
    temporal_translator: TemporalTranslatorDTO | None = None,
    single_field_schema: VPMFieldSchemaDTO | None = None,
    temporal_field_schema: VPMFieldSchemaDTO | None = None,
    split: str = "test",
) -> PromotedTestEvaluationReportDTO:
    """Evaluate the frozen promoted operating point on untouched aligned test examples."""

    if split != "test":
        raise PerceptionPromotedInferenceError("final promoted evaluation requires split='test'")
    if not test_temporal_sources:
        raise PerceptionPromotedInferenceError("test evaluation requires examples")
    seen: set[str] = set()
    examples: list[PromotedTestExampleDTO] = []
    for item in sorted(test_temporal_sources, key=lambda value: value.target_interaction_id):
        if item.target_interaction_id in seen:
            raise PerceptionPromotedInferenceError("test interaction ids must be unique")
        seen.add(item.target_interaction_id)
        if promoted.model_kind == "single_frame":
            try:
                current = current_sources[item.current_source_vpm_id]
            except KeyError as exc:
                raise PerceptionPromotedInferenceError(
                    f"missing current SourceVPMDTO for {exc.args[0]}"
                ) from exc
            result = run_promoted_inference(
                promoted,
                action_schema,
                single_translator=single_translator,
                single_field_schema=single_field_schema,
                source=current,
            )
        else:
            result = run_promoted_inference(
                promoted,
                action_schema,
                temporal_translator=temporal_translator,
                temporal_field_schema=temporal_field_schema,
                temporal_source=item,
            )
        examples.append(
            PromotedTestExampleDTO(
                interaction_id=item.target_interaction_id,
                expected_action=item.action_label,
                result_id=result.result_id,
                selected_action=result.selected_action,
                margin=result.margin,
                status=result.status,
                correct=result.selected_action == item.action_label,
            )
        )

    ordered = tuple(sorted(examples, key=lambda value: value.interaction_id))
    accepted = tuple(item for item in ordered if item.status == "accepted")
    raw_accuracy = sum(1 for item in ordered if item.correct) / len(ordered)
    accepted_accuracy = (
        sum(1 for item in accepted if item.correct) / len(accepted) if accepted else None
    )
    coverage = len(accepted) / len(ordered)
    mean_margin = sum(item.margin for item in ordered) / len(ordered)
    payload: Mapping[str, object] = {
        "accepted_accuracy": accepted_accuracy,
        "accepted_count": len(accepted),
        "calibration_id": promoted.calibration_id,
        "coverage": coverage,
        "example_count": len(ordered),
        "examples": [
            {
                "correct": item.correct,
                "expected_action": item.expected_action,
                "interaction_id": item.interaction_id,
                "margin": item.margin,
                "result_id": item.result_id,
                "selected_action": item.selected_action,
                "status": item.status,
            }
            for item in ordered
        ],
        "mean_margin": mean_margin,
        "model_id": promoted.model_id,
        "model_kind": promoted.model_kind,
        "promoted_model_id": promoted.promoted_model_id,
        "promotion_decision_id": promoted.promotion_decision_id,
        "raw_accuracy": raw_accuracy,
        "rejected_count": len(ordered) - len(accepted),
        "rejection_threshold": promoted.rejection_threshold,
        "semantics": PROMOTED_TEST_EVALUATION_SEMANTICS,
        "split": split,
        "validation_comparison_report_id": promoted.validation_comparison_report_id,
        "version": PROMOTED_TEST_EVALUATION_VERSION,
    }
    return PromotedTestEvaluationReportDTO(
        report_id=_digest(_canonical_json(payload)),
        promoted_model_id=promoted.promoted_model_id,
        model_kind=promoted.model_kind,
        model_id=promoted.model_id,
        calibration_id=promoted.calibration_id,
        promotion_decision_id=promoted.promotion_decision_id,
        validation_comparison_report_id=promoted.validation_comparison_report_id,
        split=split,
        example_count=len(ordered),
        accepted_count=len(accepted),
        rejected_count=len(ordered) - len(accepted),
        raw_accuracy=raw_accuracy,
        accepted_accuracy=accepted_accuracy,
        coverage=coverage,
        mean_margin=mean_margin,
        rejection_threshold=promoted.rejection_threshold,
        examples=ordered,
    )
