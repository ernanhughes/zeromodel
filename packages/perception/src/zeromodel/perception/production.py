"""Immutable production inference and outcome ledger for Stage P14.

P14 records each operational inference against the exact P12/P13 active-model
pointer revision. Outcomes are appended later as separate immutable records. Windowed
metrics are derived from ledger records and never mutate the underlying evidence.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Final, Mapping, Protocol

from .lifecycle import ActiveModelPointerDTO, PromotedModelLedgerEntryDTO
from .promoted_inference import PromotedInferenceResultDTO

PRODUCTION_INFERENCE_RECORD_VERSION: Final = "perception-production-inference-record/1"
PRODUCTION_OUTCOME_RECORD_VERSION: Final = "perception-production-outcome-record/1"
PRODUCTION_METRICS_REPORT_VERSION: Final = "perception-production-metrics-report/1"
PRODUCTION_INFERENCE_SEMANTICS: Final = (
    "append_only_runtime_inference_bound_to_active_model_pointer_revision"
)
PRODUCTION_OUTCOME_SEMANTICS: Final = "append_only_observed_outcome_for_runtime_inference"
PRODUCTION_METRICS_SEMANTICS: Final = (
    "windowed_operational_metrics_over_immutable_inference_and_outcome_records"
)


class PerceptionProductionLedgerError(ValueError):
    """Raised when production-ledger contracts are violated."""


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
class ProductionInferenceRecordDTO:
    record_id: str
    sequence_number: int
    pointer_id: str
    pointer_revision: int
    promoted_model_id: str
    model_id: str
    model_kind: str
    input_id: str
    interaction_id: str | None
    inference_result_id: str
    selected_action: str
    margin: float
    status: str
    rejection_threshold: float
    semantics: str = PRODUCTION_INFERENCE_SEMANTICS
    version: str = PRODUCTION_INFERENCE_RECORD_VERSION

    def __post_init__(self) -> None:
        if self.sequence_number <= 0:
            raise PerceptionProductionLedgerError("inference sequence_number must be positive")
        if self.pointer_revision <= 0:
            raise PerceptionProductionLedgerError("production inference requires active pointer revision")
        if self.model_kind not in {"single_frame", "temporal"}:
            raise PerceptionProductionLedgerError("unsupported production model kind")
        if self.status not in {"accepted", "rejected_ambiguous"}:
            raise PerceptionProductionLedgerError("unsupported production inference status")
        if not all(
            (
                self.record_id,
                self.pointer_id,
                self.promoted_model_id,
                self.model_id,
                self.input_id,
                self.inference_result_id,
                self.selected_action,
            )
        ):
            raise PerceptionProductionLedgerError("production inference identities must be non-empty")
        for value in (self.margin, self.rejection_threshold):
            if not 0.0 <= value <= 1.0:
                raise PerceptionProductionLedgerError("production inference metric outside [0, 1]")
        if self.semantics != PRODUCTION_INFERENCE_SEMANTICS:
            raise PerceptionProductionLedgerError("unsupported production inference semantics")


@dataclass(frozen=True)
class ProductionOutcomeRecordDTO:
    outcome_id: str
    inference_record_id: str
    outcome_sequence_number: int
    observed_action: str
    source: str
    correct: bool
    semantics: str = PRODUCTION_OUTCOME_SEMANTICS
    version: str = PRODUCTION_OUTCOME_RECORD_VERSION

    def __post_init__(self) -> None:
        if self.outcome_sequence_number <= 0:
            raise PerceptionProductionLedgerError("outcome sequence_number must be positive")
        if not all((self.outcome_id, self.inference_record_id, self.observed_action, self.source)):
            raise PerceptionProductionLedgerError("production outcome fields must be non-empty")
        if self.semantics != PRODUCTION_OUTCOME_SEMANTICS:
            raise PerceptionProductionLedgerError("unsupported production outcome semantics")


@dataclass(frozen=True)
class ProductionMetricsReportDTO:
    report_id: str
    promoted_model_id: str | None
    start_sequence_number: int
    end_sequence_number: int
    inference_count: int
    accepted_count: int
    rejected_count: int
    labeled_count: int
    correct_count: int
    raw_accuracy: float | None
    accepted_accuracy: float | None
    coverage: float
    mean_margin: float
    model_pointer_revisions: tuple[int, ...]
    inference_record_ids: tuple[str, ...]
    outcome_ids: tuple[str, ...]
    semantics: str = PRODUCTION_METRICS_SEMANTICS
    version: str = PRODUCTION_METRICS_REPORT_VERSION

    def __post_init__(self) -> None:
        if self.start_sequence_number <= 0 or self.end_sequence_number < self.start_sequence_number:
            raise PerceptionProductionLedgerError("invalid production metrics window")
        if self.inference_count <= 0 or self.inference_count != len(self.inference_record_ids):
            raise PerceptionProductionLedgerError("production metrics inference count is invalid")
        if self.accepted_count + self.rejected_count != self.inference_count:
            raise PerceptionProductionLedgerError("accepted and rejected counts must exhaust inferences")
        if self.correct_count > self.labeled_count or self.labeled_count != len(self.outcome_ids):
            raise PerceptionProductionLedgerError("production metrics outcome counts are invalid")
        for value in (self.coverage, self.mean_margin):
            if not 0.0 <= value <= 1.0:
                raise PerceptionProductionLedgerError("production metric outside [0, 1]")
        for value in (self.raw_accuracy, self.accepted_accuracy):
            if value is not None and not 0.0 <= value <= 1.0:
                raise PerceptionProductionLedgerError("production accuracy outside [0, 1]")
        if tuple(sorted(set(self.model_pointer_revisions))) != self.model_pointer_revisions:
            raise PerceptionProductionLedgerError("pointer revisions must be unique and sorted")
        if self.semantics != PRODUCTION_METRICS_SEMANTICS:
            raise PerceptionProductionLedgerError("unsupported production metrics semantics")


class PerceptionProductionLedgerStore(Protocol):
    def append_inference(self, record: ProductionInferenceRecordDTO) -> None: ...

    def get_inference(self, record_id: str) -> ProductionInferenceRecordDTO: ...

    def list_inferences(self) -> tuple[ProductionInferenceRecordDTO, ...]: ...

    def append_outcome(self, outcome: ProductionOutcomeRecordDTO) -> None: ...

    def get_outcome_for_inference(self, record_id: str) -> ProductionOutcomeRecordDTO | None: ...

    def list_outcomes(self) -> tuple[ProductionOutcomeRecordDTO, ...]: ...


class InMemoryPerceptionProductionLedgerStore:
    """Deterministic append-only P14 store implementation."""

    def __init__(self) -> None:
        self._inferences: list[ProductionInferenceRecordDTO] = []
        self._inference_by_id: dict[str, ProductionInferenceRecordDTO] = {}
        self._outcomes: list[ProductionOutcomeRecordDTO] = []
        self._outcome_by_inference: dict[str, ProductionOutcomeRecordDTO] = {}

    def append_inference(self, record: ProductionInferenceRecordDTO) -> None:
        expected = len(self._inferences) + 1
        if record.sequence_number != expected:
            raise PerceptionProductionLedgerError("production inference sequence is not contiguous")
        existing = self._inference_by_id.get(record.record_id)
        if existing is not None:
            if existing == record:
                return
            raise PerceptionProductionLedgerError("production inference identity conflict")
        self._inferences.append(record)
        self._inference_by_id[record.record_id] = record

    def get_inference(self, record_id: str) -> ProductionInferenceRecordDTO:
        try:
            return self._inference_by_id[record_id]
        except KeyError as exc:
            raise PerceptionProductionLedgerError(f"unknown production inference: {record_id}") from exc

    def list_inferences(self) -> tuple[ProductionInferenceRecordDTO, ...]:
        return tuple(self._inferences)

    def append_outcome(self, outcome: ProductionOutcomeRecordDTO) -> None:
        self.get_inference(outcome.inference_record_id)
        existing = self._outcome_by_inference.get(outcome.inference_record_id)
        if existing is not None:
            if existing == outcome:
                return
            raise PerceptionProductionLedgerError("inference already has a different outcome")
        expected = len(self._outcomes) + 1
        if outcome.outcome_sequence_number != expected:
            raise PerceptionProductionLedgerError("production outcome sequence is not contiguous")
        self._outcomes.append(outcome)
        self._outcome_by_inference[outcome.inference_record_id] = outcome

    def get_outcome_for_inference(self, record_id: str) -> ProductionOutcomeRecordDTO | None:
        return self._outcome_by_inference.get(record_id)

    def list_outcomes(self) -> tuple[ProductionOutcomeRecordDTO, ...]:
        return tuple(self._outcomes)


def record_production_inference(
    store: PerceptionProductionLedgerStore,
    pointer: ActiveModelPointerDTO,
    ledger_entry: PromotedModelLedgerEntryDTO,
    result: PromotedInferenceResultDTO,
) -> ProductionInferenceRecordDTO:
    """Append one runtime inference tied to the exact active pointer revision."""

    if pointer.active_promoted_model_id is None:
        raise PerceptionProductionLedgerError("cannot record production inference without active model")
    promoted = ledger_entry.promoted_model
    if pointer.active_promoted_model_id != promoted.promoted_model_id:
        raise PerceptionProductionLedgerError("ledger entry is not the active promoted model")
    if result.promoted_model_id != promoted.promoted_model_id:
        raise PerceptionProductionLedgerError("inference result does not belong to active promoted model")
    if result.model_id != promoted.model_id or result.model_kind != promoted.model_kind:
        raise PerceptionProductionLedgerError("inference result candidate does not match promoted model")
    sequence_number = len(store.list_inferences()) + 1
    payload: Mapping[str, object] = {
        "inference_result_id": result.result_id,
        "input_id": result.input_id,
        "interaction_id": result.interaction_id,
        "margin": result.margin,
        "model_id": result.model_id,
        "model_kind": result.model_kind,
        "pointer_id": pointer.pointer_id,
        "pointer_revision": pointer.revision,
        "promoted_model_id": result.promoted_model_id,
        "rejection_threshold": result.rejection_threshold,
        "selected_action": result.selected_action,
        "semantics": PRODUCTION_INFERENCE_SEMANTICS,
        "sequence_number": sequence_number,
        "status": result.status,
        "version": PRODUCTION_INFERENCE_RECORD_VERSION,
    }
    record = ProductionInferenceRecordDTO(
        record_id=_digest(payload),
        sequence_number=sequence_number,
        pointer_id=pointer.pointer_id,
        pointer_revision=pointer.revision,
        promoted_model_id=result.promoted_model_id,
        model_id=result.model_id,
        model_kind=result.model_kind,
        input_id=result.input_id,
        interaction_id=result.interaction_id,
        inference_result_id=result.result_id,
        selected_action=result.selected_action,
        margin=result.margin,
        status=result.status,
        rejection_threshold=result.rejection_threshold,
    )
    store.append_inference(record)
    return record


def record_production_outcome(
    store: PerceptionProductionLedgerStore,
    inference_record_id: str,
    *,
    observed_action: str,
    source: str,
) -> ProductionOutcomeRecordDTO:
    """Append one authoritative observed outcome for a prior production inference."""

    if not observed_action or not source:
        raise PerceptionProductionLedgerError("observed action and source must be non-empty")
    inference = store.get_inference(inference_record_id)
    sequence_number = len(store.list_outcomes()) + 1
    correct = inference.selected_action == observed_action
    payload: Mapping[str, object] = {
        "correct": correct,
        "inference_record_id": inference_record_id,
        "observed_action": observed_action,
        "outcome_sequence_number": sequence_number,
        "semantics": PRODUCTION_OUTCOME_SEMANTICS,
        "source": source,
        "version": PRODUCTION_OUTCOME_RECORD_VERSION,
    }
    outcome = ProductionOutcomeRecordDTO(
        outcome_id=_digest(payload),
        inference_record_id=inference_record_id,
        outcome_sequence_number=sequence_number,
        observed_action=observed_action,
        source=source,
        correct=correct,
    )
    store.append_outcome(outcome)
    return outcome


def build_production_metrics_report(
    store: PerceptionProductionLedgerStore,
    *,
    start_sequence_number: int,
    end_sequence_number: int | None = None,
    promoted_model_id: str | None = None,
) -> ProductionMetricsReportDTO:
    """Derive metrics over an inclusive inference sequence window."""

    all_records = store.list_inferences()
    if not all_records:
        raise PerceptionProductionLedgerError("production metrics require inference records")
    resolved_end = end_sequence_number or all_records[-1].sequence_number
    if start_sequence_number <= 0 or resolved_end < start_sequence_number:
        raise PerceptionProductionLedgerError("invalid production metrics sequence window")
    records = tuple(
        item
        for item in all_records
        if start_sequence_number <= item.sequence_number <= resolved_end
        and (promoted_model_id is None or item.promoted_model_id == promoted_model_id)
    )
    if not records:
        raise PerceptionProductionLedgerError("production metrics window contains no matching records")
    outcomes = tuple(
        outcome
        for item in records
        if (outcome := store.get_outcome_for_inference(item.record_id)) is not None
    )
    accepted = tuple(item for item in records if item.status == "accepted")
    accepted_ids = {item.record_id for item in accepted}
    accepted_outcomes = tuple(item for item in outcomes if item.inference_record_id in accepted_ids)
    correct_count = sum(1 for item in outcomes if item.correct)
    raw_accuracy = correct_count / len(outcomes) if outcomes else None
    accepted_correct = sum(1 for item in accepted_outcomes if item.correct)
    accepted_accuracy = (
        accepted_correct / len(accepted_outcomes) if accepted_outcomes else None
    )
    payload: Mapping[str, object] = {
        "accepted_accuracy": accepted_accuracy,
        "accepted_count": len(accepted),
        "correct_count": correct_count,
        "coverage": len(accepted) / len(records),
        "end_sequence_number": resolved_end,
        "inference_record_ids": [item.record_id for item in records],
        "labeled_count": len(outcomes),
        "mean_margin": sum(item.margin for item in records) / len(records),
        "model_pointer_revisions": sorted({item.pointer_revision for item in records}),
        "outcome_ids": [item.outcome_id for item in outcomes],
        "promoted_model_id": promoted_model_id,
        "raw_accuracy": raw_accuracy,
        "rejected_count": len(records) - len(accepted),
        "semantics": PRODUCTION_METRICS_SEMANTICS,
        "start_sequence_number": start_sequence_number,
        "version": PRODUCTION_METRICS_REPORT_VERSION,
    }
    return ProductionMetricsReportDTO(
        report_id=_digest(payload),
        promoted_model_id=promoted_model_id,
        start_sequence_number=start_sequence_number,
        end_sequence_number=resolved_end,
        inference_count=len(records),
        accepted_count=len(accepted),
        rejected_count=len(records) - len(accepted),
        labeled_count=len(outcomes),
        correct_count=correct_count,
        raw_accuracy=raw_accuracy,
        accepted_accuracy=accepted_accuracy,
        coverage=len(accepted) / len(records),
        mean_margin=sum(item.margin for item in records) / len(records),
        model_pointer_revisions=tuple(sorted({item.pointer_revision for item in records})),
        inference_record_ids=tuple(item.record_id for item in records),
        outcome_ids=tuple(item.outcome_id for item in outcomes),
    )
