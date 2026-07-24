from __future__ import annotations

import pytest

from zeromodel.perception import (
    InMemoryPerceptionProductionLedgerStore,
    PerceptionProductionLedgerError,
    ProductionInferenceRecordDTO,
    build_production_metrics_report,
    record_production_outcome,
)


def _record(
    sequence: int,
    *,
    status: str = "accepted",
    action: str = "LEFT",
    margin: float = 0.8,
    revision: int = 1,
    model: str = "promoted-a",
) -> ProductionInferenceRecordDTO:
    return ProductionInferenceRecordDTO(
        record_id=f"record-{sequence}",
        sequence_number=sequence,
        pointer_id=f"pointer-{revision}",
        pointer_revision=revision,
        promoted_model_id=model,
        model_id="translator-a",
        model_kind="single_frame",
        input_id=f"input-{sequence}",
        interaction_id=f"interaction-{sequence}",
        inference_result_id=f"result-{sequence}",
        selected_action=action,
        margin=margin,
        status=status,
        rejection_threshold=0.5,
    )


def test_append_only_inference_and_outcome_ledger() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    first = _record(1)
    second = _record(2, status="rejected_ambiguous", action="RIGHT", margin=0.2)
    store.append_inference(first)
    store.append_inference(second)

    outcome = record_production_outcome(
        store,
        first.record_id,
        observed_action="LEFT",
        source="environment-step",
    )

    assert store.list_inferences() == (first, second)
    assert outcome.correct is True
    assert store.get_outcome_for_inference(first.record_id) == outcome


def test_outcome_is_immutable_per_inference() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    first = _record(1)
    store.append_inference(first)
    record_production_outcome(
        store,
        first.record_id,
        observed_action="LEFT",
        source="environment-step",
    )

    with pytest.raises(PerceptionProductionLedgerError):
        record_production_outcome(
            store,
            first.record_id,
            observed_action="RIGHT",
            source="manual-correction",
        )


def test_windowed_metrics_keep_rejection_separate_from_correctness() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    first = _record(1, status="accepted", action="LEFT", margin=0.9)
    second = _record(2, status="rejected_ambiguous", action="RIGHT", margin=0.1)
    third = _record(3, status="accepted", action="RIGHT", margin=0.7, revision=2)
    for item in (first, second, third):
        store.append_inference(item)

    record_production_outcome(
        store, first.record_id, observed_action="LEFT", source="environment-step"
    )
    record_production_outcome(
        store, second.record_id, observed_action="LEFT", source="environment-step"
    )

    report = build_production_metrics_report(
        store,
        start_sequence_number=1,
        end_sequence_number=3,
    )

    assert report.inference_count == 3
    assert report.accepted_count == 2
    assert report.rejected_count == 1
    assert report.labeled_count == 2
    assert report.raw_accuracy == 0.5
    assert report.accepted_accuracy == 1.0
    assert report.coverage == pytest.approx(2 / 3)
    assert report.model_pointer_revisions == (1, 2)


def test_metrics_can_filter_one_promoted_model() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    store.append_inference(_record(1, model="promoted-a"))
    store.append_inference(_record(2, model="promoted-b"))

    report = build_production_metrics_report(
        store,
        start_sequence_number=1,
        promoted_model_id="promoted-b",
    )

    assert report.inference_count == 1
    assert report.promoted_model_id == "promoted-b"


def test_inference_sequence_must_be_contiguous() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    with pytest.raises(PerceptionProductionLedgerError):
        store.append_inference(_record(2))
