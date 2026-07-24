from zeromodel.perception import (
    InMemoryPerceptionProductionLedgerStore,
    OperationalDriftPolicyDTO,
    ProductionInferenceRecordDTO,
    ProductionOutcomeRecordDTO,
    PromotedTestEvaluationReportDTO,
    PromotedTestExampleDTO,
    build_operational_reference_profile,
    diagnose_operational_health,
)


def _test_report() -> PromotedTestEvaluationReportDTO:
    examples = (
        PromotedTestExampleDTO("t1", "left", "r1", "left", 0.8, "accepted", True),
        PromotedTestExampleDTO("t2", "right", "r2", "right", 0.6, "accepted", True),
        PromotedTestExampleDTO("t3", "left", "r3", "left", 0.4, "accepted", True),
        PromotedTestExampleDTO("t4", "right", "r4", "left", 0.2, "rejected_ambiguous", False),
    )
    return PromotedTestEvaluationReportDTO(
        report_id="test-report",
        promoted_model_id="promoted-a",
        model_kind="single_frame",
        model_id="model-a",
        calibration_id="calibration-a",
        promotion_decision_id="decision-a",
        validation_comparison_report_id="validation-a",
        split="test",
        example_count=4,
        accepted_count=3,
        rejected_count=1,
        raw_accuracy=0.75,
        accepted_accuracy=1.0,
        coverage=0.75,
        mean_margin=0.5,
        rejection_threshold=0.3,
        examples=examples,
    )


def _inference(
    sequence: int,
    action: str,
    margin: float,
    status: str,
    *,
    pointer_revision: int = 1,
) -> ProductionInferenceRecordDTO:
    return ProductionInferenceRecordDTO(
        record_id=f"production-{sequence}",
        sequence_number=sequence,
        pointer_id="pointer-1",
        pointer_revision=pointer_revision,
        promoted_model_id="promoted-a",
        model_id="model-a",
        model_kind="single_frame",
        input_id=f"input-{sequence}",
        interaction_id=f"interaction-{sequence}",
        inference_result_id=f"result-{sequence}",
        selected_action=action,
        margin=margin,
        status=status,
        rejection_threshold=0.3,
    )


def _outcome(sequence: int, observed: str, correct: bool) -> ProductionOutcomeRecordDTO:
    return ProductionOutcomeRecordDTO(
        outcome_id=f"outcome-{sequence}",
        inference_record_id=f"production-{sequence}",
        outcome_sequence_number=sequence,
        observed_action=observed,
        source="environment",
        correct=correct,
    )


def _small_policy(**changes: object) -> OperationalDriftPolicyDTO:
    values: dict[str, object] = {
        "minimum_reference_count": 4,
        "minimum_inference_count": 2,
        "minimum_labeled_count": 2,
        "minimum_accepted_labeled_count": 2,
        "minimum_label_coverage": 1.0,
    }
    values.update(changes)
    return OperationalDriftPolicyDTO(**values)  # type: ignore[arg-type]


def test_reference_profile_is_deterministic_and_preserves_test_distribution() -> None:
    first = build_operational_reference_profile(_test_report())
    second = build_operational_reference_profile(_test_report())

    assert first == second
    assert first.coverage == 0.75
    assert tuple((item.action_label, item.count) for item in first.action_distribution) == (
        ("left", 3),
        ("right", 1),
    )


def test_one_unlabeled_inference_cannot_report_drift() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    store.append_inference(_inference(1, "right", 0.0, "rejected_ambiguous"))

    report = diagnose_operational_health(
        build_operational_reference_profile(_test_report()),
        store,
        start_sequence_number=1,
    )

    assert report.overall_status == "insufficient_evidence"
    assert tuple(item.status for item in report.findings) == (
        "insufficient_evidence",
        "insufficient_evidence",
        "insufficient_evidence",
        "insufficient_evidence",
        "insufficient_evidence",
    )


def test_health_reports_insufficient_accuracy_evidence_without_labels() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    store.append_inference(_inference(1, "left", 0.7, "accepted"))
    store.append_inference(_inference(2, "right", 0.6, "accepted"))

    report = diagnose_operational_health(
        build_operational_reference_profile(_test_report()),
        store,
        start_sequence_number=1,
        policy=_small_policy(),
    )

    assert report.overall_status == "insufficient_evidence"
    assert tuple(item.status for item in report.findings[:3]) == (
        "healthy",
        "healthy",
        "healthy",
    )
    assert report.findings[3].status == "insufficient_evidence"
    assert report.findings[4].status == "insufficient_evidence"


def test_partial_labels_do_not_compare_accuracy_to_fully_labeled_reference() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    store.append_inference(_inference(1, "left", 0.7, "accepted"))
    store.append_inference(_inference(2, "right", 0.6, "accepted"))
    store.append_outcome(_outcome(1, "left", True))

    report = diagnose_operational_health(
        build_operational_reference_profile(_test_report()),
        store,
        start_sequence_number=1,
        policy=_small_policy(minimum_labeled_count=1),
    )

    assert report.findings[3].status == "insufficient_evidence"
    assert "label_coverage" in report.findings[3].rationale


def test_health_detects_supported_margin_action_and_accuracy_drift() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    records = tuple(
        _inference(sequence, "left", 0.1, "accepted") for sequence in range(1, 5)
    )
    for record in records:
        store.append_inference(record)
    for sequence in range(1, 5):
        store.append_outcome(_outcome(sequence, "right", False))

    report = diagnose_operational_health(
        build_operational_reference_profile(_test_report()),
        store,
        start_sequence_number=1,
        policy=_small_policy(
            minimum_inference_count=4,
            minimum_labeled_count=4,
            minimum_accepted_labeled_count=4,
        ),
    )

    assert report.overall_status == "drifted"
    assert report.findings[0].status == "healthy"
    assert tuple(item.status for item in report.findings[1:]) == (
        "drifted",
        "drifted",
        "drifted",
        "drifted",
    )
    assert report.action_distribution_distance == 0.25


def test_window_spanning_pointer_revisions_is_insufficient() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    store.append_inference(_inference(1, "left", 0.7, "accepted", pointer_revision=1))
    store.append_inference(_inference(2, "right", 0.6, "accepted", pointer_revision=2))
    store.append_outcome(_outcome(1, "left", True))
    store.append_outcome(_outcome(2, "right", True))

    report = diagnose_operational_health(
        build_operational_reference_profile(_test_report()),
        store,
        start_sequence_number=1,
        policy=_small_policy(),
    )

    assert report.overall_status == "insufficient_evidence"
    assert "pointer_revisions" in report.findings[0].rationale


def test_healthy_window_preserves_exact_evidence_ids() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    store.append_inference(_inference(1, "left", 0.7, "accepted"))
    store.append_inference(_inference(2, "right", 0.6, "accepted"))
    store.append_outcome(_outcome(1, "left", True))
    store.append_outcome(_outcome(2, "right", True))

    report = diagnose_operational_health(
        build_operational_reference_profile(_test_report()),
        store,
        start_sequence_number=1,
        policy=_small_policy(
            maximum_coverage_drop=0.3,
            maximum_mean_margin_drop=0.3,
            maximum_action_distribution_distance=0.3,
            maximum_raw_accuracy_drop=0.3,
            maximum_accepted_accuracy_drop=0.3,
        ),
    )

    assert report.overall_status == "healthy"
    assert report.inference_record_ids == ("production-1", "production-2")
    assert report.outcome_ids == ("outcome-1", "outcome-2")
