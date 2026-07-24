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


def _inference(sequence: int, action: str, margin: float, status: str) -> ProductionInferenceRecordDTO:
    return ProductionInferenceRecordDTO(
        record_id=f"production-{sequence}",
        sequence_number=sequence,
        pointer_id="pointer-1",
        pointer_revision=1,
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


def test_reference_profile_is_deterministic_and_preserves_test_distribution() -> None:
    first = build_operational_reference_profile(_test_report())
    second = build_operational_reference_profile(_test_report())

    assert first == second
    assert first.coverage == 0.75
    assert tuple((item.action_label, item.count) for item in first.action_distribution) == (
        ("left", 3),
        ("right", 1),
    )


def test_health_reports_insufficient_accuracy_evidence_without_labels() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    store.append_inference(_inference(1, "left", 0.7, "accepted"))
    store.append_inference(_inference(2, "right", 0.6, "accepted"))

    report = diagnose_operational_health(
        build_operational_reference_profile(_test_report()),
        store,
        start_sequence_number=1,
        policy=OperationalDriftPolicyDTO(minimum_labeled_count=2),
    )

    assert report.overall_status == "insufficient_evidence"
    assert report.findings[0].status == "healthy"
    assert report.findings[3].status == "insufficient_evidence"
    assert report.findings[4].status == "insufficient_evidence"


def test_health_detects_coverage_margin_action_and_accuracy_drift() -> None:
    store = InMemoryPerceptionProductionLedgerStore()
    records = (
        _inference(1, "left", 0.1, "rejected_ambiguous"),
        _inference(2, "left", 0.1, "rejected_ambiguous"),
        _inference(3, "left", 0.2, "rejected_ambiguous"),
        _inference(4, "left", 0.2, "rejected_ambiguous"),
    )
    for record in records:
        store.append_inference(record)
    for sequence in range(1, 5):
        store.append_outcome(_outcome(sequence, "right", False))

    report = diagnose_operational_health(
        build_operational_reference_profile(_test_report()),
        store,
        start_sequence_number=1,
        policy=OperationalDriftPolicyDTO(
            maximum_coverage_drop=0.1,
            maximum_mean_margin_drop=0.1,
            maximum_action_distribution_distance=0.2,
            maximum_raw_accuracy_drop=0.1,
            maximum_accepted_accuracy_drop=0.1,
            minimum_labeled_count=4,
        ),
    )

    assert report.overall_status == "drifted"
    assert tuple(item.status for item in report.findings[:4]) == (
        "drifted",
        "drifted",
        "drifted",
        "drifted",
    )
    assert report.findings[4].status == "insufficient_evidence"
    assert report.action_distribution_distance == 0.25


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
        policy=OperationalDriftPolicyDTO(
            maximum_coverage_drop=0.3,
            maximum_mean_margin_drop=0.3,
            maximum_action_distribution_distance=0.3,
            maximum_raw_accuracy_drop=0.3,
            maximum_accepted_accuracy_drop=0.3,
            minimum_labeled_count=2,
        ),
    )

    assert report.overall_status == "healthy"
    assert report.inference_record_ids == ("production-1", "production-2")
    assert report.outcome_ids == ("outcome-1", "outcome-2")
