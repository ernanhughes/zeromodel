from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    AdaptedDimensionDTO,
    AdaptedReportDTO,
    AdaptedSubjectDTO,
    AdaptedValueDTO,
    ReportFindingRefDTO,
    ScoreSemantics,
    SourceBindingDTO,
)
from zeromodel.artifacts.report_dto import compute_adapted_report_id
from zeromodel.core.artifact import VPMValidationError


@pytest.fixture
def base_report_parts(make_contract, make_value):
    contract = make_contract(
        adapter_id="writer.ai_artifact",
        report_kind="writer-ai-artifact",
        compatibility_id="writer-ai-artifact/v1",
    )
    subjects = (AdaptedSubjectDTO(subject_id="sentence-001"),)
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="generic_phrasing",
            label="Generic phrasing",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
    )
    values = (
        make_value(
            subject_id="sentence-001", dimension_id="generic_phrasing", raw_value=0.9
        ),
    )
    return contract, subjects, dimensions, values


def test_identical_input_produces_identical_adapted_report_identity(
    base_report_parts, make_adapted_report
):
    contract, subjects, dimensions, values = base_report_parts
    first = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values
    )
    second = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values
    )
    assert first.adapted_report_id == second.adapted_report_id


def test_attribute_tuple_order_does_not_change_identity(
    base_report_parts, make_adapted_report
):
    contract, subjects, dimensions, values = base_report_parts
    subject_a = AdaptedSubjectDTO(
        subject_id="sentence-001", attributes=(("a", "1"), ("b", "2"))
    )
    subject_b = AdaptedSubjectDTO(
        subject_id="sentence-001", attributes=(("b", "2"), ("a", "1"))
    )
    report_a = make_adapted_report(
        contract=contract, subjects=(subject_a,), dimensions=dimensions, values=values
    )
    report_b = make_adapted_report(
        contract=contract, subjects=(subject_b,), dimensions=dimensions, values=values
    )
    assert report_a.adapted_report_id == report_b.adapted_report_id


def test_changed_raw_value_changes_identity(
    base_report_parts, make_adapted_report, make_value
):
    contract, subjects, dimensions, _ = base_report_parts
    values_a = (
        make_value(
            subject_id="sentence-001", dimension_id="generic_phrasing", raw_value=0.9
        ),
    )
    values_b = (
        make_value(
            subject_id="sentence-001", dimension_id="generic_phrasing", raw_value=0.1
        ),
    )
    report_a = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values_a
    )
    report_b = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values_b
    )
    assert report_a.adapted_report_id != report_b.adapted_report_id


def test_changed_score_semantics_changes_identity(
    base_report_parts, make_adapted_report
):
    contract, subjects, _, values = base_report_parts
    dims_worse = (
        AdaptedDimensionDTO(
            dimension_id="generic_phrasing",
            label="Generic phrasing",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
    )
    dims_better = (
        AdaptedDimensionDTO(
            dimension_id="generic_phrasing",
            label="Generic phrasing",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
    )
    report_a = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dims_worse, values=values
    )
    report_b = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dims_better, values=values
    )
    assert report_a.adapted_report_id != report_b.adapted_report_id


def test_changed_source_binding_changes_identity(
    base_report_parts, make_adapted_report
):
    contract, subjects, dimensions, _ = base_report_parts
    finding_a = ReportFindingRefDTO(report_id="report-1", finding_id="finding-a")
    finding_b = ReportFindingRefDTO(report_id="report-1", finding_id="finding-b")
    binding_a = SourceBindingDTO(
        subject_id="sentence-001",
        dimension_id="generic_phrasing",
        finding_ref=finding_a,
    )
    binding_b = SourceBindingDTO(
        subject_id="sentence-001",
        dimension_id="generic_phrasing",
        finding_ref=finding_b,
    )
    value_a = AdaptedValueDTO(
        subject_id="sentence-001",
        dimension_id="generic_phrasing",
        raw_value=0.9,
        source_binding=binding_a,
    )
    value_b = AdaptedValueDTO(
        subject_id="sentence-001",
        dimension_id="generic_phrasing",
        raw_value=0.9,
        source_binding=binding_b,
    )
    report_a = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=(value_a,)
    )
    report_b = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=(value_b,)
    )
    assert report_a.adapted_report_id != report_b.adapted_report_id


def test_changed_confidence_changes_identity(base_report_parts, make_adapted_report):
    contract, subjects, dimensions, _ = base_report_parts
    finding = ReportFindingRefDTO(report_id="report-1", finding_id="f")
    binding = SourceBindingDTO(
        subject_id="sentence-001", dimension_id="generic_phrasing", finding_ref=finding
    )
    value_a = AdaptedValueDTO(
        subject_id="sentence-001",
        dimension_id="generic_phrasing",
        raw_value=0.9,
        source_binding=binding,
        confidence=0.5,
    )
    value_b = AdaptedValueDTO(
        subject_id="sentence-001",
        dimension_id="generic_phrasing",
        raw_value=0.9,
        source_binding=binding,
        confidence=0.9,
    )
    report_a = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=(value_a,)
    )
    report_b = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=(value_b,)
    )
    assert report_a.adapted_report_id != report_b.adapted_report_id


def test_changed_importance_changes_identity(base_report_parts, make_adapted_report):
    contract, subjects, dimensions, _ = base_report_parts
    finding = ReportFindingRefDTO(report_id="report-1", finding_id="f")
    binding = SourceBindingDTO(
        subject_id="sentence-001", dimension_id="generic_phrasing", finding_ref=finding
    )
    value_a = AdaptedValueDTO(
        subject_id="sentence-001",
        dimension_id="generic_phrasing",
        raw_value=0.9,
        source_binding=binding,
        importance=0.5,
    )
    value_b = AdaptedValueDTO(
        subject_id="sentence-001",
        dimension_id="generic_phrasing",
        raw_value=0.9,
        source_binding=binding,
        importance=1.0,
    )
    report_a = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=(value_a,)
    )
    report_b = make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=(value_b,)
    )
    assert report_a.adapted_report_id != report_b.adapted_report_id


def test_duplicate_subject_ids_are_rejected(base_report_parts):
    contract, _, dimensions, values = base_report_parts
    subjects = (
        AdaptedSubjectDTO(subject_id="sentence-001"),
        AdaptedSubjectDTO(subject_id="sentence-001"),
    )
    adapted_report_id = compute_adapted_report_id(
        report_id="report-1",
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
    )
    with pytest.raises(VPMValidationError, match="duplicate subject_id"):
        AdaptedReportDTO(
            adapted_report_id=adapted_report_id,
            report_id="report-1",
            report_kind=contract.report_kind,
            adapter_contract_id=contract.contract_id,
            compatibility_id=contract.compatibility_id,
            subjects=subjects,
            dimensions=dimensions,
            values=values,
        )


def test_duplicate_dimension_ids_are_rejected(base_report_parts):
    contract, subjects, _, values = base_report_parts
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="generic_phrasing",
            label="A",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
        AdaptedDimensionDTO(
            dimension_id="generic_phrasing",
            label="B",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
    )
    adapted_report_id = compute_adapted_report_id(
        report_id="report-1",
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
    )
    with pytest.raises(VPMValidationError, match="duplicate dimension_id"):
        AdaptedReportDTO(
            adapted_report_id=adapted_report_id,
            report_id="report-1",
            report_kind=contract.report_kind,
            adapter_contract_id=contract.contract_id,
            compatibility_id=contract.compatibility_id,
            subjects=subjects,
            dimensions=dimensions,
            values=values,
        )


def test_duplicate_subject_dimension_value_pair_is_rejected(
    base_report_parts, make_value
):
    contract, subjects, dimensions, _ = base_report_parts
    values = (
        make_value(
            subject_id="sentence-001", dimension_id="generic_phrasing", raw_value=0.9
        ),
        make_value(
            subject_id="sentence-001", dimension_id="generic_phrasing", raw_value=0.1
        ),
    )
    adapted_report_id = compute_adapted_report_id(
        report_id="report-1",
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
    )
    with pytest.raises(VPMValidationError, match="duplicate value"):
        AdaptedReportDTO(
            adapted_report_id=adapted_report_id,
            report_id="report-1",
            report_kind=contract.report_kind,
            adapter_contract_id=contract.contract_id,
            compatibility_id=contract.compatibility_id,
            subjects=subjects,
            dimensions=dimensions,
            values=values,
        )


def test_value_referencing_unknown_subject_is_rejected(base_report_parts, make_value):
    contract, subjects, dimensions, _ = base_report_parts
    values = (
        make_value(
            subject_id="sentence-does-not-exist",
            dimension_id="generic_phrasing",
            raw_value=0.9,
        ),
    )
    adapted_report_id = compute_adapted_report_id(
        report_id="report-1",
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
    )
    with pytest.raises(VPMValidationError, match="unknown subject_id"):
        AdaptedReportDTO(
            adapted_report_id=adapted_report_id,
            report_id="report-1",
            report_kind=contract.report_kind,
            adapter_contract_id=contract.contract_id,
            compatibility_id=contract.compatibility_id,
            subjects=subjects,
            dimensions=dimensions,
            values=values,
        )


def test_value_referencing_unknown_dimension_is_rejected(base_report_parts, make_value):
    contract, subjects, dimensions, _ = base_report_parts
    values = (
        make_value(
            subject_id="sentence-001",
            dimension_id="dimension-does-not-exist",
            raw_value=0.9,
        ),
    )
    adapted_report_id = compute_adapted_report_id(
        report_id="report-1",
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
    )
    with pytest.raises(VPMValidationError, match="unknown dimension_id"):
        AdaptedReportDTO(
            adapted_report_id=adapted_report_id,
            report_id="report-1",
            report_kind=contract.report_kind,
            adapter_contract_id=contract.contract_id,
            compatibility_id=contract.compatibility_id,
            subjects=subjects,
            dimensions=dimensions,
            values=values,
        )


def test_non_finite_raw_value_is_rejected():
    finding = ReportFindingRefDTO(report_id="report-1", finding_id="f")
    binding = SourceBindingDTO(
        subject_id="sentence-001", dimension_id="generic_phrasing", finding_ref=finding
    )
    with pytest.raises(VPMValidationError, match="finite"):
        AdaptedValueDTO(
            subject_id="sentence-001",
            dimension_id="generic_phrasing",
            raw_value=float("nan"),
            source_binding=binding,
        )
    with pytest.raises(VPMValidationError, match="finite"):
        AdaptedValueDTO(
            subject_id="sentence-001",
            dimension_id="generic_phrasing",
            raw_value=float("inf"),
            source_binding=binding,
        )


def test_target_range_semantics_without_bounds_is_rejected():
    with pytest.raises(VPMValidationError, match="target_range"):
        AdaptedDimensionDTO(
            dimension_id="sentence_length",
            label="Sentence length",
            score_semantics=ScoreSemantics.TARGET_RANGE,
        )


def test_target_range_with_inverted_bounds_is_rejected():
    with pytest.raises(VPMValidationError, match="target_min"):
        AdaptedDimensionDTO(
            dimension_id="sentence_length",
            label="Sentence length",
            score_semantics=ScoreSemantics.TARGET_RANGE,
            target_min=10,
            target_max=5,
        )


def test_source_binding_subject_mismatch_is_rejected():
    finding = ReportFindingRefDTO(report_id="report-1", finding_id="f")
    binding = SourceBindingDTO(
        subject_id="sentence-999", dimension_id="generic_phrasing", finding_ref=finding
    )
    with pytest.raises(VPMValidationError, match="subject_id"):
        AdaptedValueDTO(
            subject_id="sentence-001",
            dimension_id="generic_phrasing",
            raw_value=0.9,
            source_binding=binding,
        )
