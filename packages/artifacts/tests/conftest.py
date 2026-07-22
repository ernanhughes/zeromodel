from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    AdaptedDimensionDTO,
    AdaptedReportDTO,
    AdaptedSubjectDTO,
    AdaptedValueDTO,
    InMemoryArtifactStore,
    ReportAdapterContractDTO,
    ReportFindingRefDTO,
    ScoreSemantics,
    SourceBindingDTO,
)
from zeromodel.artifacts.report_dto import (
    compute_adapted_report_id,
    compute_report_adapter_contract_id,
)
from zeromodel.core.artifact import LayoutRecipe

# Plain module-level helpers, used internally by this file's own fixtures
# below. Not exposed for import by other test files: a bare `conftest`
# module name collides across package test directories once more than one
# package's tests are collected in the same pytest session (e.g. by
# scripts/run_fast_tests.py). Other test files must request the
# `make_value`/`make_contract`/`make_adapted_report`/`FakeAdapter` *fixtures*
# defined further below instead.


def _make_value(
    *, subject_id: str, dimension_id: str, raw_value: float, report_id: str = "report-1"
) -> AdaptedValueDTO:
    finding_ref = ReportFindingRefDTO(
        report_id=report_id, finding_id=f"{subject_id}:{dimension_id}"
    )
    source_binding = SourceBindingDTO(
        subject_id=subject_id, dimension_id=dimension_id, finding_ref=finding_ref
    )
    return AdaptedValueDTO(
        subject_id=subject_id,
        dimension_id=dimension_id,
        raw_value=raw_value,
        source_binding=source_binding,
    )


def _make_contract(
    *,
    adapter_id: str,
    report_kind: str,
    compatibility_id: str,
    adapter_version: str = "1.0.0",
    subject_kind: str = "sentence",
    dimension_namespace: str = "test",
) -> ReportAdapterContractDTO:
    contract_id = compute_report_adapter_contract_id(
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        report_kind=report_kind,
        subject_kind=subject_kind,
        dimension_namespace=dimension_namespace,
        compatibility_id=compatibility_id,
    )
    return ReportAdapterContractDTO(
        contract_id=contract_id,
        adapter_id=adapter_id,
        adapter_version=adapter_version,
        report_kind=report_kind,
        subject_kind=subject_kind,
        dimension_namespace=dimension_namespace,
        compatibility_id=compatibility_id,
    )


def _make_adapted_report(
    *,
    contract: ReportAdapterContractDTO,
    subjects: tuple,
    dimensions: tuple,
    values: tuple,
    report_id: str = "report-1",
) -> AdaptedReportDTO:
    adapted_report_id = compute_adapted_report_id(
        report_id=report_id,
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
    )
    return AdaptedReportDTO(
        adapted_report_id=adapted_report_id,
        report_id=report_id,
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values,
    )


class _FakeAdapter:
    """A minimal `ReportAdapter` test double - production adapters are
    always defined by the external application, never here."""

    def __init__(
        self, contract: ReportAdapterContractDTO, adapted: AdaptedReportDTO
    ) -> None:
        self._contract = contract
        self._adapted = adapted

    def contract(self) -> ReportAdapterContractDTO:
        return self._contract

    def adapt(self, report: object) -> AdaptedReportDTO:
        return self._adapted


@pytest.fixture
def artifact_store() -> InMemoryArtifactStore:
    return InMemoryArtifactStore()


@pytest.fixture
def source_layout_recipe() -> LayoutRecipe:
    """The simplest valid recipe: declared (source) row/column order, no
    reordering - deterministic and sufficient for compilation tests that
    aren't specifically about layout."""
    return LayoutRecipe(
        {
            "version": "vpm-layout/0",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )


@pytest.fixture
def make_value():
    return _make_value


@pytest.fixture
def make_contract():
    return _make_contract


@pytest.fixture
def make_adapted_report():
    return _make_adapted_report


@pytest.fixture
def FakeAdapter():
    return _FakeAdapter


@pytest.fixture
def ai_artifact_family():
    """A "negative" report family: higher raw score = worse."""
    contract = _make_contract(
        adapter_id="writer.ai_artifact",
        report_kind="writer-ai-artifact",
        compatibility_id="writer-ai-artifact/v1",
    )
    subjects = (
        AdaptedSubjectDTO(subject_id="sentence-001"),
        AdaptedSubjectDTO(subject_id="sentence-002"),
    )
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="generic_phrasing",
            label="Generic phrasing",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
        AdaptedDimensionDTO(
            dimension_id="over_explanation",
            label="Over-explanation",
            score_semantics=ScoreSemantics.HIGHER_IS_WORSE,
        ),
    )
    values = (
        _make_value(
            subject_id="sentence-001", dimension_id="generic_phrasing", raw_value=0.9
        ),
        _make_value(
            subject_id="sentence-001", dimension_id="over_explanation", raw_value=0.8
        ),
        _make_value(
            subject_id="sentence-002", dimension_id="generic_phrasing", raw_value=0.1
        ),
        _make_value(
            subject_id="sentence-002", dimension_id="over_explanation", raw_value=0.2
        ),
    )
    adapted = _make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values
    )
    return contract, adapted


@pytest.fixture
def quality_family():
    """A "positive" report family over the *same subject ids*: higher raw
    score = better. Proves report families stay semantically separate even
    when they share subjects."""
    contract = _make_contract(
        adapter_id="writer.sentence_quality",
        report_kind="writer-sentence-quality",
        compatibility_id="writer-sentence-quality/v1",
    )
    subjects = (
        AdaptedSubjectDTO(subject_id="sentence-001"),
        AdaptedSubjectDTO(subject_id="sentence-002"),
    )
    dimensions = (
        AdaptedDimensionDTO(
            dimension_id="quality",
            label="Quality",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
        AdaptedDimensionDTO(
            dimension_id="clarity",
            label="Clarity",
            score_semantics=ScoreSemantics.HIGHER_IS_BETTER,
        ),
    )
    values = (
        _make_value(subject_id="sentence-001", dimension_id="quality", raw_value=0.4),
        _make_value(subject_id="sentence-001", dimension_id="clarity", raw_value=0.5),
        _make_value(subject_id="sentence-002", dimension_id="quality", raw_value=0.95),
        _make_value(subject_id="sentence-002", dimension_id="clarity", raw_value=0.9),
    )
    adapted = _make_adapted_report(
        contract=contract, subjects=subjects, dimensions=dimensions, values=values
    )
    return contract, adapted
