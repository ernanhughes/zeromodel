"""Demonstrates the adapter-first Artifacts flow: two semantically distinct
report families (a negative "AI-artifact" report and a positive "quality"
report) over the *same* sentence subjects compile to two separate, source-
bound VPM artifacts without ZeroModel ever needing to know what "generic
phrasing" or "clarity" mean.

These are illustrative, synthetic in-file adapters. A real integration
would live in the external application's own package (e.g.
``writer.integrations.zeromodel``), never inside ``zeromodel.artifacts``.

Run:

    python examples/writer_report_adapters_demo.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

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
    compile_report,
)
from zeromodel.artifacts.report_dto import (
    compute_adapted_report_id,
    compute_report_adapter_contract_id,
)
from zeromodel.core.artifact import LayoutRecipe

# A minimal "source" layout: declared row/column order, no reordering.
_LAYOUT = LayoutRecipe(
    {
        "version": "vpm-layout/0",
        "row_order": {"kind": "source", "tie_break": "row_id"},
        "column_order": {"kind": "source"},
        "normalization": {"kind": "per_metric_minmax", "clip": True},
    }
)


@dataclass(frozen=True)
class WriterSentenceReport:
    """A stand-in for whatever typed report Writer's own systems produce.
    ZeroModel never sees this type - only the adapter does."""

    report_id: str
    sentence_scores: Tuple[
        Tuple[str, dict], ...
    ]  # (sentence_id, {dimension_id: raw_value})


def _build_adapted_report(
    *,
    report: WriterSentenceReport,
    contract: ReportAdapterContractDTO,
    dimensions: Tuple[AdaptedDimensionDTO, ...],
) -> AdaptedReportDTO:
    subjects = tuple(
        AdaptedSubjectDTO(subject_id=sentence_id)
        for sentence_id, _ in report.sentence_scores
    )
    values = []
    for sentence_id, scores in report.sentence_scores:
        for dimension_id, raw_value in scores.items():
            finding_ref = ReportFindingRefDTO(
                report_id=report.report_id, finding_id=f"{sentence_id}:{dimension_id}"
            )
            source_binding = SourceBindingDTO(
                subject_id=sentence_id,
                dimension_id=dimension_id,
                finding_ref=finding_ref,
            )
            values.append(
                AdaptedValueDTO(
                    subject_id=sentence_id,
                    dimension_id=dimension_id,
                    raw_value=raw_value,
                    source_binding=source_binding,
                )
            )
    values_tuple = tuple(values)
    adapted_report_id = compute_adapted_report_id(
        report_id=report.report_id,
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values_tuple,
    )
    return AdaptedReportDTO(
        adapted_report_id=adapted_report_id,
        report_id=report.report_id,
        report_kind=contract.report_kind,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        subjects=subjects,
        dimensions=dimensions,
        values=values_tuple,
    )


class AIArtifactReportAdapter:
    """Illustrative example only - a real adapter lives in the Writer codebase."""

    _DIMENSIONS = (
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

    def contract(self) -> ReportAdapterContractDTO:
        contract_id = compute_report_adapter_contract_id(
            adapter_id="writer.ai_artifact",
            adapter_version="1.0.0",
            report_kind="writer-ai-artifact",
            subject_kind="sentence",
            dimension_namespace="writer.ai_artifact",
            compatibility_id="writer-ai-artifact/v1",
        )
        return ReportAdapterContractDTO(
            contract_id=contract_id,
            adapter_id="writer.ai_artifact",
            adapter_version="1.0.0",
            report_kind="writer-ai-artifact",
            subject_kind="sentence",
            dimension_namespace="writer.ai_artifact",
            compatibility_id="writer-ai-artifact/v1",
        )

    def adapt(self, report: WriterSentenceReport) -> AdaptedReportDTO:
        return _build_adapted_report(
            report=report, contract=self.contract(), dimensions=self._DIMENSIONS
        )


class SentenceQualityReportAdapter:
    """Illustrative example only - a real adapter lives in the Writer codebase."""

    _DIMENSIONS = (
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

    def contract(self) -> ReportAdapterContractDTO:
        contract_id = compute_report_adapter_contract_id(
            adapter_id="writer.sentence_quality",
            adapter_version="1.0.0",
            report_kind="writer-sentence-quality",
            subject_kind="sentence",
            dimension_namespace="writer.sentence_quality",
            compatibility_id="writer-sentence-quality/v1",
        )
        return ReportAdapterContractDTO(
            contract_id=contract_id,
            adapter_id="writer.sentence_quality",
            adapter_version="1.0.0",
            report_kind="writer-sentence-quality",
            subject_kind="sentence",
            dimension_namespace="writer.sentence_quality",
            compatibility_id="writer-sentence-quality/v1",
        )

    def adapt(self, report: WriterSentenceReport) -> AdaptedReportDTO:
        return _build_adapted_report(
            report=report, contract=self.contract(), dimensions=self._DIMENSIONS
        )


def main() -> None:
    negative_report = WriterSentenceReport(
        report_id="report-negative-1",
        sentence_scores=(
            ("sentence-001", {"generic_phrasing": 0.9, "over_explanation": 0.8}),
            ("sentence-002", {"generic_phrasing": 0.1, "over_explanation": 0.2}),
        ),
    )
    positive_report = WriterSentenceReport(
        report_id="report-positive-1",
        sentence_scores=(
            ("sentence-001", {"quality": 0.4, "clarity": 0.5}),
            ("sentence-002", {"quality": 0.95, "clarity": 0.9}),
        ),
    )

    store = InMemoryArtifactStore()
    ai_artifact_compiled = compile_report(
        adapter=AIArtifactReportAdapter(),
        report=negative_report,
        layout_recipe=_LAYOUT,
        store=store,
    )
    quality_compiled = compile_report(
        adapter=SentenceQualityReportAdapter(),
        report=positive_report,
        layout_recipe=_LAYOUT,
        store=store,
    )

    print("Same subjects, two separate report families:")
    print(
        f"  AI-artifact report subjects: {[s.subject_id for s in ai_artifact_compiled.subjects]}"
    )
    print(
        f"  Quality report subjects:     {[s.subject_id for s in quality_compiled.subjects]}"
    )
    print(
        f"  AI-artifact compiled artifact id: {ai_artifact_compiled.artifact_ref.artifact_id}"
    )
    print(
        f"  Quality compiled artifact id:     {quality_compiled.artifact_ref.artifact_id}"
    )
    assert (
        ai_artifact_compiled.artifact_ref.artifact_id
        != quality_compiled.artifact_ref.artifact_id
    )
    print(
        "Confirmed: identical subjects, different report kinds -> different compiled artifacts."
    )


if __name__ == "__main__":
    main()
