"""Direct tests of `CompiledReportArtifactDTO`'s structural closure
validator (review finding #3): a digest proves bytes were not altered
relative to a reference, not that the bytes form a valid compiled report.
`compile_report()` always happens to produce valid data, so these tests
construct `CellBindingDTO`/`CompiledReportArtifactDTO` directly to prove
the loader-facing validation actually rejects malformed-but-digest-valid
records, the way an untrusted decoded payload could be malformed.
"""

from __future__ import annotations

import pytest

from zeromodel.artifacts import (
    ADAPTED_REPORT_ARTIFACT_KIND,
    AdaptedDimensionDTO,
    AdaptedSubjectDTO,
    ArtifactRef,
    CellBindingDTO,
    CompiledReportArtifactDTO,
    ReportFindingRefDTO,
    ScoreSemantics,
    SourceBindingDTO,
    sha256_digest,
)
from zeromodel.artifacts.compatibility_schema import (
    compute_compatibility_schema_id,
    compute_report_semantics_id,
)
from zeromodel.artifacts.compiled_artifact import (
    COMPILED_REPORT_ARTIFACT_KIND,
    CompatibilityInfo,
    CoreArtifactRefs,
    ReportSemanticsInfo,
    compute_compiled_report_artifact_id,
)
from zeromodel.artifacts.report_errors import ReportCompilationError

_MISSING_VALUE_SEMANTICS = "error"

_FAKE_ADAPTED_REPORT_REF = ArtifactRef(
    artifact_kind=ADAPTED_REPORT_ARTIFACT_KIND, artifact_id=sha256_digest(b"w")
)
_FAKE_REF = ArtifactRef(
    artifact_kind="zeromodel.core.score-table/v1", artifact_id=sha256_digest(b"x")
)
_FAKE_RECIPE_REF = ArtifactRef(
    artifact_kind="zeromodel.core.layout-recipe/v1", artifact_id=sha256_digest(b"y")
)
_FAKE_VPM_REF = ArtifactRef(
    artifact_kind="zeromodel.core.vpm-artifact/v1", artifact_id=sha256_digest(b"z")
)
_REPORT_SEMANTICS_ID = compute_report_semantics_id(
    report_kind="test-report",
    subject_kind="test-subject",
    dimension_namespace="test",
    duplicate_value_semantics="reject",
)
_REPORT_SEMANTICS = ReportSemanticsInfo(
    report_kind="test-report",
    subject_kind="test-subject",
    dimension_namespace="test",
    duplicate_value_semantics="reject",
    report_semantics_id=_REPORT_SEMANTICS_ID,
)

_SUBJECTS = (
    AdaptedSubjectDTO(subject_id="s0"),
    AdaptedSubjectDTO(subject_id="s1"),
)
_DIMENSIONS = (
    AdaptedDimensionDTO(
        dimension_id="d0", label="D0", score_semantics=ScoreSemantics.HIGHER_IS_BETTER
    ),
    AdaptedDimensionDTO(
        dimension_id="d1", label="D1", score_semantics=ScoreSemantics.HIGHER_IS_BETTER
    ),
)


def _binding(
    *,
    view_row,
    view_column,
    source_row_index,
    source_metric_index,
    subject_id,
    dimension_id,
    value_index,
) -> CellBindingDTO:
    finding_ref = ReportFindingRefDTO(
        report_id="r1", finding_id=f"{subject_id}:{dimension_id}"
    )
    source_binding = SourceBindingDTO(
        subject_id=subject_id, dimension_id=dimension_id, finding_ref=finding_ref
    )
    return CellBindingDTO(
        view_row=view_row,
        view_column=view_column,
        source_row_index=source_row_index,
        source_metric_index=source_metric_index,
        subject_id=subject_id,
        dimension_id=dimension_id,
        value_index=value_index,
        source_binding=source_binding,
    )


def _valid_bindings() -> tuple:
    return (
        _binding(
            view_row=0,
            view_column=0,
            source_row_index=0,
            source_metric_index=0,
            subject_id="s0",
            dimension_id="d0",
            value_index=0,
        ),
        _binding(
            view_row=0,
            view_column=1,
            source_row_index=0,
            source_metric_index=1,
            subject_id="s0",
            dimension_id="d1",
            value_index=1,
        ),
        _binding(
            view_row=1,
            view_column=0,
            source_row_index=1,
            source_metric_index=0,
            subject_id="s1",
            dimension_id="d0",
            value_index=2,
        ),
        _binding(
            view_row=1,
            view_column=1,
            source_row_index=1,
            source_metric_index=1,
            subject_id="s1",
            dimension_id="d1",
            value_index=3,
        ),
    )


def _build(cell_bindings: tuple) -> CompiledReportArtifactDTO:
    """Build a CompiledReportArtifactDTO whose artifact_ref is a genuinely
    correct digest over its own (possibly malformed) payload.

    This deliberately proves the review's point: a malformed record can be
    perfectly self-consistent cryptographically (the digest matches its own
    content) while still violating the semantic/structural contract -
    closure validation, not the digest check, is what must catch it.
    """
    core_refs = CoreArtifactRefs(
        score_table_ref=_FAKE_REF,
        layout_recipe_ref=_FAKE_RECIPE_REF,
        vpm_artifact_ref=_FAKE_VPM_REF,
    )
    compatibility_schema_id = compute_compatibility_schema_id(
        dimensions=_DIMENSIONS, missing_value_semantics=_MISSING_VALUE_SEMANTICS
    )
    compatibility = CompatibilityInfo(
        compatibility_id="test/v1",
        compatibility_schema_id=compatibility_schema_id,
        missing_value_semantics=_MISSING_VALUE_SEMANTICS,
    )
    artifact_id = compute_compiled_report_artifact_id(
        adapted_report_ref=_FAKE_ADAPTED_REPORT_REF,
        adapter_contract_id="contract-1",
        compatibility=compatibility,
        report_semantics=_REPORT_SEMANTICS,
        core_refs=core_refs,
        subjects=_SUBJECTS,
        dimensions=_DIMENSIONS,
        cell_bindings=cell_bindings,
    )
    return CompiledReportArtifactDTO(
        artifact_ref=ArtifactRef(
            artifact_kind=COMPILED_REPORT_ARTIFACT_KIND, artifact_id=artifact_id
        ),
        adapted_report_ref=_FAKE_ADAPTED_REPORT_REF,
        adapter_contract_id="contract-1",
        compatibility_id="test/v1",
        compatibility_schema_id=compatibility_schema_id,
        missing_value_semantics=_MISSING_VALUE_SEMANTICS,
        report_kind=_REPORT_SEMANTICS.report_kind,
        subject_kind=_REPORT_SEMANTICS.subject_kind,
        dimension_namespace=_REPORT_SEMANTICS.dimension_namespace,
        duplicate_value_semantics=_REPORT_SEMANTICS.duplicate_value_semantics,
        report_semantics_id=_REPORT_SEMANTICS_ID,
        score_table_ref=_FAKE_REF,
        layout_recipe_ref=_FAKE_RECIPE_REF,
        vpm_artifact_ref=_FAKE_VPM_REF,
        subjects=_SUBJECTS,
        dimensions=_DIMENSIONS,
        cell_bindings=cell_bindings,
    )


def test_valid_bindings_construct_cleanly() -> None:
    _build(_valid_bindings())


def test_wrong_cell_count_is_rejected() -> None:
    with pytest.raises(ReportCompilationError, match="cells"):
        _build(_valid_bindings()[:-1])


def test_duplicate_view_coordinate_is_rejected() -> None:
    bindings = list(_valid_bindings())
    bindings[3] = _binding(
        view_row=0,
        view_column=0,
        source_row_index=1,
        source_metric_index=1,
        subject_id="s1",
        dimension_id="d1",
        value_index=3,
    )
    with pytest.raises(
        ReportCompilationError, match="duplicate cell_binding view coordinate"
    ):
        _build(tuple(bindings))


def test_duplicate_value_index_is_rejected() -> None:
    bindings = list(_valid_bindings())
    bindings[3] = _binding(
        view_row=1,
        view_column=1,
        source_row_index=1,
        source_metric_index=1,
        subject_id="s1",
        dimension_id="d1",
        value_index=0,
    )
    with pytest.raises(
        ReportCompilationError, match="duplicate cell_binding value_index"
    ):
        _build(tuple(bindings))


def test_subject_id_mismatch_at_coordinate_is_rejected() -> None:
    bindings = list(_valid_bindings())
    # source_row_index=1 names s1, but subject_id claims s0.
    bindings[3] = _binding(
        view_row=1,
        view_column=1,
        source_row_index=1,
        source_metric_index=1,
        subject_id="s0",
        dimension_id="d1",
        value_index=3,
    )
    with pytest.raises(ReportCompilationError, match="subject_id"):
        _build(tuple(bindings))


def test_dimension_id_mismatch_at_coordinate_is_rejected() -> None:
    bindings = list(_valid_bindings())
    bindings[3] = _binding(
        view_row=1,
        view_column=1,
        source_row_index=1,
        source_metric_index=1,
        subject_id="s1",
        dimension_id="d0",
        value_index=3,
    )
    with pytest.raises(ReportCompilationError, match="dimension_id"):
        _build(tuple(bindings))


def test_out_of_range_source_row_index_is_rejected() -> None:
    bindings = list(_valid_bindings())
    bindings[3] = _binding(
        view_row=1,
        view_column=1,
        source_row_index=5,
        source_metric_index=1,
        subject_id="s1",
        dimension_id="d1",
        value_index=3,
    )
    with pytest.raises(ReportCompilationError, match="source_row_index"):
        _build(tuple(bindings))


def test_incomplete_view_coordinate_coverage_is_rejected() -> None:
    bindings = list(_valid_bindings())
    # Two bindings both claim (0, 0); (1, 1) is never covered. This has the
    # right *count* (4) but is not a bijection onto the coordinate grid.
    bindings[3] = _binding(
        view_row=0,
        view_column=0,
        source_row_index=1,
        source_metric_index=1,
        subject_id="s1",
        dimension_id="d1",
        value_index=3,
    )
    with pytest.raises(ReportCompilationError):
        _build(tuple(bindings))


def test_duplicate_subject_id_is_rejected() -> None:
    duplicate_subjects = (
        AdaptedSubjectDTO(subject_id="s0"),
        AdaptedSubjectDTO(subject_id="s0"),
    )
    core_refs = CoreArtifactRefs(
        score_table_ref=_FAKE_REF,
        layout_recipe_ref=_FAKE_RECIPE_REF,
        vpm_artifact_ref=_FAKE_VPM_REF,
    )
    compatibility_schema_id = compute_compatibility_schema_id(
        dimensions=_DIMENSIONS, missing_value_semantics=_MISSING_VALUE_SEMANTICS
    )
    compatibility = CompatibilityInfo(
        compatibility_id="test/v1",
        compatibility_schema_id=compatibility_schema_id,
        missing_value_semantics=_MISSING_VALUE_SEMANTICS,
    )
    artifact_id = compute_compiled_report_artifact_id(
        adapted_report_ref=_FAKE_ADAPTED_REPORT_REF,
        adapter_contract_id="contract-1",
        compatibility=compatibility,
        report_semantics=_REPORT_SEMANTICS,
        core_refs=core_refs,
        subjects=duplicate_subjects,
        dimensions=_DIMENSIONS,
        cell_bindings=_valid_bindings(),
    )
    with pytest.raises(ReportCompilationError, match="duplicate subject_id"):
        CompiledReportArtifactDTO(
            artifact_ref=ArtifactRef(
                artifact_kind=COMPILED_REPORT_ARTIFACT_KIND, artifact_id=artifact_id
            ),
            adapted_report_ref=_FAKE_ADAPTED_REPORT_REF,
            adapter_contract_id="contract-1",
            compatibility_id="test/v1",
            compatibility_schema_id=compatibility_schema_id,
            missing_value_semantics=_MISSING_VALUE_SEMANTICS,
            report_kind=_REPORT_SEMANTICS.report_kind,
            subject_kind=_REPORT_SEMANTICS.subject_kind,
            dimension_namespace=_REPORT_SEMANTICS.dimension_namespace,
            duplicate_value_semantics=_REPORT_SEMANTICS.duplicate_value_semantics,
            report_semantics_id=_REPORT_SEMANTICS_ID,
            score_table_ref=_FAKE_REF,
            layout_recipe_ref=_FAKE_RECIPE_REF,
            vpm_artifact_ref=_FAKE_VPM_REF,
            subjects=duplicate_subjects,
            dimensions=_DIMENSIONS,
            cell_bindings=_valid_bindings(),
        )
