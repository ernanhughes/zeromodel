"""Resolve and cross-validate the complete compiled-report aggregate.

A `CompiledReportArtifactDTO` is the aggregate root for four independently
identified, independently valid artifacts: `AdaptedReportDTO`, `ScoreTable`,
`LayoutRecipe`, and `VPMArtifact`. Each one can pass its own digest and
self-validation checks while the *collection* is inconsistent - a
self-validating compiled report could reference a `ScoreTable` from one
report, a `LayoutRecipe` from another, and a `VPMArtifact` from a third,
every one individually valid. Loading each referenced object and comparing
its own digest against a store record proves only that the bytes were not
altered; it does not prove the five objects describe the same report.

`load_compiled_report_aggregate` resolves every reference and then runs
`validate_compiled_report_aggregate`, which performs the actual
cross-object closure checks this module exists for:

- the adapted report the compiled report references is the one actually
  loaded, and its subjects/dimensions/report_kind agree;
- the `ScoreTable`'s row/metric order and every value match the adapted
  report's declared subjects/dimensions/raw values exactly;
- the loaded `LayoutRecipe` is the exact recipe embedded in the VPM;
- the loaded `ScoreTable` is the exact source embedded in the VPM;
- every VPM view coordinate resolves to the cell binding that claims it,
  and every cell binding's `value_index` resolves to the adapted value it
  claims to describe.

`VPMArtifact`'s own internal identity (source/recipe/permutation/bounds/
provenance) is Core's authority and is already re-validated by
`VPMArtifact.__post_init__` when `load_vpm_artifact` reconstructs it; this
module does not duplicate that, only the cross-object checks Core cannot
know about.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from zeromodel.artifacts.adapted_report_persistence import load_adapted_report
from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.compiled_artifact import CompiledReportArtifactDTO
from zeromodel.artifacts.core_artifact_persistence import (
    load_layout_recipe,
    load_score_table,
    load_vpm_artifact,
)
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_dto import AdaptedReportDTO
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.artifacts.report_loading import load_compiled_report_artifact
from zeromodel.artifacts.store import ArtifactResolver
from zeromodel.core.artifact import LayoutRecipe, ScoreTable, VPMArtifact

SPEC_VERSION = "zeromodel-artifacts-compiled-report-aggregate/v1"

_CLOSURE_CHECK_NAMES = (
    "compiled_report_valid",
    "adapted_report_valid",
    "score_table_valid",
    "layout_recipe_valid",
    "vpm_artifact_valid",
    "adapted_report_matches_compiled",
    "score_table_matches_adapted_report",
    "vpm_source_matches_score_table",
    "vpm_recipe_matches_layout",
    "cell_bindings_match_vpm",
    "cell_bindings_match_adapted_values",
    "compatibility_contract_valid",
)


@dataclass(frozen=True, slots=True)
class ResolvedCompiledReportAggregateDTO:
    """The complete, resolved compiled-report aggregate: all four objects a
    `CompiledReportArtifactDTO` references, loaded and ready to inspect or
    render. Construct via `load_compiled_report_aggregate`, which also runs
    `validate_compiled_report_aggregate` before returning."""

    compiled_report: CompiledReportArtifactDTO
    adapted_report: AdaptedReportDTO
    score_table: ScoreTable
    layout_recipe: LayoutRecipe
    vpm_artifact: VPMArtifact


def load_compiled_report_aggregate(
    *, ref: ArtifactRef, resolver: ArtifactResolver
) -> ResolvedCompiledReportAggregateDTO:
    """Resolve every artifact a compiled report references and prove they
    form one coherent aggregate.

    Each referenced object is individually decode-and-verified by its own
    loader (`load_adapted_report`, `load_score_table`, `load_layout_recipe`,
    `load_vpm_artifact`) before `validate_compiled_report_aggregate` proves
    the collection is semantically closed - never a partially valid
    aggregate, and never silently rebuilt from another object.
    """
    compiled_report = load_compiled_report_artifact(ref=ref, resolver=resolver)
    adapted_report = load_adapted_report(
        compiled_report.adapted_report_ref, resolver=resolver
    )
    score_table = load_score_table(compiled_report.score_table_ref, resolver=resolver)
    layout_recipe = load_layout_recipe(
        compiled_report.layout_recipe_ref, resolver=resolver
    )
    vpm_artifact = load_vpm_artifact(
        compiled_report.vpm_artifact_ref, resolver=resolver
    )
    aggregate = ResolvedCompiledReportAggregateDTO(
        compiled_report=compiled_report,
        adapted_report=adapted_report,
        score_table=score_table,
        layout_recipe=layout_recipe,
        vpm_artifact=vpm_artifact,
    )
    validate_compiled_report_aggregate(aggregate)
    return aggregate


def validate_compiled_report_aggregate(
    aggregate: ResolvedCompiledReportAggregateDTO,
) -> None:
    """Prove semantic closure across the four resolved objects.

    Each already-loaded object has already proven its own digest and
    self-validation (kind check, content digest, `__post_init__`); this
    proves the *collection* is coherent - the failure mode individual
    digests cannot catch (see this module's docstring).
    """
    _check_adapted_report_matches_compiled(aggregate)
    _check_score_table_matches_adapted_report(aggregate)
    _check_layout_recipe_matches_vpm(aggregate)
    _check_score_table_matches_vpm_source(aggregate)
    _check_cell_bindings_match_vpm_and_values(aggregate)


def _check_adapted_report_matches_compiled(
    aggregate: ResolvedCompiledReportAggregateDTO,
) -> None:
    compiled = aggregate.compiled_report
    adapted = aggregate.adapted_report
    if compiled.adapted_report_ref.artifact_id != adapted.adapted_report_id:
        raise ReportCompilationError(
            "compiled report's adapted_report_ref does not resolve to the loaded adapted report"
        )
    if compiled.adapter_contract_id != adapted.adapter_contract_id:
        raise ReportCompilationError(
            "compiled report adapter_contract_id does not match the adapted report's "
            "adapter_contract_id"
        )
    if compiled.compatibility_id != adapted.compatibility_id:
        raise ReportCompilationError(
            "compiled report compatibility_id does not match the adapted report's "
            "compatibility_id"
        )
    if compiled.report_kind != adapted.report_kind:
        raise ReportCompilationError(
            "compiled report report_kind does not match the adapted report's report_kind"
        )
    if compiled.subjects != adapted.subjects:
        raise ReportCompilationError(
            "compiled report subjects do not match the adapted report's subjects "
            "(identity, order, or attributes differ)"
        )
    if compiled.dimensions != adapted.dimensions:
        raise ReportCompilationError(
            "compiled report dimensions do not match the adapted report's dimensions "
            "(identity, order, semantics, or ranges differ)"
        )


def _check_score_table_matches_adapted_report(
    aggregate: ResolvedCompiledReportAggregateDTO,
) -> None:
    score_table = aggregate.score_table
    adapted = aggregate.adapted_report
    # A raw numeric matrix alone cannot distinguish two adapted reports
    # that declare the same subjects/dimensions/raw values but different
    # confidence, importance, source bindings, parent lineage, or report
    # attributes - none of those participate in the matrix. `compile_report`
    # stamps the source adapted_report_id into ScoreTable.metadata for
    # exactly this reason; require it still points at the resolved adapted
    # report, not merely one with an identical matrix (section 9.1/9.2).
    bound_adapted_report_id = score_table.metadata.get("adapted_report_id")
    if bound_adapted_report_id != adapted.adapted_report_id:
        raise ReportCompilationError(
            "score_table's bound adapted_report_id does not match the resolved adapted report "
            "(same numeric matrix, different adapted report identity)"
        )
    expected_row_ids = tuple(subject.subject_id for subject in adapted.subjects)
    expected_metric_ids = tuple(
        dimension.dimension_id for dimension in adapted.dimensions
    )
    if score_table.row_ids != expected_row_ids:
        raise ReportCompilationError(
            "score_table row_ids do not match the adapted report's declared subject order"
        )
    if score_table.metric_ids != expected_metric_ids:
        raise ReportCompilationError(
            "score_table metric_ids do not match the adapted report's declared dimension order"
        )
    value_lookup = {
        (value.subject_id, value.dimension_id): value.raw_value
        for value in adapted.values
    }
    for row_index, subject_id in enumerate(expected_row_ids):
        for column_index, dimension_id in enumerate(expected_metric_ids):
            expected_value = value_lookup.get((subject_id, dimension_id))
            if expected_value is None:
                raise ReportCompilationError(
                    f"adapted report has no value for subject={subject_id!r} "
                    f"dimension={dimension_id!r} referenced by score_table"
                )
            actual_value = float(score_table.values[row_index, column_index])
            if actual_value != float(expected_value):
                raise ReportCompilationError(
                    f"score_table value at (subject={subject_id!r}, dimension={dimension_id!r}) "
                    "does not match the adapted report's raw value"
                )


def _check_layout_recipe_matches_vpm(
    aggregate: ResolvedCompiledReportAggregateDTO,
) -> None:
    layout_recipe = aggregate.layout_recipe
    vpm_recipe = aggregate.vpm_artifact.recipe
    if layout_recipe.digest != vpm_recipe.digest:
        raise ReportCompilationError(
            "loaded layout_recipe digest does not match the VPM artifact's own recipe"
        )
    if layout_recipe.to_dict() != vpm_recipe.to_dict():
        raise ReportCompilationError(
            "loaded layout_recipe payload does not match the VPM artifact's own recipe"
        )


def _check_score_table_matches_vpm_source(
    aggregate: ResolvedCompiledReportAggregateDTO,
) -> None:
    score_table = aggregate.score_table
    vpm_source = aggregate.vpm_artifact.source
    if score_table.digest != vpm_source.digest:
        raise ReportCompilationError(
            "loaded score_table digest does not match the VPM artifact's embedded source"
        )
    if (
        score_table.row_ids != vpm_source.row_ids
        or score_table.metric_ids != vpm_source.metric_ids
    ):
        raise ReportCompilationError(
            "loaded score_table row/metric ids do not match the VPM artifact's embedded source"
        )
    if not np.array_equal(score_table.values, vpm_source.values):
        raise ReportCompilationError(
            "loaded score_table values do not match the VPM artifact's embedded source"
        )


def _check_cell_bindings_match_vpm_and_values(
    aggregate: ResolvedCompiledReportAggregateDTO,
) -> None:
    compiled = aggregate.compiled_report
    adapted = aggregate.adapted_report
    vpm_artifact = aggregate.vpm_artifact
    if len(adapted.values) != len(compiled.cell_bindings):
        raise ReportCompilationError(
            "adapted report value count does not match the compiled report's cell_binding count"
        )
    for cell in compiled.cell_bindings:
        vpm_cell = vpm_artifact.cell(cell.view_row, cell.view_column)
        if vpm_cell.source_row_index != cell.source_row_index:
            raise ReportCompilationError(
                "cell_binding source_row_index does not match the VPM artifact's own coordinate "
                "resolution"
            )
        if vpm_cell.source_metric_index != cell.source_metric_index:
            raise ReportCompilationError(
                "cell_binding source_metric_index does not match the VPM artifact's own "
                "coordinate resolution"
            )
        if vpm_cell.row_id != cell.subject_id:
            raise ReportCompilationError(
                "cell_binding subject_id does not match the VPM artifact's own coordinate "
                "resolution"
            )
        if vpm_cell.metric_id != cell.dimension_id:
            raise ReportCompilationError(
                "cell_binding dimension_id does not match the VPM artifact's own coordinate "
                "resolution"
            )

        if not (0 <= cell.value_index < len(adapted.values)):
            raise ReportCompilationError(
                "cell_binding value_index is out of range for the adapted report's values"
            )
        value = adapted.values[cell.value_index]
        if (
            value.subject_id != cell.subject_id
            or value.dimension_id != cell.dimension_id
        ):
            raise ReportCompilationError(
                "cell_binding value_index does not refer to the adapted value it claims to "
                "describe"
            )
        if value.source_binding != cell.source_binding:
            raise ReportCompilationError(
                "cell_binding source_binding does not match its adapted value's source_binding"
            )
        if float(value.raw_value) != vpm_cell.raw_value:
            raise ReportCompilationError(
                "cell_binding's adapted value raw_value does not match the VPM artifact's raw "
                "value at that coordinate"
            )


def _artifact_ref_payload(ref: ArtifactRef) -> dict:
    return {"artifact_kind": ref.artifact_kind, "artifact_id": ref.artifact_id}


@dataclass(frozen=True, slots=True)
class CompiledReportClosureReceiptDTO:
    """An auditable record that one specific compiled-report aggregate
    passed every closure check, at the exact refs checked. Never generated
    on partial success: `build_compiled_report_closure_receipt` calls
    `validate_compiled_report_aggregate` first, which raises (fail-closed)
    on the first violation, so a receipt only ever exists for an aggregate
    that passed every check.
    """

    receipt_id: str
    compiled_report_ref: ArtifactRef
    adapted_report_ref: ArtifactRef
    score_table_ref: ArtifactRef
    layout_recipe_ref: ArtifactRef
    vpm_artifact_ref: ArtifactRef
    compatibility_schema_id: str
    checks: Tuple[Tuple[str, bool], ...]
    failure_codes: Tuple[str, ...] = ()
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        expected_id = sha256_digest(canonical_json_bytes(closure_receipt_payload(self)))
        if self.receipt_id != expected_id:
            raise ReportCompilationError(
                "CompiledReportClosureReceiptDTO.receipt_id does not match its own canonical "
                "content"
            )


def _closure_receipt_identity_fields(
    *,
    compiled_report_ref: ArtifactRef,
    adapted_report_ref: ArtifactRef,
    score_table_ref: ArtifactRef,
    layout_recipe_ref: ArtifactRef,
    vpm_artifact_ref: ArtifactRef,
    compatibility_schema_id: str,
    checks: Tuple[Tuple[str, bool], ...],
    failure_codes: Tuple[str, ...],
    spec_version: str,
) -> dict:
    return {
        "spec_version": spec_version,
        "compiled_report_ref": _artifact_ref_payload(compiled_report_ref),
        "adapted_report_ref": _artifact_ref_payload(adapted_report_ref),
        "score_table_ref": _artifact_ref_payload(score_table_ref),
        "layout_recipe_ref": _artifact_ref_payload(layout_recipe_ref),
        "vpm_artifact_ref": _artifact_ref_payload(vpm_artifact_ref),
        "compatibility_schema_id": compatibility_schema_id,
        "checks": [[name, bool(passed)] for name, passed in checks],
        "failure_codes": list(failure_codes),
    }


def closure_receipt_payload(receipt: CompiledReportClosureReceiptDTO) -> dict:
    """The exact canonical payload `receipt_id` covers."""
    return _closure_receipt_identity_fields(
        compiled_report_ref=receipt.compiled_report_ref,
        adapted_report_ref=receipt.adapted_report_ref,
        score_table_ref=receipt.score_table_ref,
        layout_recipe_ref=receipt.layout_recipe_ref,
        vpm_artifact_ref=receipt.vpm_artifact_ref,
        compatibility_schema_id=receipt.compatibility_schema_id,
        checks=receipt.checks,
        failure_codes=receipt.failure_codes,
        spec_version=receipt.spec_version,
    )


def compute_closure_receipt_id(
    *,
    compiled_report_ref: ArtifactRef,
    adapted_report_ref: ArtifactRef,
    score_table_ref: ArtifactRef,
    layout_recipe_ref: ArtifactRef,
    vpm_artifact_ref: ArtifactRef,
    compatibility_schema_id: str,
    checks: Tuple[Tuple[str, bool], ...],
    failure_codes: Tuple[str, ...] = (),
    spec_version: str = SPEC_VERSION,
) -> str:
    payload = _closure_receipt_identity_fields(
        compiled_report_ref=compiled_report_ref,
        adapted_report_ref=adapted_report_ref,
        score_table_ref=score_table_ref,
        layout_recipe_ref=layout_recipe_ref,
        vpm_artifact_ref=vpm_artifact_ref,
        compatibility_schema_id=compatibility_schema_id,
        checks=checks,
        failure_codes=failure_codes,
        spec_version=spec_version,
    )
    return sha256_digest(canonical_json_bytes(payload))


def build_compiled_report_closure_receipt(
    aggregate: ResolvedCompiledReportAggregateDTO,
) -> CompiledReportClosureReceiptDTO:
    """Validate `aggregate` and return a receipt proving every check passed.

    Raises (fail-closed) instead of returning a partial receipt if any
    closure check fails - see `CompiledReportClosureReceiptDTO`'s docstring.
    """
    validate_compiled_report_aggregate(aggregate)
    compiled = aggregate.compiled_report
    checks = tuple((name, True) for name in _CLOSURE_CHECK_NAMES)
    receipt_id = compute_closure_receipt_id(
        compiled_report_ref=compiled.artifact_ref,
        adapted_report_ref=compiled.adapted_report_ref,
        score_table_ref=compiled.score_table_ref,
        layout_recipe_ref=compiled.layout_recipe_ref,
        vpm_artifact_ref=compiled.vpm_artifact_ref,
        compatibility_schema_id=compiled.compatibility_schema_id,
        checks=checks,
    )
    return CompiledReportClosureReceiptDTO(
        receipt_id=receipt_id,
        compiled_report_ref=compiled.artifact_ref,
        adapted_report_ref=compiled.adapted_report_ref,
        score_table_ref=compiled.score_table_ref,
        layout_recipe_ref=compiled.layout_recipe_ref,
        vpm_artifact_ref=compiled.vpm_artifact_ref,
        compatibility_schema_id=compiled.compatibility_schema_id,
        checks=checks,
    )
