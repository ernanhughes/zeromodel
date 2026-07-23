"""Compile an adapted report into a canonical, source-bound VPM artifact.

Sequence: read the adapter contract, adapt the report, validate structural
consistency against the contract, build the canonical source-order numeric
matrix, build `ScoreTable`/`VPMArtifact` via `zeromodel.core` (this is where
the layout recipe's row/column permutation is actually computed), build
cell bindings in VPM *view* coordinates by resolving each cell through
`VPMArtifact.cell()`, persist the Core artifacts and the compiled report
record through the injected `ArtifactStore`, and assemble the compiled
artifact record.

Does not render an image and does not compute an attention/priority
projection - those are separate, later concerns (see the package README's
claims boundary).
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TypeVar

import numpy as np

from zeromodel.artifacts.adapted_report_persistence import store_adapted_report
from zeromodel.artifacts.adapter import ReportAdapter
from zeromodel.artifacts.canonicalization import canonical_json_bytes
from zeromodel.artifacts.compatibility_schema import (
    compute_compatibility_schema_id,
    compute_report_semantics_id,
)
from zeromodel.artifacts.compiled_artifact import (
    CellBindingDTO,
    CompatibilityInfo,
    CompiledReportArtifactDTO,
    CoreArtifactRefs,
    ReportSemanticsInfo,
    compiled_report_identity_payload,
    compute_compiled_report_artifact_id,
)
from zeromodel.artifacts.core_artifact_persistence import (
    store_layout_recipe,
    store_score_table,
    store_vpm_artifact,
)
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_dto import (
    AdaptedReportDTO,
    AdaptedValueDTO,
    ReportAdapterContractDTO,
)
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.artifacts.store import ArtifactStore
from zeromodel.core.artifact import LayoutRecipe, ScoreTable, VPMArtifact, build_vpm

ReportT = TypeVar("ReportT")

_ValueLookup = Dict[Tuple[str, str], Tuple[int, AdaptedValueDTO]]


def _validate_adapted_report_matches_contract(
    adapted: AdaptedReportDTO, contract: ReportAdapterContractDTO
) -> None:
    if adapted.adapter_contract_id != contract.contract_id:
        raise ReportCompilationError(
            "adapted report's adapter_contract_id does not match the adapter's own contract "
            f"(report declares {adapted.adapter_contract_id!r}, contract is {contract.contract_id!r})"
        )
    if adapted.report_kind != contract.report_kind:
        raise ReportCompilationError(
            f"adapted report_kind {adapted.report_kind!r} is incompatible with adapter "
            f"contract report_kind {contract.report_kind!r}"
        )
    if adapted.compatibility_id != contract.compatibility_id:
        raise ReportCompilationError(
            f"adapted compatibility_id {adapted.compatibility_id!r} does not match the "
            f"adapter contract's compatibility_id {contract.compatibility_id!r}"
        )


def _build_score_table(
    adapted: AdaptedReportDTO, contract: ReportAdapterContractDTO
) -> Tuple[ScoreTable, _ValueLookup]:
    row_ids = [subject.subject_id for subject in adapted.subjects]
    metric_ids = [dimension.dimension_id for dimension in adapted.dimensions]
    value_lookup: _ValueLookup = {
        (value.subject_id, value.dimension_id): (index, value)
        for index, value in enumerate(adapted.values)
    }

    matrix = np.zeros((len(row_ids), len(metric_ids)), dtype=np.float64)
    for row_index, subject_id in enumerate(row_ids):
        for column_index, dimension_id in enumerate(metric_ids):
            found = value_lookup.get((subject_id, dimension_id))
            if found is None:
                if contract.missing_value_semantics == "error":
                    raise ReportCompilationError(
                        f"missing value for subject={subject_id!r} dimension={dimension_id!r} "
                        "(missing_value_semantics='error')"
                    )
                raise ReportCompilationError(
                    "missing_value_semantics='absent' is declared but compile_report has no "
                    "sparse VPM representation yet - see the package README's claims boundary"
                )
            _, value = found
            matrix[row_index, column_index] = value.raw_value

    score_table = ScoreTable(
        values=matrix,
        row_ids=row_ids,
        metric_ids=metric_ids,
        metadata={"adapted_report_id": adapted.adapted_report_id},
    )
    return score_table, value_lookup


def _build_cell_bindings(
    *, vpm_artifact: VPMArtifact, value_lookup: _ValueLookup
) -> Tuple[CellBindingDTO, ...]:
    """Build cell bindings in VPM *view* coordinates.

    Must run after `build_vpm()`, not before: the layout recipe's
    row/column reordering is only known once `VPMArtifact.row_order`/
    `column_order` exist. `VPMArtifact.cell(view_row, view_column)` is the
    single authority translating a view coordinate to its source
    coordinate - re-deriving that mapping here would risk it drifting out
    of sync with Core's own resolution logic.
    """
    row_count, column_count = vpm_artifact.shape
    cell_bindings: List[CellBindingDTO] = []
    for view_row in range(row_count):
        for view_column in range(column_count):
            cell = vpm_artifact.cell(view_row, view_column)
            found = value_lookup.get((cell.row_id, cell.metric_id))
            if found is None:
                raise ReportCompilationError(
                    f"no adapted value for subject={cell.row_id!r} dimension={cell.metric_id!r} "
                    "while building cell bindings"
                )
            value_index, value = found
            cell_bindings.append(
                CellBindingDTO(
                    view_row=cell.view_row,
                    view_column=cell.view_column,
                    source_row_index=cell.source_row_index,
                    source_metric_index=cell.source_metric_index,
                    subject_id=cell.row_id,
                    dimension_id=cell.metric_id,
                    value_index=value_index,
                    source_binding=value.source_binding,
                )
            )
    return tuple(cell_bindings)


def compile_report(
    *,
    adapter: "ReportAdapter[ReportT]",
    report: ReportT,
    layout_recipe: LayoutRecipe,
    store: ArtifactStore,
) -> CompiledReportArtifactDTO:
    """Compile one domain report into a canonical, stored, source-bound artifact."""
    contract = adapter.contract()
    adapted = adapter.adapt(report)
    _validate_adapted_report_matches_contract(adapted, contract)
    adapted_report_ref = store_adapted_report(adapted, store=store)

    score_table, value_lookup = _build_score_table(adapted, contract)
    vpm_artifact = build_vpm(
        score_table,
        layout_recipe,
        provenance={
            "adapted_report_id": adapted.adapted_report_id,
            "adapter_contract_id": contract.contract_id,
        },
    )
    cell_bindings = _build_cell_bindings(
        vpm_artifact=vpm_artifact, value_lookup=value_lookup
    )

    score_table_ref = store_score_table(score_table, store=store)
    layout_recipe_ref = store_layout_recipe(layout_recipe, store=store)
    vpm_artifact_ref = store_vpm_artifact(vpm_artifact, store=store)
    core_refs = CoreArtifactRefs(
        score_table_ref=score_table_ref,
        layout_recipe_ref=layout_recipe_ref,
        vpm_artifact_ref=vpm_artifact_ref,
    )

    compatibility_schema_id = compute_compatibility_schema_id(
        dimensions=adapted.dimensions,
        missing_value_semantics=contract.missing_value_semantics,
    )
    compatibility = CompatibilityInfo(
        compatibility_id=contract.compatibility_id,
        compatibility_schema_id=compatibility_schema_id,
        missing_value_semantics=contract.missing_value_semantics,
    )
    report_semantics_id = compute_report_semantics_id(
        report_kind=adapted.report_kind,
        subject_kind=contract.subject_kind,
        dimension_namespace=contract.dimension_namespace,
        duplicate_value_semantics=contract.duplicate_value_semantics,
    )
    report_semantics = ReportSemanticsInfo(
        report_kind=adapted.report_kind,
        subject_kind=contract.subject_kind,
        dimension_namespace=contract.dimension_namespace,
        duplicate_value_semantics=contract.duplicate_value_semantics,
        report_semantics_id=report_semantics_id,
    )
    artifact_id = compute_compiled_report_artifact_id(
        adapted_report_ref=adapted_report_ref,
        adapter_contract_id=contract.contract_id,
        compatibility=compatibility,
        report_semantics=report_semantics,
        core_refs=core_refs,
        subjects=adapted.subjects,
        dimensions=adapted.dimensions,
        cell_bindings=cell_bindings,
    )
    compiled = CompiledReportArtifactDTO(
        artifact_ref=ArtifactRef(
            artifact_kind="zeromodel.artifacts.compiled-report/v1",
            artifact_id=artifact_id,
        ),
        adapted_report_ref=adapted_report_ref,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        compatibility_schema_id=compatibility_schema_id,
        missing_value_semantics=contract.missing_value_semantics,
        report_kind=adapted.report_kind,
        subject_kind=contract.subject_kind,
        dimension_namespace=contract.dimension_namespace,
        duplicate_value_semantics=contract.duplicate_value_semantics,
        report_semantics_id=report_semantics_id,
        score_table_ref=score_table_ref,
        layout_recipe_ref=layout_recipe_ref,
        vpm_artifact_ref=vpm_artifact_ref,
        subjects=adapted.subjects,
        dimensions=adapted.dimensions,
        cell_bindings=cell_bindings,
    )

    canonical_bytes = canonical_json_bytes(compiled_report_identity_payload(compiled))
    ref = store.put(compiled.artifact_kind, canonical_bytes, manifest=None)
    if ref.artifact_id != artifact_id:
        raise ReportCompilationError(
            "stored compiled-report digest does not match its declared identity"
        )
    return compiled
