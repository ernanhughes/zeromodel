"""Compile an adapted report into a canonical, source-bound VPM artifact.

Sequence: read the adapter contract, adapt the report, validate structural
consistency against the contract, build the canonical numeric matrix and
its cell-to-source bindings, build (or reuse) `ScoreTable`/`LayoutRecipe`/
`VPMArtifact` via `zeromodel.core`, assemble the compiled artifact record,
and store it through the injected `ArtifactStore`.

Does not render an image and does not compute an attention/priority
projection - those are separate, later concerns (see the package README's
claims boundary).
"""

from __future__ import annotations

from typing import List, Tuple, TypeVar

import numpy as np

from zeromodel.artifacts.adapter import ReportAdapter
from zeromodel.artifacts.canonicalization import canonical_json_bytes
from zeromodel.artifacts.compiled_artifact import (
    CellBindingDTO,
    CompiledReportArtifactDTO,
    CoreArtifactIdentities,
    compiled_report_identity_payload,
    compute_compiled_report_artifact_id,
)
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_dto import AdaptedReportDTO, ReportAdapterContractDTO
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.artifacts.store import ArtifactStore
from zeromodel.core.artifact import LayoutRecipe, ScoreTable, VPMArtifact, build_vpm

ReportT = TypeVar("ReportT")


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


def _build_matrix_and_bindings(
    adapted: AdaptedReportDTO, contract: ReportAdapterContractDTO
) -> Tuple[np.ndarray, List[str], List[str], Tuple[CellBindingDTO, ...]]:
    row_ids = [subject.subject_id for subject in adapted.subjects]
    metric_ids = [dimension.dimension_id for dimension in adapted.dimensions]
    value_lookup = {
        (value.subject_id, value.dimension_id): (index, value)
        for index, value in enumerate(adapted.values)
    }

    matrix = np.zeros((len(row_ids), len(metric_ids)), dtype=np.float64)
    cell_bindings: List[CellBindingDTO] = []
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
            value_index, value = found
            matrix[row_index, column_index] = value.raw_value
            cell_bindings.append(
                CellBindingDTO(
                    row_index=row_index,
                    column_index=column_index,
                    subject_id=subject_id,
                    dimension_id=dimension_id,
                    value_index=value_index,
                    source_binding=value.source_binding,
                )
            )
    return matrix, row_ids, metric_ids, tuple(cell_bindings)


def _build_vpm_artifact(
    *,
    adapted: AdaptedReportDTO,
    contract: ReportAdapterContractDTO,
    layout_recipe: LayoutRecipe,
) -> Tuple[ScoreTable, VPMArtifact, Tuple[CellBindingDTO, ...]]:
    matrix, row_ids, metric_ids, cell_bindings = _build_matrix_and_bindings(
        adapted, contract
    )
    score_table = ScoreTable(
        values=matrix,
        row_ids=row_ids,
        metric_ids=metric_ids,
        metadata={"adapted_report_id": adapted.adapted_report_id},
    )
    vpm_artifact = build_vpm(
        score_table,
        layout_recipe,
        provenance={
            "adapted_report_id": adapted.adapted_report_id,
            "adapter_contract_id": contract.contract_id,
        },
    )
    return score_table, vpm_artifact, cell_bindings


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

    score_table, vpm_artifact, cell_bindings = _build_vpm_artifact(
        adapted=adapted, contract=contract, layout_recipe=layout_recipe
    )

    core_identities = CoreArtifactIdentities(
        score_table_identity=score_table.digest,
        layout_recipe_identity=layout_recipe.digest,
        vpm_artifact_identity=vpm_artifact.artifact_id,
    )
    artifact_id = compute_compiled_report_artifact_id(
        adapted_report_id=adapted.adapted_report_id,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        core_identities=core_identities,
        subjects=adapted.subjects,
        dimensions=adapted.dimensions,
        cell_bindings=cell_bindings,
    )
    compiled = CompiledReportArtifactDTO(
        artifact_ref=ArtifactRef(
            artifact_kind="zeromodel.artifacts.compiled-report/v1",
            artifact_id=artifact_id,
        ),
        adapted_report_id=adapted.adapted_report_id,
        adapter_contract_id=contract.contract_id,
        compatibility_id=contract.compatibility_id,
        score_table_identity=score_table.digest,
        layout_recipe_identity=layout_recipe.digest,
        vpm_artifact_identity=vpm_artifact.artifact_id,
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
