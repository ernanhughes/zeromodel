"""Load a previously compiled report artifact from canonical bytes.

Follows the same decode-and-verify pattern used by
`zeromodel.navigation.storage.load_tile`: resolve canonical bytes, recompute
the digest, require it to equal the requested ref's `artifact_id`, decode
the canonical JSON payload, and reconstruct the DTO - never trusting a
store's manifest as authoritative.
"""

from __future__ import annotations

import json
from typing import Tuple

from zeromodel.artifacts.canonicalization import sha256_digest
from zeromodel.artifacts.compiled_artifact import (
    COMPILED_REPORT_ARTIFACT_KIND,
    CellBindingDTO,
    CompiledReportArtifactDTO,
)
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_dto import (
    AdaptedDimensionDTO,
    AdaptedSubjectDTO,
    ReportFindingRefDTO,
    SourceBindingDTO,
)
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.artifacts.score_semantics import ScoreSemantics
from zeromodel.artifacts.store import ArtifactResolver


def _decode_attributes(payload: dict) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(payload.items()))


def _decode_subject(payload: dict) -> AdaptedSubjectDTO:
    return AdaptedSubjectDTO(
        subject_id=payload["subject_id"],
        label=payload["label"],
        ordinal=payload["ordinal"],
        source_ref=payload["source_ref"],
        attributes=_decode_attributes(payload["attributes"]),
    )


def _decode_dimension(payload: dict) -> AdaptedDimensionDTO:
    return AdaptedDimensionDTO(
        dimension_id=payload["dimension_id"],
        label=payload["label"],
        score_semantics=ScoreSemantics(payload["score_semantics"]),
        family=payload["family"],
        value_min=payload["value_min"],
        value_max=payload["value_max"],
        target_min=payload["target_min"],
        target_max=payload["target_max"],
        default_importance=payload["default_importance"],
        attributes=_decode_attributes(payload["attributes"]),
    )


def _decode_finding_ref(payload: dict) -> ReportFindingRefDTO:
    return ReportFindingRefDTO(
        report_id=payload["report_id"],
        finding_id=payload["finding_id"],
        finding_kind=payload["finding_kind"],
    )


def _decode_source_binding(payload: dict) -> SourceBindingDTO:
    return SourceBindingDTO(
        subject_id=payload["subject_id"],
        dimension_id=payload["dimension_id"],
        finding_ref=_decode_finding_ref(payload["finding_ref"]),
        source_uri=payload["source_uri"],
        source_start=payload["source_start"],
        source_end=payload["source_end"],
        attributes=_decode_attributes(payload["attributes"]),
    )


def _decode_cell_binding(payload: dict) -> CellBindingDTO:
    return CellBindingDTO(
        row_index=payload["row_index"],
        column_index=payload["column_index"],
        subject_id=payload["subject_id"],
        dimension_id=payload["dimension_id"],
        value_index=payload["value_index"],
        source_binding=_decode_source_binding(payload["source_binding"]),
    )


def load_compiled_report_artifact(
    *, ref: ArtifactRef, resolver: ArtifactResolver
) -> CompiledReportArtifactDTO:
    """Resolve, verify, and reconstruct a compiled report artifact.

    Never trusts the store's manifest: the returned DTO is decoded
    entirely from canonical bytes whose digest is verified to equal
    `ref.artifact_id` before decoding.
    """
    if ref.artifact_kind != COMPILED_REPORT_ARTIFACT_KIND:
        raise ReportCompilationError(
            f"load_compiled_report_artifact requires artifact_kind={COMPILED_REPORT_ARTIFACT_KIND!r}, "
            f"got {ref.artifact_kind!r}"
        )
    canonical_bytes = resolver.resolve_canonical_bytes(ref)
    actual_digest = sha256_digest(canonical_bytes)
    if actual_digest != ref.artifact_id:
        raise ReportCompilationError(
            f"resolved canonical bytes for {ref.artifact_id} do not hash to the requested id "
            f"(got {actual_digest})"
        )
    payload = json.loads(canonical_bytes)

    return CompiledReportArtifactDTO(
        artifact_ref=ref,
        adapted_report_id=payload["adapted_report_id"],
        adapter_contract_id=payload["adapter_contract_id"],
        compatibility_id=payload["compatibility_id"],
        score_table_identity=payload["score_table_identity"],
        layout_recipe_identity=payload["layout_recipe_identity"],
        vpm_artifact_identity=payload["vpm_artifact_identity"],
        subjects=tuple(_decode_subject(item) for item in payload["subjects"]),
        dimensions=tuple(_decode_dimension(item) for item in payload["dimensions"]),
        cell_bindings=tuple(
            _decode_cell_binding(item) for item in payload["cell_bindings"]
        ),
        artifact_kind=payload["artifact_kind"],
        spec_version=payload["spec_version"],
    )
