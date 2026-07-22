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
from zeromodel.artifacts.core_artifact_persistence import load_vpm_artifact
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
from zeromodel.core.artifact import VPMArtifact


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
        view_row=payload["view_row"],
        view_column=payload["view_column"],
        source_row_index=payload["source_row_index"],
        source_metric_index=payload["source_metric_index"],
        subject_id=payload["subject_id"],
        dimension_id=payload["dimension_id"],
        value_index=payload["value_index"],
        source_binding=_decode_source_binding(payload["source_binding"]),
    )


def _decode_artifact_ref(payload: dict) -> ArtifactRef:
    return ArtifactRef(
        artifact_kind=payload["artifact_kind"], artifact_id=payload["artifact_id"]
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
        compatibility_schema_id=payload["compatibility_schema_id"],
        missing_value_semantics=payload["missing_value_semantics"],
        score_table_ref=_decode_artifact_ref(payload["score_table_ref"]),
        layout_recipe_ref=_decode_artifact_ref(payload["layout_recipe_ref"]),
        vpm_artifact_ref=_decode_artifact_ref(payload["vpm_artifact_ref"]),
        subjects=tuple(_decode_subject(item) for item in payload["subjects"]),
        dimensions=tuple(_decode_dimension(item) for item in payload["dimensions"]),
        cell_bindings=tuple(
            _decode_cell_binding(item) for item in payload["cell_bindings"]
        ),
        artifact_kind=payload["artifact_kind"],
        spec_version=payload["spec_version"],
    )


def load_compiled_report_vpm(
    compiled: CompiledReportArtifactDTO, *, resolver: ArtifactResolver
) -> VPMArtifact:
    """Resolve a compiled report's actual `VPMArtifact` for rendering.

    This is the operation the persisted `vpm_artifact_ref` exists to make
    possible: load a compiled report, resolve its VPM, and render it -
    without ever having held on to the original in-memory `VPMArtifact`
    from the process that compiled it.
    """
    return load_vpm_artifact(compiled.vpm_artifact_ref, resolver=resolver)
