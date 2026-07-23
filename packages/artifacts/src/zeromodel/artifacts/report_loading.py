"""Load a previously compiled report artifact from canonical bytes.

Follows the same decode-and-verify pattern used by
`zeromodel.navigation.storage.load_tile`: resolve canonical bytes, recompute
the digest, require it to equal the requested ref's `artifact_id`, decode
the canonical JSON payload, and reconstruct the DTO - never trusting a
store's manifest as authoritative.
"""

from __future__ import annotations

import json

from zeromodel.artifacts.canonicalization import sha256_digest
from zeromodel.artifacts.compiled_artifact import (
    COMPILED_REPORT_ARTIFACT_KIND,
    CellBindingDTO,
    CompiledReportArtifactDTO,
)
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_decode import (
    decode_dimension,
    decode_source_binding,
    decode_subject,
)
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.artifacts.store import ArtifactResolver


def _decode_cell_binding(payload: dict) -> CellBindingDTO:
    return CellBindingDTO(
        view_row=payload["view_row"],
        view_column=payload["view_column"],
        source_row_index=payload["source_row_index"],
        source_metric_index=payload["source_metric_index"],
        subject_id=payload["subject_id"],
        dimension_id=payload["dimension_id"],
        value_index=payload["value_index"],
        source_binding=decode_source_binding(payload["source_binding"]),
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
        adapted_report_ref=_decode_artifact_ref(payload["adapted_report_ref"]),
        adapter_contract_ref=_decode_artifact_ref(payload["adapter_contract_ref"]),
        compatibility_id=payload["compatibility_id"],
        compatibility_schema_id=payload["compatibility_schema_id"],
        missing_value_semantics=payload["missing_value_semantics"],
        report_kind=payload["report_kind"],
        subject_kind=payload["subject_kind"],
        dimension_namespace=payload["dimension_namespace"],
        duplicate_value_semantics=payload["duplicate_value_semantics"],
        report_semantics_id=payload["report_semantics_id"],
        score_table_ref=_decode_artifact_ref(payload["score_table_ref"]),
        layout_recipe_ref=_decode_artifact_ref(payload["layout_recipe_ref"]),
        vpm_artifact_ref=_decode_artifact_ref(payload["vpm_artifact_ref"]),
        subjects=tuple(decode_subject(item) for item in payload["subjects"]),
        dimensions=tuple(decode_dimension(item) for item in payload["dimensions"]),
        cell_bindings=tuple(
            _decode_cell_binding(item) for item in payload["cell_bindings"]
        ),
        artifact_kind=payload["artifact_kind"],
        spec_version=payload["spec_version"],
    )
