"""Persist and reload `AdaptedReportDTO` as a first-class artifact.

Before this module, `AdaptedReportDTO` only ever existed as a discarded
local Python object inside `compile_report()`: the compiled report named it
by digest (`adapted_report_id`) but nothing made that digest resolvable
after the process exited. Raw values, per-value confidence, per-value
importance, source bindings, and parent-report lineage - none of which the
VPM matrix preserves completely - were unrecoverable once compilation
finished. This module closes that gap using the same decode-and-verify
pattern as `core_artifact_persistence.py`: resolve canonical bytes,
recompute the digest, require it to equal the requested ref's
`artifact_id`, decode, and reconstruct via `AdaptedReportDTO.__post_init__`
- never trusting a store's manifest as authoritative.

Canonical format: the stored payload is exactly
`adapted_report_signing_payload(adapted_report)` - the same payload
`AdaptedReportDTO.adapted_report_id` is computed over, with the
self-referential id field excluded. Because `ArtifactStore.put()` computes
`ref.artifact_id` as `sha256_digest(canonical_bytes)` over that same
payload, `ref.artifact_id == adapted_report.adapted_report_id` always
holds - the artifact-store identity and the DTO's own content identity are
the same digest, not two independent layers that happen to agree.
"""

from __future__ import annotations

import json

from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_decode import (
    decode_attributes,
    decode_dimension,
    decode_subject,
    decode_value,
)
from zeromodel.artifacts.report_dto import (
    AdaptedReportDTO,
    adapted_report_signing_payload,
)
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.artifacts.store import ArtifactResolver, ArtifactStore

ADAPTED_REPORT_ARTIFACT_KIND = "zeromodel.artifacts.adapted-report/v1"


def store_adapted_report(
    adapted_report: AdaptedReportDTO, *, store: ArtifactStore
) -> ArtifactRef:
    canonical_bytes = canonical_json_bytes(
        adapted_report_signing_payload(adapted_report)
    )
    ref = store.put(ADAPTED_REPORT_ARTIFACT_KIND, canonical_bytes, manifest=None)
    if ref.artifact_id != adapted_report.adapted_report_id:
        raise ReportCompilationError(
            "stored adapted-report digest does not match adapted_report_id "
            f"(stored={ref.artifact_id}, adapted_report_id={adapted_report.adapted_report_id})"
        )
    return ref


def load_adapted_report(
    ref: ArtifactRef, *, resolver: ArtifactResolver
) -> AdaptedReportDTO:
    if ref.artifact_kind != ADAPTED_REPORT_ARTIFACT_KIND:
        raise ReportCompilationError(
            f"load_adapted_report requires artifact_kind={ADAPTED_REPORT_ARTIFACT_KIND!r}, "
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

    return AdaptedReportDTO(
        adapted_report_id=ref.artifact_id,
        report_id=payload["report_id"],
        report_kind=payload["report_kind"],
        adapter_contract_id=payload["adapter_contract_id"],
        compatibility_id=payload["compatibility_id"],
        subjects=tuple(decode_subject(item) for item in payload["subjects"]),
        dimensions=tuple(decode_dimension(item) for item in payload["dimensions"]),
        values=tuple(decode_value(item) for item in payload["values"]),
        parent_report_ids=tuple(payload["parent_report_ids"]),
        attributes=decode_attributes(payload["attributes"]),
        spec_version=payload["spec_version"],
    )
