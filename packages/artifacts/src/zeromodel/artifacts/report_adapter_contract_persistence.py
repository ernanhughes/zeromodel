"""Persist and reload `ReportAdapterContractDTO` as a first-class artifact.

Before this module, a compiled report only ever named its governing
contract by a bare `adapter_contract_id: str`. Aggregate validation could
prove the compiled and adapted reports repeated the same id, but had no
way to resolve that id and prove the copied `subject_kind`,
`dimension_namespace`, `missing_value_semantics`, and
`duplicate_value_semantics` fields actually came from the contract that
id names - a self-validating aggregate could declare `subject_kind =
claim` while the actual contract it names declares `subject_kind =
sentence`, and nothing would catch it. Persisting the contract itself
closes that gap the same way `adapted_report_persistence.py` closed the
equivalent gap for the adapted report: resolve canonical bytes,
recompute the digest, require it to equal the requested ref's
`artifact_id`, decode, and reconstruct via `ReportAdapterContractDTO`'s
own `__post_init__` self-validation - never trusting a store's manifest.
"""

from __future__ import annotations

import json

from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_dto import (
    ReportAdapterContractDTO,
    report_adapter_contract_payload,
)
from zeromodel.artifacts.report_errors import ReportCompilationError
from zeromodel.artifacts.store import ArtifactResolver, ArtifactStore

REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND = "zeromodel.artifacts.report-adapter-contract/v1"


def store_report_adapter_contract(
    contract: ReportAdapterContractDTO, *, store: ArtifactStore
) -> ArtifactRef:
    canonical_bytes = canonical_json_bytes(report_adapter_contract_payload(contract))
    ref = store.put(
        REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND, canonical_bytes, manifest=None
    )
    if ref.artifact_id != contract.contract_id:
        raise ReportCompilationError(
            "stored adapter-contract digest does not match contract_id "
            f"(stored={ref.artifact_id}, contract_id={contract.contract_id})"
        )
    return ref


def load_report_adapter_contract(
    ref: ArtifactRef, *, resolver: ArtifactResolver
) -> ReportAdapterContractDTO:
    if ref.artifact_kind != REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND:
        raise ReportCompilationError(
            "load_report_adapter_contract requires artifact_kind="
            f"{REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND!r}, got {ref.artifact_kind!r}"
        )
    canonical_bytes = resolver.resolve_canonical_bytes(ref)
    actual_digest = sha256_digest(canonical_bytes)
    if actual_digest != ref.artifact_id:
        raise ReportCompilationError(
            f"resolved canonical bytes for {ref.artifact_id} do not hash to the requested id "
            f"(got {actual_digest})"
        )
    payload = json.loads(canonical_bytes)

    return ReportAdapterContractDTO(
        contract_id=ref.artifact_id,
        adapter_id=payload["adapter_id"],
        adapter_version=payload["adapter_version"],
        report_kind=payload["report_kind"],
        subject_kind=payload["subject_kind"],
        dimension_namespace=payload["dimension_namespace"],
        compatibility_id=payload["compatibility_id"],
        missing_value_semantics=payload["missing_value_semantics"],
        duplicate_value_semantics=payload["duplicate_value_semantics"],
        spec_version=payload["spec_version"],
    )
