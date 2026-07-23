from __future__ import annotations

from zeromodel.artifacts.adapted_report_persistence import (
    ADAPTED_REPORT_ARTIFACT_KIND,
    load_adapted_report,
    store_adapted_report,
)
from zeromodel.artifacts.adapter import ReportAdapter
from zeromodel.artifacts.aggregate import (
    CompiledReportClosureReceiptDTO,
    ResolvedCompiledReportAggregateDTO,
    build_compiled_report_closure_receipt,
    load_compiled_report_aggregate,
    load_compiled_report_vpm,
    validate_compiled_report_aggregate,
)
from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.compatibility_schema import (
    compute_compatibility_schema_id,
    compute_report_semantics_id,
)
from zeromodel.artifacts.compiled_artifact import (
    CellBindingDTO,
    CompiledReportArtifactDTO,
    CoreArtifactRefs,
)
from zeromodel.artifacts.core_artifact_persistence import (
    LAYOUT_RECIPE_ARTIFACT_KIND,
    SCORE_TABLE_ARTIFACT_KIND,
    VPM_ARTIFACT_ARTIFACT_KIND,
    load_layout_recipe,
    load_score_table,
    load_vpm_artifact,
    store_layout_recipe,
    store_score_table,
    store_vpm_artifact,
)
from zeromodel.artifacts.ref import ARTIFACT_REF_VERSION, ArtifactRef, is_sha256_digest
from zeromodel.artifacts.report_adapter_contract_persistence import (
    REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND,
    load_report_adapter_contract,
    store_report_adapter_contract,
)
from zeromodel.artifacts.report_compiler import compile_report
from zeromodel.artifacts.report_dto import (
    AdaptedDimensionDTO,
    AdaptedReportDTO,
    AdaptedSubjectDTO,
    AdaptedValueDTO,
    ReportAdapterContractDTO,
    ReportFindingRefDTO,
    SourceBindingDTO,
)
from zeromodel.artifacts.report_errors import (
    ReportAdaptationError,
    ReportCompilationError,
)
from zeromodel.artifacts.report_loading import load_compiled_report_artifact
from zeromodel.artifacts.score_semantics import ScoreSemantics
from zeromodel.artifacts.store import (
    ArtifactIntegrityError,
    ArtifactManifestConflictError,
    ArtifactNotFoundError,
    ArtifactResolver,
    ArtifactStore,
    InMemoryArtifactStore,
)

__all__ = [
    "ADAPTED_REPORT_ARTIFACT_KIND",
    "ARTIFACT_REF_VERSION",
    "LAYOUT_RECIPE_ARTIFACT_KIND",
    "REPORT_ADAPTER_CONTRACT_ARTIFACT_KIND",
    "SCORE_TABLE_ARTIFACT_KIND",
    "VPM_ARTIFACT_ARTIFACT_KIND",
    "AdaptedDimensionDTO",
    "AdaptedReportDTO",
    "AdaptedSubjectDTO",
    "AdaptedValueDTO",
    "ArtifactIntegrityError",
    "ArtifactManifestConflictError",
    "ArtifactNotFoundError",
    "ArtifactRef",
    "ArtifactResolver",
    "ArtifactStore",
    "CellBindingDTO",
    "CompiledReportArtifactDTO",
    "CompiledReportClosureReceiptDTO",
    "CoreArtifactRefs",
    "InMemoryArtifactStore",
    "ReportAdaptationError",
    "ReportAdapter",
    "ReportAdapterContractDTO",
    "ReportCompilationError",
    "ReportFindingRefDTO",
    "ResolvedCompiledReportAggregateDTO",
    "ScoreSemantics",
    "SourceBindingDTO",
    "build_compiled_report_closure_receipt",
    "canonical_json_bytes",
    "compile_report",
    "compute_compatibility_schema_id",
    "compute_report_semantics_id",
    "is_sha256_digest",
    "load_adapted_report",
    "load_compiled_report_aggregate",
    "load_compiled_report_artifact",
    "load_compiled_report_vpm",
    "load_layout_recipe",
    "load_report_adapter_contract",
    "load_score_table",
    "load_vpm_artifact",
    "sha256_digest",
    "store_adapted_report",
    "store_layout_recipe",
    "store_report_adapter_contract",
    "store_score_table",
    "store_vpm_artifact",
    "validate_compiled_report_aggregate",
]
