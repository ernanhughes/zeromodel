from __future__ import annotations

from zeromodel.artifacts.adapter import ReportAdapter
from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.compatibility_schema import compute_compatibility_schema_id
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
from zeromodel.artifacts.report_loading import (
    load_compiled_report_artifact,
    load_compiled_report_vpm,
)
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
    "ARTIFACT_REF_VERSION",
    "LAYOUT_RECIPE_ARTIFACT_KIND",
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
    "CoreArtifactRefs",
    "InMemoryArtifactStore",
    "ReportAdaptationError",
    "ReportAdapter",
    "ReportAdapterContractDTO",
    "ReportCompilationError",
    "ReportFindingRefDTO",
    "ScoreSemantics",
    "SourceBindingDTO",
    "canonical_json_bytes",
    "compile_report",
    "compute_compatibility_schema_id",
    "is_sha256_digest",
    "load_compiled_report_artifact",
    "load_compiled_report_vpm",
    "load_layout_recipe",
    "load_score_table",
    "load_vpm_artifact",
    "sha256_digest",
    "store_layout_recipe",
    "store_score_table",
    "store_vpm_artifact",
]
