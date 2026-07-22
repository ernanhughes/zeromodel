from __future__ import annotations

from zeromodel.artifacts.adapter import ReportAdapter
from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.compiled_artifact import (
    CellBindingDTO,
    CompiledReportArtifactDTO,
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
    "ARTIFACT_REF_VERSION",
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
    "is_sha256_digest",
    "load_compiled_report_artifact",
    "sha256_digest",
]
