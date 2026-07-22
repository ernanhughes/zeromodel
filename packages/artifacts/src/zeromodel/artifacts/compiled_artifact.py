"""The canonical, compiled report artifact.

`CompiledReportArtifactDTO.artifact_ref.artifact_id` is this artifact's
identity - a content digest over every field below except `artifact_ref`
itself, following the same pattern as `NavigationTileDTO.tile_id` and
`ArtifactAuthorizationDTO.authorization_id`. Visual projections derived
from this artifact (a full grid, an icon, a stripe, a priority view) are
identified separately and must never change this identity - only the
report's own content (raw values, score semantics, source bindings, the
layout recipe actually used) can.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple

from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.ref import ArtifactRef
from zeromodel.artifacts.report_dto import (
    AdaptedDimensionDTO,
    AdaptedSubjectDTO,
    SourceBindingDTO,
)
from zeromodel.artifacts.report_errors import ReportCompilationError

SPEC_VERSION = "zeromodel-artifacts-compiled-report/v1"
COMPILED_REPORT_ARTIFACT_KIND = "zeromodel.artifacts.compiled-report/v1"


def _require_nonempty_str(value: object, message: str) -> None:
    if not isinstance(value, str) or not value:
        raise ReportCompilationError(message)


class CoreArtifactIdentities(NamedTuple):
    """The three Core-owned digests a compiled report references rather
    than duplicates: `zeromodel.core`'s own `ScoreTable`, `LayoutRecipe`,
    and `VPMArtifact` identities."""

    score_table_identity: str
    layout_recipe_identity: str
    vpm_artifact_identity: str


@dataclass(frozen=True, slots=True)
class CellBindingDTO:
    """The inverse mapping from one compiled VPM coordinate back to its
    source finding. Every numeric compiled value has exactly one of these."""

    row_index: int
    column_index: int
    subject_id: str
    dimension_id: str
    value_index: int
    source_binding: SourceBindingDTO

    def __post_init__(self) -> None:
        if self.row_index < 0 or self.column_index < 0 or self.value_index < 0:
            raise ReportCompilationError(
                "CellBindingDTO.row_index/column_index/value_index must be >= 0"
            )
        _require_nonempty_str(
            self.subject_id, "CellBindingDTO.subject_id must be non-empty"
        )
        _require_nonempty_str(
            self.dimension_id, "CellBindingDTO.dimension_id must be non-empty"
        )

    def payload(self) -> dict:
        return {
            "row_index": self.row_index,
            "column_index": self.column_index,
            "subject_id": self.subject_id,
            "dimension_id": self.dimension_id,
            "value_index": self.value_index,
            "source_binding": self.source_binding.payload(),
        }


@dataclass(frozen=True, slots=True)
class CompiledReportArtifactDTO:
    """The canonical compiled report artifact.

    Does not duplicate the full VPM payload: `vpm_artifact_identity`
    references the `VPMArtifact` core already builds and identifies;
    `score_table_identity`/`layout_recipe_identity` likewise reference
    Core's own `ScoreTable`/`LayoutRecipe` digests.
    """

    artifact_ref: ArtifactRef
    adapted_report_id: str
    adapter_contract_id: str
    compatibility_id: str
    score_table_identity: str
    layout_recipe_identity: str
    vpm_artifact_identity: str
    subjects: Tuple[AdaptedSubjectDTO, ...]
    dimensions: Tuple[AdaptedDimensionDTO, ...]
    cell_bindings: Tuple[CellBindingDTO, ...]
    artifact_kind: str = COMPILED_REPORT_ARTIFACT_KIND
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        if self.artifact_ref.artifact_kind != self.artifact_kind:
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.artifact_ref.artifact_kind must equal artifact_kind"
            )
        _require_nonempty_str(
            self.adapted_report_id,
            "CompiledReportArtifactDTO.adapted_report_id must be non-empty",
        )
        if not self.subjects:
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.subjects must not be empty"
            )
        if not self.dimensions:
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.dimensions must not be empty"
            )
        if not self.cell_bindings:
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.cell_bindings must not be empty"
            )
        expected_id = sha256_digest(
            canonical_json_bytes(compiled_report_identity_payload(self))
        )
        if self.artifact_ref.artifact_id != expected_id:
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.artifact_ref.artifact_id does not match its own "
                "canonical content"
            )


def _identity_payload_fields(
    *,
    adapted_report_id: str,
    adapter_contract_id: str,
    compatibility_id: str,
    core_identities: CoreArtifactIdentities,
    subjects: Tuple[AdaptedSubjectDTO, ...],
    dimensions: Tuple[AdaptedDimensionDTO, ...],
    cell_bindings: Tuple[CellBindingDTO, ...],
    artifact_kind: str,
    spec_version: str,
) -> dict:
    return {
        "spec_version": spec_version,
        "artifact_kind": artifact_kind,
        "adapted_report_id": adapted_report_id,
        "adapter_contract_id": adapter_contract_id,
        "compatibility_id": compatibility_id,
        "score_table_identity": core_identities.score_table_identity,
        "layout_recipe_identity": core_identities.layout_recipe_identity,
        "vpm_artifact_identity": core_identities.vpm_artifact_identity,
        "subjects": [subject.payload() for subject in subjects],
        "dimensions": [dimension.payload() for dimension in dimensions],
        "cell_bindings": [cell.payload() for cell in cell_bindings],
    }


def compiled_report_identity_payload(compiled: CompiledReportArtifactDTO) -> dict:
    """The exact canonical payload `artifact_ref.artifact_id` covers."""
    return _identity_payload_fields(
        adapted_report_id=compiled.adapted_report_id,
        adapter_contract_id=compiled.adapter_contract_id,
        compatibility_id=compiled.compatibility_id,
        core_identities=CoreArtifactIdentities(
            score_table_identity=compiled.score_table_identity,
            layout_recipe_identity=compiled.layout_recipe_identity,
            vpm_artifact_identity=compiled.vpm_artifact_identity,
        ),
        subjects=compiled.subjects,
        dimensions=compiled.dimensions,
        cell_bindings=compiled.cell_bindings,
        artifact_kind=compiled.artifact_kind,
        spec_version=compiled.spec_version,
    )


def compute_compiled_report_artifact_id(
    *,
    adapted_report_id: str,
    adapter_contract_id: str,
    compatibility_id: str,
    core_identities: CoreArtifactIdentities,
    subjects: Tuple[AdaptedSubjectDTO, ...],
    dimensions: Tuple[AdaptedDimensionDTO, ...],
    cell_bindings: Tuple[CellBindingDTO, ...],
    artifact_kind: str = COMPILED_REPORT_ARTIFACT_KIND,
    spec_version: str = SPEC_VERSION,
) -> str:
    payload = _identity_payload_fields(
        adapted_report_id=adapted_report_id,
        adapter_contract_id=adapter_contract_id,
        compatibility_id=compatibility_id,
        core_identities=core_identities,
        subjects=subjects,
        dimensions=dimensions,
        cell_bindings=cell_bindings,
        artifact_kind=artifact_kind,
        spec_version=spec_version,
    )
    return sha256_digest(canonical_json_bytes(payload))
