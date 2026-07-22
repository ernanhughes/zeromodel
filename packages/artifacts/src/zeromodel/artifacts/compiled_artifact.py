"""The canonical, compiled report artifact.

`CompiledReportArtifactDTO.artifact_ref.artifact_id` is this artifact's
identity - a content digest over every field below except `artifact_ref`
itself, following the same pattern as `NavigationTileDTO.tile_id` and
`ArtifactAuthorizationDTO.authorization_id`. Visual projections derived
from this artifact (a full grid, an icon, a stripe, a priority view) are
identified separately and must never change this identity - only the
report's own content (raw values, score semantics, source bindings, the
layout recipe actually used) can.

The Core `ScoreTable`, `LayoutRecipe`, and `VPMArtifact` this record
describes are not embedded here - they are persisted separately through
the same injected `ArtifactStore` (see `core_artifact_persistence.py`) and
referenced by real `ArtifactRef` values, so a reloaded
`CompiledReportArtifactDTO` can actually resolve its own VPM rather than
merely naming a digest that once existed in a since-discarded Python
object.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Tuple

from zeromodel.artifacts.canonicalization import canonical_json_bytes, sha256_digest
from zeromodel.artifacts.compatibility_schema import compute_compatibility_schema_id
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


class CoreArtifactRefs(NamedTuple):
    """Real, resolvable references to the three Core-owned artifacts a
    compiled report describes, rather than naked digest strings naming
    Python objects that may no longer exist anywhere."""

    score_table_ref: ArtifactRef
    layout_recipe_ref: ArtifactRef
    vpm_artifact_ref: ArtifactRef


class CompatibilityInfo(NamedTuple):
    """The three ingredients of a report's compatibility claim, grouped so
    identity-payload builders take one parameter instead of three."""

    compatibility_id: str
    compatibility_schema_id: str
    missing_value_semantics: str


@dataclass(frozen=True, slots=True)
class CellBindingDTO:
    """The inverse mapping from one compiled *VPM view* coordinate back to
    its source finding. Every rendered cell has exactly one of these.

    `view_row`/`view_column` are coordinates in the VPM as actually laid
    out (after the layout recipe's row/column reordering) - the same
    coordinate space `VPMArtifact.cell(view_row, view_column)` addresses.
    `source_row_index`/`source_metric_index` are the corresponding
    positions in the *source-declared* order (the `ScoreTable`'s own row/
    metric order, before layout reordering). These two coordinate spaces
    are only guaranteed identical when the layout recipe declares
    source-order rows and columns; for any other recipe they diverge, so
    both must be recorded rather than only one.
    """

    view_row: int
    view_column: int
    source_row_index: int
    source_metric_index: int
    subject_id: str
    dimension_id: str
    value_index: int
    source_binding: SourceBindingDTO

    def __post_init__(self) -> None:
        for field_name, value in (
            ("view_row", self.view_row),
            ("view_column", self.view_column),
            ("source_row_index", self.source_row_index),
            ("source_metric_index", self.source_metric_index),
            ("value_index", self.value_index),
        ):
            if value < 0:
                raise ReportCompilationError(
                    f"CellBindingDTO.{field_name} must be >= 0"
                )
        _require_nonempty_str(
            self.subject_id, "CellBindingDTO.subject_id must be non-empty"
        )
        _require_nonempty_str(
            self.dimension_id, "CellBindingDTO.dimension_id must be non-empty"
        )
        if self.source_binding.subject_id != self.subject_id:
            raise ReportCompilationError(
                "CellBindingDTO.source_binding.subject_id does not match this cell's subject_id"
            )
        if self.source_binding.dimension_id != self.dimension_id:
            raise ReportCompilationError(
                "CellBindingDTO.source_binding.dimension_id does not match this cell's dimension_id"
            )

    def payload(self) -> dict:
        return {
            "view_row": self.view_row,
            "view_column": self.view_column,
            "source_row_index": self.source_row_index,
            "source_metric_index": self.source_metric_index,
            "subject_id": self.subject_id,
            "dimension_id": self.dimension_id,
            "value_index": self.value_index,
            "source_binding": self.source_binding.payload(),
        }


def _artifact_ref_payload(ref: ArtifactRef) -> dict:
    return {"artifact_kind": ref.artifact_kind, "artifact_id": ref.artifact_id}


@dataclass(frozen=True, slots=True)
class CompiledReportArtifactDTO:
    """The canonical compiled report artifact.

    References (never embeds) the Core `ScoreTable`/`LayoutRecipe`/
    `VPMArtifact` this compilation produced, via real, independently
    resolvable `ArtifactRef` values.
    """

    artifact_ref: ArtifactRef
    adapted_report_id: str
    adapter_contract_id: str
    compatibility_id: str
    compatibility_schema_id: str
    missing_value_semantics: str
    score_table_ref: ArtifactRef
    layout_recipe_ref: ArtifactRef
    vpm_artifact_ref: ArtifactRef
    subjects: Tuple[AdaptedSubjectDTO, ...]
    dimensions: Tuple[AdaptedDimensionDTO, ...]
    cell_bindings: Tuple[CellBindingDTO, ...]
    artifact_kind: str = COMPILED_REPORT_ARTIFACT_KIND
    spec_version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        self._validate_basic_shape()
        self._validate_closure()
        self._validate_compatibility_schema()
        expected_id = sha256_digest(
            canonical_json_bytes(compiled_report_identity_payload(self))
        )
        if self.artifact_ref.artifact_id != expected_id:
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.artifact_ref.artifact_id does not match its own "
                "canonical content"
            )

    def _validate_basic_shape(self) -> None:
        if self.artifact_ref.artifact_kind != self.artifact_kind:
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.artifact_ref.artifact_kind must equal artifact_kind"
            )
        _require_nonempty_str(
            self.adapted_report_id,
            "CompiledReportArtifactDTO.adapted_report_id must be non-empty",
        )
        _require_nonempty_str(
            self.compatibility_id,
            "CompiledReportArtifactDTO.compatibility_id must be non-empty",
        )
        _require_nonempty_str(
            self.compatibility_schema_id,
            "CompiledReportArtifactDTO.compatibility_schema_id must be non-empty",
        )
        if self.missing_value_semantics not in ("error", "absent"):
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.missing_value_semantics must be 'error' or 'absent'"
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

    def _validate_compatibility_schema(self) -> None:
        """`compatibility_id` is an opaque, caller-chosen label - it does
        not by itself prove two reports share a coordinate schema. This
        recomputes the content-derived schema digest from this record's
        own `dimensions`/`missing_value_semantics` and requires it to
        match the declared `compatibility_schema_id`, the same
        self-validating pattern used for `artifact_ref.artifact_id`.
        """
        expected_schema_id = compute_compatibility_schema_id(
            dimensions=self.dimensions,
            missing_value_semantics=self.missing_value_semantics,
        )
        if self.compatibility_schema_id != expected_schema_id:
            raise ReportCompilationError(
                "CompiledReportArtifactDTO.compatibility_schema_id does not match the content "
                "digest of its own dimensions/missing_value_semantics"
            )

    def _validate_closure(self) -> None:
        """Verify the record is structurally self-consistent, not merely
        digest-consistent: a digest proves bytes were not altered relative
        to the reference, not that the bytes form a valid compiled report.
        """
        subject_ids = [subject.subject_id for subject in self.subjects]
        if len(set(subject_ids)) != len(subject_ids):
            raise ReportCompilationError(
                "CompiledReportArtifactDTO has duplicate subject_id values"
            )
        dimension_ids = [dimension.dimension_id for dimension in self.dimensions]
        if len(set(dimension_ids)) != len(dimension_ids):
            raise ReportCompilationError(
                "CompiledReportArtifactDTO has duplicate dimension_id values"
            )

        row_count = len(self.subjects)
        column_count = len(self.dimensions)
        expected_cells = row_count * column_count
        if len(self.cell_bindings) != expected_cells:
            raise ReportCompilationError(
                f"CompiledReportArtifactDTO declares {row_count} subjects x {column_count} "
                f"dimensions ({expected_cells} cells) but has {len(self.cell_bindings)} cell_bindings"
            )

        seen_view_coords: set = set()
        seen_value_indices: set = set()
        for cell in self.cell_bindings:
            if not (0 <= cell.view_row < row_count):
                raise ReportCompilationError(
                    f"cell_binding view_row {cell.view_row} out of range for {row_count} subjects"
                )
            if not (0 <= cell.view_column < column_count):
                raise ReportCompilationError(
                    f"cell_binding view_column {cell.view_column} out of range for {column_count} dimensions"
                )
            if not (0 <= cell.source_row_index < row_count):
                raise ReportCompilationError(
                    f"cell_binding source_row_index {cell.source_row_index} out of range for {row_count} subjects"
                )
            if not (0 <= cell.source_metric_index < column_count):
                raise ReportCompilationError(
                    f"cell_binding source_metric_index {cell.source_metric_index} out of range "
                    f"for {column_count} dimensions"
                )

            view_coord = (cell.view_row, cell.view_column)
            if view_coord in seen_view_coords:
                raise ReportCompilationError(
                    f"duplicate cell_binding view coordinate {view_coord}"
                )
            seen_view_coords.add(view_coord)

            if cell.value_index in seen_value_indices:
                raise ReportCompilationError(
                    f"duplicate cell_binding value_index {cell.value_index}"
                )
            seen_value_indices.add(cell.value_index)

            expected_subject_id = self.subjects[cell.source_row_index].subject_id
            if cell.subject_id != expected_subject_id:
                raise ReportCompilationError(
                    f"cell_binding subject_id {cell.subject_id!r} does not match the subject at "
                    f"source_row_index {cell.source_row_index} ({expected_subject_id!r})"
                )
            expected_dimension_id = self.dimensions[
                cell.source_metric_index
            ].dimension_id
            if cell.dimension_id != expected_dimension_id:
                raise ReportCompilationError(
                    f"cell_binding dimension_id {cell.dimension_id!r} does not match the dimension at "
                    f"source_metric_index {cell.source_metric_index} ({expected_dimension_id!r})"
                )

        if seen_view_coords != {
            (r, c) for r in range(row_count) for c in range(column_count)
        }:
            raise ReportCompilationError(
                "cell_bindings do not cover every (view_row, view_column) coordinate exactly once"
            )
        if seen_value_indices != set(range(expected_cells)):
            raise ReportCompilationError(
                "cell_binding value_index values are not exactly {0, ..., n-1} for n cells"
            )


def _identity_payload_fields(
    *,
    adapted_report_id: str,
    adapter_contract_id: str,
    compatibility: CompatibilityInfo,
    core_refs: CoreArtifactRefs,
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
        "compatibility_id": compatibility.compatibility_id,
        "compatibility_schema_id": compatibility.compatibility_schema_id,
        "missing_value_semantics": compatibility.missing_value_semantics,
        "score_table_ref": _artifact_ref_payload(core_refs.score_table_ref),
        "layout_recipe_ref": _artifact_ref_payload(core_refs.layout_recipe_ref),
        "vpm_artifact_ref": _artifact_ref_payload(core_refs.vpm_artifact_ref),
        "subjects": [subject.payload() for subject in subjects],
        "dimensions": [dimension.payload() for dimension in dimensions],
        "cell_bindings": [cell.payload() for cell in cell_bindings],
    }


def compiled_report_identity_payload(compiled: CompiledReportArtifactDTO) -> dict:
    """The exact canonical payload `artifact_ref.artifact_id` covers."""
    return _identity_payload_fields(
        adapted_report_id=compiled.adapted_report_id,
        adapter_contract_id=compiled.adapter_contract_id,
        compatibility=CompatibilityInfo(
            compatibility_id=compiled.compatibility_id,
            compatibility_schema_id=compiled.compatibility_schema_id,
            missing_value_semantics=compiled.missing_value_semantics,
        ),
        core_refs=CoreArtifactRefs(
            score_table_ref=compiled.score_table_ref,
            layout_recipe_ref=compiled.layout_recipe_ref,
            vpm_artifact_ref=compiled.vpm_artifact_ref,
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
    compatibility: CompatibilityInfo,
    core_refs: CoreArtifactRefs,
    subjects: Tuple[AdaptedSubjectDTO, ...],
    dimensions: Tuple[AdaptedDimensionDTO, ...],
    cell_bindings: Tuple[CellBindingDTO, ...],
    artifact_kind: str = COMPILED_REPORT_ARTIFACT_KIND,
    spec_version: str = SPEC_VERSION,
) -> str:
    payload = _identity_payload_fields(
        adapted_report_id=adapted_report_id,
        adapter_contract_id=adapter_contract_id,
        compatibility=compatibility,
        core_refs=core_refs,
        subjects=subjects,
        dimensions=dimensions,
        cell_bindings=cell_bindings,
        artifact_kind=artifact_kind,
        spec_version=spec_version,
    )
    return sha256_digest(canonical_json_bytes(payload))
