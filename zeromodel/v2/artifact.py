"""Immutable Visual Policy Map artifact kernel.

This module implements the smallest useful v2 ZeroModel surface:

- ``ScoreTable``: finite numeric values plus stable row and metric identifiers.
- ``LayoutRecipe``: explicit normalization and ordering instructions.
- ``VPMArtifact``: deterministic spatial view with cell-to-source mapping.

The artifact is deliberately not a decision policy. Consumers may evaluate a
region, threshold, router, or ranking rule outside this module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cmp_to_key
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np

SPEC_VERSION = "vpm-artifact/0"
LAYOUT_VERSION = "vpm-layout/0"


JsonMap = Mapping[str, Any]


class VPMValidationError(ValueError):
    """Raised when a VPM source, recipe, or artifact violates the v0 contract."""


def _freeze_json(value: Any) -> Any:
    """Return a JSON-like immutable representation for metadata/provenance."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze_json(v) for k, v in value.items()})
    if isinstance(value, list | tuple):
        return tuple(_freeze_json(v) for v in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return a normal JSON-serializable structure from a frozen structure."""
    if isinstance(value, Mapping):
        return {str(k): _thaw_json(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(v) for v in value]
    return value


def _canonical_json_bytes(value: Any) -> bytes:
    """Serialize semantic content with stable key and separator choices."""
    return json.dumps(
        _thaw_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_hex(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_unique(values: Sequence[str], kind: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in values:
        if item in seen:
            duplicates.append(item)
        seen.add(item)
    if duplicates:
        dupes = ", ".join(sorted(set(duplicates)))
        raise VPMValidationError(f"Duplicate {kind} identifiers: {dupes}")


@dataclass(frozen=True)
class ScoreTable:
    """Rectangular source score table with stable identifiers.

    Values are canonicalized to a copied ``float64`` NumPy array. The table is
    immutable at the object level; callers should treat the exposed array as
    read-only.
    """

    values: np.ndarray
    row_ids: tuple[str, ...]
    metric_ids: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __init__(
        self,
        values: Sequence[Sequence[float]] | np.ndarray,
        row_ids: Sequence[str],
        metric_ids: Sequence[str],
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        matrix = np.array(values, dtype=np.float64, copy=True)
        rows = tuple(str(row_id) for row_id in row_ids)
        metrics = tuple(str(metric_id) for metric_id in metric_ids)

        object.__setattr__(self, "values", matrix)
        object.__setattr__(self, "row_ids", rows)
        object.__setattr__(self, "metric_ids", metrics)
        object.__setattr__(self, "metadata", _freeze_json(metadata or {}))
        self.validate()

    def validate(self) -> None:
        if self.values.ndim != 2:
            raise VPMValidationError("ScoreTable values must be a two-dimensional matrix")
        if self.values.shape != (len(self.row_ids), len(self.metric_ids)):
            raise VPMValidationError(
                "ScoreTable shape must match row_ids and metric_ids "
                f"(shape={self.values.shape}, rows={len(self.row_ids)}, "
                f"metrics={len(self.metric_ids)})"
            )
        if not self.row_ids:
            raise VPMValidationError("ScoreTable must contain at least one row")
        if not self.metric_ids:
            raise VPMValidationError("ScoreTable must contain at least one metric")
        _validate_unique(self.row_ids, "row")
        _validate_unique(self.metric_ids, "metric")
        if not np.isfinite(self.values).all():
            raise VPMValidationError(
                "ScoreTable values must be finite unless a missing-value policy is declared"
            )
        self.values.flags.writeable = False

    @property
    def shape(self) -> tuple[int, int]:
        return self.values.shape

    @property
    def digest(self) -> str:
        return _sha256_hex(self.to_identity_payload())

    def metric_index(self, metric_id: str) -> int:
        try:
            return self.metric_ids.index(metric_id)
        except ValueError as exc:
            raise VPMValidationError(f"Unknown metric_id in recipe: {metric_id}") from exc

    def to_identity_payload(self) -> dict[str, Any]:
        return {
            "values": self.values.tolist(),
            "row_ids": list(self.row_ids),
            "metric_ids": list(self.metric_ids),
            "metadata": _thaw_json(self.metadata),
        }

    def to_dict(self) -> dict[str, Any]:
        return self.to_identity_payload()

    @classmethod
    def from_dict(cls, data: JsonMap) -> "ScoreTable":
        return cls(
            values=data["values"],
            row_ids=data["row_ids"],
            metric_ids=data["metric_ids"],
            metadata=data.get("metadata") or {},
        )


@dataclass(frozen=True)
class LayoutRecipe:
    """Explicit recipe for normalizing and spatially ordering a score table."""

    data: Mapping[str, Any]

    def __init__(self, data: Mapping[str, Any]) -> None:
        frozen = _freeze_json(data)
        object.__setattr__(self, "data", frozen)
        self.validate_shape()

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayoutRecipe":
        return cls(data)

    @property
    def name(self) -> str:
        return str(self.data.get("name") or "unnamed")

    @property
    def digest(self) -> str:
        return _sha256_hex(self.to_dict())

    def validate_shape(self) -> None:
        if self.data.get("version") != LAYOUT_VERSION:
            raise VPMValidationError(
                f"LayoutRecipe version must be {LAYOUT_VERSION!r}; got {self.data.get('version')!r}"
            )

        row_order = self.data.get("row_order")
        column_order = self.data.get("column_order")
        normalization = self.data.get("normalization")
        if not isinstance(row_order, Mapping):
            raise VPMValidationError("LayoutRecipe requires row_order mapping")
        if not isinstance(column_order, Mapping):
            raise VPMValidationError("LayoutRecipe requires column_order mapping")
        if not isinstance(normalization, Mapping):
            raise VPMValidationError("LayoutRecipe requires normalization mapping")

        row_kind = row_order.get("kind")
        if row_kind not in {"source", "lexicographic", "weighted_score"}:
            raise VPMValidationError(f"Unsupported row_order kind: {row_kind!r}")
        if row_kind in {"lexicographic", "weighted_score"}:
            keys = row_order.get("keys") or row_order.get("metrics")
            if not isinstance(keys, tuple) or len(keys) == 0:
                raise VPMValidationError(f"row_order kind {row_kind!r} requires non-empty keys/metrics")
            for key in keys:
                if not isinstance(key, Mapping):
                    raise VPMValidationError("row_order keys must be mappings")
                if "metric_id" not in key:
                    raise VPMValidationError("row_order key requires metric_id")
                if key.get("direction") not in {"asc", "desc"}:
                    raise VPMValidationError("row_order key direction must be 'asc' or 'desc'")
                if row_kind == "weighted_score" and "weight" not in key:
                    raise VPMValidationError("weighted_score row_order keys require weight")
        if "tie_break" not in row_order:
            raise VPMValidationError("row_order requires explicit tie_break")
        if row_order.get("tie_break") != "row_id":
            raise VPMValidationError("Only tie_break='row_id' is supported in v0")

        column_kind = column_order.get("kind")
        if column_kind not in {"source", "explicit"}:
            raise VPMValidationError(f"Unsupported column_order kind: {column_kind!r}")
        if column_kind == "explicit":
            metric_ids = column_order.get("metric_ids")
            if not isinstance(metric_ids, tuple) or len(metric_ids) == 0:
                raise VPMValidationError("explicit column_order requires non-empty metric_ids")
            _validate_unique([str(metric_id) for metric_id in metric_ids], "column metric")

        normalization_kind = normalization.get("kind")
        if normalization_kind != "per_metric_minmax":
            raise VPMValidationError(
                f"Unsupported normalization kind: {normalization_kind!r}"
            )

    def validate_against(self, table: ScoreTable) -> None:
        row_order = self.data["row_order"]
        row_kind = row_order["kind"]
        if row_kind in {"lexicographic", "weighted_score"}:
            keys = row_order.get("keys") or row_order.get("metrics")
            for key in keys:
                table.metric_index(str(key["metric_id"]))

        column_order = self.data["column_order"]
        if column_order["kind"] == "explicit":
            for metric_id in column_order["metric_ids"]:
                table.metric_index(str(metric_id))

    def to_dict(self) -> dict[str, Any]:
        return _thaw_json(self.data)


@dataclass(frozen=True)
class VPMCell:
    """Resolved view cell and its source coordinates."""

    view_row: int
    view_column: int
    source_row_index: int
    source_metric_index: int
    row_id: str
    metric_id: str
    raw_value: float
    normalized_value: float


@dataclass(frozen=True)
class VPMRegion:
    """Resolved region summary for a rectangular view slice."""

    rows: tuple[int, ...]
    columns: tuple[int, ...]
    cells: tuple[VPMCell, ...]

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.rows), len(self.columns))

    @property
    def normalized_mean(self) -> float:
        if not self.cells:
            raise VPMValidationError("Cannot summarize an empty region")
        return float(np.mean([cell.normalized_value for cell in self.cells]))


@dataclass(frozen=True)
class VPMArtifact:
    """Immutable deterministic Visual Policy Map artifact."""

    source: ScoreTable
    recipe: LayoutRecipe
    normalized_values: np.ndarray
    row_order: tuple[int, ...]
    column_order: tuple[int, ...]
    provenance: Mapping[str, Any]
    artifact_id: str
    spec_version: str = SPEC_VERSION

    def __init__(
        self,
        source: ScoreTable,
        recipe: LayoutRecipe,
        normalized_values: Sequence[Sequence[float]] | np.ndarray,
        row_order: Sequence[int],
        column_order: Sequence[int],
        provenance: Mapping[str, Any] | None = None,
        artifact_id: str | None = None,
        spec_version: str = SPEC_VERSION,
    ) -> None:
        matrix = np.array(normalized_values, dtype=np.float64, copy=True)
        rows = tuple(int(index) for index in row_order)
        columns = tuple(int(index) for index in column_order)
        frozen_provenance = _freeze_json(provenance or {})

        object.__setattr__(self, "source", source)
        object.__setattr__(self, "recipe", recipe)
        object.__setattr__(self, "normalized_values", matrix)
        object.__setattr__(self, "row_order", rows)
        object.__setattr__(self, "column_order", columns)
        object.__setattr__(self, "provenance", frozen_provenance)
        object.__setattr__(self, "spec_version", spec_version)

        computed_id = artifact_id or self.compute_artifact_id()
        object.__setattr__(self, "artifact_id", computed_id)
        self.validate()

    @property
    def shape(self) -> tuple[int, int]:
        return self.normalized_values.shape

    def validate(self) -> None:
        if self.spec_version != SPEC_VERSION:
            raise VPMValidationError(
                f"VPMArtifact spec_version must be {SPEC_VERSION!r}; got {self.spec_version!r}"
            )
        self.source.validate()
        self.recipe.validate_against(self.source)

        row_count, metric_count = self.source.shape
        if sorted(self.row_order) != list(range(row_count)):
            raise VPMValidationError("row_order must be a full permutation of source rows")
        if sorted(self.column_order) != list(range(metric_count)):
            raise VPMValidationError("column_order must be a full permutation of source metrics")
        if self.normalized_values.shape != (row_count, metric_count):
            raise VPMValidationError(
                "normalized_values must have source shape after view ordering "
                f"(shape={self.normalized_values.shape}, source={self.source.shape})"
            )
        if not np.isfinite(self.normalized_values).all():
            raise VPMValidationError("normalized_values must be finite")
        if self.normalized_values.size and (
            self.normalized_values.min() < -1e-12 or self.normalized_values.max() > 1 + 1e-12
        ):
            raise VPMValidationError("normalized_values must be in the [0, 1] range")

        expected_id = self.compute_artifact_id()
        if self.artifact_id != expected_id:
            raise VPMValidationError(
                f"artifact_id mismatch: expected {expected_id}, got {self.artifact_id}"
            )
        self.normalized_values.flags.writeable = False

    def compute_artifact_id(self) -> str:
        return _sha256_hex(self.to_identity_payload())

    def to_identity_payload(self) -> dict[str, Any]:
        return {
            "spec_version": self.spec_version,
            "source": self.source.to_identity_payload(),
            "layout_recipe": self.recipe.to_dict(),
            "view": {
                "normalized_values": self.normalized_values.tolist(),
                "row_order": list(self.row_order),
                "column_order": list(self.column_order),
            },
            "provenance": _thaw_json(self.provenance),
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.to_identity_payload()
        payload["artifact_id"] = self.artifact_id
        return payload

    @classmethod
    def from_dict(cls, data: JsonMap) -> "VPMArtifact":
        view = data["view"]
        return cls(
            source=ScoreTable.from_dict(data["source"]),
            recipe=LayoutRecipe.from_dict(data["layout_recipe"]),
            normalized_values=view["normalized_values"],
            row_order=view["row_order"],
            column_order=view["column_order"],
            provenance=data.get("provenance") or {},
            artifact_id=data.get("artifact_id"),
            spec_version=data.get("spec_version", SPEC_VERSION),
        )

    def cell(self, view_row: int, view_column: int) -> VPMCell:
        if not (0 <= view_row < len(self.row_order)):
            raise IndexError(f"view_row out of range: {view_row}")
        if not (0 <= view_column < len(self.column_order)):
            raise IndexError(f"view_column out of range: {view_column}")

        source_row_index = self.row_order[view_row]
        source_metric_index = self.column_order[view_column]
        return VPMCell(
            view_row=view_row,
            view_column=view_column,
            source_row_index=source_row_index,
            source_metric_index=source_metric_index,
            row_id=self.source.row_ids[source_row_index],
            metric_id=self.source.metric_ids[source_metric_index],
            raw_value=float(self.source.values[source_row_index, source_metric_index]),
            normalized_value=float(self.normalized_values[view_row, view_column]),
        )

    def region(self, rows: slice, columns: slice) -> VPMRegion:
        row_indices = tuple(range(*rows.indices(len(self.row_order))))
        column_indices = tuple(range(*columns.indices(len(self.column_order))))
        cells = tuple(
            self.cell(row, column)
            for row in row_indices
            for column in column_indices
        )
        if not row_indices or not column_indices:
            raise VPMValidationError("region must select at least one row and one column")
        return VPMRegion(rows=row_indices, columns=column_indices, cells=cells)


def _normalize_per_metric_minmax(table: ScoreTable, clip: bool) -> np.ndarray:
    values = table.values.astype(np.float64, copy=True)
    mins = values.min(axis=0)
    maxs = values.max(axis=0)
    ranges = maxs - mins
    normalized = np.zeros_like(values, dtype=np.float64)
    non_constant = ranges > 0
    normalized[:, non_constant] = (values[:, non_constant] - mins[non_constant]) / ranges[non_constant]
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    return normalized


def _compare_lexicographic(table: ScoreTable, keys: Sequence[Mapping[str, Any]], left: int, right: int) -> int:
    for key in keys:
        metric_index = table.metric_index(str(key["metric_id"]))
        left_value = float(table.values[left, metric_index])
        right_value = float(table.values[right, metric_index])
        if left_value < right_value:
            return -1 if key["direction"] == "asc" else 1
        if left_value > right_value:
            return 1 if key["direction"] == "asc" else -1
    return (table.row_ids[left] > table.row_ids[right]) - (table.row_ids[left] < table.row_ids[right])


def _weighted_score(table: ScoreTable, keys: Sequence[Mapping[str, Any]], row_index: int) -> float:
    total = 0.0
    for key in keys:
        metric_index = table.metric_index(str(key["metric_id"]))
        value = float(table.values[row_index, metric_index])
        direction = str(key["direction"])
        signed_value = value if direction == "desc" else -value
        total += float(key["weight"]) * signed_value
    return total


def _compile_row_order(table: ScoreTable, recipe: LayoutRecipe) -> tuple[int, ...]:
    row_order = recipe.data["row_order"]
    kind = row_order["kind"]
    row_indices = tuple(range(len(table.row_ids)))

    if kind == "source":
        return row_indices

    keys = tuple(row_order.get("keys") or row_order.get("metrics"))
    if kind == "lexicographic":
        return tuple(
            sorted(
                row_indices,
                key=cmp_to_key(lambda left, right: _compare_lexicographic(table, keys, left, right)),
            )
        )

    if kind == "weighted_score":
        return tuple(
            sorted(
                row_indices,
                key=lambda index: (-_weighted_score(table, keys, index), table.row_ids[index]),
            )
        )

    raise VPMValidationError(f"Unsupported row_order kind: {kind!r}")


def _compile_column_order(table: ScoreTable, recipe: LayoutRecipe) -> tuple[int, ...]:
    column_order = recipe.data["column_order"]
    kind = column_order["kind"]
    if kind == "source":
        return tuple(range(len(table.metric_ids)))
    if kind == "explicit":
        return tuple(table.metric_index(str(metric_id)) for metric_id in column_order["metric_ids"])
    raise VPMValidationError(f"Unsupported column_order kind: {kind!r}")


def build_vpm(
    score_table: ScoreTable,
    recipe: LayoutRecipe,
    provenance: Mapping[str, Any] | None = None,
) -> VPMArtifact:
    """Build an immutable VPM artifact from a score table and layout recipe."""
    score_table.validate()
    recipe.validate_against(score_table)

    normalization = recipe.data["normalization"]
    normalized_source = _normalize_per_metric_minmax(
        score_table,
        clip=bool(normalization.get("clip", True)),
    )
    row_order = _compile_row_order(score_table, recipe)
    column_order = _compile_column_order(score_table, recipe)
    normalized_view = normalized_source[np.ix_(row_order, column_order)]

    default_provenance = {
        "source_digest": f"sha256:{score_table.digest}",
        "recipe_digest": f"sha256:{recipe.digest}",
        "parents": [],
    }
    if provenance:
        default_provenance.update(dict(provenance))

    return VPMArtifact(
        source=score_table,
        recipe=recipe,
        normalized_values=normalized_view,
        row_order=row_order,
        column_order=column_order,
        provenance=default_provenance,
    )
