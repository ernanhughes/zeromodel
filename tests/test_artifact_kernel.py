from __future__ import annotations

import json

import numpy as np
import pytest

from zeromodel.core.artifact import (
    LayoutRecipe,
    ScoreTable,
    VPMArtifact,
    build_vpm,
)
from zeromodel.core.artifact import VPMValidationError


GOLDEN_SAMPLE_ARTIFACT_ID = "32f801671139b73e349c756570c27c06d39c422a4d9a277782e1c997a473083b"


def quality_recipe() -> LayoutRecipe:
    return LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "quality-first",
            "row_order": {
                "kind": "lexicographic",
                "keys": [
                    {"metric_id": "quality", "direction": "desc"},
                    {"metric_id": "uncertainty", "direction": "asc"},
                ],
                "tie_break": "row_id",
            },
            "column_order": {
                "kind": "explicit",
                "metric_ids": ["quality", "uncertainty"],
            },
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )


def sample_table() -> ScoreTable:
    return ScoreTable(
        values=[
            [0.9, 0.1],
            [0.4, 0.9],
            [0.7, 0.2],
        ],
        row_ids=["b", "a", "c"],
        metric_ids=["quality", "uncertainty"],
        metadata={"source": "unit-test"},
    )


def test_build_vpm_is_deterministic() -> None:
    first = build_vpm(sample_table(), quality_recipe())
    second = build_vpm(sample_table(), quality_recipe())

    assert first.artifact_id == second.artifact_id
    assert first.compute_artifact_id() == first.artifact_id
    assert first.normalized_values.flags.writeable is False


def test_golden_artifact_id_pins_public_identity_contract() -> None:
    artifact = build_vpm(sample_table(), quality_recipe())

    assert artifact.artifact_id == GOLDEN_SAMPLE_ARTIFACT_ID


def test_cell_maps_view_coordinates_to_source_coordinates() -> None:
    artifact = build_vpm(sample_table(), quality_recipe())

    assert artifact.row_order == (0, 2, 1)
    assert artifact.column_order == (0, 1)

    cell = artifact.cell(view_row=1, view_column=0)
    assert cell.row_id == "c"
    assert cell.metric_id == "quality"
    assert cell.source_row_index == 2
    assert cell.source_metric_index == 0
    assert cell.raw_value == pytest.approx(0.7)
    assert cell.normalized_value == pytest.approx(0.6)


def test_region_summary_uses_view_cells() -> None:
    artifact = build_vpm(sample_table(), quality_recipe())

    region = artifact.region(rows=slice(0, 2), columns=slice(0, 2))

    assert region.shape == (2, 2)
    assert [cell.row_id for cell in region.cells] == ["b", "b", "c", "c"]
    assert region.normalized_mean == pytest.approx(np.mean([1.0, 0.0, 0.6, 0.125]))


def test_ties_are_resolved_by_row_id() -> None:
    table = ScoreTable(
        values=[[0.5], [0.5], [0.5]],
        row_ids=["row-c", "row-a", "row-b"],
        metric_ids=["quality"],
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "tie-break",
            "row_order": {
                "kind": "lexicographic",
                "keys": [{"metric_id": "quality", "direction": "desc"}],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )

    artifact = build_vpm(table, recipe)

    assert artifact.row_order == (1, 2, 0)
    assert [artifact.cell(row, 0).row_id for row in range(3)] == ["row-a", "row-b", "row-c"]


def test_constant_columns_preserve_clipped_raw_signal() -> None:
    table = ScoreTable(
        values=[[1.0], [1.0], [1.0]],
        row_ids=["a", "b", "c"],
        metric_ids=["risk"],
    )
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "name": "constant-risk",
            "row_order": {"kind": "source", "tie_break": "row_id"},
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )

    artifact = build_vpm(table, recipe)

    assert artifact.normalized_values.tolist() == [[1.0], [1.0], [1.0]]


def test_score_table_rejects_duplicate_row_ids() -> None:
    with pytest.raises(VPMValidationError, match="Duplicate row"):
        ScoreTable(values=[[0.1], [0.2]], row_ids=["same", "same"], metric_ids=["quality"])


def test_score_table_rejects_non_finite_values_without_policy() -> None:
    with pytest.raises(VPMValidationError, match="finite"):
        ScoreTable(values=[[float("nan")]], row_ids=["row"], metric_ids=["quality"])


def test_score_table_rejects_non_json_metadata_scalars() -> None:
    with pytest.raises(VPMValidationError, match="plain JSON scalar"):
        ScoreTable(values=[[0.1]], row_ids=["row"], metric_ids=["quality"], metadata={"bad": np.int64(1)})


def test_recipe_rejects_unknown_shape_kinds() -> None:
    with pytest.raises(VPMValidationError, match="Unsupported row_order kind"):
        LayoutRecipe.from_dict(
            {
                "version": "vpm-layout/0",
                "row_order": {"kind": "magic", "tie_break": "row_id"},
                "column_order": {"kind": "source"},
                "normalization": {"kind": "per_metric_minmax", "clip": True},
            }
        )


def test_build_rejects_unknown_metrics() -> None:
    recipe = LayoutRecipe.from_dict(
        {
            "version": "vpm-layout/0",
            "row_order": {
                "kind": "lexicographic",
                "keys": [{"metric_id": "missing", "direction": "desc"}],
                "tie_break": "row_id",
            },
            "column_order": {"kind": "source"},
            "normalization": {"kind": "per_metric_minmax", "clip": True},
        }
    )

    with pytest.raises(VPMValidationError, match="Unknown metric_id"):
        build_vpm(sample_table(), recipe)


def test_artifact_round_trips_through_json_dict() -> None:
    artifact = build_vpm(sample_table(), quality_recipe())
    payload = json.loads(json.dumps(artifact.to_dict(), sort_keys=True))

    loaded = VPMArtifact.from_dict(payload)

    assert loaded.artifact_id == artifact.artifact_id
    assert loaded.row_order == artifact.row_order
    assert loaded.column_order == artifact.column_order
    assert loaded.cell(0, 0) == artifact.cell(0, 0)


def test_artifact_rejects_tampered_identity() -> None:
    artifact = build_vpm(sample_table(), quality_recipe())
    payload = artifact.to_dict()
    payload["artifact_id"] = "not-the-real-id"

    with pytest.raises(VPMValidationError, match="artifact_id mismatch"):
        VPMArtifact.from_dict(payload)
