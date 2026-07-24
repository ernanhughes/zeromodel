from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zeromodel.perception import (
    PerceptionFieldError,
    SourceImageEncoderSpecDTO,
    build_grid_field_schema,
    encode_source_array,
    extract_source_fields,
    mask_source_fields,
    reconstruct_source_array,
)


def _source(values: np.ndarray, *, color_space: str = "L"):
    return encode_source_array(
        values,
        SourceImageEncoderSpecDTO(color_space=color_space),
    )


def test_grid_schema_is_deterministic_and_edge_aware() -> None:
    source = _source(np.arange(15, dtype=np.uint8).reshape(3, 5))

    first = build_grid_field_schema(source, tile_width=2, tile_height=2)
    second = build_grid_field_schema(source, tile_width=2, tile_height=2)

    assert first == second
    assert first.field_schema_id == second.field_schema_id
    assert len(first.fields) == 6
    assert sorted((field.width, field.height) for field in first.fields) == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 1),
        (2, 2),
        (2, 2),
    ]


def test_extract_and_reconstruct_are_exact_for_grayscale() -> None:
    values = np.arange(35, dtype=np.uint8).reshape(5, 7)
    source = _source(values)
    schema = build_grid_field_schema(source, tile_width=3, tile_height=2)

    samples = extract_source_fields(source, schema)
    reconstructed = reconstruct_source_array(reversed(samples), schema)

    np.testing.assert_array_equal(reconstructed, values)
    assert tuple(sample.field_id for sample in samples) == tuple(
        field.field_id for field in schema.fields
    )


def test_separate_channel_partition_reconstructs_rgb_exactly() -> None:
    values = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    source = _source(values, color_space="RGB")
    schema = build_grid_field_schema(
        source,
        tile_width=3,
        tile_height=3,
        channel_mode="separate",
    )

    assert len(schema.fields) == 12
    assert all(field.channels == 1 for field in schema.fields)
    np.testing.assert_array_equal(
        reconstruct_source_array(extract_source_fields(source, schema), schema),
        values,
    )


def test_joint_channel_partition_uses_one_field_per_spatial_tile() -> None:
    values = np.zeros((4, 4, 3), dtype=np.uint8)
    source = _source(values, color_space="RGB")
    schema = build_grid_field_schema(
        source,
        tile_width=2,
        tile_height=2,
        channel_mode="joint",
    )

    assert len(schema.fields) == 4
    assert all(field.channels == 3 for field in schema.fields)


def test_keep_and_remove_masks_are_exact_complements_for_selected_field() -> None:
    values = np.arange(16, dtype=np.uint8).reshape(4, 4)
    source = _source(values)
    schema = build_grid_field_schema(source, tile_width=2, tile_height=2)
    selected = schema.fields[0]

    kept = mask_source_fields(
        source,
        schema,
        [selected.field_id],
        mode="keep",
        neutral_value=0,
    )
    removed = mask_source_fields(
        source,
        schema,
        [selected.field_id],
        mode="remove",
        neutral_value=0,
    )

    selected_mask = np.zeros_like(values, dtype=bool)
    selected_mask[selected.y0 : selected.y1, selected.x0 : selected.x1] = True
    np.testing.assert_array_equal(kept[selected_mask], values[selected_mask])
    assert np.all(kept[~selected_mask] == 0)
    assert np.all(removed[selected_mask] == 0)
    np.testing.assert_array_equal(removed[~selected_mask], values[~selected_mask])


def test_schema_rejects_invalid_partition_parameters() -> None:
    source = _source(np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(PerceptionFieldError, match="tile dimensions"):
        build_grid_field_schema(source, tile_width=0, tile_height=1)
    with pytest.raises(PerceptionFieldError, match="channel_mode"):
        build_grid_field_schema(
            source,
            tile_width=1,
            tile_height=1,
            channel_mode="unknown",
        )


def test_schema_rejects_incompatible_source_shape_and_spec() -> None:
    source = _source(np.zeros((2, 2), dtype=np.uint8))
    schema = build_grid_field_schema(source, tile_width=1, tile_height=1)
    changed_shape = _source(np.zeros((3, 2), dtype=np.uint8))
    changed_spec = encode_source_array(
        np.zeros((2, 2, 3), dtype=np.uint8),
        SourceImageEncoderSpecDTO(color_space="RGB"),
    )

    with pytest.raises(PerceptionFieldError, match="shape"):
        extract_source_fields(changed_shape, schema)
    with pytest.raises(PerceptionFieldError, match="encoder spec"):
        extract_source_fields(changed_spec, schema)


def test_reconstruction_rejects_missing_or_tampered_samples() -> None:
    source = _source(np.arange(9, dtype=np.uint8).reshape(3, 3))
    schema = build_grid_field_schema(source, tile_width=2, tile_height=2)
    samples = extract_source_fields(source, schema)

    with pytest.raises(PerceptionFieldError, match="exactly one"):
        reconstruct_source_array(samples[:-1], schema)

    tampered = replace(samples[0], values=bytes([0]) * len(samples[0].values))
    with pytest.raises(PerceptionFieldError, match="digest"):
        reconstruct_source_array((tampered, *samples[1:]), schema)


def test_mask_rejects_unknown_field_and_invalid_neutral_value() -> None:
    source = _source(np.zeros((2, 2), dtype=np.uint8))
    schema = build_grid_field_schema(source, tile_width=1, tile_height=1)

    with pytest.raises(PerceptionFieldError, match="unknown field"):
        mask_source_fields(source, schema, ["sha256:missing"], mode="keep")
    with pytest.raises(PerceptionFieldError, match="neutral_value"):
        mask_source_fields(
            source,
            schema,
            [schema.fields[0].field_id],
            mode="remove",
            neutral_value=256,
        )
