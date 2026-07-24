from __future__ import annotations

from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from zeromodel.perception import (
    DiscreteActionSchemaDTO,
    PerceptionRepresentationError,
    SourceImageEncoderSpecDTO,
    decode_discrete_action,
    encode_discrete_action,
    encode_source_array,
    encode_source_image_bytes,
)


def _png_payload(values: np.ndarray, mode: str) -> bytes:
    buffer = BytesIO()
    Image.fromarray(values, mode=mode).save(buffer, format="PNG")
    return buffer.getvalue()


def test_source_array_encoding_is_deterministic_and_round_trips() -> None:
    values = np.array(
        [
            [[0, 1, 2], [3, 4, 5]],
            [[6, 7, 8], [9, 10, 11]],
        ],
        dtype=np.uint8,
    )

    first = encode_source_array(values)
    second = encode_source_array(values.copy())

    assert first == second
    assert first.width == 2
    assert first.height == 2
    assert first.channels == 3
    assert first.color_space == "RGB"
    np.testing.assert_array_equal(first.to_array(), values)


def test_source_byte_and_array_paths_share_canonical_identity() -> None:
    values = np.array(
        [
            [[20, 30, 40], [50, 60, 70]],
            [[80, 90, 100], [110, 120, 130]],
        ],
        dtype=np.uint8,
    )

    from_array = encode_source_array(values)
    from_png = encode_source_image_bytes(_png_payload(values, "RGB"))

    assert from_png.source_vpm_id == from_array.source_vpm_id
    assert from_png.pixel_digest == from_array.pixel_digest
    assert from_png.png_bytes == from_array.png_bytes


def test_source_identity_changes_with_pixels_or_encoder_contract() -> None:
    values = np.zeros((2, 2, 3), dtype=np.uint8)
    changed = values.copy()
    changed[0, 0, 0] = 1

    baseline = encode_source_array(values)
    pixel_change = encode_source_array(changed)
    contract_change = encode_source_array(
        values,
        SourceImageEncoderSpecDTO(max_pixels=32),
    )

    assert baseline.source_vpm_id != pixel_change.source_vpm_id
    assert baseline.pixel_digest != pixel_change.pixel_digest
    assert baseline.source_vpm_id != contract_change.source_vpm_id
    assert baseline.pixel_digest == contract_change.pixel_digest


def test_source_encoding_rejects_wrong_dtype_shape_and_bounds() -> None:
    with pytest.raises(PerceptionRepresentationError, match="dtype uint8"):
        encode_source_array(np.zeros((2, 2, 3), dtype=np.float32))

    with pytest.raises(PerceptionRepresentationError, match="shape"):
        encode_source_array(np.zeros((2, 2), dtype=np.uint8))

    with pytest.raises(PerceptionRepresentationError, match="exceed"):
        encode_source_array(
            np.zeros((3, 3, 3), dtype=np.uint8),
            SourceImageEncoderSpecDTO(max_width=2),
        )

    payload = _png_payload(np.zeros((1, 1, 3), dtype=np.uint8), "RGB")
    with pytest.raises(PerceptionRepresentationError, match="payload size"):
        encode_source_image_bytes(
            payload,
            SourceImageEncoderSpecDTO(max_input_bytes=1),
        )


def test_source_encoding_rejects_invalid_image_bytes() -> None:
    with pytest.raises(PerceptionRepresentationError, match="supported image"):
        encode_source_image_bytes(b"not an image")


def test_action_schema_is_canonical_and_content_addressed() -> None:
    first = DiscreteActionSchemaDTO.from_labels(["RIGHT", "FIRE", "LEFT"])
    second = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT", "FIRE"])

    assert first.labels == ("FIRE", "LEFT", "RIGHT")
    assert first.action_schema_id == second.action_schema_id

    with pytest.raises(PerceptionRepresentationError, match="unique"):
        DiscreteActionSchemaDTO(labels=("LEFT", "LEFT"))

    with pytest.raises(PerceptionRepresentationError, match="canonical sorted"):
        DiscreteActionSchemaDTO(labels=("RIGHT", "LEFT"))


def test_discrete_action_encoding_is_deterministic_and_round_trips() -> None:
    schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT", "FIRE"])

    first = encode_discrete_action("RIGHT", schema)
    second = encode_discrete_action("RIGHT", schema)

    assert first == second
    assert first.width == 3
    assert first.height == 1
    assert first.channels == 1
    assert first.scores() == (0, 0, 255)
    assert decode_discrete_action(first, schema) == "RIGHT"


def test_action_encoding_rejects_unknown_label() -> None:
    schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])

    with pytest.raises(PerceptionRepresentationError, match="not present"):
        encode_discrete_action("FIRE", schema)


def test_action_decoding_rejects_digest_schema_and_metadata_tampering() -> None:
    schema = DiscreteActionSchemaDTO.from_labels(["LEFT", "RIGHT"])
    target = encode_discrete_action("LEFT", schema)

    with pytest.raises(PerceptionRepresentationError, match="digest mismatch"):
        decode_discrete_action(replace(target, png_bytes=target.png_bytes + b"x"), schema)

    other_schema = DiscreteActionSchemaDTO.from_labels(["FIRE", "LEFT", "RIGHT"])
    with pytest.raises(PerceptionRepresentationError, match="schema identity mismatch"):
        decode_discrete_action(target, other_schema)

    with pytest.raises(PerceptionRepresentationError, match="metadata disagrees"):
        decode_discrete_action(replace(target, action_label="RIGHT"), schema)
