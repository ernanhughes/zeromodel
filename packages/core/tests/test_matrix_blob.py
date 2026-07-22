from __future__ import annotations

import copy

import numpy as np
import pytest

from zeromodel.core import MatrixBlob, VPMValidationError


def test_matrix_blob_round_trip_preserves_identity_and_values() -> None:
    values = np.array([[-4, 0, 7], [12, -8, 3]], dtype=np.int8)
    blob = MatrixBlob.from_array(
        values, scale=0.125, zero_point=0, metadata={"kind": "prototype-bank"}
    )
    loaded = MatrixBlob.from_dict(blob.to_dict())

    assert loaded.blob_id == blob.blob_id
    assert loaded.dtype == "int8"
    assert loaded.shape == (2, 3)
    assert loaded.nbytes == values.nbytes
    assert np.array_equal(loaded.to_array(), values)
    assert loaded.to_array().flags.writeable is False
    assert np.allclose(
        loaded.to_array(dequantize=True), values.astype(np.float32) * 0.125
    )


def test_matrix_blob_identity_changes_with_metadata_or_quantization() -> None:
    values = np.arange(6, dtype=np.int8).reshape(2, 3)
    first = MatrixBlob.from_array(values, scale=0.1, metadata={"name": "a"})

    assert (
        first.blob_id
        != MatrixBlob.from_array(values, scale=0.1, metadata={"name": "b"}).blob_id
    )
    assert (
        first.blob_id
        != MatrixBlob.from_array(values, scale=0.2, metadata={"name": "a"}).blob_id
    )


def test_matrix_blob_rejects_tamper_and_invalid_quantization() -> None:
    blob = MatrixBlob.from_array(np.arange(4, dtype=np.uint8).reshape(2, 2))
    payload = copy.deepcopy(blob.to_dict())
    payload["data_base64"] = "AQIDBQ=="
    with pytest.raises(VPMValidationError, match="id mismatch"):
        MatrixBlob.from_dict(payload)
    with pytest.raises(VPMValidationError, match="positive and finite"):
        MatrixBlob.from_array(np.array([1], dtype=np.int8), scale=0.0)
    with pytest.raises(VPMValidationError, match="integer storage"):
        MatrixBlob.from_array(np.array([1.0], dtype=np.float32), scale=0.1)


def test_matrix_blob_float_identity_is_canonical_across_native_endianness() -> None:
    native = np.array([[0.25, -1.5]], dtype=np.float32)
    big_endian = native.astype(">f4")

    assert (
        MatrixBlob.from_array(native).blob_id
        == MatrixBlob.from_array(big_endian).blob_id
    )
