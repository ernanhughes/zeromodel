from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

import zeromodel.video_action_set_benchmark as benchmark
from test_video_observation_rmdto import _pixels, sample_record
from zeromodel.artifact import VPMValidationError
from zeromodel.domains.video_action_set.canonical_json import canonical_sha256
from zeromodel.domains.video_action_set.contracts import TRANSFORMATION_FAMILY_VERSION
from zeromodel.domains.video_action_set.observation_dto import (
    MaterializedObservationDTO,
)
from zeromodel.domains.video_action_set.pixel_digest import (
    array_digest,
    pixel_digest,
    pixel_digest_from_bytes,
)
from zeromodel.domains.video_action_set.transformations import (
    _apply_transformation,
    _transformation_contract,
    _transformation_parameters,
    _translation_values_for_seed,
    _validate_transformation_parameters,
)


EXPECTED_TRANSFORMATION_CONTRACT = {
    "version": "zeromodel-video-action-set-transformation-family/v1",
    "families": {
        "exact": {
            "classification": "valid",
            "dx": [0, 0],
            "dy": [0, 0],
            "scale_percent": [100, 100],
            "offset": [0, 0],
            "occlusion": False,
        },
        "bounded_translation": {
            "classification": "valid",
            "dx": [-1, 1],
            "dy": [-1, 1],
            "scale_percent": [100, 100],
            "offset": [0, 0],
            "occlusion": False,
        },
        "bounded_photometric": {
            "classification": "valid",
            "dx": [0, 0],
            "dy": [0, 0],
            "scale_percent": [90, 105],
            "offset": [0, 5],
            "occlusion": False,
        },
        "bounded_translation_photometric": {
            "classification": "valid",
            "dx": [-1, 1],
            "dy": [-1, 1],
            "scale_percent": [90, 105],
            "offset": [0, 5],
            "occlusion": False,
        },
        "bounded_translation_occlusion": {
            "classification": "valid",
            "dx": [-1, 1],
            "dy": [-1, 1],
            "scale_percent": [100, 100],
            "offset": [0, 0],
            "occlusion": True,
        },
        "compound_bounded": {
            "classification": "valid",
            "dx": [-1, 1],
            "dy": [-1, 1],
            "scale_percent": [90, 105],
            "offset": [0, 5],
            "occlusion": True,
        },
    },
    "occlusion_bounds": {
        "top": [0, 2],
        "left": [0, 2],
        "height": 2,
        "width": 3,
        "value": 64,
    },
}

EXPECTED_CONTRACT_DIGEST = (
    "sha256:b7159f26df7e721680edfe9d52948797a10afa0f4c7dafc7396baacaf3eb2450"
)

EXPECTED_PARAMETERS = {
    "exact": {
        "version": "zeromodel-video-action-set-transformation-family/v1",
        "family": "exact",
        "seed": 12345,
        "dx": 0,
        "dy": 0,
        "scale_percent": 100,
        "offset": 0,
        "occlusion": None,
        "parameter_digest": (
            "sha256:7ad8100c9edae0dbbb370d41d6c2f04b9d95dab4ce2f0a371477649c62311c32"
        ),
    },
    "bounded_translation": {
        "version": "zeromodel-video-action-set-transformation-family/v1",
        "family": "bounded_translation",
        "seed": 12345,
        "dx": 0,
        "dy": 1,
        "scale_percent": 100,
        "offset": 0,
        "occlusion": None,
        "parameter_digest": (
            "sha256:322ef2669264ea17418cc65b8657b39fa964b51ce53ef11949ee866e6d8a10ad"
        ),
    },
    "bounded_photometric": {
        "version": "zeromodel-video-action-set-transformation-family/v1",
        "family": "bounded_photometric",
        "seed": 12345,
        "dx": 0,
        "dy": 0,
        "scale_percent": 103,
        "offset": 5,
        "occlusion": None,
        "parameter_digest": (
            "sha256:31331886c5f37c7326441bbf70c5d0b759b380f132cb6ae8e7394d351c6509c1"
        ),
    },
    "bounded_translation_photometric": {
        "version": "zeromodel-video-action-set-transformation-family/v1",
        "family": "bounded_translation_photometric",
        "seed": 12345,
        "dx": 0,
        "dy": 1,
        "scale_percent": 103,
        "offset": 5,
        "occlusion": None,
        "parameter_digest": (
            "sha256:19f71181a4ed8077aff31a318c3033358c467667d382aa2351d06b7e6aae20ce"
        ),
    },
    "bounded_translation_occlusion": {
        "version": "zeromodel-video-action-set-transformation-family/v1",
        "family": "bounded_translation_occlusion",
        "seed": 12345,
        "dx": 0,
        "dy": 1,
        "scale_percent": 100,
        "offset": 0,
        "occlusion": {"top": 1, "left": 2, "height": 2, "width": 3, "value": 64},
        "parameter_digest": (
            "sha256:c4d6388f784d81c8ad6286a703396719d2a9cb3016ff8cff894e94fc5c60dafe"
        ),
    },
    "compound_bounded": {
        "version": "zeromodel-video-action-set-transformation-family/v1",
        "family": "compound_bounded",
        "seed": 12345,
        "dx": 0,
        "dy": 1,
        "scale_percent": 103,
        "offset": 5,
        "occlusion": {"top": 0, "left": 1, "height": 2, "width": 3, "value": 64},
        "parameter_digest": (
            "sha256:a2c0c56598a4c1666d05ca76c0782996a5183f2c2c726f26ef1246aabed1f8c3"
        ),
    },
}

OUTPUT_GOLDENS = {
    "exact": {
        "bytes": "000102030405060708090a0b0c0d0e0f10111213",
        "digest": "sha256:e7aebf577f60412f0312d442c70a1fa6148c090bf5bab404caec29482ae779e8",
        "changed": 0,
    },
    "bounded_translation": {
        "bytes": "0000000000000102030405060708090a0b0c0d0e",
        "digest": "sha256:debdac7459ccb5129c1580891fdb80f5e265122cc26db0b3b8e27aed95f78949",
        "changed": 19,
    },
    "bounded_photometric": {
        "bytes": "05060708090a0b0c0d0e0f101112131415171819",
        "digest": "sha256:772f6f803b902cef207fe455f849aad1c69f6880e9fce0f4c265664133176dd7",
        "changed": 20,
    },
    "bounded_translation_photometric": {
        "bytes": "050505050505060708090a0b0c0d0e0f10111213",
        "digest": "sha256:ce88113f7db54c5dda0e5127290938d36ba9ef951e8046d18e7d50bdbb9b64c0",
        "changed": 5,
    },
    "bounded_translation_occlusion": {
        "bytes": "0000000000000140404005064040400a0b0c0d0e",
        "digest": "sha256:33b6b91a06ffd9ae0a6874f3ef6f825899c9d9b48c2126220fc2c62e7e9713b7",
        "changed": 19,
    },
    "compound_bounded": {
        "bytes": "054040400505404040090a0b0c0d0e0f10111213",
        "digest": "sha256:bd98bb3cd4a51d1b76d380ea09669391507e49667f955234f0a26e4dc443e999",
        "changed": 8,
    },
}

SOURCE_DIGEST = (
    "sha256:e7aebf577f60412f0312d442c70a1fa6148c090bf5bab404caec29482ae779e8"
)


def test_transformation_contract_payload_is_frozen() -> None:
    contract = _transformation_contract()

    assert TRANSFORMATION_FAMILY_VERSION == (
        "zeromodel-video-action-set-transformation-family/v1"
    )
    assert contract == EXPECTED_TRANSFORMATION_CONTRACT
    assert canonical_sha256(contract) == EXPECTED_CONTRACT_DIGEST
    assert set(contract["families"]) == {
        "exact",
        "bounded_translation",
        "bounded_photometric",
        "bounded_translation_photometric",
        "bounded_translation_occlusion",
        "compound_bounded",
    }
    assert contract["occlusion_bounds"] == {
        "top": [0, 2],
        "left": [0, 2],
        "height": 2,
        "width": 3,
        "value": 64,
    }


def test_transformation_parameters_are_frozen_for_fixed_seed() -> None:
    assert _translation_values_for_seed(12345) == (0, 1)
    for family, expected in EXPECTED_PARAMETERS.items():
        assert _transformation_parameters(family, 12345) == expected

    with pytest.raises(VPMValidationError, match="unsupported transformation family"):
        _transformation_parameters("unsupported", 12345)


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda item: item.__setitem__("family", "unsupported"),
            "unsupported transformation family",
        ),
        (lambda item: item.__setitem__("dx", 2), "transformation dx out of bounds"),
        (lambda item: item.__setitem__("dy", 2), "transformation dy out of bounds"),
        (
            lambda item: item.__setitem__("scale_percent", 106),
            "transformation scale out of bounds",
        ),
        (
            lambda item: item.__setitem__("offset", 6),
            "transformation offset out of bounds",
        ),
    ],
)
def test_transformation_validation_rejects_scalar_bounds(mutator, message: str) -> None:
    params = deepcopy(EXPECTED_PARAMETERS["bounded_translation_photometric"])
    mutator(params)

    with pytest.raises(VPMValidationError, match=message):
        _validate_transformation_parameters(params)


@pytest.mark.parametrize(
    ("params", "message", "image_shape"),
    [
        (
            EXPECTED_PARAMETERS["bounded_translation_occlusion"] | {"occlusion": None},
            "occlusion parameters required",
            (16, 28),
        ),
        (
            EXPECTED_PARAMETERS["exact"]
            | {
                "occlusion": {"top": 0, "left": 0, "height": 2, "width": 3, "value": 64}
            },
            "occlusion parameters supplied for non-occlusion family",
            (16, 28),
        ),
        (
            {
                **EXPECTED_PARAMETERS["bounded_translation_occlusion"],
                "occlusion": {
                    "top": 3,
                    "left": 2,
                    "height": 2,
                    "width": 3,
                    "value": 64,
                },
            },
            "occlusion top out of bounds",
            (16, 28),
        ),
        (
            {
                **EXPECTED_PARAMETERS["bounded_translation_occlusion"],
                "occlusion": {
                    "top": 1,
                    "left": 2,
                    "height": 3,
                    "width": 3,
                    "value": 64,
                },
            },
            "occlusion dimensions or value out of bounds",
            (16, 28),
        ),
        (
            EXPECTED_PARAMETERS["bounded_translation_occlusion"],
            "occlusion exceeds image bounds",
            (2, 5),
        ),
        (
            EXPECTED_PARAMETERS["exact"] | {"parameter_digest": "sha256:" + "0" * 64},
            "foreign transformation parameter digest",
            (16, 28),
        ),
    ],
)
def test_transformation_validation_rejects_occlusion_and_digest(
    params: dict[str, object],
    message: str,
    image_shape: tuple[int, int],
) -> None:
    with pytest.raises(VPMValidationError, match=message):
        _validate_transformation_parameters(deepcopy(params), image_shape=image_shape)


@pytest.mark.parametrize(
    "family",
    [
        "exact",
        "bounded_translation",
        "bounded_photometric",
        "bounded_translation_photometric",
        "bounded_translation_occlusion",
        "compound_bounded",
    ],
)
def test_transformation_output_bytes_and_trace_are_frozen(family: str) -> None:
    source = np.arange(20, dtype=np.uint8).reshape(4, 5)
    result, trace = _apply_transformation(source, EXPECTED_PARAMETERS[family])
    expected = OUTPUT_GOLDENS[family]
    expected_bytes = bytes.fromhex(expected["bytes"])

    assert result.dtype == np.uint8
    assert result.flags.c_contiguous
    assert result.shape == source.shape
    assert result.tobytes(order="C") == expected_bytes
    assert trace == {
        "source_observation_digest": SOURCE_DIGEST,
        "transformed_observation_digest": expected["digest"],
        "transformation_parameter_digest": EXPECTED_PARAMETERS[family][
            "parameter_digest"
        ],
        "changed_pixel_count": expected["changed"],
    }
    assert tuple(trace) == (
        "source_observation_digest",
        "transformed_observation_digest",
        "transformation_parameter_digest",
        "changed_pixel_count",
    )
    assert array_digest(source) == SOURCE_DIGEST
    assert pixel_digest_from_bytes(expected_bytes) == expected["digest"]


def test_pixel_digest_uses_contiguous_c_order_bytes_for_views() -> None:
    view = np.arange(24, dtype=np.uint8).reshape(4, 6)[:, ::2]

    assert not view.flags.c_contiguous
    assert view.tolist() == [[0, 2, 4], [6, 8, 10], [12, 14, 16], [18, 20, 22]]
    assert pixel_digest(None) is None
    assert array_digest(view) == (
        "sha256:eeab3b7b19f2560d5b64621880ab7f06559c9567526cd513a1ad0e4d55092335"
    )
    assert pixel_digest_from_bytes(bytes.fromhex("00020406080a0c0e10121416")) == (
        "sha256:eeab3b7b19f2560d5b64621880ab7f06559c9567526cd513a1ad0e4d55092335"
    )


def test_legacy_benchmark_facade_exposes_moved_symbols() -> None:
    source = np.arange(20, dtype=np.uint8).reshape(4, 5)

    assert benchmark.TRANSFORMATION_FAMILY_VERSION == TRANSFORMATION_FAMILY_VERSION
    assert benchmark._pixel_digest(source) == pixel_digest(source)
    assert benchmark._array_digest(source) == array_digest(source)
    assert benchmark._transformation_contract() == _transformation_contract()
    assert benchmark._translation_values_for_seed(
        12345
    ) == _translation_values_for_seed(12345)
    assert benchmark._transformation_parameters(
        "compound_bounded", 12345
    ) == _transformation_parameters("compound_bounded", 12345)
    benchmark_output, benchmark_trace = benchmark._apply_transformation(
        source,
        EXPECTED_PARAMETERS["compound_bounded"],
    )
    output, trace = _apply_transformation(
        source,
        EXPECTED_PARAMETERS["compound_bounded"],
    )
    np.testing.assert_array_equal(benchmark_output, output)
    assert benchmark_trace == trace


def test_materialized_observation_uses_historical_pixel_digest() -> None:
    record = sample_record(pixels=_pixels())

    assert record["observation_pixel_digest"] == pixel_digest(record["pixels"])
    item = MaterializedObservationDTO.from_record(record)

    assert item.matrix_blob is not None
    assert item.observation.observation_pixel_digest == pixel_digest(record["pixels"])
    assert (
        pixel_digest_from_bytes(item.matrix_blob.data)
        == record["observation_pixel_digest"]
    )
    assert (
        item.to_record(include_pixels=False)["observation_pixel_digest"]
        == record["observation_pixel_digest"]
    )


@pytest.mark.parametrize("record_pixels_kind", ["list", "int16"])
def test_materialized_observation_normalizes_record_pixels_to_uint8(
    record_pixels_kind: str,
) -> None:
    pixels = _pixels()
    record = sample_record(pixels=pixels)
    if record_pixels_kind == "list":
        record["pixels"] = pixels.tolist()
    else:
        record["pixels"] = pixels.astype(np.int16)

    item = MaterializedObservationDTO.from_record(record)

    assert item.matrix_blob is not None
    restored = item.matrix_blob.to_array()
    assert restored.dtype == np.uint8
    np.testing.assert_array_equal(restored, pixels)
    assert (
        item.observation.observation_pixel_digest == record["observation_pixel_digest"]
    )
