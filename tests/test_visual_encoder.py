from __future__ import annotations

import numpy as np
import pytest

from zeromodel.core.artifact import VPMValidationError
from research.visual.visual_encoder import EncoderManifest, square_letterbox_uint8


def _manifest() -> EncoderManifest:
    return EncoderManifest(
        provider_kind="fixture_encoder",
        model_id="fixture/model",
        revision="abc123",
        architecture="fixture",
        weights_digest="sha256:weights",
        preprocessing_digest="sha256:preprocess",
        output_dimension=4,
        normalization="l2",
        framework="numpy",
        framework_version="fixture",
        license_id="MIT",
        source_record="fixture",
        metadata={"output": "global"},
    )


def test_encoder_manifest_is_identity_bearing_and_round_trips() -> None:
    manifest = _manifest()
    restored = EncoderManifest.from_dict(manifest.to_dict())

    assert restored.manifest_id == manifest.manifest_id
    assert restored.to_dict() == manifest.to_dict()


def test_encoder_manifest_rejects_tamper_and_invalid_dimension() -> None:
    payload = _manifest().to_dict()
    payload["revision"] = "different"
    with pytest.raises(VPMValidationError, match="manifest id mismatch"):
        EncoderManifest.from_dict(payload)

    with pytest.raises(VPMValidationError, match="output_dimension"):
        EncoderManifest(
            provider_kind="fixture",
            model_id="fixture",
            revision="v1",
            architecture="fixture",
            weights_digest="digest",
            preprocessing_digest="digest",
            output_dimension=0,
            normalization="l2",
            framework="numpy",
            framework_version="fixture",
            license_id="MIT",
            source_record="fixture",
        )


def test_square_letterbox_preserves_every_source_pixel_without_aliasing() -> None:
    source = np.arange(2 * 6 * 3, dtype=np.uint8).reshape(2, 6, 3)
    before = source.copy()

    canvas = square_letterbox_uint8(source, canvas_side=10, fill=7)

    assert canvas.shape == (10, 10, 3)
    assert canvas.flags.writeable is False
    assert np.array_equal(canvas[4:6, 2:8], source)
    assert np.all(canvas[:4] == 7)
    assert np.array_equal(source, before)
    assert source.flags.writeable

    source[0, 0, 0] = 255
    assert canvas[4, 2, 0] != 255


def test_square_letterbox_rejects_crop_and_invalid_rgb_contract() -> None:
    rgb = np.zeros((2, 6, 3), dtype=np.uint8)
    with pytest.raises(VPMValidationError, match="cannot crop"):
        square_letterbox_uint8(rgb, canvas_side=5)
    with pytest.raises(VPMValidationError, match="HxWx3"):
        square_letterbox_uint8(np.zeros((2, 6), dtype=np.uint8), canvas_side=6)
