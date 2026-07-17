from __future__ import annotations

import pytest

from zeromodel.artifact import VPMValidationError
from zeromodel.visual_encoder import EncoderManifest


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
