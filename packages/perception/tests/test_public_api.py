from __future__ import annotations

from zeromodel.perception import (
    PERCEPTION_PACKAGE_VERSION,
    PERCEPTION_STAGE,
    DiscreteActionSchemaDTO,
    SourceImageEncoderSpecDTO,
)


def test_phase_one_public_contract() -> None:
    assert PERCEPTION_PACKAGE_VERSION == "1.0.13"
    assert PERCEPTION_STAGE == "P1"
    assert SourceImageEncoderSpecDTO().color_space == "RGB"
    assert DiscreteActionSchemaDTO.from_labels(["RIGHT", "LEFT"]).labels == (
        "LEFT",
        "RIGHT",
    )
